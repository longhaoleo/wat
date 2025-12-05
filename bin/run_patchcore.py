# run_patchcore.py
from json import load
import os
import argparse
import logging
from typing import Dict, Any, Tuple, Callable, List, Optional

import numpy as np
import torch
from sklearn.metrics import classification_report, roc_auc_score
import csv

import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
from patchcore.evaluation import Evaluator, build_feature_matrix
from patchcore.datasets.tiny_genimage import Dataset, DatasetSplit
import patchcore.utils


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("run_patchcore")


# -----------------------------
# FAISS NN 选择
# -----------------------------
def choose_nn_method(
    nn: str,
    target_embed_dimension: int,
    est_num_features: int,
    faiss_on_gpu: bool,
    faiss_num_workers: int,
):
    """根据特征维度与样本量，选择最合适的 FAISS 索引实现。"""
    from patchcore import common as pc_common

    SMALL_N = 2000
    dim_ok_for_pq = (target_embed_dimension % 64 == 0)  # 本实现 PQ 的 M=64，需 d%64==0

    if nn == "flat":
        return pc_common.FaissNN(on_gpu=faiss_on_gpu, num_workers=faiss_num_workers)

    if nn == "ivfpq":
        if not dim_ok_for_pq or est_num_features < SMALL_N:
            LOGGER.warning(
                "Requested ivfpq 但 d%%64==0 和/或样本量不足未满足：d=%d, estN=%d。回退 Flat。",
                target_embed_dimension, est_num_features,
            )
            return pc_common.FaissNN(on_gpu=faiss_on_gpu, num_workers=faiss_num_workers)
        return pc_common.ApproximateFaissNN(on_gpu=faiss_on_gpu, num_workers=faiss_num_workers)

    # auto
    if dim_ok_for_pq and est_num_features >= SMALL_N:
        return pc_common.ApproximateFaissNN(on_gpu=faiss_on_gpu, num_workers=faiss_num_workers)
    else:
        return pc_common.FaissNN(on_gpu=faiss_on_gpu, num_workers=faiss_num_workers)


# -----------------------------
# 推断输入形状
# -----------------------------
def infer_input_shape(dl):
    """从任意 DataLoader 抽一批数据，推断 PatchCore 期望的 (C,H,W)。"""
    x = next(iter(dl))
    if isinstance(x, dict):
        x = x.get("image", next(iter(x.values())))
    if isinstance(x, (tuple, list)):
        x = x[0]
    if x.ndim == 4:
        return tuple(x.shape[1:4])  # (C,H,W)
    if x.ndim == 3:
        return tuple(x.shape)
    raise RuntimeError(f"Unexpected shape: {getattr(x,'shape',None)}")


# -----------------------------
# PatchCore 构造器
# -----------------------------
def get_patchcore(
    backbone_name: str,
    layers_to_extract_from: List[str],
    pretrain_embed_dimension: int,
    target_embed_dimension: int,
    patchsize: int,
    anomaly_scorer_k: int,
):
    def _builder(input_shape: Tuple[int, int, int], featuresampler, device: torch.device, nn_method):
        """闭包：按输入形状动态构造 PatchCore 并加载对应骨干网络。"""
        backbone = patchcore.backbones.load(backbone_name)
        backbone.name = backbone_name

        pc = patchcore.patchcore.PatchCore(device)
        pc.load(
            backbone=backbone,
            layers_to_extract_from=layers_to_extract_from,
            device=device,
            input_shape=input_shape,
            pretrain_embed_dimension=pretrain_embed_dimension,
            target_embed_dimension=target_embed_dimension,
            patchsize=patchsize,
            anomaly_score_num_nn=anomaly_scorer_k,
            featuresampler=featuresampler,
            nn_method=nn_method,
        )
        return pc
    return _builder


# -----------------------------
# Sampler 选择
# -----------------------------
def get_sampler(name: str, percentage: float, device: torch.device):
    """根据配置返回特征采样器，主要用于控制记忆库容量。"""
    name = name.lower()
    if name == "identity":
        return patchcore.sampler.IdentitySampler()
    if name == "greedy_coreset":
        return patchcore.sampler.GreedyCoresetSampler(percentage, device)
    if name == "approx_greedy_coreset":
        return patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)
    if name == "random":
        return patchcore.sampler.RandomSampler(percentage)
    # 中心采样：主体优先
    if name == "central":
        return patchcore.sampler.CentralSampler(percentage, use_mahalanobis=False)
    if name == "central_mahal":
        return patchcore.sampler.CentralSampler(percentage, use_mahalanobis=True)
    # 密度采样：高密度优先（可带后缀 n_neighbors，如 density:10）
    if name.startswith("density"):
        n_neighbors = 50
        if ":" in name:
            try:
                n_neighbors = int(name.split(":", 1)[1])
            except Exception:
                pass
        return patchcore.sampler.DensitySampler(percentage, n_neighbors=n_neighbors)
    # KMeans / TopC：kmeans_topc[:C]（默认 C=2），普通 kmeans 等同于 topc:1
    if name.startswith("kmeans_topc") or name == "kmeans":
        topk = 1
        if name.startswith("kmeans_topc"):
            topk = 2
            if ":" in name:
                try:
                    topk = int(name.split(":", 1)[1])
                except Exception:
                    pass
        km = patchcore.sampler.KMeansSampler(percentage, per_cluster_topk=topk)
        # 默认用 MiniBatchKMeans，提高大规模性能
        km.use_minibatch = True
        return km
    raise ValueError(f"Unknown sampler: {name}")


# -----------------------------
# 数据集封装：返回 get_dataloaders(seed, bankname)
# - 训练：仅 train/<bankname>
# - 验证/测试：优先 data_path/val 和 data_path/test；若只有其中之一，则对唯一集 50/50 划分
# -----------------------------
def get_dataloaders(
    data_path: str,
    dataset_names: List[str],
    batch_size: int,
    resize: int,
    imagesize: int,
    workers: int,
) -> Callable[[int, Optional[str]], Dict[str, Any]]:
    """生成一个工厂函数：按 seed/bankname 构建 train/val/test 三个 DataLoader。"""
    dataset_names = list(dataset_names or [])
    dataset_roots = {ds: os.path.join(data_path, ds) for ds in dataset_names}

    # 验证/测试集采用“公共”划分，同一 split 只生成一次索引，所有 bank 共用
    val_test_indices: Dict[Tuple[str, DatasetSplit], Tuple[np.ndarray, np.ndarray]] = {}

    def _build_loader_for_dataset(
        dataset_root: str,
        split: DatasetSplit,
        bankname: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        '''
        定义 _build_loader：按传入的 split/bankname/seed 实例化 Dataset，
        再包成 DataLoader；训练集根据 split 是否为 TRAIN 决定是否打乱（train 打乱，其它不打乱），
        drop_last=False 保留末尾不足批次的数据。
        '''
        ds = Dataset(
            source=dataset_root,
            split=split,
            bankname=bankname,
            resize=resize,
            imagesize=imagesize,
            seed=seed,
        )
        should_shuffle = bool(split == DatasetSplit.TRAIN)

        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=should_shuffle,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )
        return ds, dl

    def _get_val_test_indices(
        dataset_root: str,
        seed: Optional[int],
        split: DatasetSplit,
    ) -> Tuple[np.ndarray, np.ndarray]:
        '''
        为给定的 split（VAL 或 TEST 目录）计算并缓存一个“公共”的 50/50 切分索引，
        用于在没有同时存在 val/ 与 test/ 目录时，把同一目录均分成验证/测试两部分；
        所有 bank 共享这套索引，保证对齐与可重复。
        '''
        # 使用  nonlocal 关键字声明对外层变量的引用
        nonlocal val_test_indices
        key = (dataset_root, split)
        if key in val_test_indices:
            return val_test_indices[key]

        reference_ds, _ = _build_loader_for_dataset(dataset_root, split, bankname=None, seed=seed)
        # 创建一个新的 NumPy 随机数生成器
        rng = np.random.default_rng(seed)
        idxs = rng.permutation(len(reference_ds))
        mid = int(len(idxs) * 0.5)
        val_idx, test_idx = idxs[:mid], idxs[mid:]
        val_test_indices[key] = (val_idx, test_idx)
        return val_test_indices[key]

    def _choose_val_test_loaders(
        seed: Optional[int],
        dataset_root: str,
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        val_dir = os.path.join(dataset_root, "val")
        test_dir = os.path.join(dataset_root, "test")
        has_val = os.path.isdir(val_dir)
        has_test = os.path.isdir(test_dir)

        if has_val and has_test:
            # 情况 1：val/ 与 test/ 均存在，直接分别构建
            # 注意：这里强制 bankname=None，表示公共评估集，不对类别（ai/nature）做过滤
            _, val_loader = _build_loader_for_dataset(dataset_root, DatasetSplit.VAL, bankname=None, seed=seed)
            _, test_loader = _build_loader_for_dataset(dataset_root, DatasetSplit.TEST, bankname=None, seed=seed)
            LOGGER.info(
                "Use distinct val/ (%d) and test/ (%d) splits under %s.",
                len(val_loader.dataset),
                len(test_loader.dataset),
                dataset_root,
            )
            return val_loader, test_loader

        # 情况 2：只有一个目录（val/ 或 test/）可用，需内部 50/50 切分
        single_split = DatasetSplit.VAL if has_val else DatasetSplit.TEST
        source_dir = val_dir if has_val else test_dir

        # 生成该目录的“公共”均分索引（val/test 各一半），所有 bank 共用
        val_idx, test_idx = _get_val_test_indices(dataset_root, seed, split=single_split)

        # 读取该目录全部样本（bankname=None 表示不做 bank 过滤）
        full_ds, _ = _build_loader_for_dataset(dataset_root, single_split, bankname=None, seed=seed)

        # 用索引构造子集
        val_subset = torch.utils.data.Subset(full_ds, val_idx)
        test_subset = torch.utils.data.Subset(full_ds, test_idx)

        # 构建 DataLoader（这里未显式 shuffle；是否打乱由更高一层策略决定）
        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_subset,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=True,
        )

        LOGGER.info(
            "Only %s/ found under %s, split 50/50 -> VAL=%d | TEST=%d.",
            single_split.value,
            source_dir,
            len(val_subset),
            len(test_subset),
        )
        return val_loader, test_loader


    def return_dataloaders(seed: int, bankname: Optional[str]) -> Dict[str, Any]:
        # 1) 训练集：将多个数据集的 train/ 拼接在一起用于记忆库构建
        train_datasets = []
        for ds_name, ds_root in dataset_roots.items():
            ds, _ = _build_loader_for_dataset(
                ds_root, DatasetSplit.TRAIN, bankname=bankname, seed=seed
            )
            if len(ds) == 0:
                LOGGER.warning("Dataset %s has empty train split under %s, skipping.", ds_name, ds_root)
                continue
            train_datasets.append(ds)
        if not train_datasets:
            raise RuntimeError(f"No train samples found under {dataset_roots}.")
        train_ds = train_datasets[0] if len(train_datasets) == 1 else torch.utils.data.ConcatDataset(train_datasets)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )

        # 2) 验证/测试：为每个数据集单独构建 val/test Loader
        val_loaders: Dict[str, torch.utils.data.DataLoader] = {}
        test_loaders: Dict[str, torch.utils.data.DataLoader] = {}
        for ds_name, ds_root in dataset_roots.items():
            val_loader, test_loader = _choose_val_test_loaders(seed=seed, dataset_root=ds_root)
            val_loader.name = f"{ds_name}-val"
            test_loader.name = f"{ds_name}-test"
            val_loaders[ds_name] = val_loader
            test_loaders[ds_name] = test_loader

        train_loader.name = f"{'+'.join(dataset_names)}-train[{bankname}]"

        LOGGER.info(
            "Dataloaders ready: train=%d | val datasets=%s | test datasets=%s",
            len(train_ds),
            {k: len(v.dataset) for k, v in val_loaders.items()},
            {k: len(v.dataset) for k, v in test_loaders.items()},
        )
        return {"train": train_loader, "val": val_loaders, "test": test_loaders}

    return return_dataloaders


def train_memorybank(build_pc_fn, sampler, device, nn_method, train_loader) -> patchcore.patchcore.PatchCore:
    """单独训练某个记忆库：提取训练集特征并填充 KNN 内存。"""
    input_shape = infer_input_shape(train_loader)
    pc = build_pc_fn(input_shape=input_shape, featuresampler=sampler, device=device, nn_method=nn_method)
    pc.fit(train_loader)  # 记忆库构建
    return pc

def get_image_scores(patchcore_model, dataloader) -> np.ndarray:
    """
    用某个 PatchCore 模型（某 bank 的“正常库”）对 dataloader 打分，返回图像级异常分。
    - 不依赖 predict 返回的 labels，避免“相对/绝对”语义混淆。
    - 对 NaN/Inf 做填充，保证后续统计稳定。
    """
    prediction = patchcore_model.predict(dataloader)
    image_scores = np.asarray(prediction[0], dtype=float).reshape(-1)
    image_scores = np.asarray(image_scores, dtype=float).reshape(-1)

    return image_scores

# -----------------------------
# 主流程
# -----------------------------
def main(args):
    """主流程：加载数据 -> 训练记忆库 -> 公共评估 -> 可视化。"""
    np.seterr(all='raise')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    # 数据路径与数据集列表
    dataset_names = args.dataset_names 
    data_path = os.path.expanduser(args.data_path)

    # Data
    build_dataloader = get_dataloaders(
        data_path=data_path,
        dataset_names=dataset_names,
        batch_size=args.batch_size,
        resize=args.resize,
        imagesize=args.imagesize,
        workers=args.workers,
    )

    # 读取 banknames（要求 2 个，用于双库评估）
    banknames = args.banknames
    if len(banknames) != 2:
        raise ValueError(f"需要提供恰好 2 个 bank 名称，例如 --banknames ai nature，收到: {banknames}")


    # Sampler
    sampler = get_sampler(args.sampler_name, args.coreset_percentage, device)

    # PatchCore builder
    build_pc = get_patchcore(
        backbone_name=args.backbone_name,
        layers_to_extract_from=args.layers_to_extract_from,
        pretrain_embed_dimension=args.pretrain_embed_dimension,
        target_embed_dimension=args.target_embed_dimension,
        patchsize=args.patchsize,
        anomaly_scorer_k=args.anomaly_scorer_k,
    )


    # 选择 NN 方法（用可能更大的训练集规模估计）
    pcs: Dict[str, patchcore.patchcore.PatchCore] = {}
    bank_loaders: Dict[str, Dict[str, torch.utils.data.DataLoader]] = {}
    
    # 逐个 bank 构建“只含本域样本”的训练 Loader，并训练对应记忆库
    for bank in banknames:
        loaders_for_bank = build_dataloader(seed=args.seed, bankname=bank)
        bank_loaders[bank] = loaders_for_bank

        estN = len(loaders_for_bank["train"].dataset)
        nn_method = choose_nn_method(
            nn=args.nn_method,
            target_embed_dimension=args.target_embed_dimension,
            est_num_features=max(estN, 1),
            faiss_on_gpu=args.faiss_on_gpu,
            faiss_num_workers=args.faiss_num_workers,
        )
        pc = train_memorybank(build_pc, sampler, device, nn_method, loaders_for_bank["train"])
        pcs[bank] = pc

    # 公共 Loader（不筛 bank）用于验证/测试阶段，保证两套记忆库看到同一批样本
    common_loaders = build_dataloader(seed=args.seed, bankname=None)

    bank_ai, bank_nature = banknames[0], banknames[1]

    summary_logs: Dict[str, float] = {}
    visualization_cache: Dict[str, Dict[str, Any]] = {}
    csv_rows: List[List[Any]] = []

    for ds_name in dataset_names:
        val_loader = common_loaders["val"].get(ds_name)
        test_loader = common_loaders["test"].get(ds_name)
        if val_loader is None or test_loader is None:
            LOGGER.warning("Dataset %s missing val/test loaders, skip evaluation.", ds_name)
            continue

        # val/test 推理
        val_scores_a, val_masks_a, y_val,  _ = pcs[bank_ai].predict(val_loader)

        val_scores_b, val_masks_b, y_val,  _ = pcs[bank_nature].predict(val_loader)

        test_scores_a, test_masks_a, y_test,  test_path_a = pcs[bank_ai].predict(test_loader)
        test_scores_b, test_masks_b, y_test,  test_path_b = pcs[bank_nature].predict(test_loader)

        evaluator = Evaluator(save_classifier_path=None)

        features_val = build_feature_matrix(val_scores_a, val_scores_b)
        features_test = build_feature_matrix(test_scores_a, test_scores_b)

        evaluator.fit(features_val, y_val)
        prob_val = evaluator.predict(features_val)
        evaluator.select_threshold(prob_val, y_val)
        prob_test = evaluator.predict(features_test)
        val_auc_b, _, _ = evaluator.evaluate(prob_val, y_val)
        test_auc_b, y_pred, rep_b = evaluator.evaluate(prob_test, y_test)

        LOGGER.info(
            "[%s] τ=%.6f | VAL AUC=%.4f | TEST AUC=%.4f",
            ds_name,
            evaluator.threshold,
            val_auc_b,
            test_auc_b,
        )

        # 组合可视化注释：为每张图显示两个原始分数（ai/nature）
        vis_ann = [
            f"ai:{a:.3f} | nature:{b:.3f} | label_oring:{c} | label_pred:{d}"
            for a, b, c, d in zip(test_scores_a, test_scores_b, y_test, y_pred)
        ]
        visualization_cache[ds_name] = {
            "masks_a": test_masks_a,
            "masks_b": test_masks_b,
            "paths": [str(p) for p in test_path_a],
            "ann": vis_ann,
        }

        if rep_b:
            LOGGER.info("[%s] TEST report:\n%s", ds_name, rep_b)
        summary_logs.update({
            f"{ds_name}_logreg_val_auc": float(val_auc_b),
            f"{ds_name}_logreg_test_auc": float(test_auc_b),
        })

        # 累积 csv 行
        for i, (pid, a, b, y) in enumerate(zip(test_path_a, test_scores_a, test_scores_b, y_test)):
            row = [ds_name, str(pid), float(a), float(b), int(y), float(prob_test[i])]
            csv_rows.append(row)

    # 保存每个测试样本的两个原始分数（ai/nature）与 LogReg 概率到 CSV（覆盖全部 test 样本）
    try:
        if csv_rows:
            os.makedirs("runs", exist_ok=True)
            out_csv = os.path.join("runs", "test_scores.csv")
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = ["dataset", "id", "score_ai", "score_nature", "label_is_ai", "prob_logreg_ai"]
                writer.writerow(header)
                writer.writerows(csv_rows)
            LOGGER.info("Saved per-image test scores to %s", out_csv)
    except Exception as e:
        LOGGER.warning("Failed to save per-image test scores: %s", e)

    # 上文已在分支中完成评估，这里统一打印 summary（若存在）
    try:
        if 'summary_logs' in locals() and summary_logs:
            LOGGER.info("Summary: %s", {k: round(v, 6) for k, v in summary_logs.items()})
    except Exception:
        pass

    # 按数据集分别做可视化
    try:
        if not visualization_cache:
            raise RuntimeError("visualization cache is empty (no predictions recorded).")
        for ds_name, vis in visualization_cache.items():
            patchcore.utils.plot_random_segmentations(
                image_paths=vis["paths"],
                segmentations_a=vis["masks_a"],
                segmentations_b=vis["masks_b"],
                savefolder=os.path.join("output_images", ds_name),
                num_samples=min(50, len(vis["paths"])),
                resize=args.resize,
                imagesize=args.imagesize,
                annotations=vis.get("ann"),
            )
    except Exception as e:
        LOGGER.warning("Segmentation visualization skipped: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 数据
    parser.add_argument("--data_path", type=str, default=os.path.expanduser("~/datasets/tiny_genimage"))
    parser.add_argument("--dataset_names", nargs="+", default=[
        # 'adm',
        # 'biggan', 
        # 'glide', 
        # 'midjourney', 
        'sdv5', 
        # 'vqdm', 
        # 'wukong',
        # 'Chameleon',
        # 'sdv5_bigval'
    ],help="数据集列表")
    parser.add_argument("--seed", type=int, default=0)

    # dataloader
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=224)

    # backbone
    parser.add_argument(
        "--backbone_name",
        default="clip_vit_b16",
        help=(
            "骨干网络名称，例如 resnet50 / vit_base / vit_swin_base / clip_vit_b16"
        ),
    )
    parser.add_argument(
        "--layers_to_extract_from", nargs="+",
        # default=["layer2"]     # resnet50 的 layer2 
        # default=["layers.1"] #  vit_swin_base
        # default=["blocks.1"]
        # default=["blocks.6"] # vit_small 的第 7（差不多） 个block 输出，效果最好
        # default=["blocks.3"] # vit_base 的第 3（差不多） 个block 输出，效果最好
        # default=["blocks.3"] # 
        default=["transformer.resblocks.3"]   # clip_vit_b16  效果最好
        # default=["transformer.resblocks.1","transformer.resblocks.2","transformer.resblocks.3"]   # clip_vit_b16  TEST AUC=0.9900
        # default=["blocks.2", "blocks.3", "blocks.4", "blocks.5", "blocks.6", "blocks.8", "blocks.10"]
        # default=[
    #         # "features.8", # vgg19的conv3_3
    #         # "features.4", 
    #         "features.2" # inception_v4
    #             ]
    )

    # 预处理映射维度；0 表示自动按抽取层通道数（取最大值）推断
    parser.add_argument("--pretrain_embed_dimension", type=int, default=0,
                        help="预处理输出维度；设为 0 时按选定层的通道数自动推断")
    parser.add_argument("--target_embed_dimension", type=int, default=512)
    parser.add_argument("--patchsize", type=int, default=3)
    parser.add_argument("--anomaly_scorer_k", type=int, default=3)

    # sampler / coreset
    parser.add_argument("--sampler_name", 
                        # default="central_mahal",
                        # default="density",
                        default="random",
                        # default="approx_greedy_coreset"
                        )
    parser.add_argument("--coreset_percentage", type=float, default=0.1)
    # FAISS / 设备
    parser.add_argument("--faiss_on_gpu", action="store_true")
    parser.add_argument("--faiss_num_workers", type=int, default=12)
    parser.add_argument("--nn_method", choices=["auto", "flat", "ivfpq"], default="flat")

    # 两个 bank 名称
    parser.add_argument("--banknames", nargs="+", default=['ai','nature'],)

    args = parser.parse_args()
    main(args)
