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

    SMALL_N = 50000
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
    if name == "pca":
        return patchcore.common.PCASampler(percentage)
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
    """生成一个工厂函数：按 seed/bankname 构建 train/test DataLoader。

    约定：
        - train：使用 <dataset>/train/<bankname> 作为记忆库构建数据；
        - test：优先使用 <dataset>/test/*，若不存在则退化为 <dataset>/val/* 作为测试集。
    """
    dataset_names = list(dataset_names or [])
    dataset_roots = {ds: os.path.join(data_path, ds) for ds in dataset_names}

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

    def _build_test_loader_for_dataset(
        seed: Optional[int],
        dataset_root: str,
    ) -> torch.utils.data.DataLoader:
        """
        为给定数据集构建“公共”测试集 DataLoader。
        规则：
            - 若存在 <root>/test/，则直接使用该目录；
            - 否则若存在 <root>/val/，则将其视作测试集使用；
            - 若两者都不存在，则抛出异常。
        """
        val_dir = os.path.join(dataset_root, "val")
        test_dir = os.path.join(dataset_root, "test")
        has_val = os.path.isdir(val_dir)
        has_test = os.path.isdir(test_dir)

        if has_val and has_test:
            # val/ 与 test/ 均存在，优先选择 test/ 作为测试集。
            _, test_loader = _build_loader_for_dataset(
                dataset_root, DatasetSplit.TEST, bankname=None, seed=seed
            )
            LOGGER.info(
                "Use existing test/ (%d) under %s.",
                len(test_loader.dataset),
                dataset_root,
            )
            return test_loader

        if has_test:
            # 仅存在 test/ 目录
            _, test_loader = _build_loader_for_dataset(
                dataset_root, DatasetSplit.TEST, bankname=None, seed=seed
            )
            LOGGER.info(
                "Use test/ (%d) as evaluation split under %s.",
                len(test_loader.dataset),
                dataset_root,
            )
            return test_loader

        if has_val:
            # 仅存在 val/ 目录，将其视作测试集
            _, test_loader = _build_loader_for_dataset(
                dataset_root, DatasetSplit.VAL, bankname=None, seed=seed
            )
            LOGGER.info(
                "Use val/ (%d) as evaluation split under %s (no test/ found).",
                len(test_loader.dataset),
                dataset_root,
            )
            return test_loader

        raise RuntimeError(
            f"Neither 'val/' nor 'test/' exists under dataset root: {dataset_root}"
        )


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

        # 2) 测试：为每个数据集单独构建“公共” test Loader
        test_loaders: Dict[str, torch.utils.data.DataLoader] = {}
        for ds_name, ds_root in dataset_roots.items():
            test_loader = _build_test_loader_for_dataset(seed=seed, dataset_root=ds_root)
            test_loader.name = f"{ds_name}-test"
            test_loaders[ds_name] = test_loader

        train_loader.name = f"{'+'.join(dataset_names)}-train[{bankname}]"

        LOGGER.info(
            "Dataloaders ready: train=%d | test datasets=%s",
            len(train_ds),
            {k: len(v.dataset) for k, v in test_loaders.items()},
        )
        return {"train": train_loader, "test": test_loaders}

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
    """
    主流程：
        - phase == 'train'：
            1) 在指定 train 数据集上，为每个 bank 训练 PatchCore 记忆库并保存到磁盘；
            2) 在“公共 train”（不区分 bank）的一个子集上训练逻辑回归分类器，并保存到磁盘。
        - phase == 'infer'：
            1) 从磁盘加载已训练好的 PatchCore 记忆库和逻辑回归分类器；
            2) 在指定 test 数据集上执行推理/评估（不再依赖 val）。
    """
    np.seterr(all='raise')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    # 数据路径
    data_path = os.path.expanduser(args.data_path)

    # 训练 / 测试数据集名称
    train_dataset_names = list(args.dataset_names or [])
    if not train_dataset_names:
        raise RuntimeError("至少需要提供一个用于训练的 dataset 名称，通过 --dataset_names 指定。")
    test_dataset_names = list(args.test_dataset_names) if args.test_dataset_names else train_dataset_names
    LOGGER.info(
        "Train datasets: %s | Test datasets: %s",
        train_dataset_names,
        test_dataset_names,
    )

    # 读取 banknames
    banknames = args.banknames
    bank_ai, bank_nature = banknames[0], banknames[1]

    # Sampler
    sampler = get_sampler(args.sampler_name, args.coreset_percentage, device)

    # -----------------------------
    # Phase 1: 训练 PatchCore 记忆库 + 逻辑回归分类器，并保存到磁盘
    # -----------------------------
    if args.phase in ("train", "both"):
        # 1) 构建 PatchCore builder，用于后续为各 bank 训练记忆库
        build_pc = get_patchcore(
            backbone_name=args.backbone_name,
            layers_to_extract_from=args.layers_to_extract_from,
            pretrain_embed_dimension=args.pretrain_embed_dimension,
            target_embed_dimension=args.target_embed_dimension,
            patchsize=args.patchsize,
            anomaly_scorer_k=args.anomaly_scorer_k,
        )

        # 2) 基于“训练数据集列表”构建 DataLoader 工厂
        build_dataloader_train = get_dataloaders(
            data_path=data_path,
            dataset_names=train_dataset_names,
            batch_size=args.batch_size,
            resize=args.resize,
            imagesize=args.imagesize,
            workers=args.workers,
        )

        # 3) 为每个 bank 训练一个 PatchCore 记忆库，并保存到磁盘
        pcs: Dict[str, patchcore.patchcore.PatchCore] = {}
        for bank in banknames:
            loaders_for_bank = build_dataloader_train(seed=args.seed, bankname=bank)
            estN = len(loaders_for_bank["train"].dataset)
            nn_method = choose_nn_method(
                nn=args.nn_method,
                target_embed_dimension=args.target_embed_dimension,
                est_num_features=max(estN, 1),
                faiss_on_gpu=args.faiss_on_gpu,
                faiss_num_workers=args.faiss_num_workers,
            )
            pc = train_memorybank(build_pc, sampler, device, nn_method, loaders_for_bank["train"])

            # 记忆库保存路径：<pc_save_root>/<bank>/
            if args.pc_save_root:
                save_dir = os.path.join(args.pc_save_root, bank)
                os.makedirs(save_dir, exist_ok=True)
                pc.save_to_path(save_dir)
                LOGGER.info("Saved PatchCore memory bank for '%s' to %s", bank, save_dir)

            pcs[bank] = pc

        # 4) 公共 Loader（不筛 bank），从“训练数据集列表”构建：
        #    - 用于给逻辑回归分类器提供训练数据（只采样一部分 train）。
        common_loaders = build_dataloader_train(seed=args.seed, bankname=None)

        # 1) 取“公共 train”作为分类器可用的样本池（包含 ai/nature 两类）
        base_train_loader = common_loaders["train"]
        base_train_dataset = base_train_loader.dataset
        total_train = len(base_train_dataset)
        if total_train == 0:
            raise RuntimeError("No training samples found for classifier training.")

        # 2) 按给定比例随机采样一部分样本用来训练逻辑回归（避免全量 train 过拟合 / 过慢）
        frac = float(args.cls_train_fraction)
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"cls_train_fraction must be in (0,1], got {frac}.")
        n_sample = max(1, int(total_train * frac))

        rng = np.random.default_rng(args.seed)
        sampled_indices = rng.choice(total_train, size=n_sample, replace=False)
        sampled_dataset = torch.utils.data.Subset(base_train_dataset, sampled_indices)
        sampled_loader = torch.utils.data.DataLoader(
            sampled_dataset,
            batch_size=args.batch_size,
            shuffle=False,          # 保持顺序一致，便于双 bank 对齐
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )
        LOGGER.info(
            "Training logistic classifier on %d / %d train samples (fraction=%.3f).",
            n_sample,
            total_train,
            frac,
        )

        # 3) 从原始数据集中直接读取标签（is_ai），避免重复依赖 PatchCore.predict 的返回结构
        y_train_list: List[int] = []
        for idx in sampled_indices:
            sample = base_train_dataset[idx]
            if isinstance(sample, dict):
                label = sample.get("is_ai")
            else:
                # 兼容形如 (image, label, ...) 的返回格式
                label = sample[1] if len(sample) > 1 else 0
            if isinstance(label, torch.Tensor):
                label = int(label.item())
            else:
                label = int(label)
            y_train_list.append(label)
        y_train = np.asarray(y_train_list, dtype=int).reshape(-1)

        # 4) 用两套 PatchCore 记忆库分别对这部分样本打分，得到 (score_ai, score_nature)
        #    这里使用辅助函数 get_image_scores，仅取图像级分数，避免在此处重复解析 predict 的完整返回。
        train_scores_a = get_image_scores(pcs[bank_ai], sampled_loader)
        train_scores_b = get_image_scores(pcs[bank_nature], sampled_loader)

        features_train = build_feature_matrix(
            np.asarray(train_scores_a, dtype=float),
            np.asarray(train_scores_b, dtype=float),
        )

        # 5) 构建 Evaluator，训练逻辑回归分类器并保存到指定路径
        if args.classifier_path:
            os.makedirs(os.path.dirname(args.classifier_path), exist_ok=True)
        evaluator = Evaluator(save_classifier_path=args.classifier_path)
        evaluator.fit(features_train, y_train)

        # 若仅训练，则此处结束；若 phase == "both"，则继续进入推理阶段。
        if args.phase == "train":
            return

    # -----------------------------
    # Phase 2: 仅推理/评估
    #   - 假设 PatchCore 记忆库与逻辑回归分类器已在 Phase 1 中训练好并保存；
    #   - 这里只负责：加载 PatchCore + 分类器，在指定 test 数据集上做推理/评估；
    #   - 不再依赖 val。
    # -----------------------------
    if args.phase not in ("infer", "both"):
        raise ValueError(f"Unknown phase: {args.phase}. Expected 'train', 'infer' or 'both'.")

    # 1) 从磁盘加载 PatchCore 记忆库（每个 bank 一份）
    if not args.pc_save_root:
        raise RuntimeError("Phase 'infer' requires a valid --pc_save_root to load PatchCore memory banks.")

    pcs: Dict[str, patchcore.patchcore.PatchCore] = {}
    for bank in banknames:
        load_dir = os.path.join(args.pc_save_root, bank)
        nn_method = patchcore.common.FaissNN(on_gpu=args.faiss_on_gpu, num_workers=args.faiss_num_workers)
        pc = patchcore.patchcore.PatchCore(device)
        pc.load_from_path(load_path=load_dir, device=device, nn_method=nn_method)
        pcs[bank] = pc
        LOGGER.info("Loaded PatchCore memory bank for '%s' from %s", bank, load_dir)

    # 2) 基于“测试数据集列表”构建 DataLoader 工厂，并拿到公共 test Loader
    build_dataloader_test = get_dataloaders(
        data_path=data_path,
        dataset_names=test_dataset_names,
        batch_size=args.batch_size,
        resize=args.resize,
        imagesize=args.imagesize,
        workers=args.workers,
    )
    common_loaders = build_dataloader_test(seed=args.seed, bankname=None)

    # 3) 加载已经训练好的逻辑回归分类器
    if not args.classifier_path:
        raise RuntimeError("Phase 'infer' requires a valid --classifier_path to load the trained classifier.")

    evaluator = Evaluator(save_classifier_path=args.classifier_path)
    evaluator.load_classifier(args.classifier_path)
    LOGGER.info("Loaded logistic classifier from: %s", args.classifier_path)

    summary_logs: Dict[str, float] = {}
    visualization_cache: Dict[str, Dict[str, Any]] = {}
    csv_rows: List[List[Any]] = []

    for ds_name in test_dataset_names:
        test_loader = common_loaders["test"].get(ds_name)
        if test_loader is None:
            LOGGER.warning("Dataset %s missing test loader, skip evaluation.", ds_name)
            continue

        # 仅对 test 做推理/评估：双 bank 分别打分，然后送入已经训练好的逻辑回归分类器
        test_scores_a, test_masks_a, y_test,  test_path_a = pcs[bank_ai].predict(test_loader)
        test_scores_b, test_masks_b, y_test,  test_path_b = pcs[bank_nature].predict(test_loader)

        features_test = build_feature_matrix(test_scores_a, test_scores_b)

        prob_test = evaluator.predict(features_test)
        test_auc_b, y_pred, rep_b = evaluator.evaluate(prob_test, y_test)

        LOGGER.info(
            "[%s] TEST AUC=%.4f",
            ds_name,
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
    HOME = os.path.expanduser("~/dreamycore")
    # 数据
    parser.add_argument("--data_path", type=str, default=os.path.expanduser("~/datasets/tiny_genimage"))
    parser.add_argument("--dataset_names", nargs="+", default=[
        'adm',
        'biggan', 
        'glide', 
        'midjourney', 
        'sdv5', 
        'vqdm', 
        'wukong',
        # 'Chameleon',
        # 'sdv5_bigval'
    ], help="用于训练 PatchCore 记忆库和逻辑回归分类器的数据集名称列表（train split）。")
    parser.add_argument(
        "--test_dataset_names",
        nargs="+",
        # default=None,
        default=[
        'adm',
        'biggan', 
        'glide', 
        'midjourney', 
        'sdv5', 
        'vqdm', 
        'wukong',
        'Chameleon',
        'sdv5_bigval',
        ],
        help="用于推理/评估的数据集名称列表（test/val split）；若不指定，则默认与 --dataset_names 相同。",
    )
    parser.add_argument("--seed", type=int, default=0)

    # dataloader
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=224)

    # backbone
    parser.add_argument(
        "--backbone_name",
        default="clip_vit_b16",
        help=("骨干网络名称，例如 resnet50 / vit_base / vit_swin_base / clip_vit_b16"))
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

    # 逻辑回归分类器 / 阶段控制相关：
    #   - phase = train ：在公共 train 上（按比例采样）训练 PatchCore 记忆库 + 逻辑回归，并保存到磁盘；
    #   - phase = infer ：从磁盘加载已训练好的 PatchCore 记忆库与逻辑回归，只在 test 上做推理/评估；
    #   - phase = both  ：先执行 train 流程，再在同一次运行中立即执行 infer 流程（默认行为）。
    parser.add_argument(
        "--phase",type=str,choices=["train", "infer", "both"],
        default="infer",
        help=("运行阶段：'train' 仅训练 PatchCore 记忆库和逻辑回归；"
            "'infer' 仅加载已训练模型并在 test 上推理/评估；"
            "'both' 先训练再在同一次运行中执行推理/评估（默认）。"))
    parser.add_argument(
        "--classifier_path",type=str,
        default=os.path.join( HOME, "classifier", "logreg_classifier.joblib"),
        help="保存 / 加载逻辑回归分类器的路径。",
    )
    parser.add_argument(
        "--pc_save_root",type=str,
        default=os.path.join( HOME, "memorybanks"),
        help="保存 / 加载 PatchCore 记忆库的根目录，每个 bank 将保存在该目录下以 bank 名称命名的子目录中。",
    )
    parser.add_argument(
        "--cls_train_fraction",type=float,default=0.1,
        help="用于训练逻辑回归分类器的训练集采样比例 (0,1]，例如 0.1 表示随机采样 10% 的 train 样本。",
    )

    # sampler / coreset
    parser.add_argument("--sampler_name", 
                        # default="central",
                        # default="central_mahal",
                        # default="density",
                        default="random",
                        # default="pca"
                        )
    parser.add_argument("--coreset_percentage", type=float, default=0.001)
    # FAISS / 设备
    parser.add_argument("--faiss_on_gpu", action="store_true")
    parser.add_argument("--faiss_num_workers", type=int, default=12)
    parser.add_argument("--nn_method", choices=["auto", "flat", "ivfpq"], default="auto")

    # 两个 bank 名称
    parser.add_argument("--banknames", nargs="+", default=['ai','nature'],)

    args = parser.parse_args()
    main(args)
