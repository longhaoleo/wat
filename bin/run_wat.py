# run_wat.py
import os
import argparse
import logging
from typing import Dict, Any, Tuple, Callable, List, Optional

import numpy as np
import torch
import csv

import wat.backbones
import wat.common
import wat.wat
import wat.sampler
from wat.datasets.tiny_genimage import Dataset, DatasetSplit


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("run_wat")


def fuse_ai_margin_with_conf(
    raw_margin: np.ndarray,
    ai_conf: np.ndarray,
    conf_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    将 AI 库 generator 置信度融合进二分类 margin。
    - raw_margin = score_nature - score_ai，越大越偏向 AI；
    - gate = conf_floor + (1-conf_floor) * clip(ai_conf, 0, 1)；
    - adj_margin = raw_margin * gate。
    """
    conf = np.asarray(ai_conf, dtype=float).reshape(-1)
    conf = np.where(np.isfinite(conf), conf, 0.0)
    conf = np.clip(conf, 0.0, 1.0)
    conf_floor = float(np.clip(conf_floor, 0.0, 1.0))
    gate = conf_floor + (1.0 - conf_floor) * conf
    return np.asarray(raw_margin, dtype=float).reshape(-1) * gate, gate


# -----------------------------
# KNN 选择（优先 FAISS，缺失则退回 BruteNN）
# -----------------------------
def choose_nn_method(
    nn: str,
    target_embed_dimension: int,
    est_num_features: int,
    faiss_on_gpu: bool,
    faiss_num_workers: int,
):
    """根据配置选择 KNN 方法。"""
    common = wat.common
    if nn == "faiss" or nn == "ivfpq" or nn == "flat":
        if common.HAS_FAISS:
            return common.FaissNN(on_gpu=faiss_on_gpu, num_workers=faiss_num_workers)
        LOGGER.warning("faiss 未安装，回退 BruteNN。")
        return common.BruteNN()
    # auto：若装了 faiss 且数据量较大，优先 faiss
    if common.HAS_FAISS and est_num_features > 50000:
        return common.FaissNN(on_gpu=faiss_on_gpu, num_workers=faiss_num_workers)
    return common.BruteNN()


# -----------------------------
# 推断输入形状
# -----------------------------
def infer_input_shape(dl):
    """从任意 DataLoader 抽一批数据，推断 WAT 期望的 (C,H,W)。"""
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
# WAT 构造器
# -----------------------------
def get_wat(
    backbone_name: str,
    layers_to_extract_from: List[str],
    pretrain_embed_dimension: int,
    target_embed_dimension: int,
    patchsize: int,
    anomaly_scorer_k: int,
):
    def _builder(input_shape: Tuple[int, int, int], featuresampler, device: torch.device, nn_method):
        """闭包：按输入形状动态构造 WAT 并加载对应骨干网络。"""
        backbone = wat.backbones.load(backbone_name)
        backbone.name = backbone_name

        pc = wat.wat.WAT(device)
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
        return wat.common.PCASampler(percentage)
    if name == "random":
        return wat.sampler.RandomSampler(percentage)
    # 中心采样：主体优先
    if name == "central":
        return wat.sampler.CentralSampler(percentage, use_mahalanobis=False)
    if name == "central_mahal":
        return wat.sampler.CentralSampler(percentage, use_mahalanobis=True)
    # 密度采样：高密度优先（可带后缀 n_neighbors，如 density:10）
    if name.startswith("density"):
        n_neighbors = 50
        if ":" in name:
            try:
                n_neighbors = int(name.split(":", 1)[1])
            except Exception:
                pass
        return wat.sampler.DensitySampler(percentage, n_neighbors=n_neighbors)
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
        km = wat.sampler.KMeansSampler(percentage, per_cluster_topk=topk)
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


def train_memorybank(build_pc_fn, sampler, device, nn_method, train_loader) -> wat.wat.WAT:
    """单独训练某个记忆库：提取训练集特征并填充 KNN 内存。"""
    input_shape = infer_input_shape(train_loader)
    pc = build_pc_fn(input_shape=input_shape, featuresampler=sampler, device=device, nn_method=nn_method)
    pc.fit(train_loader)  # 记忆库构建
    return pc



# -----------------------------
# 主流程
# -----------------------------
def main(args):
    """
    主流程：
        - phase == 'train'：
            1) 在指定 train 数据集上，为每个 bank 训练 WAT 记忆库并保存到磁盘；
        - phase == 'infer'：
            1) 从磁盘加载已训练好的 WAT 记忆库；
            2) 在指定 test/val split 上执行推理/评估。
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
    # Phase 1: 训练 WAT 记忆库并保存到磁盘
    # -----------------------------
    if args.phase in ("train", "both"):
        # 1) 构建 WAT builder，用于后续为各 bank 训练记忆库
        build_pc = get_wat(
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

        # 3) 为每个 bank 训练一个 WAT 记忆库，并保存到磁盘
        pcs: Dict[str, wat.wat.WAT] = {}
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
                LOGGER.info("Saved WAT memory bank for '%s' to %s", bank, save_dir)

            pcs[bank] = pc

        # 训练阶段只需写出记忆库
        if args.phase == "train":
            return

    # -----------------------------
    # Phase 2: 仅推理/评估
    #   - 假设 WAT 记忆库已在 Phase 1 中训练好并保存；
    #   - 这里只负责：加载 WAT 记忆库，在 test 上做推理/评估；
    #   - 不再依赖 val，也无需额外分类器。
    # -----------------------------
    if args.phase not in ("infer", "both"):
        raise ValueError(f"Unknown phase: {args.phase}. Expected 'train', 'infer' or 'both'.")

    # 1) 从磁盘加载 WAT 记忆库（每个 bank 一份）
    if not args.pc_save_root:
        raise RuntimeError("Phase 'infer' requires a valid --pc_save_root to load WAT memory banks.")

    pcs: Dict[str, wat.wat.WAT] = {}
    for bank in banknames:
        load_dir = os.path.join(args.pc_save_root, bank)
        nn_method = wat.common.BruteNN()
        pc = wat.wat.WAT(device)
        pc.load_from_path(load_path=load_dir, device=device, nn_method=nn_method)
        pcs[bank] = pc
        LOGGER.info("Loaded WAT memory bank for '%s' from %s", bank, load_dir)

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

    summary_allacc_logs: Dict[str, float] = {}
    summary_certainacc_logs: Dict[str, float] = {}
    summary_uncertain_logs: Dict[str, float] = {}
    genacc_logs: Dict[str, float] = {}
    csv_rows: List[List[Any]] = []
    all_labels_total: List[int] = []
    all_preds_total: List[int] = []
    all_preds_certain: List[int] = []
    all_labels_certain: List[int] = []
    all_uncertain_flags: List[int] = []
    all_gen_gt: List[str] = []
    all_gen_pred: List[str] = []
    all_gen_is_ai: List[int] = []
    all_gen_is_tp_ai: List[int] = []

    for ds_name in test_dataset_names:
        test_loader = common_loaders["test"].get(ds_name)
        if test_loader is None:
            LOGGER.warning("Dataset %s missing test loader, skip evaluation.", ds_name)
            continue

        # 仅对 test 做推理/评估：双 bank 分别打分
        (
            test_scores_a,
            _,
            y_test,
            test_path_a,
            gen_pred_a,
            gen_conf_a,
            gt_generators,
            gt_dataset_names,
        ) = pcs[bank_ai].predict_with_meta(test_loader)
        test_scores_b, _, y_test, test_path_b, nat_pred_b, nat_conf_b, _, _ = pcs[bank_nature].predict_with_meta(test_loader)

        test_scores_a_np = np.asarray(test_scores_a, dtype=float).reshape(-1)
        test_scores_b_np = np.asarray(test_scores_b, dtype=float).reshape(-1)
        y_test_np = np.asarray(y_test, dtype=int).reshape(-1)

        paths = [str(p) for p in test_path_a] if test_path_a is not None else ["-"] * len(test_scores_a_np)
        gen_a = gen_pred_a or ["unknown"] * len(test_scores_a_np)
        gen_a_conf = np.asarray(gen_conf_a or [float("nan")] * len(test_scores_a_np), dtype=float).reshape(-1)
        nat_b = nat_pred_b or ["nature"] * len(test_scores_a_np)
        gt_gen = gt_generators or ["unknown"] * len(test_scores_a_np)
        gt_ds = gt_dataset_names or [ds_name] * len(test_scores_a_np)

        raw_margin = (test_scores_b_np - test_scores_a_np)  # >0 偏向 AI，<0 偏向 nature
        adj_margin, conf_gate = fuse_ai_margin_with_conf(
            raw_margin=raw_margin,
            ai_conf=gen_a_conf,
            conf_floor=args.ai_conf_floor,
        )
        uncertain_mask = np.abs(adj_margin) < float(args.uncertain_eps)
        preds = np.where(uncertain_mask, -1, (adj_margin > 0.0).astype(int))  # -1: uncertain
        certain_mask = preds != -1

        # all_acc: 全样本准确率；uncertain(-1) 会自然记为错误
        all_acc = float((preds == y_test_np).mean()) if len(y_test_np) else float("nan")
        if certain_mask.any():
            acc_certain = float((preds[certain_mask] == y_test_np[certain_mask]).mean())
        else:
            acc_certain = float("nan")
        coverage = float(certain_mask.mean()) if len(certain_mask) else 0.0
        uncertain_rate = 1.0 - coverage
        summary_allacc_logs[ds_name] = all_acc
        summary_certainacc_logs[ds_name] = acc_certain
        summary_uncertain_logs[ds_name] = uncertain_rate
        LOGGER.info(
            "[%s] acc_certain=%.4f, all_acc=%.4f, coverage=%.4f, uncertain_rate=%.4f",
            ds_name, acc_certain, all_acc, coverage, uncertain_rate
        )

        all_labels_total.extend(y_test_np.tolist())
        all_preds_total.extend(preds.astype(int).tolist())
        all_uncertain_flags.extend(uncertain_mask.astype(int).tolist())
        if certain_mask.any():
            all_labels_certain.extend(y_test_np[certain_mask].tolist())
            all_preds_certain.extend(preds[certain_mask].tolist())

        # generator accuracy：只在真实 AI 样本上统计（nature 没有 generator 语义）
        try:
            ga_np = np.asarray([str(x) for x in gen_a], dtype=object)
            gg_np = np.asarray([str(x) for x in gt_gen], dtype=object)
            is_ai = (y_test_np == 1)
            is_tp_ai = is_ai & (preds == 1)
            n_ai = int(is_ai.sum())
            n_tp_ai = int(is_tp_ai.sum())
            gen_acc_ai = float((ga_np[is_ai] == gg_np[is_ai]).mean()) if n_ai else float("nan")
            gen_acc_tp = float((ga_np[is_tp_ai] == gg_np[is_tp_ai]).mean()) if n_tp_ai else float("nan")
            genacc_logs[ds_name] = gen_acc_ai
            LOGGER.info(
                "[%s] gen_acc(gt_ai)=%.4f (n=%d) | gen_acc(tp_ai)=%.4f (n=%d)",
                ds_name,
                gen_acc_ai,
                n_ai,
                gen_acc_tp,
                n_tp_ai,
            )
            all_gen_gt.extend(gg_np.tolist())
            all_gen_pred.extend(ga_np.tolist())
            all_gen_is_ai.extend(is_ai.astype(int).tolist())
            all_gen_is_tp_ai.extend(is_tp_ai.astype(int).tolist())
        except Exception as e:
            LOGGER.warning("[%s] Failed to compute generator accuracy: %s", ds_name, e)

        for path, sa, sb, yt, pr, ga, gc, nb, ggt, dgt, m_raw, m_adj, gate, un in zip(
            paths, test_scores_a_np, test_scores_b_np, y_test_np, preds, gen_a, gen_a_conf, nat_b, gt_gen, gt_ds,
            raw_margin, adj_margin, conf_gate, uncertain_mask
        ):
            rel = (sa - sb) / (abs(sb) + 1e-8)
            sym = (sa - sb) / (abs(sa) + abs(sb) + 1e-8)
            if int(pr) == 1:
                pred_gen = ga
                pred_label = "ai"
            elif int(pr) == 0:
                pred_gen = "nature"
                pred_label = "nature"
            else:
                pred_gen = "uncertain"
                pred_label = "uncertain"
            csv_rows.append([
                ds_name, path, sa, sb, rel, sym, yt, pr, pred_label, int(un),
                float(m_raw), float(m_adj), float(gate), pred_gen, ga, gc, nb, ggt, dgt
            ])


    # 保存每个测试样本的两个原始分数到 CSV（覆盖全部 test 样本）
    try:
        if csv_rows:
            os.makedirs("runs", exist_ok=True)
            out_csv = os.path.join("runs", "test_scores.csv")
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = [
                    "dataset",
                    "id",
                    "score_ai",
                    "score_nature",
                    "rel_diff_to_b",
                    "sym_diff",
                    "label_is_ai",
                    "pred_is_ai",
                    "pred_label",
                    "is_uncertain",
                    "margin_raw",
                    "margin_adj",
                    "ai_conf_gate",
                    "pred_generator",
                    "ai_generator",
                    "ai_generator_conf",
                    "nature_label",
                    "gt_generator",
                    "gt_dataset_name",
                ]
                writer.writerow(header)
                writer.writerows(csv_rows)
            LOGGER.info("Saved per-image test scores to %s", out_csv)
    except Exception as e:
        LOGGER.warning("Failed to save per-image test scores: %s", e)

    # 打印 summary 与全局精度
    try:
        if summary_certainacc_logs:
            LOGGER.info("Per-dataset acc_certain: %s", {k: round(v, 6) for k, v in summary_certainacc_logs.items()})
        if summary_allacc_logs:
            LOGGER.info("Per-dataset all_acc: %s", {k: round(v, 6) for k, v in summary_allacc_logs.items()})
        if summary_uncertain_logs:
            LOGGER.info("Per-dataset uncertain_rate: %s", {k: round(v, 6) for k, v in summary_uncertain_logs.items()})
    except Exception:
        pass

    try:
        if genacc_logs:
            LOGGER.info("Per-dataset gen_acc(gt_ai): %s", {k: round(v, 6) for k, v in genacc_logs.items()})
    except Exception:
        pass

    try:
        if all_labels_total:
            total_samples = len(all_labels_total)
            uncertain_rate = float(np.mean(np.asarray(all_uncertain_flags, dtype=int)))
            coverage = 1.0 - uncertain_rate
            LOGGER.info("Overall coverage=%.4f, uncertain_rate=%.4f (samples=%d)", coverage, uncertain_rate, total_samples)
        if all_labels_total and all_preds_total:
            labels_all_np = np.asarray(all_labels_total, dtype=int)
            preds_all_np = np.asarray(all_preds_total, dtype=int)
            overall_all_acc = float((labels_all_np == preds_all_np).mean())
            LOGGER.info("Overall all_acc: %.4f (samples=%d)", overall_all_acc, len(labels_all_np))

        if all_labels_certain:
            labels_np = np.array(all_labels_certain, dtype=int)
            preds_np = np.array(all_preds_certain, dtype=int)
            overall_acc = float((labels_np == preds_np).mean())
            LOGGER.info("Overall acc_certain: %.4f (samples=%d)", overall_acc, len(labels_np))
    except Exception as e:
        LOGGER.warning("Failed to compute overall metrics: %s", e)

    # overall generator accuracy
    try:
        if all_gen_gt and all_gen_pred and all_gen_is_ai:
            gg = np.asarray(all_gen_gt, dtype=object)
            ga = np.asarray(all_gen_pred, dtype=object)
            is_ai = np.asarray(all_gen_is_ai, dtype=int) == 1
            is_tp_ai = np.asarray(all_gen_is_tp_ai, dtype=int) == 1
            n_ai = int(is_ai.sum())
            n_tp_ai = int(is_tp_ai.sum())
            gen_acc_ai = float((ga[is_ai] == gg[is_ai]).mean()) if n_ai else float("nan")
            gen_acc_tp = float((ga[is_tp_ai] == gg[is_tp_ai]).mean()) if n_tp_ai else float("nan")
            LOGGER.info("Overall gen_acc(gt_ai)=%.4f (n=%d) | gen_acc(tp_ai)=%.4f (n=%d)", gen_acc_ai, n_ai, gen_acc_tp, n_tp_ai)
    except Exception as e:
        LOGGER.warning("Failed to compute overall generator accuracy: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    HOME = os.path.expanduser("~/wat")
    # 数据
    parser.add_argument("--data_path", type=str, default=os.path.expanduser("~/datasets/tiny_genimage"))
    parser.add_argument("--dataset_names", nargs="+", default=[
        # 'adm',
        # 'biggan', 
        'glide', 
        'midjourney', 
        'sdv5', 
        # 'vqdm', 
        # 'wukong',
        # 'Chameleon',
        # 'sdv5_bigval',
        # 'sdv5 mini', 
    ], help="用于训练 WAT 记忆库的数据集名称列表（train split）。")
    parser.add_argument(
        "--test_dataset_names",
        nargs="+",
        # default=None,
        default=[
        # 'adm',
        # 'biggan', 
        'glide', 
        'midjourney', 
        'sdv5', 
        # 'vqdm', 
        # 'wukong',
        # 'Chameleon',
        # 'sdv5_bigval',
        # 'sdv5 mini', 
        ],
        help="用于推理/评估的数据集名称列表（test/val split）；若不指定，则默认与 --dataset_names 相同。",
    )
    parser.add_argument("--seed", type=int, default=0)

    # dataloader
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=13)
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
    parser.add_argument("--patchsize", type=int, default=3, help="兼容参数（当前图像级聚合，不切 patch）")
    parser.add_argument("--anomaly_scorer_k", type=int, default=20)

    # 阶段控制与模型保存
    parser.add_argument(
        "--phase",type=str,choices=["train", "infer", "both"],
        default="infer",
        help=("运行阶段：'train' 仅训练并保存 WAT 记忆库；"
            "'infer' 仅加载已训练记忆库并在 test 上推理/评估；"
            "'both' 先训练再在同一次运行中执行推理/评估（默认）。"))
    parser.add_argument(
        "--pc_save_root",type=str,
        default=os.path.join( HOME, "memorybanks"),
        help="保存 / 加载 WAT 记忆库的根目录，每个 bank 将保存在该目录下以 bank 名称命名的子目录中。",
    )


    # sampler / coreset
    parser.add_argument("--sampler_name", 
                        # default="central",
                        # default="central_mahal",
                        # default="density",
                        default="random",
                        # default="pca"
                        )
    parser.add_argument("--coreset_percentage", type=float, default=0.1)
    # FAISS / 设备
    parser.add_argument("--faiss_on_gpu", action="store_true")
    parser.add_argument("--faiss_num_workers", type=int, default=12)
    parser.add_argument("--nn_method", choices=["auto", "flat", "ivfpq"], default="auto")
    parser.add_argument(
        "--ai_conf_floor",
        type=float,
        default=0.35,
        help="将 ai_generator_conf 融入分类 margin 时的最小保留权重（0~1，越小越依赖 conf）。",
    )
    parser.add_argument(
        "--uncertain_eps",
        type=float,
        default=0.01,
        help="不确定区间阈值：|adjusted_margin| < eps 时输出 uncertain。",
    )

    # 两个 bank 名称
    parser.add_argument("--banknames", nargs="+", default=['ai','nature'],)

    args = parser.parse_args()
    main(args)
