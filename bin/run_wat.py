"""
WAT 训练/推理主入口。

本文件对应 `PROJECT_DETAILED_COMMENTS.md` 第 1 节：
1) 构建与数据准备；
2) 训练阶段（memory bank 构建）；
3) 推理评估阶段（调用 `wat.eval_tools`）；
4) CSV 结果输出；
5) overall 指标聚合口径。
"""

# run_wat.py
import os
import argparse
import logging
from typing import Dict, Any, Tuple, Callable, List, Optional

import numpy as np
import torch

import wat.backbones
import wat.common
import wat.wat
import wat.sampler
import wat.eval_tools as eval_tools
from wat.datasets.tiny_genimage import Dataset, DatasetSplit


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("run_wat")




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
    def _parse_optional_int_suffix(text: str, default_value: int) -> int:
        """
        解析类似 `density:50`、`kmeans_topc:3` 这种后缀整数配置。
        - 没有 `:` 时直接返回默认值；
        - `:` 后不是纯整数时也回退默认值；
        - 不使用 try/except，避免把配置错误静默吞掉。
        """
        if ":" not in text:
            return default_value
        suffix = text.split(":", 1)[1].strip()
        if suffix.isdigit():
            return int(suffix)
        return default_value

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
        n_neighbors = _parse_optional_int_suffix(name, default_value=50)
        return wat.sampler.DensitySampler(percentage, n_neighbors=n_neighbors)
    # KMeans / TopC：kmeans_topc[:C]（默认 C=2），普通 kmeans 等同于 topc:1
    if name.startswith("kmeans_topc") or name == "kmeans":
        topk = 1
        if name.startswith("kmeans_topc"):
            topk = _parse_optional_int_suffix(name, default_value=2)
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
    # [1.6-入口] 数值问题尽早显式抛错，避免静默污染指标。
    np.seterr(all='raise')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    # [1.6-路径] 展开 ~，避免跨环境启动时路径歧义。
    data_path = os.path.expanduser(args.data_path)

    # [1.6-数据集] 训练集必须显式存在；测试集未给时默认复用训练集列表。
    train_dataset_names = list(args.dataset_names or [])
    if not train_dataset_names:
        raise RuntimeError("至少需要提供一个用于训练的 dataset 名称，通过 --dataset_names 指定。")
    test_dataset_names = list(args.test_dataset_names) if args.test_dataset_names else train_dataset_names
    LOGGER.info(
        "Train datasets: %s | Test datasets: %s",
        train_dataset_names,
        test_dataset_names,
    )

    # [1.6-bank] 固定语义：banknames[0]=ai, banknames[1]=nature。
    banknames = args.banknames
    bank_ai, bank_nature = banknames[0], banknames[1]

    # [1.6-sampler] 训练建库采样器（identity/random/central/...）。
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
            # 单个 bank 的完整建库过程：抽特征 -> sampler -> KNN fit。
            pc = train_memorybank(build_pc, sampler, device, nn_method, loaders_for_bank["train"])

            # 记忆库保存路径：<pc_save_root>/<bank>/
            if args.pc_save_root:
                save_dir = os.path.join(args.pc_save_root, bank)
                os.makedirs(save_dir, exist_ok=True)
                pc.save_to_path(save_dir)
                LOGGER.info("Saved WAT memory bank for '%s' to %s", bank, save_dir)

            pcs[bank] = pc

        # [1.6-提前返回] 纯 train 模式到此结束，不进入推理评估。
        if args.phase == "train":
            return

    # -----------------------------
    # Phase 2: 仅推理/评估
    #   - 假设 WAT 记忆库已在 Phase 1 中训练好并保存；
    #   - 这里只负责：加载 WAT 记忆库，在 test 上做推理/评估；
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
        # 推理阶段允许用当前 CLI 的 --anomaly_scorer_k 覆盖已保存 bank 的 top-k。
        # 便于快速做 k 敏感性分析（不必重训 memory bank）。
        if int(args.anomaly_scorer_k) > 0 and pc.anomaly_scorer.n_nearest_neighbours != int(args.anomaly_scorer_k):
            LOGGER.info(
                "Override loaded top-k for bank '%s': %d -> %d",
                bank,
                pc.anomaly_scorer.n_nearest_neighbours,
                int(args.anomaly_scorer_k),
            )
            pc.anomaly_scorer.n_nearest_neighbours = int(args.anomaly_scorer_k)
        pcs[bank] = pc
        LOGGER.info("Loaded WAT memory bank for '%s' from %s", bank, load_dir)

    # [1.6-测试载入] 每个 dataset 各自构建公共 test/val loader。
    build_dataloader_test = get_dataloaders(
        data_path=data_path,
        dataset_names=test_dataset_names,
        batch_size=args.batch_size,
        resize=args.resize,
        imagesize=args.imagesize,
        workers=args.workers,
    )
    common_loaders = build_dataloader_test(seed=args.seed, bankname=None)

    # -----------------------------
    # [1.6-统计容器] 数据集级日志 + CSV 行缓存 + overall 全局缓冲区
    # -----------------------------
    # 1) 二分类总体准确率（ai vs nature）
    summary_classification_accuracy_with_uncertainty_logs: Dict[str, float] = {}
    summary_classification_accuracy_on_certain_samples_logs: Dict[str, float] = {}
    summary_classification_uncertainty_rate_logs: Dict[str, float] = {}

    # 2) 真实 AI 样本上的“判别为 AI”准确率
    summary_ai_detection_accuracy_with_uncertainty_logs: Dict[str, float] = {}
    summary_ai_detection_accuracy_on_certain_samples_logs: Dict[str, float] = {}
    summary_ai_detection_certain_sample_coverage_logs: Dict[str, float] = {}

    # 3) 各类 CSV 输出缓存
    per_image_rows: List[List[Any]] = []
    per_dataset_rows: List[List[Any]] = []
    per_dataset_per_generator_rows: List[List[Any]] = []
    overall_per_generator_rows: List[List[Any]] = []

    # 4) 全局统计缓存（跨数据集合并）
    all_labels_total: List[int] = []
    all_predictions_total: List[int] = []
    all_labels_on_certain_samples: List[int] = []
    all_predictions_on_certain_samples: List[int] = []
    all_uncertainty_flags: List[int] = []
    all_ground_truth_generators: List[str] = []
    all_ground_truth_is_ai_flags: List[int] = []
    all_predicted_is_ai_flags: List[int] = []

    # [1.6-逐数据集评估] 每个数据集都走完整 infer -> metrics -> merge 流程。
    for ds_name in test_dataset_names:
        test_loader = common_loaders["test"].get(ds_name)
        if test_loader is None:
            LOGGER.warning("Dataset %s missing test loader, skip evaluation.", ds_name)
            continue

        dataset_result = eval_tools.evaluate_single_test_dataset(
            dataset_name=ds_name,
            test_loader=test_loader,
            model_ai=pcs[bank_ai],
            model_nature=pcs[bank_nature],
            ai_conf_floor=args.ai_conf_floor,
            uncertain_eps=args.uncertain_eps,
            logger=LOGGER,
        )

        # (a) 写入数据集级 summary 字典（供日志打印）
        summary_classification_accuracy_with_uncertainty_logs[ds_name] = dataset_result[
            "classification_accuracy_with_uncertainty"
        ]
        summary_classification_accuracy_on_certain_samples_logs[ds_name] = dataset_result[
            "classification_accuracy_on_certain_samples"
        ]
        summary_classification_uncertainty_rate_logs[ds_name] = dataset_result["classification_uncertainty_rate"]
        summary_ai_detection_accuracy_with_uncertainty_logs[ds_name] = dataset_result[
            "ai_detection_accuracy_with_uncertainty"
        ]
        summary_ai_detection_accuracy_on_certain_samples_logs[ds_name] = dataset_result[
            "ai_detection_accuracy_on_certain_samples"
        ]
        summary_ai_detection_certain_sample_coverage_logs[ds_name] = dataset_result[
            "ai_detection_certain_sample_coverage"
        ]
        # (b) 写入 CSV 缓存（样本级 / 数据集级 / 数据集-生成器级）
        per_dataset_rows.append(dataset_result["per_dataset_row"])
        per_dataset_per_generator_rows.extend(dataset_result["per_dataset_per_generator_rows"])
        per_image_rows.extend(dataset_result["per_image_rows"])

        # (c) 写入 overall 缓冲区（跨数据集合并）
        global_buffers = dataset_result["global_buffers"]
        all_labels_total.extend(global_buffers["labels_total"])
        all_predictions_total.extend(global_buffers["predictions_total"])
        all_labels_on_certain_samples.extend(global_buffers["labels_on_certain_samples"])
        all_predictions_on_certain_samples.extend(global_buffers["predictions_on_certain_samples"])
        all_uncertainty_flags.extend(global_buffers["uncertainty_flags"])
        all_ground_truth_generators.extend(global_buffers["ground_truth_generators"])
        all_ground_truth_is_ai_flags.extend(global_buffers["ground_truth_is_ai_flags"])
        all_predicted_is_ai_flags.extend(global_buffers["predicted_is_ai_flags"])
    # [1.4] 按样本导出明细（用于后续误判排查）。
    eval_tools.save_csv_rows(
        output_csv_path=os.path.join("runs", "test_scores.csv"),
        header=[
            "dataset_name",
            "image_path",
            "anomaly_score_from_ai_bank",
            "anomaly_score_from_nature_bank",
            "relative_difference_to_nature_score",
            "symmetric_score_difference",
            "ground_truth_is_ai",
            "predicted_is_ai_with_uncertainty",
            "predicted_label_text",
            "uncertainty_flag",
            "raw_margin_nature_minus_ai",
            "confidence_adjusted_margin_nature_minus_ai",
            "ai_confidence_gate_weight",
            "predicted_generator_for_final_label",
            "predicted_generator_from_ai_bank",
            "predicted_generator_confidence_from_ai_bank",
            "predicted_generator_base_confidence_before_diversity_penalty",
            "predicted_generator_diversity_penalty",
            "topk_unique_label_count",
            "topk_entropy_normalized",
            "topk_unique_ratio",
            "predicted_label_from_nature_bank",
            "ground_truth_generator_name",
            "ground_truth_dataset_name",
        ],
        rows=per_image_rows,
        success_message="Saved per-image test scores to %s",
        logger=LOGGER,
    )

    # [1.4] 按数据集导出汇总指标。
    eval_tools.save_csv_rows(
        output_csv_path=os.path.join("runs", "per_dataset_ai_evaluation_summary.csv"),
        header=[
            "dataset_name",
            "total_sample_count",
            "certain_prediction_sample_count",
            "true_ai_sample_count",
            "true_ai_certain_prediction_sample_count",
            "true_ai_predicted_as_ai_sample_count",
            "classification_accuracy_with_uncertainty",
            "classification_accuracy_on_certain_samples",
            "classification_certain_sample_coverage",
            "classification_uncertainty_rate",
            "ai_detection_accuracy_with_uncertainty",
            "ai_detection_accuracy_on_certain_samples",
            "ai_detection_certain_sample_coverage",
        ],
        rows=per_dataset_rows,
        success_message="Saved per-dataset AI evaluation summary to %s",
        logger=LOGGER,
    )

    # [1.4] 按数据集 + 按生成器导出细分指标。
    eval_tools.save_csv_rows(
        output_csv_path=os.path.join("runs", "per_dataset_per_ai_generator_evaluation_summary.csv"),
        header=[
            "dataset_name",
            "ground_truth_ai_generator_name",
            "true_ai_sample_count",
            "true_ai_certain_prediction_sample_count",
            "true_ai_predicted_as_ai_sample_count",
            "ai_detection_accuracy_with_uncertainty",
            "ai_detection_accuracy_on_certain_samples",
            "ai_detection_certain_sample_coverage",
        ],
        rows=per_dataset_per_generator_rows,
        success_message="Saved per-dataset per-AI-generator summary to %s",
        logger=LOGGER,
    )

    # [1.6-日志摘要] 仅在对应 summary 字典非空时打印。
    if summary_classification_accuracy_with_uncertainty_logs:
        LOGGER.info(
            "Per-dataset classification_accuracy_with_uncertainty: %s",
            {k: round(v, 6) for k, v in summary_classification_accuracy_with_uncertainty_logs.items()},
        )
    if summary_classification_accuracy_on_certain_samples_logs:
        LOGGER.info(
            "Per-dataset classification_accuracy_on_certain_samples: %s",
            {k: round(v, 6) for k, v in summary_classification_accuracy_on_certain_samples_logs.items()},
        )
    if summary_classification_uncertainty_rate_logs:
        LOGGER.info(
            "Per-dataset classification_uncertainty_rate: %s",
            {k: round(v, 6) for k, v in summary_classification_uncertainty_rate_logs.items()},
        )
    if summary_ai_detection_accuracy_with_uncertainty_logs:
        LOGGER.info(
            "Per-dataset ai_detection_accuracy_with_uncertainty: %s",
            {k: round(v, 6) for k, v in summary_ai_detection_accuracy_with_uncertainty_logs.items()},
        )
    if summary_ai_detection_accuracy_on_certain_samples_logs:
        LOGGER.info(
            "Per-dataset ai_detection_accuracy_on_certain_samples: %s",
            {k: round(v, 6) for k, v in summary_ai_detection_accuracy_on_certain_samples_logs.items()},
        )
    overall_classification_accuracy_with_uncertainty = float("nan")
    overall_classification_accuracy_on_certain_samples = float("nan")
    overall_classification_certain_sample_coverage = float("nan")
    overall_classification_uncertainty_rate = float("nan")
    overall_ai_detection_accuracy_with_uncertainty = float("nan")
    overall_ai_detection_accuracy_on_certain_samples = float("nan")
    overall_ai_detection_certain_sample_coverage = float("nan")
    overall_true_ai_sample_count = 0
    overall_true_ai_certain_sample_count = 0
    overall_true_ai_samples_predicted_as_ai_count = 0

    # [1.5-overall] 先算 uncertain 比率与 certain 覆盖率。
    if all_labels_total:
        total_sample_count = len(all_labels_total)
        overall_classification_uncertainty_rate = float(np.mean(np.asarray(all_uncertainty_flags, dtype=int)))
        overall_classification_certain_sample_coverage = 1.0 - overall_classification_uncertainty_rate
        LOGGER.info(
            "Overall classification_certain_sample_coverage=%.4f, classification_uncertainty_rate=%.4f (samples=%d)",
            overall_classification_certain_sample_coverage,
            overall_classification_uncertainty_rate,
            total_sample_count,
        )

    # [1.5-overall] 在全样本上算二分类准确率，并在真实 AI 子集上算检测指标。
    if all_labels_total and all_predictions_total:
        labels_all_np = np.asarray(all_labels_total, dtype=int)
        predictions_all_np = np.asarray(all_predictions_total, dtype=int)
        overall_classification_accuracy_with_uncertainty = float((labels_all_np == predictions_all_np).mean())
        LOGGER.info(
            "Overall classification_accuracy_with_uncertainty: %.4f (samples=%d)",
            overall_classification_accuracy_with_uncertainty,
            len(labels_all_np),
        )
        (
            overall_ai_detection_accuracy_with_uncertainty,
            overall_ai_detection_accuracy_on_certain_samples,
            overall_ai_detection_certain_sample_coverage,
            overall_true_ai_sample_count,
            overall_true_ai_certain_sample_count,
        ) = eval_tools.compute_ai_detection_metrics(
            predicted_is_ai=predictions_all_np,
            true_ai_mask=(labels_all_np == 1),
        )
        LOGGER.info(
            "Overall ai_detection_accuracy_with_uncertainty=%.4f, "
            "ai_detection_accuracy_on_certain_samples=%.4f, ai_detection_certain_sample_coverage=%.4f "
            "(true_ai_samples=%d, true_ai_certain_samples=%d)",
            overall_ai_detection_accuracy_with_uncertainty,
            overall_ai_detection_accuracy_on_certain_samples,
            overall_ai_detection_certain_sample_coverage,
            overall_true_ai_sample_count,
            overall_true_ai_certain_sample_count,
        )

    # [1.5-overall] 在 certain 子集上单独计算分类准确率。
    if all_labels_on_certain_samples:
        labels_on_certain_samples_np = np.asarray(all_labels_on_certain_samples, dtype=int)
        predictions_on_certain_samples_np = np.asarray(all_predictions_on_certain_samples, dtype=int)
        overall_classification_accuracy_on_certain_samples = float(
            (labels_on_certain_samples_np == predictions_on_certain_samples_np).mean()
        )
        LOGGER.info(
            "Overall classification_accuracy_on_certain_samples: %.4f (samples=%d)",
            overall_classification_accuracy_on_certain_samples,
            len(labels_on_certain_samples_np),
        )

    # [1.5-overall] 全局按 generator 细分统计 AI 检测指标。
    if all_ground_truth_generators and all_ground_truth_is_ai_flags:
        # 把列表转成 numpy，便于做掩码切片统计。
        ground_truth_generator_np = np.asarray(all_ground_truth_generators, dtype=object)
        ground_truth_is_ai_mask = np.asarray(all_ground_truth_is_ai_flags, dtype=int) == 1
        predictions_all_np = np.asarray(all_predictions_total, dtype=int)
        predicted_is_ai_mask = predictions_all_np == 1

        # 样本长度必须一致，否则说明上游聚合逻辑错位，直接显式报错。
        if len(predictions_all_np) != len(ground_truth_generator_np):
            raise RuntimeError("Global prediction length does not match global generator label length.")

        # 仅保留计数：真实 AI 中有多少被判为 AI。
        true_ai_samples_predicted_as_ai_mask = ground_truth_is_ai_mask & predicted_is_ai_mask
        overall_true_ai_samples_predicted_as_ai_count = int(true_ai_samples_predicted_as_ai_mask.sum())

        # [1.5-overall] 逐 generator 统计（全局口径）。
        unique_generators = sorted(
            {str(x) for x in ground_truth_generator_np[ground_truth_is_ai_mask].tolist()}
        ) if int(ground_truth_is_ai_mask.sum()) > 0 else []
        for generator_name in unique_generators:
            generator_true_ai_mask = ground_truth_is_ai_mask & (ground_truth_generator_np == generator_name)
            (
                generator_ai_detection_accuracy_with_uncertainty,
                generator_ai_detection_accuracy_on_certain_samples,
                generator_ai_detection_certain_sample_coverage,
                generator_true_ai_sample_count,
                generator_true_ai_certain_sample_count,
            ) = eval_tools.compute_ai_detection_metrics(
                predicted_is_ai=predictions_all_np,
                true_ai_mask=generator_true_ai_mask,
            )

            generator_true_ai_predicted_as_ai_mask = generator_true_ai_mask & predicted_is_ai_mask
            generator_true_ai_predicted_as_ai_count = int(generator_true_ai_predicted_as_ai_mask.sum())

            overall_per_generator_rows.append([
                generator_name,
                generator_true_ai_sample_count,
                generator_true_ai_certain_sample_count,
                generator_true_ai_predicted_as_ai_count,
                generator_ai_detection_accuracy_with_uncertainty,
                generator_ai_detection_accuracy_on_certain_samples,
                generator_ai_detection_certain_sample_coverage,
            ])

    # [1.4] 保存 overall 汇总（单行）。
    eval_tools.save_csv_rows(
        output_csv_path=os.path.join("runs", "overall_ai_evaluation_summary.csv"),
        header=[
            "classification_accuracy_with_uncertainty",
            "classification_accuracy_on_certain_samples",
            "classification_certain_sample_coverage",
            "classification_uncertainty_rate",
            "ai_detection_accuracy_with_uncertainty",
            "ai_detection_accuracy_on_certain_samples",
            "ai_detection_certain_sample_coverage",
            "true_ai_sample_count",
            "true_ai_certain_prediction_sample_count",
            "true_ai_predicted_as_ai_sample_count",
        ],
        rows=[[
            overall_classification_accuracy_with_uncertainty,
            overall_classification_accuracy_on_certain_samples,
            overall_classification_certain_sample_coverage,
            overall_classification_uncertainty_rate,
            overall_ai_detection_accuracy_with_uncertainty,
            overall_ai_detection_accuracy_on_certain_samples,
            overall_ai_detection_certain_sample_coverage,
            overall_true_ai_sample_count,
            overall_true_ai_certain_sample_count,
            overall_true_ai_samples_predicted_as_ai_count,
        ]],
        success_message="Saved overall AI evaluation summary to %s",
        logger=LOGGER,
    )

    # [1.4] 保存 overall 按 generator 细分汇总。
    eval_tools.save_csv_rows(
        output_csv_path=os.path.join("runs", "overall_per_ai_generator_evaluation_summary.csv"),
        header=[
            "ground_truth_ai_generator_name",
            "true_ai_sample_count",
            "true_ai_certain_prediction_sample_count",
            "true_ai_predicted_as_ai_sample_count",
            "ai_detection_accuracy_with_uncertainty",
            "ai_detection_accuracy_on_certain_samples",
            "ai_detection_certain_sample_coverage",
        ],
        rows=overall_per_generator_rows,
        success_message="Saved overall per-AI-generator summary to %s",
        logger=LOGGER,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    HOME = os.path.expanduser("~/wat")
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
        # 'sdv5_bigval',
        # 'sdv5 mini', 
    ], help="用于训练 WAT 记忆库的数据集名称列表（train split）。")
    parser.add_argument(
        "--test_dataset_names",
        nargs="+",
        default=[
        'adm',
        'biggan', 
        'glide', 
        'midjourney', 
        'sdv5', 
        'vqdm', 
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
    parser.add_argument("--anomaly_scorer_k", type=int, default=20, help="异常分数计算时使用的 k 值")

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
    parser.add_argument("--sampler_name", default="random",)
    parser.add_argument("--coreset_percentage", type=float, default=0.1)
    # FAISS / 设备
    parser.add_argument("--faiss_on_gpu", action="store_true")
    parser.add_argument("--faiss_num_workers", type=int, default=12)
    parser.add_argument("--nn_method", choices=["auto", "flat", "ivfpq"], default="auto")
    parser.add_argument(
        "--ai_conf_floor",
        type=float,
        default=1.0,
        help="将 ai_generator_conf 融入分类 margin 时的最小保留权重（0~1，越小越依赖 conf；1.0 表示不衰减 margin）。",
    )
    parser.add_argument(
        "--uncertain_eps",
        type=float,
        default=0.002,
        help="不确定区间阈值：|adjusted_margin| < eps 时输出 uncertain。",
    )

    # 两个 bank 名称
    parser.add_argument("--banknames", nargs="+", default=['ai','nature'],)

    args = parser.parse_args()
    main(args)
                                                                                                                                
                                                                                                                                 