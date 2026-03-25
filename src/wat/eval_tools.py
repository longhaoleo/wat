"""
评估工具模块（Evaluation Tools）。

本文件专门承载“推理后评估”相关逻辑，避免 `bin/run_wat.py` 中堆叠大量指标与 CSV 细节。
设计原则：
1) `run_wat.py` 只做流程编排（加载模型、循环数据集、写结果）；
2) 本模块负责“如何计算指标”的细节；
3) 指标命名尽量完整，方便后续论文复现实验与结果审计。

本文件对应 `PROJECT_DETAILED_COMMENTS.md` 第 2 节：
- `fuse_ai_margin_with_conf(...)`
- `compute_ai_detection_metrics(...)`
- `_run_dual_bank_inference(...)`
- `_compute_classification_metrics(...)`
- `_compute_ai_generator_metrics(...)`
- `_build_per_image_rows(...)`
- `evaluate_single_test_dataset(...)`
- `save_csv_rows(...)`
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch


LOGGER = logging.getLogger(__name__)


def fuse_ai_margin_with_conf(
    raw_margin: np.ndarray,
    ai_conf: np.ndarray,
    conf_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    用 AI generator 置信度修正二分类 margin。

    输入口径：
    - `raw_margin = score_nature - score_ai`
      - raw_margin > 0：更偏向 AI；
      - raw_margin < 0：更偏向 nature。
    - `ai_conf`：AI memory bank 给出的 generator 置信度（理论范围 [0, 1]）。
    - `conf_floor`：保底权重，避免低置信度时 margin 被压到几乎 0。

    计算：
    - `gate = conf_floor + (1 - conf_floor) * clip(ai_conf, 0, 1)`
    - `adjusted_margin = raw_margin * gate`

    返回：
    - `adjusted_margin`：融合置信度后的 margin；
    - `gate`：每个样本对应的融合权重（用于审计和 CSV 输出）。
    """
    conf = np.asarray(ai_conf, dtype=float).reshape(-1)
    # 非有限数统一视为 0，避免 NaN/Inf 传播到后续计算。
    conf = np.where(np.isfinite(conf), conf, 0.0)
    conf = np.clip(conf, 0.0, 1.0)

    # floor 也做一次裁剪，防御异常入参。
    conf_floor = float(np.clip(conf_floor, 0.0, 1.0))
    gate = conf_floor + (1.0 - conf_floor) * conf
    return np.asarray(raw_margin, dtype=float).reshape(-1) * gate, gate


def compute_ai_detection_metrics(
    predicted_is_ai: np.ndarray,
    true_ai_mask: np.ndarray,
) -> tuple[float, float, float, int, int]:
    """
    统计“判别为 AI”的准确率（只在真实 AI 样本上）。

    这是一个容易混淆的指标，明确两种口径：
    1) `accuracy_with_uncertainty`
       - 基于真实 AI 样本全集；
       - 仅当预测为 1 才算对；
       - uncertain(-1) 和预测为 0 都算错。
    2) `accuracy_on_certain_samples`
       - 先筛掉 uncertain(-1)；
       - 只在 certain 的真实 AI 样本上计算预测为 1 的比例。

    额外返回：
    - `certain_sample_coverage`：真实 AI 样本中，被模型“确定给出 ai/nature”判断的覆盖率；
    - 样本数信息，便于日志里解释数值可信度。
    """
    predictions = np.asarray(predicted_is_ai, dtype=int).reshape(-1)
    mask = np.asarray(true_ai_mask, dtype=bool).reshape(-1)
    true_ai_sample_count = int(mask.sum())

    if true_ai_sample_count == 0:
        return float("nan"), float("nan"), 0.0, 0, 0

    predictions_on_true_ai = predictions[mask]
    accuracy_with_uncertainty = float((predictions_on_true_ai == 1).mean())

    certain_mask = predictions_on_true_ai != -1
    true_ai_certain_sample_count = int(certain_mask.sum())
    if true_ai_certain_sample_count > 0:
        accuracy_on_certain_samples = float((predictions_on_true_ai[certain_mask] == 1).mean())
    else:
        accuracy_on_certain_samples = float("nan")

    certain_sample_coverage = float(true_ai_certain_sample_count / true_ai_sample_count)
    return (
        accuracy_with_uncertainty,
        accuracy_on_certain_samples,
        certain_sample_coverage,
        true_ai_sample_count,
        true_ai_certain_sample_count,
    )


def _run_dual_bank_inference(
    dataset_name: str,
    test_loader: torch.utils.data.DataLoader,
    model_ai: Any,
    model_nature: Any,
    ai_conf_floor: float,
    uncertain_eps: float,
) -> Dict[str, Any]:
    """
    执行双 bank 推理并统一整理成结构化字段。

    该函数把“模型输出”转成“评估输入”：
    - 原始分数（ai/nature）；
    - 真实标签与 generator 真值；
    - 预测 generator 与置信度；
    - 最终三值分类（1/0/-1）及相关中间量（margin/gate/uncertainty）。
    """
    # 1) AI bank 推理：得到 ai score + 最近邻 generator 及其置信度。
    ai_predict_outputs = model_ai.predict_with_meta(test_loader)
    if len(ai_predict_outputs) >= 13:
        (
            test_scores_ai,
            _,
            labels_is_ai,
            test_paths,
            predicted_generators_from_ai_bank,
            predicted_generator_confidences,
            ground_truth_generators,
            ground_truth_dataset_names,
            predicted_generator_base_confidences,
            predicted_generator_diversity_penalties,
            topk_unique_label_counts,
            topk_entropy_normalized_values,
            topk_unique_ratio_values,
        ) = ai_predict_outputs
    else:
        (
            test_scores_ai,
            _,
            labels_is_ai,
            test_paths,
            predicted_generators_from_ai_bank,
            predicted_generator_confidences,
            ground_truth_generators,
            ground_truth_dataset_names,
        ) = ai_predict_outputs
        predicted_generator_base_confidences = None
        predicted_generator_diversity_penalties = None
        topk_unique_label_counts = None
        topk_entropy_normalized_values = None
        topk_unique_ratio_values = None

    # 2) nature bank 推理：得到 nature score + 最近邻 nature 标签。
    nature_predict_outputs = model_nature.predict_with_meta(test_loader)
    (
        test_scores_nature,
        _,
        labels_is_ai,
        _,
        predicted_nature_labels,
        _,
        _,
        _,
    ) = nature_predict_outputs[:8]

    test_scores_ai_np = np.asarray(test_scores_ai, dtype=float).reshape(-1)
    test_scores_nature_np = np.asarray(test_scores_nature, dtype=float).reshape(-1)
    labels_is_ai_np = np.asarray(labels_is_ai, dtype=int).reshape(-1)

    image_paths = [str(p) for p in test_paths] if test_paths is not None else ["-"] * len(test_scores_ai_np)
    predicted_generators = [str(x) for x in (predicted_generators_from_ai_bank or ["unknown"] * len(test_scores_ai_np))]
    predicted_generator_confidence_np = np.asarray(
        predicted_generator_confidences or [float("nan")] * len(test_scores_ai_np),
        dtype=float,
    ).reshape(-1)
    predicted_generator_base_confidence_np = np.asarray(
        predicted_generator_base_confidences or [float("nan")] * len(test_scores_ai_np),
        dtype=float,
    ).reshape(-1)
    predicted_generator_diversity_penalty_np = np.asarray(
        predicted_generator_diversity_penalties or [float("nan")] * len(test_scores_ai_np),
        dtype=float,
    ).reshape(-1)
    topk_unique_label_count_np = np.asarray(
        topk_unique_label_counts or [float("nan")] * len(test_scores_ai_np),
        dtype=float,
    ).reshape(-1)
    topk_entropy_normalized_np = np.asarray(
        topk_entropy_normalized_values or [float("nan")] * len(test_scores_ai_np),
        dtype=float,
    ).reshape(-1)
    topk_unique_ratio_np = np.asarray(
        topk_unique_ratio_values or [float("nan")] * len(test_scores_ai_np),
        dtype=float,
    ).reshape(-1)
    predicted_nature = [str(x) for x in (predicted_nature_labels or ["nature"] * len(test_scores_ai_np))]
    ground_truth_generator_labels = [str(x) for x in (ground_truth_generators or ["unknown"] * len(test_scores_ai_np))]
    ground_truth_dataset_labels = [str(x) for x in (ground_truth_dataset_names or [dataset_name] * len(test_scores_ai_np))]

    # 3) 先融合 generator 置信度，再做 uncertain 判定。
    raw_margin = (test_scores_nature_np - test_scores_ai_np)
    adjusted_margin, confidence_gate = fuse_ai_margin_with_conf(
        raw_margin=raw_margin,
        ai_conf=predicted_generator_confidence_np,
        conf_floor=ai_conf_floor,
    )
    # margin 接近 0 的样本输出 uncertain(-1)。
    uncertainty_mask = np.abs(adjusted_margin) < float(uncertain_eps)
    predictions_is_ai = np.where(uncertainty_mask, -1, (adjusted_margin > 0.0).astype(int))
    certain_mask = predictions_is_ai != -1
    true_ai_mask = (labels_is_ai_np == 1)

    return {
        "test_scores_ai_np": test_scores_ai_np,
        "test_scores_nature_np": test_scores_nature_np,
        "labels_is_ai_np": labels_is_ai_np,
        "image_paths": image_paths,
        "predicted_generators": predicted_generators,
        "predicted_generator_confidence_np": predicted_generator_confidence_np,
        "predicted_generator_base_confidence_np": predicted_generator_base_confidence_np,
        "predicted_generator_diversity_penalty_np": predicted_generator_diversity_penalty_np,
        "topk_unique_label_count_np": topk_unique_label_count_np,
        "topk_entropy_normalized_np": topk_entropy_normalized_np,
        "topk_unique_ratio_np": topk_unique_ratio_np,
        "predicted_nature": predicted_nature,
        "ground_truth_generator_labels": ground_truth_generator_labels,
        "ground_truth_dataset_labels": ground_truth_dataset_labels,
        "raw_margin": raw_margin,
        "adjusted_margin": adjusted_margin,
        "confidence_gate": confidence_gate,
        "uncertainty_mask": uncertainty_mask,
        "predictions_is_ai": predictions_is_ai,
        "certain_mask": certain_mask,
        "true_ai_mask": true_ai_mask,
    }


def _compute_classification_metrics(
    predictions_is_ai: np.ndarray,
    labels_is_ai_np: np.ndarray,
    certain_mask: np.ndarray,
) -> Dict[str, float]:
    """
    计算二分类总体指标（ai vs nature）。

    指标定义：
    - `classification_accuracy_with_uncertainty`：uncertain(-1) 自动算错；
    - `classification_accuracy_on_certain_samples`：仅在 certain 样本上统计；
    - `classification_certain_sample_coverage`：certain 占比；
    - `classification_uncertainty_rate`：1 - coverage。
    """
    classification_accuracy_with_uncertainty = (
        float((predictions_is_ai == labels_is_ai_np).mean()) if len(labels_is_ai_np) else float("nan")
    )
    if certain_mask.any():
        classification_accuracy_on_certain_samples = float(
            (predictions_is_ai[certain_mask] == labels_is_ai_np[certain_mask]).mean()
        )
    else:
        classification_accuracy_on_certain_samples = float("nan")

    classification_certain_sample_coverage = float(certain_mask.mean()) if len(certain_mask) else 0.0
    classification_uncertainty_rate = 1.0 - classification_certain_sample_coverage
    return {
        "classification_accuracy_with_uncertainty": classification_accuracy_with_uncertainty,
        "classification_accuracy_on_certain_samples": classification_accuracy_on_certain_samples,
        "classification_certain_sample_coverage": classification_certain_sample_coverage,
        "classification_uncertainty_rate": classification_uncertainty_rate,
    }


def _compute_ai_generator_metrics(
    dataset_name: str,
    predictions_is_ai: np.ndarray,
    true_ai_mask: np.ndarray,
    ground_truth_generator_labels: List[str],
) -> Dict[str, Any]:
    """
    按 generator 分组统计 AI 检测指标（不含 generator 分类准确率）。
    """
    ground_truth_generator_np = np.asarray([str(x) for x in ground_truth_generator_labels], dtype=object)

    true_ai_sample_count = int(true_ai_mask.sum())

    # per-dataset-per-generator 行：直接用于 CSV 输出。
    per_dataset_per_generator_rows: List[List[Any]] = []
    unique_generators = sorted(
        {str(x) for x in ground_truth_generator_np[true_ai_mask].tolist()}
    ) if true_ai_sample_count > 0 else []

    # 对当前数据集中的每个真实 generator 单独计算检测/分类指标。
    for generator_name in unique_generators:
        generator_true_ai_mask = true_ai_mask & (ground_truth_generator_np == generator_name)
        (
            generator_ai_detection_accuracy_with_uncertainty,
            generator_ai_detection_accuracy_on_certain_samples,
            generator_ai_detection_certain_sample_coverage,
            generator_true_ai_sample_count,
            generator_true_ai_certain_sample_count,
        ) = compute_ai_detection_metrics(
            predicted_is_ai=predictions_is_ai,
            true_ai_mask=generator_true_ai_mask,
        )

        generator_true_ai_predicted_as_ai_mask = generator_true_ai_mask & (predictions_is_ai == 1)
        generator_true_ai_predicted_as_ai_count = int(generator_true_ai_predicted_as_ai_mask.sum())

        per_dataset_per_generator_rows.append([
            dataset_name,
            generator_name,
            generator_true_ai_sample_count,
            generator_true_ai_certain_sample_count,
            generator_true_ai_predicted_as_ai_count,
            generator_ai_detection_accuracy_with_uncertainty,
            generator_ai_detection_accuracy_on_certain_samples,
            generator_ai_detection_certain_sample_coverage,
        ])

    return {
        "per_dataset_per_generator_rows": per_dataset_per_generator_rows,
    }


def _build_per_image_rows(
    dataset_name: str,
    inference_outputs: Dict[str, Any],
) -> List[List[Any]]:
    """
    构造逐样本 CSV 行。

    这里把推理中间量（raw/adjusted margin、gate）与最终预测一起输出，
    目的是支持后续误判溯源和阈值调参分析。
    """
    per_image_rows: List[List[Any]] = []
    for (
        path,
        score_ai,
        score_nature,
        label_is_ai,
        prediction_is_ai,
        predicted_generator,
        predicted_generator_confidence,
        predicted_generator_base_confidence,
        predicted_generator_diversity_penalty,
        topk_unique_label_count,
        topk_entropy_normalized,
        topk_unique_ratio,
        predicted_nature_label,
        gt_generator,
        gt_dataset_name,
        margin_raw,
        margin_adjusted,
        gate,
        is_uncertain,
    ) in zip(
        inference_outputs["image_paths"],
        inference_outputs["test_scores_ai_np"],
        inference_outputs["test_scores_nature_np"],
        inference_outputs["labels_is_ai_np"],
        inference_outputs["predictions_is_ai"],
        inference_outputs["predicted_generators"],
        inference_outputs["predicted_generator_confidence_np"],
        inference_outputs["predicted_generator_base_confidence_np"],
        inference_outputs["predicted_generator_diversity_penalty_np"],
        inference_outputs["topk_unique_label_count_np"],
        inference_outputs["topk_entropy_normalized_np"],
        inference_outputs["topk_unique_ratio_np"],
        inference_outputs["predicted_nature"],
        inference_outputs["ground_truth_generator_labels"],
        inference_outputs["ground_truth_dataset_labels"],
        inference_outputs["raw_margin"],
        inference_outputs["adjusted_margin"],
        inference_outputs["confidence_gate"],
        inference_outputs["uncertainty_mask"],
    ):
        relative_difference_to_nature_score = (score_ai - score_nature) / (abs(score_nature) + 1e-8)
        symmetric_difference = (score_ai - score_nature) / (abs(score_ai) + abs(score_nature) + 1e-8)
        # 二分类结果转可读标签：
        # 1 -> ai, 0 -> nature, -1 -> uncertain。
        if int(prediction_is_ai) == 1:
            predicted_generator_for_output = predicted_generator
            predicted_label = "ai"
        elif int(prediction_is_ai) == 0:
            predicted_generator_for_output = "nature"
            predicted_label = "nature"
        else:
            predicted_generator_for_output = "uncertain"
            predicted_label = "uncertain"

        per_image_rows.append([
            dataset_name,
            path,
            score_ai,
            score_nature,
            relative_difference_to_nature_score,
            symmetric_difference,
            label_is_ai,
            prediction_is_ai,
            predicted_label,
            int(is_uncertain),
            float(margin_raw),
            float(margin_adjusted),
            float(gate),
            predicted_generator_for_output,
            predicted_generator,
            predicted_generator_confidence,
            predicted_generator_base_confidence,
            predicted_generator_diversity_penalty,
            topk_unique_label_count,
            topk_entropy_normalized,
            topk_unique_ratio,
            predicted_nature_label,
            gt_generator,
            gt_dataset_name,
        ])
    return per_image_rows


def evaluate_single_test_dataset(
    dataset_name: str,
    test_loader: torch.utils.data.DataLoader,
    model_ai: Any,
    model_nature: Any,
    ai_conf_floor: float,
    uncertain_eps: float,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    对单个测试数据集执行完整评估（对外入口）。

    返回内容分成四组：
    1) 数据集级 summary 指标；
    2) 按 generator 分组的 summary 行；
    3) 逐样本输出行；
    4) 全局聚合缓冲区（供上层累计成 overall 指标）。

    返回字段说明（核心）：
    - `classification_accuracy_with_uncertainty`
      二分类准确率，uncertain(-1) 记错。
    - `classification_accuracy_on_certain_samples`
      仅在 `pred_is_ai != -1` 子集上统计二分类准确率。
    - `classification_uncertainty_rate`
      `1 - certain_coverage`。
    - `ai_detection_accuracy_with_uncertainty`
      在真实 AI 样本上，`pred_is_ai == 1` 的比例（uncertain 记错）。
    - `ai_detection_accuracy_on_certain_samples`
      在真实 AI 且 certain 子集上，`pred_is_ai == 1` 的比例。
    - `ai_detection_certain_sample_coverage`
      真实 AI 样本中，被模型给出确定判断的覆盖率。
    """
    log = logger or LOGGER
    # Stage 1) 双 bank 推理，整理统一字段。
    inference_outputs = _run_dual_bank_inference(
        dataset_name=dataset_name,
        test_loader=test_loader,
        model_ai=model_ai,
        model_nature=model_nature,
        ai_conf_floor=ai_conf_floor,
        uncertain_eps=uncertain_eps,
    )

    # Stage 2) 二分类总体指标。
    classification_metrics = _compute_classification_metrics(
        predictions_is_ai=inference_outputs["predictions_is_ai"],
        labels_is_ai_np=inference_outputs["labels_is_ai_np"],
        certain_mask=inference_outputs["certain_mask"],
    )
    log.info(
        "[%s] classification_accuracy_with_uncertainty=%.4f, classification_accuracy_on_certain_samples=%.4f, "
        "classification_certain_sample_coverage=%.4f, classification_uncertainty_rate=%.4f",
        dataset_name,
        classification_metrics["classification_accuracy_with_uncertainty"],
        classification_metrics["classification_accuracy_on_certain_samples"],
        classification_metrics["classification_certain_sample_coverage"],
        classification_metrics["classification_uncertainty_rate"],
    )

    # Stage 3) 真实 AI 子集上的“判为 AI”检测指标。
    (
        ai_detection_accuracy_with_uncertainty,
        ai_detection_accuracy_on_certain_samples,
        ai_detection_certain_sample_coverage,
        true_ai_sample_count,
        true_ai_certain_sample_count,
    ) = compute_ai_detection_metrics(
        predicted_is_ai=inference_outputs["predictions_is_ai"],
        true_ai_mask=inference_outputs["true_ai_mask"],
    )
    log.info(
        "[%s] ai_detection_accuracy_with_uncertainty=%.4f, ai_detection_accuracy_on_certain_samples=%.4f, "
        "ai_detection_certain_sample_coverage=%.4f (true_ai_samples=%d, true_ai_certain_samples=%d)",
        dataset_name,
        ai_detection_accuracy_with_uncertainty,
        ai_detection_accuracy_on_certain_samples,
        ai_detection_certain_sample_coverage,
        true_ai_sample_count,
        true_ai_certain_sample_count,
    )

    # Stage 4) 按 generator 分组统计 AI 检测指标。
    ai_generator_metrics = _compute_ai_generator_metrics(
        dataset_name=dataset_name,
        predictions_is_ai=inference_outputs["predictions_is_ai"],
        true_ai_mask=inference_outputs["true_ai_mask"],
        ground_truth_generator_labels=inference_outputs["ground_truth_generator_labels"],
    )
    true_ai_samples_predicted_as_ai_count = int(
        (inference_outputs["true_ai_mask"] & (inference_outputs["predictions_is_ai"] == 1)).sum()
    )

    # Stage 5) 样本级明细行（用于 test_scores.csv）。
    per_image_rows = _build_per_image_rows(
        dataset_name=dataset_name,
        inference_outputs=inference_outputs,
    )

    return {
        "dataset_name": dataset_name,
        "classification_accuracy_with_uncertainty": classification_metrics["classification_accuracy_with_uncertainty"],
        "classification_accuracy_on_certain_samples": classification_metrics["classification_accuracy_on_certain_samples"],
        "classification_uncertainty_rate": classification_metrics["classification_uncertainty_rate"],
        "ai_detection_accuracy_with_uncertainty": ai_detection_accuracy_with_uncertainty,
        "ai_detection_accuracy_on_certain_samples": ai_detection_accuracy_on_certain_samples,
        "ai_detection_certain_sample_coverage": ai_detection_certain_sample_coverage,
        "per_dataset_row": [
            dataset_name,
            len(inference_outputs["labels_is_ai_np"]),
            int(inference_outputs["certain_mask"].sum()),
            true_ai_sample_count,
            true_ai_certain_sample_count,
            true_ai_samples_predicted_as_ai_count,
            classification_metrics["classification_accuracy_with_uncertainty"],
            classification_metrics["classification_accuracy_on_certain_samples"],
            classification_metrics["classification_certain_sample_coverage"],
            classification_metrics["classification_uncertainty_rate"],
            ai_detection_accuracy_with_uncertainty,
            ai_detection_accuracy_on_certain_samples,
            ai_detection_certain_sample_coverage,
        ],
        "per_dataset_per_generator_rows": ai_generator_metrics["per_dataset_per_generator_rows"],
        "per_image_rows": per_image_rows,
        "global_buffers": {
            "labels_total": inference_outputs["labels_is_ai_np"].tolist(),
            "predictions_total": inference_outputs["predictions_is_ai"].astype(int).tolist(),
            "labels_on_certain_samples": (
                inference_outputs["labels_is_ai_np"][inference_outputs["certain_mask"]].tolist()
                if inference_outputs["certain_mask"].any()
                else []
            ),
            "predictions_on_certain_samples": (
                inference_outputs["predictions_is_ai"][inference_outputs["certain_mask"]].tolist()
                if inference_outputs["certain_mask"].any()
                else []
            ),
            "uncertainty_flags": inference_outputs["uncertainty_mask"].astype(int).tolist(),
            "ground_truth_generators": [str(x) for x in inference_outputs["ground_truth_generator_labels"]],
            "predicted_generators": [str(x) for x in inference_outputs["predicted_generators"]],
            "ground_truth_is_ai_flags": inference_outputs["true_ai_mask"].astype(int).tolist(),
            "predicted_is_ai_flags": (inference_outputs["predictions_is_ai"] == 1).astype(int).tolist(),
        },
    }


def save_csv_rows(
    output_csv_path: str,
    header: List[str],
    rows: List[List[Any]],
    success_message: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    通用 CSV 写出函数。

    使用规范：
    - `rows` 为空时直接返回，不创建空文件；
    - 自动创建父目录；
    - 成功后记录统一日志，避免上层重复样板代码。
    """
    if not rows:
        return
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    (logger or LOGGER).info(success_message, output_csv_path)
