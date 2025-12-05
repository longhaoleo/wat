import numpy as np
from sklearn import metrics

# 图像级别检索指标计算
def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    计算图像级别的检索统计数据（AUROC、FPR、TPR、精度、召回率、F1、FNR）

    参数:
        anomaly_prediction_weights: [np.array 或 list] 图像的异常预测权重，值越大表示越有可能是异常。
        anomaly_ground_truth_labels: [np.array 或 list] 二进制标签，如果图像是异常则为1，正常则为0。
    
    返回:
        字典，包含 AUROC、精度、召回率、F1 分数、准确率和最佳阈值等统计指标。
    """
    # 计算 AUROC
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )

    # 计算精度、召回率、PR曲线的阈值
    precision, recall, pr_thresholds = metrics.precision_recall_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    
    m = 1  # F1分数的参数
    # 计算 F1 分数
    fm_scores = (1 + m**2) * (precision * recall) / (m**2 * precision + recall)
    optimal_threshold = pr_thresholds[np.argmax(fm_scores)]  # 获取最大 F1 对应的阈值

    # 在最佳阈值下计算精度、召回率、F1 和准确率
    predictions = (anomaly_prediction_weights >= optimal_threshold).astype(int)
    precision_at_optimal = metrics.precision_score(
        anomaly_ground_truth_labels, predictions
    )
    recall_at_optimal = metrics.recall_score(
        anomaly_ground_truth_labels, predictions
    )
    f1_at_optimal = metrics.f1_score(
        anomaly_ground_truth_labels, predictions
    )
    accuracy = metrics.accuracy_score(anomaly_ground_truth_labels, predictions)

    # 返回所有计算的指标
    return {
        "auroc": auroc,
        "precision": precision_at_optimal,
        "recall": recall_at_optimal,
        "f1_score": f1_at_optimal,
        "accuracy": accuracy,
        "optimal_threshold": optimal_threshold,
    }

# 像素级别检索指标计算
def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    计算像素级别的检索统计数据（AUROC、FPR、TPR）用于异常分割结果和地面真值掩码。

    参数:
        anomaly_segmentations: [np.array 或 list] 生成的异常分割掩码，形状为 NxHxW。
        ground_truth_masks: [np.array 或 list] 地面真值掩码，形状为 NxHxW。

    返回:
        字典，包含 AUROC、FPR、TPR、最佳阈值、最佳 FPR 和 FNR 等统计指标。
    """
    # 将列表转换为 NumPy 数组，如果输入是列表的话
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    # 展开所有像素点
    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    # 计算 FPR 和 TPR 以及 ROC 曲线的阈值
    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    # 计算 AUROC
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    # 计算精度、召回率、阈值
    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    # 计算 F1 分数
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    # 获取最佳阈值（即 F1 分数最大的点）
    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    # 计算最佳 FPR 和 FNR
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    # 返回所有计算的指标
    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }
