# patchcore/evaluator.py
from typing import Dict, Any, Tuple, Callable, List, Optional
import numpy as np
import logging

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)


LOGGER = logging.getLogger("evaluator")

def robust_median_normalize(values: np.ndarray) -> np.ndarray:
    """对 1D 数组做 median/MAD 归一化，弱化长尾影响。"""
    arr = np.asarray(values, dtype=float).reshape(-1)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med)) + 1e-9
    return (arr - med) / mad

def build_feature_matrix(
    scores_from_ai_bank: np.ndarray,
    scores_from_nature_bank: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    统一生成 (N, 3) 的特征矩阵：
        [ai_norm, nature_norm]
    """
    ai_norm = robust_median_normalize(scores_from_ai_bank)
    nature_norm = robust_median_normalize(scores_from_nature_bank)
    feature_matrix = np.stack([ai_norm, nature_norm], axis=1)
    return feature_matrix


class Evaluator:
    def __init__(
        self,
        save_classifier_path: Optional[str] = None,
    ) -> None:
        """
        负责逻辑回归分类器的训练 / 预测与简单评估。

        参数:
            save_classifier_path:
                - 若不为 None，则在 fit() 之后自动将训练好的分类器持久化到该路径；
                - infer 阶段可以通过 load_classifier(...) 从磁盘恢复分类器权重。
        """
        self.save_classifier_path = save_classifier_path
        self.threshold: Optional[float] = None
        self.classifier: Optional[LogisticRegression] = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> LogisticRegression:
        """
        在给定特征/标签上训练一个逻辑回归分类器。

        - features: 形状 [N, D] 的特征矩阵；
        - labels:   形状 [N] 的 0/1 标签向量。

        若构造时提供了 save_classifier_path，则在训练结束后将模型持久化保存。
        """
        self.classifier = LogisticRegression(max_iter=2000)
        self.classifier.fit(features, labels)
        if self.save_classifier_path:
            joblib.dump(self.classifier, self.save_classifier_path)
        return self.classifier

    def load_classifier(self, path: Optional[str] = None) -> LogisticRegression:
        """
        从磁盘加载已经训练好的逻辑回归分类器。

        参数:
            path: 分类器模型文件路径；若为 None，则使用构造时传入的 save_classifier_path。

        返回:
            已加载的 LogisticRegression 实例，并同步赋值到 self.classifier。
        """
        load_path = path or self.save_classifier_path
        if not load_path:
            raise RuntimeError("No classifier path specified for loading.")
        self.classifier = joblib.load(load_path)
        return self.classifier

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("Logistic classifier not fitted yet.")
        return self.classifier.predict_proba(features)[:, 1]

    def evaluate(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[float, Optional[np.ndarray], Optional[str]]:
        auc = roc_auc_score(labels, probabilities)
        preds = None
        report = None
        threshold = self.threshold if self.threshold is not None else 0.5
        preds = (probabilities >= threshold).astype(int)
        report = classification_report(labels, preds, digits=4)
        return auc, preds, report

    def select_threshold(self, scores_for_positive: np.ndarray, labels: np.ndarray):
        """在 PR 曲线 F1 最大点选择阈值，若异常则退化为中位数位置。"""
        precision, recall, thresholds = precision_recall_curve(labels, scores_for_positive)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        best_index = int(np.argmax(f1[:-1]))
        if len(thresholds) / 4 < best_index < 3 * len(thresholds) / 4:
            self.threshold = float(thresholds[best_index])
        self.threshold = float(thresholds[len(thresholds) // 2])
