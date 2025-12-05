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
        self.save_classifier_path = save_classifier_path
        self.threshold: Optional[float] = None
        self.classifier: Optional[LogisticRegression] = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> LogisticRegression:
        self.classifier = LogisticRegression(max_iter=2000)
        self.classifier.fit(features, labels)
        if self.save_classifier_path:
            joblib.dump(self.classifier, self.save_classifier_path)
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

