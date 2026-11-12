"""Fraud-focused metrics and validation-only F1 decision threshold tuning."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)


@dataclass
class FraudMetrics:
    precision: float
    recall: float
    f1: float
    average_precision: float
    tn: int
    fp: int
    fn: int
    tp: int


def validation_f1_threshold(y_true: np.ndarray, fraud_prob: np.ndarray) -> float:
    """Optimize the classification threshold on VALIDATION data, never test."""
    precision, recall, thresholds = precision_recall_curve(y_true, fraud_prob)
    if len(thresholds) == 0:
        return 0.5
    # The final precision/recall entry has no corresponding threshold.
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(thresholds[int(np.argmax(f1))])


def compute_fraud_metrics(
    y_true: np.ndarray, fraud_prob: np.ndarray, threshold: float
) -> FraudMetrics:
    pred = (fraud_prob >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return FraudMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        average_precision=float(average_precision_score(y_true, fraud_prob)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )
