from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def classification_metrics(y_true, y_pred, num_classes=None):
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must contain the same number of labels.")

    if num_classes is None:
        num_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)

    labels = np.arange(num_classes)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    total = matrix.sum()

    true_positive = np.diag(matrix).astype(float)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive
    true_negative = total - true_positive - false_positive - false_negative

    def divide(numerator, denominator):
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=denominator != 0,
        )

    precision = divide(true_positive, true_positive + false_positive)
    sensitivity = divide(true_positive, true_positive + false_negative)
    specificity = divide(true_negative, true_negative + false_positive)
    f1 = divide(2.0 * precision * sensitivity, precision + sensitivity)
    accuracy = float(true_positive.sum() / total) if total else 0.0

    return {
        "accuracy": accuracy,
        "precision_macro": float(precision.mean()),
        "specificity_macro": float(specificity.mean()),
        "sensitivity_macro": float(sensitivity.mean()),
        "f1_macro": float(f1.mean()),
        "per_class": {
            str(index): {
                "precision": float(precision[index]),
                "specificity": float(specificity[index]),
                "sensitivity": float(sensitivity[index]),
                "f1": float(f1[index]),
            }
            for index in labels
        },
        "confusion_matrix": matrix.tolist(),
    }
