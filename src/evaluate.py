"""
Model Evaluation and Metric Calculation Engine.

Calculates:
- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1-Score
- Confusion Matrix (TN, FP, FN, TP)
"""

from typing import Dict, Any, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)


def calculate_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive clinical diagnostic evaluation metrics.
    
    Returns dictionary with:
    - accuracy
    - precision
    - recall
    - specificity
    - f1_score
    - roc_auc (if probabilities provided)
    - confusion_matrix: dict with TN, FP, FN, TP and 2x2 list
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    cm = confusion_matrix(y_true_arr, y_pred_arr)
    
    # Extract TN, FP, FN, TP
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Edge case handling if only 1 class in sample
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(np.unique(y_true_arr)) == 1 and y_true_arr[0] == 0:
            tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
            fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
        elif len(np.unique(y_true_arr)) == 1 and y_true_arr[0] == 1:
            fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
            tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))

    # Metrics
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    
    # Specificity = TN / (TN + FP)
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics = {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'specificity': round(spec, 4),
        'f1_score': round(f1, 4),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]
        }
    }

    if y_prob is not None:
        try:
            if len(np.shape(y_prob)) > 1:
                prob_pos = y_prob[:, 1]
            else:
                prob_pos = y_prob
            roc_auc = float(roc_auc_score(y_true_arr, prob_pos))
            metrics['roc_auc'] = round(roc_auc, 4)
        except Exception:
            metrics['roc_auc'] = None

    return metrics
