"""
Evaluation metrics for multi-label odor classification.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    roc_auc_score,
    average_precision_score,
    classification_report,
)


def compute_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Compute comprehensive metrics for multi-label classification.
    
    Args:
        y_true: (N, num_classes) binary ground truth labels
        y_pred_probs: (N, num_classes) predicted probabilities [0, 1]
        threshold: Threshold for converting probabilities to binary predictions
        
    Returns:
        dict with all computed metrics
    """
    y_pred_binary = (y_pred_probs >= threshold).astype(np.float32)
    
    metrics = {}
    
    # ─── Sample-level metrics ────────────────────────────────────────────
    # Exact match ratio (strict accuracy)
    metrics["exact_match"] = accuracy_score(y_true, y_pred_binary)
    
    # Hamming loss (fraction of wrong labels)
    metrics["hamming_loss"] = hamming_loss(y_true, y_pred_binary)
    
    # ─── Label-level metrics ─────────────────────────────────────────────
    # Micro F1 (aggregate TP/FP/FN across all labels)
    metrics["f1_micro"] = f1_score(y_true, y_pred_binary, average="micro", zero_division=0)
    
    # Macro F1 (average F1 per label)
    metrics["f1_macro"] = f1_score(y_true, y_pred_binary, average="macro", zero_division=0)
    
    # Weighted F1
    metrics["f1_weighted"] = f1_score(y_true, y_pred_binary, average="weighted", zero_division=0)
    
    # ─── Ranking metrics ─────────────────────────────────────────────────
    try:
        # Macro AUC-ROC (only for classes with both positive and negative samples)
        # Filter classes with at least one positive sample
        valid_classes = y_true.sum(axis=0) > 0
        if valid_classes.sum() > 0:
            metrics["auc_roc_macro"] = roc_auc_score(
                y_true[:, valid_classes],
                y_pred_probs[:, valid_classes],
                average="macro",
            )
        else:
            metrics["auc_roc_macro"] = 0.0
    except ValueError:
        metrics["auc_roc_macro"] = 0.0
    
    try:
        # Mean Average Precision (mAP)
        valid_classes = y_true.sum(axis=0) > 0
        if valid_classes.sum() > 0:
            metrics["mAP"] = average_precision_score(
                y_true[:, valid_classes],
                y_pred_probs[:, valid_classes],
                average="macro",
            )
        else:
            metrics["mAP"] = 0.0
    except ValueError:
        metrics["mAP"] = 0.0
    
    # ─── Per-class accuracy ──────────────────────────────────────────────
    per_class_acc = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:
            correct = (y_pred_binary[:, i] == y_true[:, i]).sum()
            per_class_acc.append(correct / len(y_true))
    metrics["per_class_accuracy_mean"] = np.mean(per_class_acc) if per_class_acc else 0.0
    
    return metrics


def format_metrics(metrics: dict) -> str:
    """Format metrics dictionary as a readable string."""
    lines = []
    lines.append("=" * 50)
    lines.append("  EVALUATION METRICS")
    lines.append("=" * 50)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key:30s}: {value:.4f}")
        else:
            lines.append(f"  {key:30s}: {value}")
    
    lines.append("=" * 50)
    return "\n".join(lines)
