"""Evaluation metrics for Legal LoRA Tagger."""

import time
from collections import Counter

from app.data_prep import CLAUSE_LABELS
from app.utils import get_logger, count_parameters

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_model(model, tokenizer, test_data: list, device=None) -> dict:
    """
    Evaluate model on test data.
    Returns dict with accuracy, precision, recall, f1, confusion_matrix,
    classification_report, avg_inference_time, total_samples, y_true, y_pred.
    """
    from app.inference import predict_clause
    from app.utils import get_device

    if device is None:
        device = get_device()

    y_true = []
    y_pred = []
    inference_times = []

    logger.info(f"Evaluating on {len(test_data)} samples...")

    for item in test_data:
        t0 = time.time()
        result = predict_clause(item["text"], model, tokenizer, device=device)
        t1 = time.time()
        inference_times.append(t1 - t0)
        y_true.append(item["label"])
        y_pred.append(result["label"])

    # Compute metrics using sklearn when available; fall back to manual
    try:
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
            classification_report,
        )

        labels_present = sorted(set(y_true + y_pred))
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels_present)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels_present)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels_present)
        cm = confusion_matrix(y_true, y_pred, labels=labels_present)
        report = classification_report(y_true, y_pred, zero_division=0)

        logger.info(f"Accuracy:  {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall:    {recall:.4f}")
        logger.info(f"F1:        {f1:.4f}")

    except ImportError:
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = correct / len(y_true) if y_true else 0
        precision = accuracy
        recall = accuracy
        f1 = accuracy
        cm = None
        report = f"Accuracy: {accuracy:.4f}"

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "avg_inference_time": sum(inference_times) / len(inference_times) if inference_times else 0,
        "total_samples": len(test_data),
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def evaluate_baseline(test_data: list) -> dict:
    """Evaluate a majority-class baseline."""
    all_labels = [item["label"] for item in test_data]
    most_common = Counter(all_labels).most_common(1)[0][0] if all_labels else "Unknown"
    y_pred = [most_common] * len(all_labels)
    correct = sum(1 for t, p in zip(all_labels, y_pred) if t == p)
    accuracy = correct / len(all_labels) if all_labels else 0
    return {
        "baseline_type": "majority_class",
        "majority_label": most_common,
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# Efficiency report
# ---------------------------------------------------------------------------

def report_model_efficiency(model) -> dict:
    """Print and return model efficiency stats."""
    stats = count_parameters(model)
    print("\n=== Model Efficiency Report ===")
    print(f"Total parameters:     {stats['total']:,}")
    print(f"Trainable parameters: {stats['trainable']:,}")
    print(f"Frozen parameters:    {stats['frozen']:,}")
    print(f"Trainable percentage: {stats['trainable_pct']:.4f}%")
    return stats
