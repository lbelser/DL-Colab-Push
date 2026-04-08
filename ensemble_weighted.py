"""
Per-class weighted ensemble for the WikiArt pipeline.

Instead of equal-weight averaging, each model gets a per-class weight
based on how well it did on validation data for that specific artist.
Weights are normalised so they sum to 1 per class.
"""

import numpy as np
from sklearn.metrics import accuracy_score

from config import NUM_CLASSES


# ──────────────────────────────────────────────
# Weight matrix computation
# ──────────────────────────────────────────────

def compute_perclass_weights(models, val_ds, val_labels):
    """
    Compute weight matrix (n_models, NUM_CLASSES) from per-class val accuracy.
    Each column sums to 1 — models are weighted by how good they are per artist.
    """
    from evaluation import predict_on_dataset

    n_models = len(models)
    raw_weights = np.zeros((n_models, NUM_CLASSES), dtype=np.float64)

    for i, model in enumerate(models):
        print(f"  Computing per-class accuracy for model {i + 1}/{n_models}: "
              f"{getattr(model, 'name', str(i))}...")

        probs, _ = predict_on_dataset(model, val_ds)
        preds = np.argmax(probs, axis=1)

        for c in range(NUM_CLASSES):
            class_mask = val_labels == c
            n_samples = class_mask.sum()
            if n_samples > 0:
                raw_weights[i, c] = (preds[class_mask] == val_labels[class_mask]).mean()
            else:
                raw_weights[i, c] = 0.0

    # Normalise per class
    col_sums = raw_weights.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums == 0, 1.0, col_sums)
    weights = raw_weights / col_sums

    return weights


def print_weight_matrix(weights, model_names, class_names):
    """Print which model is trusted most per artist."""
    print(f"\n{'='*70}")
    print(f"{'PER-CLASS WEIGHT MATRIX':^70}")
    print(f"{'='*70}")
    header = f"{'Artist':<25}" + "".join(f"{n[:8]:>10}" for n in model_names)
    print(header)
    print("-" * 70)

    dominant_model_counts = np.zeros(len(model_names), dtype=int)
    for c in range(NUM_CLASSES):
        row = f"{class_names[c].replace('_', ' '):<25}"
        for i in range(len(model_names)):
            row += f"{weights[i, c]:>10.3f}"
        best = np.argmax(weights[:, c])
        dominant_model_counts[best] += 1
        print(row + f"  <- {model_names[best][:8]}")

    print("-" * 70)
    print("Dominant model count:")
    for i, name in enumerate(model_names):
        print(f"  {name}: {dominant_model_counts[i]} classes")


# ──────────────────────────────────────────────
# Ensemble inference
# ──────────────────────────────────────────────

def weighted_ensemble_predict(models, weights, dataset):
    """
    Weighted ensemble: for each class c, final prob =
    sum_i(weights[i,c] * model_i_prob[n,c]).
    """
    from evaluation import predict_on_dataset

    all_probs = []
    true_labels = None

    for model in models:
        probs, labels = predict_on_dataset(model, dataset)
        all_probs.append(probs)
        if true_labels is None:
            true_labels = labels

    stacked = np.stack(all_probs, axis=0)  # (n_models, N, NUM_CLASSES)
    w = weights[:, np.newaxis, :]          # (n_models, 1, NUM_CLASSES)
    final_probs = (stacked * w).sum(axis=0)

    return final_probs, true_labels


def equal_weight_predict(models, dataset):
    """Simple equal-weight average as a baseline for comparison."""
    from evaluation import predict_on_dataset

    all_probs = []
    true_labels = None

    for model in models:
        probs, labels = predict_on_dataset(model, dataset)
        all_probs.append(probs)
        if true_labels is None:
            true_labels = labels

    return np.mean(all_probs, axis=0), true_labels


# ──────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────

def evaluate_ensemble(probs, true_labels, model_name="Ensemble"):
    """Print accuracy metrics for ensemble probability outputs."""
    from evaluation import compute_topk_accuracy

    y_pred = np.argmax(probs, axis=1)
    acc = accuracy_score(true_labels, y_pred)

    print(f"\n{'='*60}")
    print(f"Ensemble Results: {model_name}")
    print(f"{'='*60}")
    print(f"  Top-1 Accuracy: {acc:.4f} ({acc:.1%})")

    topk = compute_topk_accuracy(true_labels, probs)
    return {"accuracy": acc, **topk}
