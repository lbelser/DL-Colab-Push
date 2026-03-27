"""
Per-class weighted ensemble for the WikiArt artist classification project.

Motivation:
  Different models have different per-class strengths due to their architectural
  inductive biases and the dataset imbalance. A simple average treats all models
  equally for all classes. Per-class weighting lets the ensemble defer to the
  model that is strongest for each specific artist.

  Weights are learned from validation performance — NOT hand-tuned.
  This is a simplified form of stacked generalisation (Wolpert 1992).

Weight matrix:
  Shape: (n_models, NUM_CLASSES)
  weights[i, c] = how much model i's prediction is trusted for class c
  Each column sums to 1.0 (weights across models for a given class sum to 1)

  Derived from per-class validation accuracy of each model.

Usage:
    from ensemble_weighted import compute_perclass_weights, weighted_ensemble_predict

    # Compute weights from validation set
    weights = compute_perclass_weights([resnet, effnet, vit], val_ds, val_labels)

    # Weighted ensemble inference on test set
    final_probs, true_labels = weighted_ensemble_predict(
        [resnet, effnet, vit], weights, test_ds
    )
"""

import numpy as np
from sklearn.metrics import accuracy_score

from config import NUM_CLASSES


# ──────────────────────────────────────────────
# Weight matrix computation
# ──────────────────────────────────────────────

def compute_perclass_weights(models, val_ds, val_labels):
    """
    Compute per-class weight matrix from validation set performance.

    For each model:
      - Run inference on the validation set
      - Compute accuracy separately for each of the 23 artist classes
    Then normalise each column so model weights for a given class sum to 1.

    Parameters
    ----------
    models     : list of keras.Model / EnsemblePredictor
    val_ds     : tf.data.Dataset — validation set (no augmentation)
    val_labels : np.ndarray, shape (N,) — integer true labels for val set

    Returns
    -------
    weights : np.ndarray, shape (n_models, NUM_CLASSES), float64
              weights[i, c] = model i's normalised contribution for class c
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

    # Normalise per class: each column sums to 1
    col_sums = raw_weights.sum(axis=0, keepdims=True)
    # Avoid division by zero for classes where all models score 0
    col_sums = np.where(col_sums == 0, 1.0, col_sums)
    weights = raw_weights / col_sums

    return weights


def print_weight_matrix(weights, model_names, class_names):
    """
    Pretty-print the weight matrix: which model is trusted most per class.

    Parameters
    ----------
    weights     : np.ndarray, shape (n_models, NUM_CLASSES)
    model_names : list of str
    class_names : list of str (length NUM_CLASSES)
    """
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
        print(row + f"  ← {model_names[best][:8]}")

    print("-" * 70)
    print("Dominant model count:")
    for i, name in enumerate(model_names):
        print(f"  {name}: {dominant_model_counts[i]} classes")


# ──────────────────────────────────────────────
# Ensemble inference
# ──────────────────────────────────────────────

def weighted_ensemble_predict(models, weights, dataset):
    """
    Weighted ensemble: combine softmax outputs using the per-class weight matrix.

    For each class c, the final probability is:
        p_final[n, c] = sum_i( weights[i, c] * p_i[n, c] )

    Since weights[:, c] sum to 1 for each class c, this is a proper
    weighted average of model probabilities per class.

    Parameters
    ----------
    models  : list of keras.Model / EnsemblePredictor
    weights : np.ndarray, shape (n_models, NUM_CLASSES)
    dataset : tf.data.Dataset — test set (no augmentation)

    Returns
    -------
    final_probs : np.ndarray, shape (N, NUM_CLASSES)
    true_labels : np.ndarray, shape (N,)
    """
    from evaluation import predict_on_dataset

    all_probs = []
    true_labels = None

    for model in models:
        probs, labels = predict_on_dataset(model, dataset)
        all_probs.append(probs)
        if true_labels is None:
            true_labels = labels

    # stacked: (n_models, N, NUM_CLASSES)
    stacked = np.stack(all_probs, axis=0)

    # Expand weights for broadcasting: (n_models, 1, NUM_CLASSES)
    w = weights[:, np.newaxis, :]

    # Weighted sum across models: (N, NUM_CLASSES)
    final_probs = (stacked * w).sum(axis=0)

    return final_probs, true_labels


def equal_weight_predict(models, dataset):
    """
    Simple equal-weight average of softmax outputs from all models.
    Used as a baseline comparison for the per-class weighted ensemble.

    Parameters
    ----------
    models  : list of keras.Model / EnsemblePredictor
    dataset : tf.data.Dataset

    Returns
    -------
    avg_probs   : np.ndarray, shape (N, NUM_CLASSES)
    true_labels : np.ndarray, shape (N,)
    """
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
    """
    Compute and print accuracy metrics for an ensemble's probability outputs.

    Parameters
    ----------
    probs       : np.ndarray, shape (N, NUM_CLASSES)
    true_labels : np.ndarray, shape (N,)
    model_name  : str — label for printing

    Returns
    -------
    dict with "accuracy", "Top-3", "Top-5"
    """
    from evaluation import compute_topk_accuracy

    y_pred = np.argmax(probs, axis=1)
    acc = accuracy_score(true_labels, y_pred)

    print(f"\n{'='*60}")
    print(f"Ensemble Results: {model_name}")
    print(f"{'='*60}")
    print(f"  Top-1 Accuracy: {acc:.4f} ({acc:.1%})")

    topk = compute_topk_accuracy(true_labels, probs)
    return {"accuracy": acc, **topk}
