"""
Model evaluation: metrics, confusion matrix, and error analysis.

After training, we need to objectively measure how well the model
performs on data it has NEVER seen (the test set). This module
provides all the tools for that analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    top_k_accuracy_score,
)
from PIL import Image

from config import CLASS_NAMES, NUM_CLASSES, PLOT_DIR
from utils import save_figure


# ──────────────────────────────────────────────
# 1. Generate predictions
# ──────────────────────────────────────────────
def predict_on_dataset(model, dataset):
    """
    Run the model on a full tf.data.Dataset and collect
    predicted probabilities and true labels.
    """
    all_probs = []
    all_labels = []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_probs, all_labels


# ──────────────────────────────────────────────
# 2. Classification report
# ──────────────────────────────────────────────
def print_classification_report(y_true, y_pred):
    """
    Print per-class precision, recall, F1-score, and support.

    Why these metrics:
      - Accuracy alone can be misleading with imbalanced classes
      - Precision: of all images we labelled as Artist X, how many
        were actually by Artist X?
      - Recall: of all actual Artist X paintings, how many did we
        correctly identify?
      - F1: harmonic mean of precision and recall — a single number
        that balances both
    """
    artist_labels = [name.replace("_", " ") for name in CLASS_NAMES]
    report = classification_report(
        y_true, y_pred, target_names=artist_labels, digits=3
    )
    print(report)
    return report


# ──────────────────────────────────────────────
# 3. Confusion matrix
# ──────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name="model", normalize=True):
    """
    Heatmap showing which artists the model confuses with each other.

    Why: the confusion matrix reveals systematic errors.
    For example, if the model frequently confuses Monet and Pissarro,
    this makes artistic sense — both are Impressionists with similar
    colour palettes.

    Parameters
    ----------
    normalize : bool
        If True, show percentages instead of raw counts (easier to
        compare across classes of different sizes).
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(16, 14))
    artist_labels = [name.replace("_", " ") for name in CLASS_NAMES]
    fmt = ".1%" if normalize else "d"
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=artist_labels,
        yticklabels=artist_labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted artist")
    ax.set_ylabel("True artist")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    fig.tight_layout()
    save_figure(fig, f"{model_name}_confusion_matrix.png")
    plt.show()

    return cm


# ──────────────────────────────────────────────
# 4. Top-k accuracy
# ──────────────────────────────────────────────
def compute_topk_accuracy(y_true, y_probs, k_values=(1, 3, 5)):
    """
    Compute top-k accuracy for multiple values of k.

    Why: with 23 artists, the model's second or third guess may
    still be correct. Top-5 accuracy shows how often the correct
    artist is among the model's top 5 predictions — useful for
    understanding the model's "near misses".
    """
    results = {}
    for k in k_values:
        if k >= NUM_CLASSES:
            continue
        acc = top_k_accuracy_score(y_true, y_probs, k=k, labels=range(NUM_CLASSES))
        results[f"Top-{k}"] = acc
        print(f"Top-{k} Accuracy: {acc:.4f}")
    return results


# ──────────────────────────────────────────────
# 5. Error analysis — show misclassified images
# ──────────────────────────────────────────────
def show_misclassified(test_paths, y_true, y_pred, n=12):
    """
    Display a grid of images that the model got wrong.

    Why: looking at actual misclassifications often reveals more
    about model weaknesses than any single number can. We might
    discover that the model struggles with certain styles, periods,
    or subject matter.
    """
    wrong_mask = y_true != y_pred
    wrong_indices = np.where(wrong_mask)[0]

    if len(wrong_indices) == 0:
        print("No misclassifications — perfect score!")
        return

    # Pick a random sample of errors
    rng = np.random.RandomState(42)
    sample = rng.choice(wrong_indices, size=min(n, len(wrong_indices)), replace=False)

    cols = 4
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for i, idx in enumerate(sample):
        img = Image.open(test_paths[idx]).convert("RGB")
        axes[i].imshow(img)
        true_name = CLASS_NAMES[y_true[idx]].replace("_", " ")
        pred_name = CLASS_NAMES[y_pred[idx]].replace("_", " ")
        axes[i].set_title(f"True: {true_name}\nPred: {pred_name}",
                          fontsize=9, color="red")
        axes[i].axis("off")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Misclassified images", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "misclassified_samples.png")
    plt.show()


# ──────────────────────────────────────────────
# 6. Full evaluation pipeline
# ──────────────────────────────────────────────
def evaluate_model(model, test_ds, test_paths, test_labels, model_name="model"):
    """
    Run the complete evaluation suite on the test set:
      1. Generate predictions
      2. Print classification report
      3. Plot confusion matrix
      4. Compute top-k accuracy
      5. Show misclassified examples

    Returns a dict of summary metrics.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}\n")

    # Predictions
    y_probs, y_true = predict_on_dataset(model, test_ds)
    y_pred = np.argmax(y_probs, axis=1)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}\n")

    print_classification_report(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, model_name=model_name)
    topk = compute_topk_accuracy(y_true, y_probs)
    show_misclassified(test_paths, y_true, y_pred)

    return {"accuracy": acc, **topk}
