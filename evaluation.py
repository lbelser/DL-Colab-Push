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

from collections import Counter
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


# ──────────────────────────────────────────────
# 7. Per-artist accuracy vs class size
# ──────────────────────────────────────────────
def plot_accuracy_vs_class_size(y_true, y_pred, all_labels, model_name="model"):
    """
    Scatter plot: does having more training images help?

    Each dot is one artist. X-axis = number of training images,
    Y-axis = per-class accuracy. If there's a positive correlation,
    it means the model benefits from more data and underrepresented
    artists are at a disadvantage.

    Parameters
    ----------
    all_labels : list of all original string labels (before split)
                 used to count total images per artist
    """
    # Count images per artist in the full dataset
    class_counts = Counter(all_labels)

    # Per-class accuracy from predictions
    per_class_acc = {}
    for idx, name in enumerate(CLASS_NAMES):
        mask = y_true == idx
        if mask.sum() > 0:
            per_class_acc[name] = (y_pred[mask] == idx).mean()

    # Build aligned arrays
    artists = list(per_class_acc.keys())
    accs = [per_class_acc[a] for a in artists]
    sizes = [class_counts[a] for a in artists]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(sizes, accs, s=80, alpha=0.7, edgecolors="black", linewidths=0.5)

    # Label each point with the artist name
    for i, artist in enumerate(artists):
        ax.annotate(
            artist.replace("_", " "),
            (sizes[i], accs[i]),
            fontsize=7, ha="left", va="bottom",
            xytext=(5, 3), textcoords="offset points",
        )

    # Trend line
    z = np.polyfit(sizes, accs, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(sizes), max(sizes), 100)
    ax.plot(x_line, p(x_line), "--", color="red", alpha=0.5, label=f"Trend (slope={z[0]:.5f})")

    ax.set_xlabel("Number of images in dataset")
    ax.set_ylabel("Per-class accuracy (test set)")
    ax.set_title(f"Accuracy vs Class Size — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, f"{model_name}_accuracy_vs_class_size.png")
    plt.show()

    # Compute correlation
    corr = np.corrcoef(sizes, accs)[0, 1]
    print(f"Pearson correlation (class size vs accuracy): {corr:.3f}")
    return corr


# ──────────────────────────────────────────────
# 8. Confusion cluster analysis
# ──────────────────────────────────────────────
def analyse_confusion_clusters(y_true, y_pred, model_name="model", top_n=10):
    """
    Identify the most common artist-pair confusions.

    Why: these pairs often reflect real art-historical relationships.
    For example, Impressionists (Monet, Pissarro, Renoir) share
    colour palettes and brushwork. Surfacing these helps us write
    a richer analysis in the report.
    """
    cm = confusion_matrix(y_true, y_pred)
    # Zero the diagonal (correct predictions don't count)
    np.fill_diagonal(cm, 0)

    # Find the top-N off-diagonal entries
    pairs = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if cm[i, j] > 0:
                pairs.append((cm[i, j], CLASS_NAMES[i], CLASS_NAMES[j]))

    pairs.sort(reverse=True)

    print(f"\nTop {top_n} confusions — {model_name}")
    print(f"{'Count':>5}  {'True artist':<25} → {'Predicted as':<25}")
    print("─" * 60)
    for count, true, pred in pairs[:top_n]:
        print(f"{count:5d}  {true.replace('_', ' '):<25} → {pred.replace('_', ' '):<25}")

    return pairs[:top_n]


# ──────────────────────────────────────────────
# 9. Multi-model comparison bar chart
# ──────────────────────────────────────────────
def plot_model_comparison(results_dict):
    """
    Grouped bar chart comparing all models across metrics.

    Parameters
    ----------
    results_dict : dict of {model_name: {"accuracy": ..., "Top-3": ..., "Top-5": ...}}
    """
    models = list(results_dict.keys())
    metrics = ["accuracy", "Top-3", "Top-5"]
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        values = [results_dict[m].get(metric, 0) for m in models]
        bars = ax.bar(x + i * width, values, width, label=metric)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison — All Metrics")
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    save_figure(fig, "model_comparison_bars.png")
    plt.show()


# ──────────────────────────────────────────────
# 10. Per-artist F1 comparison across models
# ──────────────────────────────────────────────
def plot_per_artist_f1(model_reports, filename="per_artist_f1_comparison.png"):
    """
    Grouped bar chart comparing per-artist F1 scores across models.

    Parameters
    ----------
    model_reports : dict of {model_name: (y_true, y_pred)}
        Each entry provides the true and predicted labels for a model.
    """
    from sklearn.metrics import f1_score

    artist_labels = [name.replace("_", " ") for name in CLASS_NAMES]
    model_names = list(model_reports.keys())
    n_models = len(model_names)

    # Compute per-class F1 for each model
    f1_per_model = {}
    for name, (y_true, y_pred) in model_reports.items():
        f1s = f1_score(y_true, y_pred, labels=range(NUM_CLASSES), average=None)
        f1_per_model[name] = f1s

    # Sort artists by ensemble (last model) F1 for readability
    last_model = model_names[-1]
    sort_idx = np.argsort(f1_per_model[last_model])

    fig, ax = plt.subplots(figsize=(12, 10))
    y_pos = np.arange(NUM_CLASSES)
    bar_height = 0.8 / n_models

    for i, name in enumerate(model_names):
        f1s_sorted = f1_per_model[name][sort_idx]
        ax.barh(y_pos + i * bar_height, f1s_sorted, bar_height,
                label=name, alpha=0.85)

    ax.set_yticks(y_pos + bar_height * (n_models - 1) / 2)
    ax.set_yticklabels([artist_labels[j] for j in sort_idx], fontsize=9)
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-Artist F1 Score — Model Comparison")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    save_figure(fig, filename)
    plt.show()
