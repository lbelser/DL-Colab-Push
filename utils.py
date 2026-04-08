"""
Utility functions: seeds, plotting helpers.
"""

import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from config import SEED, PLOT_DIR


# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
def set_seeds(seed: int = SEED):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ──────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────
def save_figure(fig, filename: str):
    """Save a matplotlib figure to the plots directory."""
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved plot → {path}")


def plot_training_history(history, model_name: str = "model"):
    """
    Plot accuracy and loss curves for training and validation.

    Parameters
    ----------
    history : keras History object returned by model.fit()
    model_name : label used in the plot title and filename
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Accuracy ---
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title(f"{model_name} — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Loss ---
    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title(f"{model_name} — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, f"{model_name}_training_curves.png")
    plt.show()
