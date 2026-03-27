"""
Test-Time Augmentation (TTA) for the WikiArt artist classification project.

TTA approximates Bayesian model averaging: instead of predicting on a single
deterministic view of each test image, we generate N randomly-augmented views
and average their softmax outputs. This reduces prediction variance caused by
small input perturbations (digitisation differences, lighting, scanner artefacts).

Augmentation policy for TTA:
  ONLY gentle transforms — horizontal flip + tiny brightness/contrast.
  NO random crops. Project finding: spatial crops destroy composition cues
  that are important style signals for paintings (see decision_log.md).

Usage:
    from tta import tta_predict
    avg_probs, true_labels = tta_predict(model, test_paths, test_labels, n_augments=5)
    y_pred = np.argmax(avg_probs, axis=1)
    accuracy = accuracy_score(true_labels, y_pred)
"""

import numpy as np
import tensorflow as tf

from config import IMG_SIZE, BATCH_SIZE
from data_loader import load_and_preprocess


# ──────────────────────────────────────────────
# TTA-specific augmentation (gentler than training)
# ──────────────────────────────────────────────

def augment_tta(img, label):
    """
    Gentle augmentation for test-time use.

    Deliberately conservative compared to training augmentation:
      - Random horizontal flip        (paintings have no inherent orientation)
      - Very small brightness shift   (±5%, simulates minor scan differences)
      - Very small contrast shift     (±5%, same reasoning)
      - NO crops, NO saturation, NO hue shifts

    Why no crops: experiments showed random crops are destructive for art
    classification because composition IS a style signal. We never apply
    crops at test time.

    Parameters
    ----------
    img   : tf.Tensor, shape (H, W, C), float32 [0, 1]
    label : tf.Tensor — passed through unchanged

    Returns
    -------
    (augmented_img, label)
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.05)
    img = tf.image.random_contrast(img, lower=0.95, upper=1.05)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


def build_tta_dataset(paths, labels, batch_size=BATCH_SIZE):
    """
    Build one augmented view of the test set.
    Each call produces a different random augmentation (different seed).

    Parameters
    ----------
    paths      : array-like of file path strings
    labels     : array-like of integer class indices
    batch_size : int

    Returns
    -------
    tf.data.Dataset yielding (augmented_images, labels) batches
    """
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(augment_tta, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ──────────────────────────────────────────────
# TTA inference
# ──────────────────────────────────────────────

def tta_predict(model, test_paths, test_labels, n_augments=5, verbose=True):
    """
    Apply Test-Time Augmentation: average predictions over N augmented views.

    Interface is identical to predict_on_dataset() from evaluation.py,
    so results can be passed directly to compute_topk_accuracy(),
    plot_confusion_matrix(), etc.

    Parameters
    ----------
    model        : keras.Model or EnsemblePredictor — must have a .predict() method
    test_paths   : array of image file path strings
    test_labels  : numpy array of integer class indices (0..NUM_CLASSES-1)
    n_augments   : int — number of augmented views to average (default 5)
    verbose      : bool — print progress per augmentation round

    Returns
    -------
    avg_probs : np.ndarray, shape (N, NUM_CLASSES) — averaged softmax probabilities
    labels    : np.ndarray, shape (N,) — true integer labels (unchanged)
    """
    from evaluation import predict_on_dataset

    all_probs = []

    for i in range(n_augments):
        if verbose:
            print(f"  TTA round {i + 1}/{n_augments}...", end="\r")

        ds = build_tta_dataset(test_paths, test_labels)
        probs, labels = predict_on_dataset(model, ds)
        all_probs.append(probs)

    if verbose:
        print(f"  TTA complete ({n_augments} rounds)        ")

    avg_probs = np.mean(all_probs, axis=0)
    return avg_probs, labels


def tta_accuracy(model, test_paths, test_labels, n_augments=5):
    """
    Convenience wrapper: run TTA and return top-1 accuracy.

    Returns
    -------
    float — top-1 accuracy after TTA
    """
    from sklearn.metrics import accuracy_score
    avg_probs, labels = tta_predict(model, test_paths, test_labels, n_augments)
    y_pred = np.argmax(avg_probs, axis=1)
    return accuracy_score(labels, y_pred)
