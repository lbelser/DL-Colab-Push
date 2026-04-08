"""
Test-Time Augmentation (TTA) for the WikiArt pipeline.

Generate N augmented views of each test image and average their softmax outputs
to reduce prediction variance from input perturbations (scan quality, lighting, etc.).

Only gentle augmentation here — no random crops (composition matters for style).
"""

import numpy as np
import tensorflow as tf

from config import IMG_SIZE, BATCH_SIZE
from data_loader import load_and_preprocess


# ──────────────────────────────────────────────
# TTA augmentation (gentler than training)
# ──────────────────────────────────────────────

def augment_tta(img, label):
    """Gentle: flip + tiny brightness/contrast shift. No crops."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.05)
    img = tf.image.random_contrast(img, lower=0.95, upper=1.05)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


def build_tta_dataset(paths, labels, batch_size=BATCH_SIZE):
    """Build one augmented view of the test set (each call gives different random aug)."""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(augment_tta, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ──────────────────────────────────────────────
# TTA inference
# ──────────────────────────────────────────────

def tta_predict(model, test_paths, test_labels, n_augments=5, verbose=True):
    """
    Average predictions over N augmented views of the test set.
    Returns (avg_probs, true_labels) — same interface as predict_on_dataset().
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
    """Run TTA and return top-1 accuracy."""
    from sklearn.metrics import accuracy_score
    avg_probs, labels = tta_predict(model, test_paths, test_labels, n_augments)
    y_pred = np.argmax(avg_probs, axis=1)
    return accuracy_score(labels, y_pred)
