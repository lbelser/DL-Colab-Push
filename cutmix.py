"""
CutMix data augmentation for the WikiArt artist classification project.

Yun et al., "CutMix: Training Strategy that Makes Use of Sample Combinations",
ICCV 2019. https://arxiv.org/abs/1905.04899

Key differences from standard augmentation:
  - Operates at BATCH level (mixes pairs within a batch), not per-image
  - Requires ONE-HOT labels (produces soft mixed targets)
  - Must be compiled with CategoricalCrossentropy, NOT SparseCategoricalCrossentropy

Usage:
    from cutmix import build_dataset_cutmix
    train_ds = build_dataset_cutmix(train_paths, train_labels_int, is_training=True)
    # Then compile model with CategoricalCrossentropy(label_smoothing=0.1)
"""

import numpy as np
import tensorflow as tf

from config import IMG_SIZE, NUM_CLASSES, BATCH_SIZE, SEED


# ──────────────────────────────────────────────
# Core CutMix operation (batch-level)
# ──────────────────────────────────────────────

def cutmix_batch(images, labels_onehot, alpha=1.0):
    """
    Apply CutMix to a single batch.

    Algorithm (Yun et al. 2019):
      1. Sample λ ~ Beta(α, α)  → controls how much of image B to paste
      2. Compute a random bounding box with area ≈ (1-λ) × full image
      3. Paste the box from a shuffled version of the batch onto the originals
      4. Mix labels proportionally: y_mix = λ·y_A + (1-λ)·y_B

    Parameters
    ----------
    images       : tf.Tensor, shape (B, H, W, C), dtype float32, range [0,1]
    labels_onehot: tf.Tensor, shape (B, NUM_CLASSES), dtype float32
    alpha        : float — Beta distribution parameter (1.0 = uniform)

    Returns
    -------
    mixed_images : tf.Tensor, same shape as images
    mixed_labels : tf.Tensor, same shape as labels_onehot (soft targets)
    """
    h, w = IMG_SIZE

    # Sample λ from Beta(α, α) — controls the cut size
    lam = float(np.random.beta(alpha, alpha))

    # Compute bounding box: area ≈ (1-λ) × total
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    x1 = int(np.clip(cx - cut_w // 2, 0, w))
    x2 = int(np.clip(cx + cut_w // 2, 0, w))
    y1 = int(np.clip(cy - cut_h // 2, 0, h))
    y2 = int(np.clip(cy + cut_h // 2, 0, h))

    # Recompute λ from actual box area (may differ due to clipping)
    lam = 1.0 - float((x2 - x1) * (y2 - y1)) / float(h * w)

    # Build binary mask: 1 = keep original pixel, 0 = take from shuffled image
    mask = np.ones((h, w, 1), dtype=np.float32)
    mask[y1:y2, x1:x2, :] = 0.0
    mask_tf = tf.constant(mask)  # (H, W, 1) — broadcasts over batch

    # Shuffle indices within the batch to get the "donor" images
    batch_size = tf.shape(images)[0]
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels_onehot, indices)

    # Apply mask: paste donor patch into original images
    mixed_images = images * mask_tf + shuffled_images * (1.0 - mask_tf)

    # Mix labels with the same λ
    mixed_labels = lam * labels_onehot + (1.0 - lam) * shuffled_labels

    return mixed_images, mixed_labels


# ──────────────────────────────────────────────
# Dataset builder with one-hot labels + CutMix
# ──────────────────────────────────────────────

def build_dataset_cutmix(paths, labels_int, is_training=False,
                          batch_size=BATCH_SIZE, augmentation="strong"):
    """
    Build a tf.data.Dataset for CutMix training.

    Differences from the standard build_dataset():
      - Labels are converted to one-hot floats (required for soft CutMix targets)
      - CutMix is applied after batching (training only)

    For evaluation/validation, use this function with is_training=False
    to get one-hot labels consistent with CategoricalCrossentropy.

    Parameters
    ----------
    paths        : list/array of image file paths
    labels_int   : numpy array of integer class indices (0..NUM_CLASSES-1)
    is_training  : bool — whether to apply augmentation and CutMix
    batch_size   : int
    augmentation : "strong" | "basic" — augmentation applied before CutMix

    Returns
    -------
    tf.data.Dataset yielding (images, one_hot_labels) per batch
    """
    from data_loader import augment_strong, augment

    # Convert integer labels to one-hot once (not per-step)
    labels_onehot = tf.one_hot(labels_int.astype("int32"), NUM_CLASSES)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels_onehot))

    if is_training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)

    # Load and preprocess images (replicates load_and_preprocess but accepts one-hot labels)
    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = img / 255.0
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    # Apply image augmentation (label is one-hot — pass dummy int to aug functions)
    if is_training:
        if augmentation == "strong":
            aug_fn = augment_strong
        else:
            aug_fn = augment

        def _aug(img, label):
            img, _ = aug_fn(img, tf.constant(0, dtype=tf.int32))
            return img, label

        ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)

    # Apply CutMix at the batch level (training only)
    if is_training:
        def _cutmix_wrapper(imgs, lbls):
            mixed_imgs, mixed_lbls = tf.py_function(
                func=cutmix_batch,
                inp=[imgs, lbls],
                Tout=[tf.float32, tf.float32],
            )
            # Restore static shapes (py_function loses them)
            mixed_imgs.set_shape([None, *IMG_SIZE, 3])
            mixed_lbls.set_shape([None, NUM_CLASSES])
            return mixed_imgs, mixed_lbls

        ds = ds.map(_cutmix_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)


# ──────────────────────────────────────────────
# Evaluation helper for CutMix-trained models
# ──────────────────────────────────────────────

def build_eval_dataset_onehot(paths, labels_int, batch_size=BATCH_SIZE):
    """
    Standard (no-augmentation) evaluation dataset with one-hot labels.
    Use for val/test evaluation of CutMix-trained models.
    """
    labels_onehot = tf.one_hot(labels_int.astype("int32"), NUM_CLASSES)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels_onehot))

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = img / 255.0
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
