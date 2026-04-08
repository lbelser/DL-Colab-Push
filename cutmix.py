"""
CutMix augmentation (Yun et al., ICCV 2019).

Operates at batch level — mixes pairs within a batch and produces soft labels.
Requires one-hot labels and CategoricalCrossentropy (not Sparse).
"""

import numpy as np
import tensorflow as tf

from config import IMG_SIZE, NUM_CLASSES, BATCH_SIZE, SEED


# ──────────────────────────────────────────────
# Core CutMix (batch-level)
# ──────────────────────────────────────────────

def cutmix_batch(images, labels_onehot, alpha=1.0):
    """
    Apply CutMix to a batch:
    1. Sample lambda from Beta(alpha, alpha)
    2. Random bounding box with area ~ (1-lambda)
    3. Paste box from shuffled batch onto originals
    4. Mix labels proportionally
    """
    h, w = IMG_SIZE

    lam = float(np.random.beta(alpha, alpha))

    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    x1 = int(np.clip(cx - cut_w // 2, 0, w))
    x2 = int(np.clip(cx + cut_w // 2, 0, w))
    y1 = int(np.clip(cy - cut_h // 2, 0, h))
    y2 = int(np.clip(cy + cut_h // 2, 0, h))

    # Recompute lambda from actual box area (clipping changes it)
    lam = 1.0 - float((x2 - x1) * (y2 - y1)) / float(h * w)

    # Binary mask: 1 = keep original, 0 = take from donor
    mask = np.ones((h, w, 1), dtype=np.float32)
    mask[y1:y2, x1:x2, :] = 0.0
    mask_tf = tf.constant(mask)

    batch_size = tf.shape(images)[0]
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels_onehot, indices)

    mixed_images = images * mask_tf + shuffled_images * (1.0 - mask_tf)
    mixed_labels = lam * labels_onehot + (1.0 - lam) * shuffled_labels

    return mixed_images, mixed_labels


# ──────────────────────────────────────────────
# Dataset builder with CutMix
# ──────────────────────────────────────────────

def build_dataset_cutmix(paths, labels_int, is_training=False,
                          batch_size=BATCH_SIZE, augmentation="strong"):
    """
    tf.data pipeline with one-hot labels + CutMix (training only).
    For eval, call with is_training=False to get one-hot labels
    compatible with CategoricalCrossentropy.
    """
    from data_loader import augment_strong, augment

    labels_onehot = tf.one_hot(labels_int.astype("int32"), NUM_CLASSES)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels_onehot))

    if is_training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = img / 255.0
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        aug_fn = augment_strong if augmentation == "strong" else augment

        def _aug(img, label):
            img, _ = aug_fn(img, tf.constant(0, dtype=tf.int32))
            return img, label

        ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)

    if is_training:
        def _cutmix_wrapper(imgs, lbls):
            mixed_imgs, mixed_lbls = tf.py_function(
                func=cutmix_batch,
                inp=[imgs, lbls],
                Tout=[tf.float32, tf.float32],
            )
            mixed_imgs.set_shape([None, *IMG_SIZE, 3])
            mixed_lbls.set_shape([None, NUM_CLASSES])
            return mixed_imgs, mixed_lbls

        ds = ds.map(_cutmix_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)


# ──────────────────────────────────────────────
# Eval dataset (one-hot, no augmentation)
# ──────────────────────────────────────────────

def build_eval_dataset_onehot(paths, labels_int, batch_size=BATCH_SIZE):
    """Standard eval dataset with one-hot labels for CutMix-trained models."""
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
