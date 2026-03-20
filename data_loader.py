"""
Data loading, splitting, and augmentation pipeline.

This module handles the full journey from raw image files on disk
to ready-to-train tf.data.Dataset objects.
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import (
    IMG_SIZE, BATCH_SIZE, SEED,
    VAL_SPLIT, TEST_SPLIT, CLASS_NAMES, NUM_CLASSES,
)


# ──────────────────────────────────────────────
# 1. Stratified train / val / test split
# ──────────────────────────────────────────────
def split_dataset(file_paths, labels):
    """
    Split the dataset into training, validation, and test sets
    using stratified sampling.

    Stratified means every split has roughly the same proportion
    of each artist — critical when classes are imbalanced.

    Returns three tuples: (paths, labels) for train, val, test.
    """
    # First split: separate the test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        file_paths, labels,
        test_size=TEST_SPLIT,
        stratify=labels,
        random_state=SEED,
    )

    # Second split: separate validation from training
    # val_split is relative to the remaining data (not the original)
    relative_val = VAL_SPLIT / (1 - TEST_SPLIT)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=relative_val,
        stratify=train_val_labels,
        random_state=SEED,
    )

    print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


# ──────────────────────────────────────────────
# 2. Label encoding
# ──────────────────────────────────────────────
def encode_labels(labels):
    """
    Convert artist name strings to integer indices.
    The mapping is alphabetical (matches CLASS_NAMES order).
    """
    label_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    return np.array([label_to_idx[lbl] for lbl in labels])


# ──────────────────────────────────────────────
# 3. Image loading & preprocessing
# ──────────────────────────────────────────────
def load_and_preprocess(path, label):
    """
    TensorFlow function that reads an image file, decodes it,
    resizes it, and scales pixel values to [0, 1].

    This runs inside the tf.data pipeline, so it must use
    TensorFlow ops (not NumPy / PIL).
    """
    # Read raw bytes from disk
    img = tf.io.read_file(path)
    # Decode JPEG into a 3-channel tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Resize to the target size (e.g. 224×224)
    img = tf.image.resize(img, IMG_SIZE)
    # Scale from [0, 255] to [0, 1]
    img = img / 255.0
    return img, label


# ──────────────────────────────────────────────
# 4. Data augmentation (training only)
# ──────────────────────────────────────────────
def augment(img, label):
    """
    Apply random transformations to training images to increase
    the effective dataset size and reduce overfitting.

    Augmentations chosen:
      - Random horizontal flip  : paintings have no inherent left/right
      - Random rotation (±10 %) : slight tilt for robustness
      - Random brightness shift : handles varying photo conditions
      - Random contrast shift   : same reason
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    # Clip to valid range after augmentation
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


# ──────────────────────────────────────────────
# 5. Build tf.data pipelines
# ──────────────────────────────────────────────
def build_dataset(paths, labels, is_training=False, batch_size=BATCH_SIZE):
    """
    Assemble a tf.data.Dataset from file paths and integer labels.

    For training data we shuffle and apply augmentation.
    For validation / test data we only load, resize, and batch.

    Prefetching lets the CPU prepare the next batch while the
    GPU is busy training — a simple but important optimisation.
    """
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if is_training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def prepare_all_datasets(file_paths, labels):
    """
    One-call convenience: split → encode → build tf.data pipelines.

    Returns
    -------
    train_ds, val_ds, test_ds : tf.data.Dataset
    test_paths, test_labels   : kept for per-image error analysis later
    """
    # Split
    (train_p, train_l), (val_p, val_l), (test_p, test_l) = split_dataset(file_paths, labels)

    # Encode string labels to integers
    train_y = encode_labels(train_l)
    val_y = encode_labels(val_l)
    test_y = encode_labels(test_l)

    # Build datasets
    train_ds = build_dataset(train_p, train_y, is_training=True)
    val_ds = build_dataset(val_p, val_y, is_training=False)
    test_ds = build_dataset(test_p, test_y, is_training=False)

    return train_ds, val_ds, test_ds, test_p, test_y
