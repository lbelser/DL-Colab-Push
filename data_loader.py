"""
Data loading, splitting, and augmentation for the WikiArt pipeline.
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import (
    IMG_SIZE, BATCH_SIZE, SEED,
    VAL_SPLIT, TEST_SPLIT, CLASS_NAMES, NUM_CLASSES,
)

# Rotation layer for randaugment — created once to avoid re-instantiation in map()
_RA_ROTATE = None
try:
    import keras as _keras
    _RA_ROTATE = _keras.layers.RandomRotation(factor=10.0 / 360.0, fill_mode="reflect")
except Exception:
    pass


# ──────────────────────────────────────────────
# 1. Stratified train / val / test split
# ──────────────────────────────────────────────
def split_dataset(file_paths, labels):
    """Split into train/val/test with stratified sampling."""
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        file_paths, labels,
        test_size=TEST_SPLIT,
        stratify=labels,
        random_state=SEED,
    )

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
    """Convert artist name strings to integer indices (alphabetical order)."""
    label_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    return np.array([label_to_idx[lbl] for lbl in labels])


# ──────────────────────────────────────────────
# 3. Image loading & preprocessing
# ──────────────────────────────────────────────
def load_and_preprocess(path, label):
    """Read, decode, resize, and scale an image to [0, 1]."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img, label


# ──────────────────────────────────────────────
# 4. Data augmentation (training only)
# ──────────────────────────────────────────────
def augment(img, label):
    """Basic augmentation — flip, brightness, contrast."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


def augment_strong(img, label):
    """
    Stronger augmentation (Phase 2) — adds saturation, hue, random crop.
    Used to combat the train-val gap from fine-tuning.
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    img = tf.image.random_hue(img, max_delta=0.02)

    # Random crop 85-100% then resize back
    crop_size = tf.random.uniform([], 0.85, 1.0)
    h = tf.cast(tf.cast(IMG_SIZE[0], tf.float32) * crop_size, tf.int32)
    w = tf.cast(tf.cast(IMG_SIZE[1], tf.float32) * crop_size, tf.int32)
    img = tf.image.random_crop(img, size=[h, w, 3])
    img = tf.image.resize(img, IMG_SIZE)

    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


# ──────────────────────────────────────────────
# 4b. RandAugment-style pipeline (Week 5 practical)
# ──────────────────────────────────────────────
def augment_randaugment(img, label):
    """
    Augmentation matching the Week 5 practical's RandAugment(factor=0.1).
    Flip + brightness + contrast + gentle rotation (±10°).
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)

    if _RA_ROTATE is not None:
        img = _RA_ROTATE(tf.expand_dims(img, 0))[0]

    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


# ──────────────────────────────────────────────
# 4c. Augmentation as in-model Keras layer (Week 4 pattern)
# ──────────────────────────────────────────────
def build_augmentation_layer():
    """
    Week 4 practical pattern: augmentation embedded inside the model.
    When using this, set augmentation="none" in build_dataset() to avoid
    double-augmenting.
    """
    import keras
    from keras.layers import RandomBrightness, RandomFlip, RandomRotation
    return keras.Sequential(
        [
            RandomBrightness(factor=0.1, value_range=(0.0, 1.0)),
            RandomFlip(),
            RandomRotation(factor=10.0 / 360.0, fill_mode="reflect"),
        ],
        name="augmentation_layer",
    )


# ──────────────────────────────────────────────
# 5. Build tf.data pipelines
# ──────────────────────────────────────────────
def build_dataset(paths, labels, is_training=False, batch_size=BATCH_SIZE,
                  augmentation="basic"):
    """
    Build a tf.data.Dataset from file paths and integer labels.
    Augmentation options: "basic", "strong", "randaugment", "none".
    """
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if is_training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training and augmentation != "none":
        if augmentation == "strong":
            aug_fn = augment_strong
        elif augmentation == "randaugment":
            aug_fn = augment_randaugment
        else:
            aug_fn = augment  # "basic"
        ds = ds.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def prepare_all_datasets(file_paths, labels, augmentation="basic"):
    """One-call convenience: split -> encode -> build tf.data pipelines."""
    (train_p, train_l), (val_p, val_l), (test_p, test_l) = split_dataset(file_paths, labels)

    train_y = encode_labels(train_l)
    val_y = encode_labels(val_l)
    test_y = encode_labels(test_l)

    train_ds = build_dataset(train_p, train_y, is_training=True, augmentation=augmentation)
    val_ds = build_dataset(val_p, val_y, is_training=False)
    test_ds = build_dataset(test_p, test_y, is_training=False)

    return train_ds, val_ds, test_ds, test_p, test_y
