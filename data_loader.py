"""
Data loading, splitting, and augmentation pipeline.

This module handles the full journey from raw image files on disk
to ready-to-train tf.data.Dataset objects.
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Module-level rotation layer — created once at import time so it is not
# re-instantiated on every tf.data.map() call (performance + thread safety).
_RA_ROTATE = None
try:
    import keras as _keras
    _RA_ROTATE = _keras.layers.RandomRotation(factor=10.0 / 360.0, fill_mode="reflect")
except Exception:
    pass

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

    BASIC augmentation — used in Phase 1 experiments.
      - Random horizontal flip  : paintings have no inherent left/right
      - Random brightness shift : handles varying photo conditions
      - Random contrast shift   : same reason
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    # Clip to valid range after augmentation
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


def augment_strong(img, label):
    """
    STRONGER augmentation — used in Phase 2 after observing
    the train-val gap (overfitting) in fine-tuned models.

    Additional transforms over the basic set:
      - Random saturation   : paintings vary widely in colour intensity
      - Random hue shift    : small shift to handle colour reproduction
      - Random JPEG quality : simulates varying scan/photo quality
      - Random crop + resize: forces model to handle partial views,
        reduces reliance on composition and encourages texture learning

    Why stronger: ResNet50 fine-tuned showed a ~11 point train-val gap
    (92% train vs 81% val). More augmentation acts as regularisation
    to close this gap without reducing model capacity.
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    img = tf.image.random_hue(img, max_delta=0.02)

    # Random crop: take 85-100% of the image, then resize back
    # This forces the model to recognise style from partial views
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
    Augmentation pipeline inspired by the RandAugment layer used in the
    Week 5 practical (keras_cv.layers.RandAugment, factor=0.1).

    The practical placed RandAugment inside the model's call() method.
    Here we apply equivalent operations in the tf.data pipeline, which
    is the recommended pattern for GPU-based preprocessing in TF2.

    Operations (factor ≈ 0.1, i.e. gentle):
      - RandomBrightness ±10 %   (same delta as practical factor=0.1)
      - RandomFlip horizontal    (paintings have no inherent orientation)
      - RandomRotation ±10°      (very gentle; practicals used factor=0.1
                                  → 10 % of 360° ≈ 36°; we use 10° to
                                  respect painting composition)
      - Clip to [0, 1]

    Why NOT the basic/strong pipelines:
      The practical used RandAugment as the single augmentation choice
      for transfer learning. We use this pipeline in the Week 5 section
      (ResNet50) so that section matches the practical's approach.
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)

    # Random rotation via module-level layer (created once at import time)
    if _RA_ROTATE is not None:
        img = _RA_ROTATE(tf.expand_dims(img, 0))[0]

    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


# ──────────────────────────────────────────────
# 4c. Augmentation as a Keras Pipeline layer (Week 4 in-model pattern)
# ──────────────────────────────────────────────
def build_augmentation_layer():
    """
    Return the Week 4 practical augmentation pipeline as a Keras layer.

    The Week 4 practical (MyTinyRegularizedCNN) embedded augmentation
    INSIDE the model using keras.layers.Pipeline (Keras CV). We replicate
    the same transforms here using keras.Sequential, which is available in
    all standard keras/tf.keras builds without extra dependencies.

    This function returns that layer so model definitions can mirror the
    Week 4 pattern (used in build_baseline_cnn_regularised()).

    The project's main pipeline applies augmentation via tf.data.map()
    (see augment(), augment_strong(), augment_randaugment()) — the
    GPU-friendly modern approach. This function provides the Week 4
    in-model equivalent for educational comparison.

    When this layer is embedded inside a model, train that model with
    augmentation="none" in build_dataset() to avoid double-augmenting.

    Returns
    -------
    keras.Sequential
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
    Assemble a tf.data.Dataset from file paths and integer labels.

    For training data we shuffle and apply augmentation.
    For validation / test data we only load, resize, and batch.

    Prefetching lets the CPU prepare the next batch while the
    GPU is busy training — a simple but important optimisation.

    Parameters
    ----------
    augmentation : "basic" | "strong" | "randaugment" | "none"
        Which augmentation pipeline to apply to training data.
        "basic"        — flips, brightness, contrast (Phase 1 / Week 3 baseline)
        "strong"       — adds saturation, hue, crop+resize (Phase 2 extensions)
        "randaugment"  — gentle flip + brightness + rotation, mirroring the
                         RandAugment(factor=0.1) shown in the Week 5 practical
        "none"         — no augmentation (for val/test, or ablation)
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
    """
    One-call convenience: split → encode → build tf.data pipelines.

    Parameters
    ----------
    augmentation : "basic" | "strong"
        Passed through to build_dataset for the training set.

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
    train_ds = build_dataset(train_p, train_y, is_training=True, augmentation=augmentation)
    val_ds = build_dataset(val_p, val_y, is_training=False)
    test_ds = build_dataset(test_p, test_y, is_training=False)

    return train_ds, val_ds, test_ds, test_p, test_y
