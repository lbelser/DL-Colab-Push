"""
Model architectures for artist classification.
Baseline CNN -> transfer learning (ResNet50, EfficientNet) -> ensemble.
"""

import numpy as np
import keras
from keras import layers, Model
from config import IMG_SIZE, NUM_CLASSES


# ──────────────────────────────────────────────
# 1. Baseline CNN (from scratch)
# ──────────────────────────────────────────────
def build_baseline_cnn():
    """
    3-block CNN as a performance baseline.
    Conv(32) -> Conv(64) -> Conv(128) -> GAP -> Dense -> Softmax
    """
    inputs = layers.Input(shape=(*IMG_SIZE, 3), name="input_image")

    # Block 1
    x = layers.Conv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs, outputs, name="Baseline_CNN")
    return model


# ──────────────────────────────────────────────
# 1b. Baseline CNN — Week 4 regularised version
# ──────────────────────────────────────────────
def build_baseline_cnn_regularised():
    """
    Week 4 version of the baseline CNN with regularisation:
    LeakyReLU, L2 on Conv2D, Dropout(0.3) after each block,
    and augmentation embedded inside the model.
    Use augmentation="none" in build_dataset() with this model.
    """
    from keras.regularizers import l2
    from data_loader import build_augmentation_layer

    inputs = layers.Input(shape=(*IMG_SIZE, 3), name="input_image")
    x = build_augmentation_layer()(inputs)

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D()(x)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return Model(inputs, outputs, name="Baseline_CNN_Regularised")


# ──────────────────────────────────────────────
# 2. Transfer learning — ResNet50
# ──────────────────────────────────────────────
def build_resnet50(freeze_base=True):
    """ResNet50 (ImageNet) with a custom classification head."""
    base = keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = not freeze_base

    inputs = layers.Input(shape=(*IMG_SIZE, 3), name="input_image")
    # Our pipeline normalises to [0,1], but ResNet expects [0,255] + its own preprocessing
    x = layers.Rescaling(255.0)(inputs)
    x = keras.applications.resnet50.preprocess_input(x)
    x = base(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs, outputs, name="ResNet50_Transfer")
    return model


# ──────────────────────────────────────────────
# 3. Transfer learning — EfficientNetB0
# ──────────────────────────────────────────────
def build_efficientnet(freeze_base=True):
    """EfficientNetB0 (ImageNet) with a custom classification head."""
    base = keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = not freeze_base

    inputs = layers.Input(shape=(*IMG_SIZE, 3), name="input_image")
    x = layers.Rescaling(255.0)(inputs)
    x = base(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs, outputs, name="EfficientNetB0_Transfer")
    return model


# ──────────────────────────────────────────────
# 4. Transfer learning — EfficientNetB2
# ──────────────────────────────────────────────
def build_efficientnet_b2(freeze_base=True):
    """
    EfficientNetB2 (ImageNet) — upgraded from B0 for more capacity.
    B0 was too small (5.3M params) to match ResNet50 (25.6M).
    B2 has ~9.2M params. Still using 224x224 for fair comparison.
    """
    base = keras.applications.EfficientNetB2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = not freeze_base

    inputs = layers.Input(shape=(*IMG_SIZE, 3), name="input_image")
    x = layers.Rescaling(255.0)(inputs)
    x = base(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs, outputs, name="EfficientNetB2_Transfer")
    return model


# ──────────────────────────────────────────────
# 5. Ensemble
# ──────────────────────────────────────────────
def build_ensemble(models):
    """Create an ensemble that averages softmax predictions from multiple models."""
    return EnsemblePredictor(models)


class EnsemblePredictor:
    """Averages predictions from multiple models. Has a .predict() method
    so it works with our evaluation functions."""
    def __init__(self, models):
        self.models = models
        self.name = "Ensemble"

    def predict(self, x, verbose=0):
        preds = [m.predict(x, verbose=verbose) for m in self.models]
        return np.mean(preds, axis=0)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def unfreeze_base(model, unfreeze_from_layer=None):
    """
    Unfreeze the backbone for fine-tuning (Phase 2).
    If unfreeze_from_layer is given, only layers from that index onward
    are unfrozen — keeps early generic layers frozen.
    """
    base = None
    for layer in model.layers:
        if hasattr(layer, "layers") and len(layer.layers) > 10:
            base = layer
            break

    if base is None:
        print("Could not find base model to unfreeze.")
        return

    base.trainable = True

    if unfreeze_from_layer is not None:
        for layer in base.layers[:unfreeze_from_layer]:
            layer.trainable = False
        n_unfrozen = len(base.layers) - unfreeze_from_layer
        print(f"Unfroze {n_unfrozen} / {len(base.layers)} layers in {base.name}")
    else:
        print(f"Unfroze all {len(base.layers)} layers in {base.name}")
