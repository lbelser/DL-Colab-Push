"""
Model architectures for artist classification.

We follow an iterative approach (simple → complex):
  1. Baseline CNN       — built from scratch, establishes a lower bound
  2. Transfer learning  — pretrained models (fine-tuned), expected to
                          perform much better because they already
                          understand visual features from ImageNet

All models output NUM_CLASSES logits with softmax activation.
"""

import numpy as np
import keras
from keras import layers, Model
from config import IMG_SIZE, NUM_CLASSES


# ──────────────────────────────────────────────
# 1. Baseline CNN (built from scratch)
# ──────────────────────────────────────────────
def build_baseline_cnn():
    """
    A simple convolutional neural network with 3 conv blocks.

    Purpose: establish a performance baseline. Because this model
    has no pretrained knowledge, it must learn ALL visual features
    from our ~13 k images alone. We expect it to underperform
    transfer-learning approaches, but it gives us a reference point.

    Architecture:
      Conv(32) → Conv(64) → Conv(128) → GlobalAvgPool → Dense → Softmax
    Each conv block = Conv2D + BatchNorm + ReLU + MaxPool
    """
    inputs = layers.Input(shape=(*IMG_SIZE, 3), name="input_image")

    # Block 1 — detect simple edges and colour patterns
    x = layers.Conv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    # Block 2 — combine edges into textures and shapes
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    # Block 3 — higher-level patterns (brushstrokes, composition)
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)  # regularisation to fight overfitting
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs, outputs, name="Baseline_CNN")
    return model


# ──────────────────────────────────────────────
# 2. Transfer learning — ResNet50
# ──────────────────────────────────────────────
def build_resnet50(freeze_base=True):
    """
    ResNet50 pretrained on ImageNet with a custom classification head.

    Why ResNet50:
      - Deep residual connections solve the vanishing-gradient problem
      - Strong ImageNet performance transfers well to art (colour,
        texture, and composition features are shared)
      - Well-studied architecture, good baseline for transfer learning

    Parameters
    ----------
    freeze_base : bool
        If True, freeze all ResNet layers and only train the new head.
        Set to False later for fine-tuning the entire network.
    """
    # Load the pretrained backbone WITHOUT the original 1000-class head
    base = keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = not freeze_base

    inputs = layers.Input(shape=(*IMG_SIZE, 3), name="input_image")
    # ResNet expects pixel values in [0, 255] and applies its own
    # preprocessing internally — but since our pipeline normalises
    # to [0, 1], we rescale back before feeding into the base.
    x = layers.Rescaling(255.0)(inputs)
    x = keras.applications.resnet50.preprocess_input(x)
    x = base(x, training=False)

    # Custom classification head
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
    """
    EfficientNetB0 pretrained on ImageNet with a custom head.

    Why EfficientNet:
      - Achieves better accuracy-per-parameter than ResNet by
        systematically scaling width, depth, and resolution
      - Smaller and faster than ResNet50 while often matching
        or exceeding its accuracy
      - Good counterpoint to ResNet for comparative analysis

    Parameters
    ----------
    freeze_base : bool
        If True, freeze backbone weights (feature extraction mode).
    """
    base = keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = not freeze_base

    inputs = layers.Input(shape=(*IMG_SIZE, 3), name="input_image")
    # EfficientNet expects [0, 255] internally
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
    EfficientNetB2 pretrained on ImageNet with a custom head.

    Why B2 instead of B0:
      B0 achieved 73.9% vs ResNet50's 78.8%. The likely bottleneck
      was model capacity — B0 has only 9M params while ResNet50 has
      25.6M. B2 scales up to ~9.2M params with native 260×260
      resolution, giving it more feature channels and depth to
      capture the diversity of 23 artistic styles.

    We still use 224×224 input to keep the comparison fair and
    avoid reprocessing the data pipeline.
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
# 5. Ensemble — average predictions from multiple models
# ──────────────────────────────────────────────
def build_ensemble(models):
    """
    Combine multiple trained models into an ensemble that averages
    their softmax predictions.

    Why ensembling works: different architectures learn different
    features and make different errors. Averaging their predictions
    cancels out individual mistakes — the ensemble is typically more
    accurate than any single model.

    Parameters
    ----------
    models : list of trained keras Models (must share the same input shape)

    Returns an EnsemblePredictor (not a keras Model, but has a predict method).
    """
    return EnsemblePredictor(models)


class EnsemblePredictor:
    """
    Lightweight wrapper that averages predictions from multiple models.
    Compatible with our evaluation pipeline (has a .predict method).
    """
    def __init__(self, models):
        self.models = models
        self.name = "Ensemble"

    def predict(self, x, verbose=0):
        """Average the softmax outputs of all models."""
        preds = [m.predict(x, verbose=verbose) for m in self.models]
        return np.mean(preds, axis=0)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def unfreeze_base(model, unfreeze_from_layer=None):
    """
    Unfreeze the backbone for fine-tuning.

    After the classification head has converged (with a frozen base),
    we unfreeze some or all base layers and continue training with a
    much lower learning rate. This lets the pretrained features adapt
    to the specific characteristics of paintings.

    Parameters
    ----------
    model : keras.Model
    unfreeze_from_layer : int or None
        If given, only layers from this index onward are unfrozen.
        Keeps early (generic) layers frozen, fine-tunes later (specific) ones.
    """
    # The base model is always the second layer (after Input + Rescaling)
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
        # Freeze everything before the given layer index
        for layer in base.layers[:unfreeze_from_layer]:
            layer.trainable = False
        n_unfrozen = len(base.layers) - unfreeze_from_layer
        print(f"Unfroze {n_unfrozen} / {len(base.layers)} layers in {base.name}")
    else:
        print(f"Unfroze all {len(base.layers)} layers in {base.name}")
