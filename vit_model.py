"""
Vision Transformer (ViT-B/16) model builder for the WikiArt project.

Uses a pre-trained ViT-B/16 backbone from TensorFlow Hub (ImageNet-21k weights).
Follows the same two-phase training strategy as ResNet50/EfficientNet:
  Phase 1: frozen backbone, train classification head only
  Phase 2: unfreeze full backbone, fine-tune at LR = 1e-5

Why ViT here:
  CNNs (ResNet, EfficientNet) capture LOCAL features through sliding filters.
  ViT splits the image into 16×16 patches and processes them with self-attention,
  capturing GLOBAL composition across the full canvas. Artistic style often
  depends on whole-painting composition and patch relationships that CNNs miss.
  ViT's orthogonal inductive bias makes it an ideal ensemble partner.

TF Hub model: sayakpaul/vit_b16_fe/1
  - Pre-trained on ImageNet-21k (21 000 categories, 14M images)
  - Outputs a 768-dimensional feature vector per image
  - Expects float32 inputs in [0, 1] range — compatible with our pipeline

Usage:
    from vit_model import build_vit, unfreeze_vit
    model = build_vit(freeze_base=True)       # Phase 1
    unfreeze_vit(model)                        # Phase 2 prep
"""

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers, Model

from config import IMG_SIZE, NUM_CLASSES

# ViT-B/16 pre-trained on ImageNet-21k, feature extraction variant
# Outputs: (batch_size, 768) — the [CLS] token embedding
VIT_URL = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"


def build_vit(freeze_base=True):
    """
    Build a ViT-B/16 transfer learning model.

    Architecture:
        Input (224×224×3)
        → ViT-B/16 backbone (hub.KerasLayer, 86M params)
            Splits image into 14×14 = 196 patches of 16×16 pixels
            Each patch → linear embedding (768-dim)
            12 transformer encoder blocks with multi-head self-attention
            Outputs [CLS] token: (batch, 768)
        → Dropout(0.5)
        → Dense(256, ReLU)
        → Dropout(0.3)
        → Dense(23, Softmax)

    Parameters
    ----------
    freeze_base : bool
        True  → Phase 1 (only train the head, backbone frozen)
        False → Phase 2 (full model trainable, use with LR=1e-5)

    Returns
    -------
    keras.Model named "ViT_B16"
    """
    vit_layer = hub.KerasLayer(
        VIT_URL,
        trainable=not freeze_base,
        name="vit_b16",
    )

    inputs = layers.Input(shape=(*IMG_SIZE, 3), name="input_image")

    # ViT expects [0, 1] inputs — our pipeline normalises to [0, 1] already
    x = vit_layer(inputs)   # (batch, 768) CLS token

    # Classification head — same structure as ResNet50/EfficientNet for consistency
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs, outputs, name="ViT_B16")
    return model


def unfreeze_vit(model):
    """
    Unfreeze the ViT backbone for Phase 2 fine-tuning.

    Unlike ResNet50 (where we unfreeze from a specific layer index),
    ViT's hub.KerasLayer does not expose indexed sub-layers. Instead,
    we unfreeze the entire backbone and rely on the very low learning
    rate (1e-5) to make gentle updates without catastrophic forgetting.

    Call this before recompiling with LR=1e-5 and label_smoothing=0.1.

    Parameters
    ----------
    model : keras.Model — the ViT model returned by build_vit()
    """
    unfrozen = False
    for layer in model.layers:
        if isinstance(layer, hub.KerasLayer):
            layer.trainable = True
            n_weights = len(layer.trainable_weights)
            print(f"[ViT] Unfrozen: '{layer.name}' ({n_weights} weight tensors)")
            unfrozen = True
            break

    if not unfrozen:
        print("[ViT] Warning: no hub.KerasLayer found in model — nothing unfrozen.")

    total_trainable = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    print(f"[ViT] Total trainable parameters after unfreeze: {total_trainable:,}")
