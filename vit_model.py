"""
Vision Transformer (ViT-B/16) for the WikiArt pipeline.

Uses a pre-trained ViT-B/16 from TF Hub (ImageNet-21k weights).
Same two-phase training as ResNet50/EfficientNet:
  Phase 1: frozen backbone, train head only
  Phase 2: unfreeze full backbone, fine-tune at LR=1e-5

ViT processes images as 16x16 patches with self-attention, so it can
capture global composition patterns that CNNs miss. This makes it a
good third model for the ensemble since its mistakes will be different.
"""

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers, Model

from config import IMG_SIZE, NUM_CLASSES

VIT_URL = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"


def build_vit(freeze_base=True):
    """
    ViT-B/16 with a custom classification head.
    The hub layer outputs a 768-dim CLS token embedding.
    """
    vit_layer = hub.KerasLayer(
        VIT_URL,
        trainable=not freeze_base,
        name="vit_b16",
    )

    inputs = layers.Input(shape=(*IMG_SIZE, 3), name="input_image")
    x = vit_layer(inputs)   # (batch, 768)

    # Same head as ResNet/EfficientNet for fair comparison
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs, outputs, name="ViT_B16")
    return model


def unfreeze_vit(model):
    """
    Unfreeze the ViT backbone for fine-tuning.
    hub.KerasLayer doesn't expose sub-layers, so we unfreeze
    everything and use a very low LR (1e-5) instead.
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
        print("[ViT] Warning: no hub.KerasLayer found.")

    total_trainable = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    print(f"[ViT] Total trainable parameters after unfreeze: {total_trainable:,}")
