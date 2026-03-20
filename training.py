"""
Training loop, callbacks, and compilation helpers.

Separating training logic from model definitions keeps the
notebook clean and makes it easy to train different models
with the same setup.
"""

import os
import numpy as np
import keras
from sklearn.utils.class_weight import compute_class_weight

from config import (
    LEARNING_RATE, EPOCHS, MODEL_DIR, LOG_DIR,
    EARLY_STOP_PATIENCE, LR_REDUCE_PATIENCE, LR_REDUCE_FACTOR,
    CLASS_NAMES,
)


# ──────────────────────────────────────────────
# 1. Compile a model with standard settings
# ──────────────────────────────────────────────
def compile_model(model, learning_rate=LEARNING_RATE):
    """
    Compile a model with:
      - Adam optimiser (adaptive learning rate, works well out of the box)
      - Sparse categorical crossentropy (integer labels, not one-hot)
      - Accuracy metric (easy to interpret during training)
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ──────────────────────────────────────────────
# 2. Callbacks
# ──────────────────────────────────────────────
def get_callbacks(model_name: str):
    """
    Set up training callbacks:

    - ModelCheckpoint: save the best model (by val_accuracy) so we
      never lose progress even if training degrades later.
    - EarlyStopping: stop training when val_accuracy hasn't improved
      for EARLY_STOP_PATIENCE epochs — prevents wasting compute and
      overfitting.
    - ReduceLROnPlateau: if val_loss plateaus, reduce the learning
      rate to allow finer optimisation steps.
    - TensorBoard: log training metrics for interactive visualisation.
    """
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, f"{model_name}_best.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    lr_reduce = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=LR_REDUCE_FACTOR,
        patience=LR_REDUCE_PATIENCE,
        min_lr=1e-7,
        verbose=1,
    )

    tensorboard = keras.callbacks.TensorBoard(
        log_dir=os.path.join(LOG_DIR, model_name),
    )

    return [checkpoint, early_stop, lr_reduce, tensorboard]


# ──────────────────────────────────────────────
# 3. Class weights (handle imbalanced data)
# ──────────────────────────────────────────────
def compute_weights(train_labels):
    """
    Compute class weights inversely proportional to class frequency.

    Why: artists with fewer paintings (e.g. Salvador Dali, 336 images)
    would otherwise contribute less to the loss than artists with many
    paintings (e.g. Van Gogh, 1322 images). Class weights tell the
    model to pay MORE attention to under-represented artists.
    """
    unique_classes = np.unique(train_labels)
    weights = compute_class_weight("balanced", classes=unique_classes, y=train_labels)
    class_weight_dict = dict(zip(unique_classes, weights))
    print("Class weights computed (sample):")
    for cls_idx in list(class_weight_dict.keys())[:5]:
        print(f"  {CLASS_NAMES[cls_idx]:25s} → {class_weight_dict[cls_idx]:.3f}")
    print(f"  ... ({len(class_weight_dict)} classes total)")
    return class_weight_dict


# ──────────────────────────────────────────────
# 4. Train a model (one-call convenience)
# ──────────────────────────────────────────────
def train_model(model, train_ds, val_ds, train_labels,
                model_name="model", epochs=EPOCHS, learning_rate=LEARNING_RATE):
    """
    Full training pipeline: compile → compute weights → fit.

    Parameters
    ----------
    model       : uncompiled keras Model
    train_ds    : training tf.data.Dataset
    val_ds      : validation tf.data.Dataset
    train_labels: integer numpy array (needed for class-weight computation)
    model_name  : used in checkpoint filenames and TensorBoard
    epochs      : maximum number of training epochs
    learning_rate : initial learning rate

    Returns
    -------
    history : keras History object (contains loss/accuracy per epoch)
    """
    compile_model(model, learning_rate=learning_rate)
    class_weights = compute_weights(train_labels)
    callbacks = get_callbacks(model_name)

    print(f"\n{'='*60}")
    print(f"Training {model_name}  ({model.count_params():,} parameters)")
    print(f"{'='*60}\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    return history
