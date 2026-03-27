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
    CLASS_NAMES, BASELINE_LR, BASELINE_WEIGHT_DECAY, LR_DECAY_FACTOR,
)


# ──────────────────────────────────────────────
# 1. Compile a model with standard settings
# ──────────────────────────────────────────────
def compile_model(model, learning_rate=LEARNING_RATE, label_smoothing=0.0):
    """
    Compile a model with:
      - Adam optimiser (adaptive learning rate, works well out of the box)
      - Sparse categorical crossentropy (integer labels, not one-hot)
      - Accuracy metric (easy to interpret during training)
      - AUC metric    (Week 3 practical tracked this alongside accuracy)
      - F1Score macro (Week 3 practical tracked this; more informative than
                       accuracy for imbalanced classes — artists with fewer
                       paintings should not be ignored in the metric)

    Parameters
    ----------
    label_smoothing : float (0.0 to 0.2 typical)
        Replaces hard 0/1 targets with soft targets (e.g. 0.9 / 0.004).
        Why: with 23 classes and subjective boundaries between artists,
        label smoothing prevents the model from becoming overconfident
        and acts as a form of regularisation.
    """
    if label_smoothing > 0:
        # With label smoothing we need a loss that accepts a smoothing param.
        # SparseCategoricalCrossentropy doesn't support it directly, so we
        # use CategoricalCrossentropy with a custom wrapper that one-hot
        # encodes on the fly. Alternatively, we can use the from_logits
        # approach. But the simplest: use Keras's built-in loss object.
        loss = keras.losses.SparseCategoricalCrossentropy()
        # Actually, Keras 3 doesn't have label_smoothing on Sparse loss.
        # We use CategoricalCrossentropy + one-hot conversion in the pipeline.
        # For simplicity, keep sparse and apply smoothing via the loss:
        loss = _smoothed_sparse_loss(label_smoothing)
    else:
        loss = "sparse_categorical_crossentropy"

    # Week 3 practical used: CategoricalAccuracy + AUC + F1Score(macro)
    # We replicate that metric set here so training logs are directly
    # comparable to the practical outputs.
    try:
        extra_metrics = [
            keras.metrics.AUC(name="auc"),
            keras.metrics.F1Score(average="macro", name="f1_macro"),
        ]
    except AttributeError:
        # Older Keras builds may not expose F1Score; fall back gracefully
        extra_metrics = []

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"] + extra_metrics,
    )
    return model


def _smoothed_sparse_loss(smoothing):
    """
    Create a loss function that applies label smoothing to sparse
    integer labels by one-hot encoding them first.
    """
    import tensorflow as tf
    from config import NUM_CLASSES

    base_loss = keras.losses.CategoricalCrossentropy(label_smoothing=smoothing)

    def loss_fn(y_true, y_pred):
        y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), NUM_CLASSES)
        return base_loss(y_true_onehot, y_pred)

    return loss_fn


# ──────────────────────────────────────────────
# 1b. Compile — Week 3 / 4 practical setup (SGD)
# ──────────────────────────────────────────────
def compile_model_sgd(model, learning_rate=BASELINE_LR,
                      weight_decay=0.0):
    """
    Compile a model with SGD, mirroring the Week 3 and Week 4 practical setup.

    Week 3 practical used: SGD(learning_rate=0.01)
    Week 4 practical added: SGD(learning_rate=0.01, weight_decay=0.01)

    Metrics match the practical:
      - accuracy (CategoricalAccuracy equivalent with sparse labels)
      - F1Score macro-averaged across all 23 artist classes

    Parameters
    ----------
    learning_rate : float  — initial LR; decayed by LearningRateScheduler
    weight_decay  : float  — L2 penalty baked into the SGD update
                             (0.0 for Week 3 plain CNN; 0.01 for Week 4)
    """
    try:
        # Keras 3 / keras-core
        f1_metric = keras.metrics.F1Score(average="macro", name="f1_macro")
    except AttributeError:
        # Fallback: F1 not built-in; track only accuracy during training
        f1_metric = None

    metrics = ["accuracy"] + ([f1_metric] if f1_metric is not None else [])

    model.compile(
        optimizer=keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=metrics,
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


def get_callbacks_sgd(model_name: str, initial_lr: float = BASELINE_LR):
    """
    Callback set that matches the Week 4 practical exactly:

      - ModelCheckpoint : saves best weights by val_loss
                          (practical monitored val_loss, not val_accuracy)
      - CSVLogger       : writes every epoch's metrics to a CSV file —
                          introduced in the Week 4 practical for reproducible
                          run logging
      - LearningRateScheduler : exponential decay lr *= 0.95 per epoch,
                          matching the Week 4 practical scheduler
      - EarlyStopping   : stops if val_loss stalls (implied in the practical)

    Intentionally does NOT include ReduceLROnPlateau — the practical used
    an explicit exponential schedule instead.
    """
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, f"{model_name}_best.keras"),
        monitor="val_loss",          # practical monitors val_loss
        save_best_only=True,
        verbose=1,
    )

    csv_logger = keras.callbacks.CSVLogger(
        filename=os.path.join(LOG_DIR, f"{model_name}_training_log.csv"),
        separator=",",
        append=False,
    )

    def exponential_decay(epoch, lr):
        """Decay factor 0.95 per epoch — matches the Week 4 practical."""
        return float(lr * LR_DECAY_FACTOR)

    lr_scheduler = keras.callbacks.LearningRateScheduler(
        exponential_decay,
        verbose=0,
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    return [checkpoint, csv_logger, lr_scheduler, early_stop]


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
                model_name="model", epochs=EPOCHS, learning_rate=LEARNING_RATE,
                label_smoothing=0.0):
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
    label_smoothing : float, soft-target regularisation (0.0 = off)

    Returns
    -------
    history : keras History object (contains loss/accuracy per epoch)
    """
    compile_model(model, learning_rate=learning_rate, label_smoothing=label_smoothing)
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


# ──────────────────────────────────────────────
# 5. Train — Week 3 / 4 practical setup (SGD)
# ──────────────────────────────────────────────
def train_model_sgd(model, train_ds, val_ds, train_labels,
                    model_name="model", epochs=EPOCHS,
                    learning_rate=BASELINE_LR,
                    weight_decay=BASELINE_WEIGHT_DECAY):
    """
    Full training pipeline using the Week 3/4 practical setup.

    Differences from train_model():
      - SGD optimiser with momentum + weight_decay  (not Adam)
      - Exponential LR decay scheduler              (not ReduceLROnPlateau)
      - CSVLogger writes metrics to disk each epoch (not TensorBoard only)
      - ModelCheckpoint monitors val_loss            (not val_accuracy)

    Call with weight_decay=0.0 for the plain Week 3 CNN.
    Call with weight_decay=BASELINE_WEIGHT_DECAY for the Week 4 version.

    Parameters
    ----------
    model        : uncompiled keras Model
    train_ds     : training tf.data.Dataset
    val_ds       : validation tf.data.Dataset
    train_labels : integer numpy array (needed for class-weight computation)
    model_name   : used in checkpoint and CSV filenames
    epochs       : maximum training epochs
    learning_rate: initial SGD learning rate
    weight_decay : L2 coefficient for SGD (0.0 = Week 3; 0.01 = Week 4)

    Returns
    -------
    history : keras History object
    """
    compile_model_sgd(model, learning_rate=learning_rate,
                      weight_decay=weight_decay)
    class_weights = compute_weights(train_labels)
    callbacks = get_callbacks_sgd(model_name, initial_lr=learning_rate)

    wd_str = f", weight_decay={weight_decay}" if weight_decay > 0 else ""
    print(f"\n{'='*60}")
    print(f"Training {model_name}  ({model.count_params():,} parameters)")
    print(f"  Optimiser: SGD  LR={learning_rate}{wd_str}")
    print(f"  LR schedule: exponential decay (×{LR_DECAY_FACTOR} per epoch)")
    print(f"{'='*60}\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )
    return history
