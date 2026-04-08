"""
Training loop, callbacks, and compilation helpers.
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
# 1. Compile (Adam)
# ──────────────────────────────────────────────
def compile_model(model, learning_rate=LEARNING_RATE, label_smoothing=0.0):
    """Compile with Adam + sparse CE. Optionally applies label smoothing."""
    if label_smoothing > 0:
        loss = _smoothed_sparse_loss(label_smoothing)
    else:
        loss = "sparse_categorical_crossentropy"

    # AUC + F1 to match the Week 3 practical metric set
    try:
        extra_metrics = [
            keras.metrics.AUC(name="auc"),
            keras.metrics.F1Score(average="macro", name="f1_macro"),
        ]
    except AttributeError:
        extra_metrics = []

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"] + extra_metrics,
    )
    return model


def _smoothed_sparse_loss(smoothing):
    """Label smoothing wrapper for sparse integer labels (one-hot encodes internally)."""
    import tensorflow as tf
    from config import NUM_CLASSES

    base_loss = keras.losses.CategoricalCrossentropy(label_smoothing=smoothing)

    def loss_fn(y_true, y_pred):
        y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), NUM_CLASSES)
        return base_loss(y_true_onehot, y_pred)

    return loss_fn


# ──────────────────────────────────────────────
# 1b. Compile (SGD) — Week 3/4 practical setup
# ──────────────────────────────────────────────
def compile_model_sgd(model, learning_rate=BASELINE_LR, weight_decay=0.0):
    """
    Compile with SGD + momentum, matching the Week 3/4 practicals.
    weight_decay=0.0 for Week 3, 0.01 for Week 4.
    """
    try:
        f1_metric = keras.metrics.F1Score(average="macro", name="f1_macro")
    except AttributeError:
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
    """ModelCheckpoint + EarlyStopping + ReduceLROnPlateau + TensorBoard."""
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
    Week 4 practical callback set:
    ModelCheckpoint (val_loss) + CSVLogger + exponential LR decay + EarlyStopping.
    No ReduceLROnPlateau — the practical used an explicit schedule instead.
    """
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, f"{model_name}_best.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    csv_logger = keras.callbacks.CSVLogger(
        filename=os.path.join(LOG_DIR, f"{model_name}_training_log.csv"),
        separator=",",
        append=False,
    )

    def exponential_decay(epoch, lr):
        return float(lr * LR_DECAY_FACTOR)

    lr_scheduler = keras.callbacks.LearningRateScheduler(
        exponential_decay, verbose=0,
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    return [checkpoint, csv_logger, lr_scheduler, early_stop]


# ──────────────────────────────────────────────
# 3. Class weights
# ──────────────────────────────────────────────
def compute_weights(train_labels):
    """Compute balanced class weights so minority artists aren't ignored."""
    unique_classes = np.unique(train_labels)
    weights = compute_class_weight("balanced", classes=unique_classes, y=train_labels)
    class_weight_dict = dict(zip(unique_classes, weights))
    print("Class weights computed (sample):")
    for cls_idx in list(class_weight_dict.keys())[:5]:
        print(f"  {CLASS_NAMES[cls_idx]:25s} -> {class_weight_dict[cls_idx]:.3f}")
    print(f"  ... ({len(class_weight_dict)} classes total)")
    return class_weight_dict


# ──────────────────────────────────────────────
# 4. Train (Adam)
# ──────────────────────────────────────────────
def train_model(model, train_ds, val_ds, train_labels,
                model_name="model", epochs=EPOCHS, learning_rate=LEARNING_RATE,
                label_smoothing=0.0):
    """Compile + compute class weights + fit. Returns keras History."""
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
# 5. Train (SGD) — Week 3/4 practical setup
# ──────────────────────────────────────────────
def train_model_sgd(model, train_ds, val_ds, train_labels,
                    model_name="model", epochs=EPOCHS,
                    learning_rate=BASELINE_LR,
                    weight_decay=BASELINE_WEIGHT_DECAY):
    """
    Same as train_model() but with SGD + exponential LR decay + CSVLogger.
    weight_decay=0.0 for Week 3, BASELINE_WEIGHT_DECAY for Week 4.
    """
    compile_model_sgd(model, learning_rate=learning_rate,
                      weight_decay=weight_decay)
    class_weights = compute_weights(train_labels)
    callbacks = get_callbacks_sgd(model_name, initial_lr=learning_rate)

    wd_str = f", weight_decay={weight_decay}" if weight_decay > 0 else ""
    print(f"\n{'='*60}")
    print(f"Training {model_name}  ({model.count_params():,} parameters)")
    print(f"  Optimiser: SGD  LR={learning_rate}{wd_str}")
    print(f"  LR schedule: exponential decay (x{LR_DECAY_FACTOR} per epoch)")
    print(f"{'='*60}\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )
    return history
