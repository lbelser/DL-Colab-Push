"""
Training module (main)
"""

from pathlib import Path
from functools import partial

from keras.optimizers import SGD
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy, AUC, F1Score
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler

from src.model import AugmentedVGG16
from src.utils import load_ds as _load_ds
from src.utils import exp_decay_lr_scheduler


def train(epochs: int, batch_size: int) -> None:
    """
    """

    root_dir_path = Path(__file__).parent

    data_dir_path = root_dir_path / "flowers"

    train_dir_path = data_dir_path / "train"
    val_dir_path = data_dir_path / "val"
    test_dir_path = data_dir_path / "test"

    checkpoint_file_path = root_dir_path / "checkpoint.keras"
    metrics_file_path = root_dir_path / "metrics.csv"

    batch_size = 32
    input_shape = (224, 224, 3)

    load_ds = partial(_load_ds, batch_size=batch_size, input_shape=input_shape)
    train_ds = load_ds(train_dir_path)
    val_ds = load_ds(val_dir_path, shuffle=False)
    test_ds = load_ds(test_dir_path, shuffle=False)

    initial_lr = 0.01
    final_lr = 0.001
    factor = (final_lr / initial_lr) ** (1 / epochs)
    lr_scheduler = partial(exp_decay_lr_scheduler, factor=factor)

    model = AugmentedVGG16()
    optimizer = SGD(learning_rate=0.01, name="optimizer")
    loss = CategoricalCrossentropy(name="loss")

    # metrics
    categorical_accuracy = CategoricalAccuracy(name="accuracy")
    auc = AUC(name="auc")
    f1_score = F1Score(average="macro", name="f1_score")
    metrics = [categorical_accuracy, auc, f1_score]

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        checkpoint_file_path,
        monitor="val_loss",
        verbose=0
    )
    metrics_callback = CSVLogger(metrics_file_path)
    lr_scheduler_callback = LearningRateScheduler(lr_scheduler)
    callbacks = [
        checkpoint_callback,
        metrics_callback,
        lr_scheduler_callback
    ]

    # traces the computation
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # train the model
    _ = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=0
    )

    evaluation_dict = model.evaluate(
        test_ds,
        return_dict=True,
        verbose=0
    )

    print(evaluation_dict)


def main() -> None:
    """
    """

    from argparse import ArgumentParser

    parser = ArgumentParser(prog="augmented_vgg16 training")
    parser.add_argument(
        "--epochs",
        type=int,
        required=True
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True
    )

    args = parser.parse_args()
    train(args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
