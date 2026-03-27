"""
Training module (main)
"""

from pathlib import Path
from functools import partial

from keras.optimizers import SGD
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy, AUC, F1Score
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler

from src.model import MyTinyRegularizedCNN
from src.utils import load_cifar10_sample, exp_decay_lr_scheduler


def train(epochs: int, batch_size: int, n_train: int, n_test: int) -> None:
    """
    Training function
    """

    root_dir_path = Path(__file__).parent
    checkpoint_file_path = root_dir_path / "checkpoint.keras"
    metrics_file_path = root_dir_path = root_dir_path / "metrics.csv"

    # get data
    X_train, y_train, X_test, y_test = load_cifar10_sample(n_train, n_test)

    # define scheduling behaviour
    initial_lr = 0.01
    final_lr = 0.001
    factor = (final_lr / initial_lr) ** (1 / epochs)
    lr_scheduler = partial(exp_decay_lr_scheduler, factor=factor)

    model = MyTinyRegularizedCNN()
    optimizer = SGD(learning_rate=0.01, name="optimizer", weight_decay = 0.01)
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
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0
    )

    evaluation_dict = model.evaluate(
        X_test,
        y_test,
        batch_size=batch_size,
        return_dict=True,
        verbose=0
    )

    print(evaluation_dict)


def main() -> None:
    """
    Module's main function
    """

    from argparse import ArgumentParser

    parser = ArgumentParser(prog="my_tiny_cnn training")
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
    parser.add_argument(
        "--n_train",
        type=int,
        default=1024
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=128
    )

    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.n_train, args.n_test)


if __name__ == "__main__":
    main()
