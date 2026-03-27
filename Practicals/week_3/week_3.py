"""
Training module (main)
"""

from functools import partial

from keras.optimizers import SGD
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy, AUC, F1Score

from src.model import MyTinyCNN
from src.utils import load_cifar10_sample


def train(epochs: int, batch_size: int, n_train: int, n_test: int) -> None:
    """
    Training function
    """

    # get data
    X_train, y_train, X_test, y_test = load_cifar10_sample(n_train, n_test)

    model = MyTinyCNN()
    optimizer = SGD(learning_rate=0.01, name="optimizer")
    loss = CategoricalCrossentropy(name="loss")

    # metrics
    categorical_accuracy = CategoricalAccuracy(name="accuracy")
    auc = AUC(name="auc")
    f1_score = F1Score(average="macro", name="f1_score")
    metrics = [categorical_accuracy, auc, f1_score]

    # traces the computation
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # train the model
    _ = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
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
