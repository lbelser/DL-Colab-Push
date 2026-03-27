"""
Module containing dataset and training utilities,
When ran as main, will show a example from the train subset
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from random import sample

from numpy import ndarray
from keras.datasets.cifar10 import load_data as load_cifar10
from keras.utils import to_categorical

from matplotlib.pyplot import subplots, show


def normalize(array: ndarray) -> ndarray:
    """
    Converts the type of an ndarray to float32 and normalizes over the pixel
    range of values
    """

    return array.astype("float32") / 255.0


def load_cifar10_sample(n_train: int, n_test: int) -> tuple[ndarray, ...]:
    """
    Gets a normalized subset of the CIFAR-10 dataset
    """

    n_classes = 10

    (X_train, y_train), (X_test, y_test) = load_cifar10()

    train_idxs = sample(range(len(X_train)), k=n_train)
    X_train = normalize(X_train[train_idxs])
    y_train = to_categorical(y_train[train_idxs], n_classes)

    test_idxs = sample(range(len(X_test)), k=n_test)
    X_test = normalize(X_test[test_idxs])
    y_test = to_categorical(y_test[test_idxs], n_classes)

    return X_train, y_train, X_test, y_test


def show_image(array: ndarray) -> None:
    """
    Prints image encoded as a numpy array (uint8)
    """

    figure, axis = subplots(frameon=False)
    axis.imshow(array, aspect="equal")
    axis.set_axis_off()
    show()


def exp_decay_lr_scheduler(
    epoch: int,
    current_lr: float,
    factor: float = 0.95
) -> float:
    """
    Exponential decay learning rate scheduler
    """

    current_lr *= factor

    return current_lr


def main() -> None:
    """
    Module's main function
    """

    X_train, _, _, _ = load_cifar10_sample(1, 1)

    train_example = X_train[0]
    show_image(train_example)

if __name__ == "__main__":
    main()
