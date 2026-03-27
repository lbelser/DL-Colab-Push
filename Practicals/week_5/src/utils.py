"""
Module containing image dataset and training utilities,
"""

from typing import Any
from pathlib import Path

from numpy import ndarray
from matplotlib.pyplot import subplots, show
from keras.utils import image_dataset_from_directory


def show_image(array: ndarray) -> None:
    """
    Prints image encoded as a numpy array (uint8)
    """

    figure, axis = subplots(frameon=False)
    axis.imshow(array, aspect="equal")
    axis.set_axis_off()
    show()


def load_ds(
    dir_path: Path,
    batch_size: int,
    input_shape: tuple[int, ...],
    shuffle: bool = True
) -> Any:
    """
    """

    height, width, n_channels = input_shape
    image_size = (height, width)

    ds = image_dataset_from_directory(
        dir_path,
        label_mode="categorical",
        batch_size=batch_size,
        image_size=image_size,
        interpolation="bilinear",
        shuffle=shuffle,
        verbose=False
    )

    return ds


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

    pass

if __name__ == "__main__":
    main()
