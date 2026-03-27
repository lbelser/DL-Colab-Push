"""
Module containing MyTinyCNN definition,
When ran as main, shows the model's summary
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from typing import Self, Any

from keras import Model
from keras.layers import Rescaling, RandAugment
from keras.layers import Flatten, Dense
from keras.applications import VGG16


class AugmentedVGG16(Model):
    """
    Pre-trained VG16 + RandAugment
    """

    def __init__(self: Self) -> None:
        """
        Initialization
        """

        super().__init__()

        self.n_classes = 3
        self.rescale_layer = Rescaling(scale=1 / 255.0)
        self.augmentation_layer = RandAugment(value_range=(0.0, 1.0))
        self.pre_trained_architecture = VGG16(include_top=False, classes=32)
        self.flatten_layer = Flatten()
        self.dense_layer = Dense(self.n_classes, activation="softmax")

    def call(self: Self, inputs: Any) -> Any:
        """
        Forward call
        """

        x = self.rescale_layer(inputs)
        x = self.augmentation_layer(x)
        x = self.pre_trained_architecture(x)
        x = self.flatten_layer(x)

        return self.dense_layer(x)


def main() -> None:
    """
    Module's main function
    """

    from keras import Input

    input_shape = (224, 224, 3)
    model = AugmentedVGG16()

    inputs = Input(shape=input_shape)
    _ = model.call(inputs)

    model.summary()


if __name__ == "__main__":
    main()
