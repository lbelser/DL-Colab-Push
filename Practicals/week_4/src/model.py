"""
Module containing a regularized MyTinyCNN definition,
When ran as main, shows the model's summary
"""

from typing import Self, Any

from keras import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.layers import RandomBrightness, RandomFlip, RandomRotation
from keras.layers import Pipeline
from keras.ops import add


augmentation_layer = Pipeline(
    [
        RandomBrightness(factor=0.1, value_range=(0.0, 1.0)),
        RandomFlip(),
        RandomRotation(factor=0.1, fill_mode="reflect")
    ],
    name="augmentation_layer"
)


class MyTinyRegularizedCNN(Model):
    """
    MyTinyCNN class, inherets from keras' Model class
    """

    def __init__(self: Self, activation: str = "relu") -> None:
        """
        Initialization
        """

        super().__init__(name="my_tiny_oo_cnn")
        self.n_classes = 10

        self.augmentation_layer = augmentation_layer

        self.conv_layer_1 = Conv2D(
            filters=3 * 8,
            kernel_size=(3, 3),
            activation=activation,
            name="conv_layer_1"
        )
        self.max_pool_layer_1 = MaxPooling2D(
            pool_size=(2, 2),
            name="max_pool_layer_1"
        )

        # exemplify non-sequential nature of computation possible with
        # the functional and object-oriented methods
        self.conv_layer_2l = Conv2D(
            filters=3 * 16,
            kernel_size=(3, 3),
            activation=activation,
            name="conv_layer_2l",
            padding="same"
        )
        self.conv_layer_2r = Conv2D(
            filters=3 * 16,
            kernel_size=(2, 2),
            activation=activation,
            name="conv_layer_2r",
            padding="same"
        )
        self.max_pool_layer_2 = MaxPooling2D(
            pool_size=(2, 2),
            name="max_pool_layer_2"
        )

        self.flatten_layer = Flatten(name="flatten_layer")
        self.dropout = Dropout(rate=0.3)
        self.dense_layer = Dense(
            self.n_classes,
            activation="softmax",
            name="classification_head"
        )

    def call(self: Self, inputs: Any) -> Any:
        """
        Forward call
        """

        x = self.augmentation_layer(inputs)


        x = self.conv_layer_1(x)
        x = self.max_pool_layer_1(x)

        # exemplify non-sequential nature of computation possible with
        # the functional and object-oriented methods
        x_l = self.conv_layer_2l(x)
        x_r = self.conv_layer_2r(x)
        x = add(x_l, x_r)
        x = self.max_pool_layer_2(x)

        x = self.flatten_layer(x)
        x = self.dropout(x)

        return  self.dense_layer(x)


def main() -> None:
    """
    Module's main function
    """

    from keras import Input

    input_shape = (32, 32, 3)
    model = MyTinyRegularizedCNN()

    inputs = Input(shape=input_shape)
    _ = model.call(inputs)

    model.summary()


if __name__ == "__main__":
    main()
