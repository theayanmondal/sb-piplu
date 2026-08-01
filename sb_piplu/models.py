from __future__ import annotations

from collections.abc import Callable

from tensorflow import keras

from .activations import kernel_initializer_for, make_activation_layer


def model_names() -> tuple[str, ...]:
    return ("lenet5", "alexnet", "nin", "vgg11", "resnet18")


def _activation_factory(name: str, k_init: float) -> Callable[[], keras.layers.Layer]:
    return lambda: make_activation_layer(name, k_init=k_init)


def build_lenet5(input_shape, num_classes, activation="sb_piplu", k_init=21.0):
    act = _activation_factory(activation, k_init)
    init = kernel_initializer_for(activation)

    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(6, 3, kernel_initializer=init)(inputs)
    x = act()(x)
    x = keras.layers.AveragePooling2D()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(16, 3, kernel_initializer=init)(x)
    x = act()(x)
    x = keras.layers.AveragePooling2D()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(120, kernel_initializer=init)(x)
    x = act()(x)
    x = keras.layers.Dense(84, kernel_initializer=init)(x)
    x = act()(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="lenet5")


def build_alexnet(
    input_shape,
    num_classes,
    activation="sb_piplu",
    k_init=21.0,
    dropout_rate=0.4,
):
    act = _activation_factory(activation, k_init)
    init = kernel_initializer_for(activation)

    inputs = keras.Input(shape=input_shape)
    x = inputs
    for filters, kernel, stride, pool in (
        (96, 11, 4, True),
        (256, 5, 1, True),
        (384, 3, 1, False),
        (384, 3, 1, False),
        (256, 3, 1, True),
    ):
        x = keras.layers.Conv2D(
            filters,
            kernel,
            strides=stride,
            padding="same",
            kernel_initializer=init,
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = act()(x)
        if pool:
            x = keras.layers.MaxPooling2D(2, strides=2, padding="same")(x)

    x = keras.layers.Flatten()(x)
    for units in (4096, 4096, 1000):
        x = keras.layers.Dense(units, kernel_initializer=init)(x)
        x = keras.layers.BatchNormalization()(x)
        x = act()(x)
        x = keras.layers.Dropout(dropout_rate)(x)

    x = keras.layers.Dense(num_classes)(x)
    x = keras.layers.BatchNormalization()(x)
    outputs = keras.layers.Activation("softmax")(x)
    return keras.Model(inputs, outputs, name="alexnet")


def build_nin(
    input_shape,
    num_classes,
    activation="sb_piplu",
    k_init=21.0,
    weight_decay=1e-6,
):
    act = _activation_factory(activation, k_init)
    init = kernel_initializer_for(activation)
    regularizer = keras.regularizers.l2(weight_decay)

    inputs = keras.Input(shape=input_shape)
    x = inputs

    for filters, kernel in ((192, 5), (160, 1), (96, 1)):
        x = keras.layers.Conv2D(
            filters,
            kernel,
            padding="same",
            kernel_regularizer=regularizer,
            kernel_initializer=init,
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = act()(x)
    x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)
    x = keras.layers.Dropout(0.2)(x)

    for filters, kernel in ((192, 5), (192, 1), (192, 1)):
        x = keras.layers.Conv2D(
            filters,
            kernel,
            padding="same",
            kernel_regularizer=regularizer,
            kernel_initializer=init,
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = act()(x)
    x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)
    x = keras.layers.Dropout(0.2)(x)

    for filters, kernel in ((192, 3), (192, 1), (num_classes, 1)):
        x = keras.layers.Conv2D(
            filters,
            kernel,
            padding="same",
            kernel_regularizer=regularizer,
            kernel_initializer=init,
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = act()(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Activation("softmax")(x)
    return keras.Model(inputs, outputs, name="nin")


def build_vgg11(
    input_shape,
    num_classes,
    activation="sb_piplu",
    k_init=21.0,
    dropout_rate=0.0,
):
    act = _activation_factory(activation, k_init)
    init = kernel_initializer_for(activation)

    inputs = keras.Input(shape=input_shape)
    x = inputs
    for filters in (32, 64, 128, 256):
        for _ in range(2):
            x = keras.layers.Conv2D(
                filters,
                3,
                padding="same",
                kernel_initializer=init,
            )(x)
            x = act()(x)
        x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, kernel_initializer=init)(x)
    x = act()(x)
    if dropout_rate:
        x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(128, kernel_initializer=init)(x)
    x = act()(x)
    if dropout_rate:
        x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="vgg11")


@keras.utils.register_keras_serializable(package="SbPiPLU")
class ResNetBasicBlock(keras.layers.Layer):
    def __init__(
        self,
        channels,
        activation="sb_piplu",
        k_init=21.0,
        downsample=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.activation_name = activation
        self.k_init = float(k_init)
        self.downsample = bool(downsample)
        stride = 2 if downsample else 1
        init = kernel_initializer_for(activation)

        self.conv1 = keras.layers.Conv2D(
            channels, 3, strides=stride, padding="same", kernel_initializer=init
        )
        self.bn1 = keras.layers.BatchNormalization()
        self.act1 = make_activation_layer(activation, k_init=k_init)
        self.conv2 = keras.layers.Conv2D(
            channels, 3, padding="same", kernel_initializer=init
        )
        self.bn2 = keras.layers.BatchNormalization()
        self.act2 = make_activation_layer(activation, k_init=k_init)

        if downsample:
            self.shortcut_conv = keras.layers.Conv2D(
                channels, 1, strides=2, padding="same", kernel_initializer=init
            )
            self.shortcut_bn = keras.layers.BatchNormalization()
        else:
            self.shortcut_conv = None
            self.shortcut_bn = None

    def call(self, inputs, training=None):
        residual = inputs
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        if self.shortcut_conv is not None:
            residual = self.shortcut_conv(residual)
            residual = self.shortcut_bn(residual, training=training)

        return self.act2(x + residual)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "activation": self.activation_name,
                "k_init": self.k_init,
                "downsample": self.downsample,
            }
        )
        return config


def build_resnet18(input_shape, num_classes, activation="sb_piplu", k_init=21.0):
    init = kernel_initializer_for(activation)
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(
        64, 7, strides=2, padding="same", kernel_initializer=init
    )(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = make_activation_layer(activation, k_init=k_init)(x)
    x = keras.layers.MaxPooling2D(2, strides=2, padding="same")(x)

    for channels, downsample in (
        (64, False),
        (64, False),
        (128, True),
        (128, False),
        (256, True),
        (256, False),
        (512, True),
        (512, False),
    ):
        x = ResNetBasicBlock(
            channels,
            activation=activation,
            k_init=k_init,
            downsample=downsample,
        )(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="resnet18")


def build_model(
    name,
    input_shape,
    num_classes,
    activation="sb_piplu",
    k_init=21.0,
    **kwargs,
):
    key = name.lower().replace("-", "")
    builders = {
        "lenet5": build_lenet5,
        "alexnet": build_alexnet,
        "nin": build_nin,
        "vgg11": build_vgg11,
        "resnet18": build_resnet18,
    }
    try:
        builder = builders[key]
    except KeyError as exc:
        valid = ", ".join(model_names())
        raise ValueError(f"Unknown model '{name}'. Choose from: {valid}") from exc
    return builder(
        input_shape=input_shape,
        num_classes=num_classes,
        activation=activation,
        k_init=k_init,
        **kwargs,
    )
