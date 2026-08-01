from __future__ import annotations

from collections.abc import Callable

import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="SbPiPLU")
class SbPiPLU(keras.layers.Layer):
    """Sb-PiPLU with one trainable scalar k per layer."""

    def __init__(self, k_init: float = 21.0, **kwargs):
        super().__init__(**kwargs)
        if k_init <= 0:
            raise ValueError("k_init must be greater than zero.")
        self.k_init = float(k_init)

    def build(self, input_shape):
        self.k = self.add_weight(
            name="k",
            shape=(),
            initializer=keras.initializers.Constant(self.k_init),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        dtype = x.dtype

        zero = tf.cast(0.0, dtype)
        one = tf.cast(1.0, dtype)
        two = tf.cast(2.0, dtype)
        half = tf.cast(0.5, dtype)
        k = tf.cast(self.k, dtype)

        softsign_x = x / (one + tf.abs(x))
        negative = two * softsign_x + half * tf.square(softsign_x)
        positive = tf.where(x <= k, x, x / k)

        return tf.where(x <= zero, negative, positive)

    def get_config(self):
        config = super().get_config()
        config.update({"k_init": self.k_init})
        return config


@keras.utils.register_keras_serializable(package="SbPiPLU")
def tanh_relu(x):
    x = tf.convert_to_tensor(x)
    return tf.maximum(x, tf.math.tanh(x))


@keras.utils.register_keras_serializable(package="SbPiPLU")
def elish(x):
    x = tf.convert_to_tensor(x)
    sigmoid_x = tf.math.sigmoid(x)
    positive = x * sigmoid_x
    negative = tf.math.expm1(x) * sigmoid_x
    return tf.where(x >= tf.cast(0.0, x.dtype), positive, negative)


@keras.utils.register_keras_serializable(package="SbPiPLU")
def mish(x):
    x = tf.convert_to_tensor(x)
    return x * tf.math.tanh(tf.math.softplus(x))


@keras.utils.register_keras_serializable(package="SbPiPLU")
def gelu_approximate(x):
    return tf.nn.gelu(x, approximate=True)


def activation_names() -> tuple[str, ...]:
    return (
        "relu",
        "leaky_relu",
        "tanh",
        "tanh_relu",
        "elu",
        "selu",
        "gelu",
        "softsign",
        "swish",
        "elish",
        "mish",
        "sb_piplu",
    )


def make_activation_layer(
    name: str,
    k_init: float = 21.0,
    **kwargs,
) -> keras.layers.Layer:
    key = name.lower().replace("-", "_")

    if key == "sb_piplu":
        return SbPiPLU(k_init=k_init, **kwargs)
    if key == "leaky_relu":
        return keras.layers.LeakyReLU(negative_slope=0.01, **kwargs)

    functions: dict[str, str | Callable] = {
        "relu": "relu",
        "tanh": "tanh",
        "tanh_relu": tanh_relu,
        "elu": keras.activations.elu,
        "selu": "selu",
        "gelu": gelu_approximate,
        "softsign": "softsign",
        "swish": keras.activations.silu,
        "elish": elish,
        "mish": mish,
    }

    try:
        activation = functions[key]
    except KeyError as exc:
        choices = ", ".join(activation_names())
        raise ValueError(f"Unknown activation '{name}'. Choose from: {choices}") from exc

    return keras.layers.Activation(activation, **kwargs)


def kernel_initializer_for(name: str) -> str:
    key = name.lower().replace("-", "_")
    if key in {"tanh", "tanh_relu", "softsign"}:
        return "glorot_uniform"
    return "he_normal"
