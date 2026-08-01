import numpy as np
import tensorflow as tf

from sb_piplu.activations import SbPiPLU, elish, tanh_relu


def test_sb_piplu_matches_equation():
    layer = SbPiPLU(k_init=21.0)
    x = tf.constant([-2.0, 0.0, 10.0, 21.0, 42.0], dtype=tf.float32)
    actual = layer(x).numpy()

    softsign_negative = -2.0 / 3.0
    expected = np.array(
        [
            2.0 * softsign_negative + 0.5 * softsign_negative**2,
            0.0,
            10.0,
            21.0,
            2.0,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_k_is_trainable():
    layer = SbPiPLU(k_init=21.0)
    x = tf.constant([42.0], dtype=tf.float32)

    with tf.GradientTape() as tape:
        output = tf.reduce_sum(layer(x))
    gradient = tape.gradient(output, layer.k)

    assert layer.k.trainable
    np.testing.assert_allclose(
        gradient.numpy(),
        -42.0 / (21.0**2),
        rtol=1e-6,
    )


def test_k_gradient_is_zero_outside_upper_branch():
    layer = SbPiPLU(k_init=21.0)
    x = tf.constant([-2.0, 10.0], dtype=tf.float32)

    with tf.GradientTape() as tape:
        output = tf.reduce_sum(layer(x))
    gradient = tape.gradient(output, layer.k)

    np.testing.assert_allclose(gradient.numpy(), 0.0, atol=1e-7)


def test_tanh_relu_matches_original_expression():
    x = tf.constant([-2.0, -0.5, 0.0, 1.0, 3.0])
    expected = tf.maximum(x, tf.math.tanh(x))
    np.testing.assert_allclose(tanh_relu(x).numpy(), expected.numpy())


def test_elish_matches_original_expression():
    x = tf.constant([-2.0, -0.5, 0.0, 1.0, 3.0], dtype=tf.float32)
    one = tf.ones_like(x)
    zero = tf.zeros_like(x)

    original = tf.maximum(zero, x / (one + tf.exp(-x)))
    original += tf.minimum(zero, (tf.exp(x) - one) / (one + tf.exp(-x)))

    np.testing.assert_allclose(
        elish(x).numpy(),
        original.numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
