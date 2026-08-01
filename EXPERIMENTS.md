# Experiment definitions

The model builders follow the layer sequences used in the original
TensorFlow/Keras experiment scripts. Input channels and the final class count
are supplied by the selected dataset.

## LeNet-5

```text
Conv2D(6, 3 x 3)
activation
AveragePooling2D
BatchNormalization

Conv2D(16, 3 x 3)
activation
AveragePooling2D
BatchNormalization

Flatten
Dense(120)
activation
Dense(84)
activation
Dense(num_classes, softmax)
```

## AlexNet

```text
Conv2D(96, 11 x 11, stride 4) -> BN -> activation -> max pool
Conv2D(256, 5 x 5)            -> BN -> activation -> max pool
Conv2D(384, 3 x 3)            -> BN -> activation
Conv2D(384, 3 x 3)            -> BN -> activation
Conv2D(256, 3 x 3)            -> BN -> activation -> max pool
Flatten
Dense(4096) -> BN -> activation -> dropout
Dense(4096) -> BN -> activation -> dropout
Dense(1000) -> BN -> activation -> dropout
Dense(num_classes) -> BN -> softmax
```

The shared builder uses dropout `0.4`. The SVHN source script used `0.5`; that
value can be passed directly to `build_alexnet()` when that exact variant is
needed.

## Network in Network (NIN)

```text
Conv2D(192, 5 x 5) -> BN -> activation
Conv2D(160, 1 x 1) -> BN -> activation
Conv2D(96,  1 x 1) -> BN -> activation
MaxPooling2D(3 x 3, stride 2)
Dropout(0.2)

Conv2D(192, 5 x 5) -> BN -> activation
Conv2D(192, 1 x 1) -> BN -> activation
Conv2D(192, 1 x 1) -> BN -> activation
MaxPooling2D(3 x 3, stride 2)
Dropout(0.2)

Conv2D(192,         3 x 3) -> BN -> activation
Conv2D(192,         1 x 1) -> BN -> activation
Conv2D(num_classes, 1 x 1) -> BN -> activation
GlobalAveragePooling2D
softmax
```

The convolutional kernels use L2 regularization with coefficient `1e-6`.

## VGG11

```text
2 x Conv2D(32,  3 x 3) -> max pool
2 x Conv2D(64,  3 x 3) -> max pool
2 x Conv2D(128, 3 x 3) -> max pool
2 x Conv2D(256, 3 x 3) -> max pool
Flatten
Dense(256) -> activation
Dense(128) -> activation
Dense(num_classes, softmax)
```

An activation follows every convolution. The shared builder leaves dropout
disabled by default because the amount varied among the dataset-specific
scripts.

## ResNet18

```text
Conv2D(64, 7 x 7, stride 2) -> BN -> activation
MaxPooling2D(2 x 2, stride 2)

2 residual blocks with 64 channels
2 residual blocks with 128 channels
2 residual blocks with 256 channels
2 residual blocks with 512 channels

GlobalAveragePooling2D
Dense(num_classes, softmax)
```

The first block of each new channel stage downsamples with stride 2 and uses a
`1 x 1` convolution in the shortcut.

## Activation placement

Sb-PiPLU is a separate Keras layer rather than a function passed through the
`activation=` argument. This allows every activation layer to register its own
trainable scalar `k`.

The comparison set is:

```text
ReLU
Leaky ReLU (negative slope 0.01)
Tanh
Tanh-ReLU
ELU (alpha 1.0)
SELU
approximate GELU
Softsign
Swish (beta 1)
ELiSH
Mish
Sb-PiPLU
```

## Training

The settings explicitly stated in the paper are Adam with learning rate
`0.001`, categorical cross-entropy, at most 50 epochs, early-stopping patience
10, and learning-rate reduction by a factor of 0.2.

The original scripts commonly used batch size 64, scheduler patience 3,
early-stopping `min_delta=0.001`, and scheduler `min_delta=0.0001`. These are
the repository defaults but are not presented as universal paper settings.

Gradient clipping can be enabled with `--clipnorm`. The paper discusses its
use but does not give a clipping threshold, so no threshold is enabled by
default.

## Result files

The CSV files under `results/` contain values printed in the paper. Files
under `runs/` are produced by new executions and must not be cited as the
published results unless the experiment has been independently verified.
