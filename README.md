# Sb-PiPLU

TensorFlow/Keras implementation of the Softsign-based Piecewise Parametric
Linear Unit (Sb-PiPLU) and the image-classification experiments described in:

> Ayan Mondal, Vimal K. Shrivastava, Ayan Chatterjee, and Raghavendra
> Ramachandra, “Sb-PiPLU: A Novel Parametric Activation Function for Deep
> Learning,” *IEEE Access*, vol. 13, 2025.<br>
> DOI: [10.1109/ACCESS.2025.3561464](https://doi.org/10.1109/ACCESS.2025.3561464)

The study was originally run with separate scripts for each model and dataset.
This repository collects the common activation, model, data-loading, training,
and evaluation code in one place.

## Activation function

Sb-PiPLU is defined as

$$
\mathrm{Sb-PiPLU}(x,k)=
\begin{cases}
2\~\mathrm{softsign}(x)+\frac{1}{2}\left(\mathrm{softsign}(x)\right)^2, & x\leq 0 \\
x, & 0\lt x\leq k \\
\frac{x}{k}, & x\gt k
\end{cases}
$$

where

$$
\mathrm{softsign}(x)=\frac{x}{1+|x|}.
$$

`SbPiPLU` is implemented as a Keras layer because `k` is trainable. Every layer
instance has its own scalar `k`, initialized to 21 by default.

```python
import tensorflow as tf

from sb_piplu import SbPiPLU

inputs = tf.keras.Input(shape=(32, 32, 3))
x = tf.keras.layers.Conv2D(64, 3, padding="same")(inputs)
x = SbPiPLU(k_init=21.0)(x)
```

## Contents

```text
sb_piplu/
    activations.py      Sb-PiPLU and the comparison activations
    data.py             Dataset loading, resizing, normalization, and splits
    metrics.py          Accuracy, precision, specificity, sensitivity, and F1
    models.py           LeNet-5, AlexNet, NIN, VGG11, and ResNet18
    training.py         Training, evaluation, timing, and saved outputs
train.py                Run one model/dataset/activation experiment
compare_activations.py  Compare several activations with one model
sweep_k.py              Repeat an experiment with different k initializations
model_summary.py        Print a model summary
results/                Values transcribed from the published tables
DATASETS.md             Dataset layouts accepted by the loaders
EXPERIMENTS.md          Model definitions and experiment settings
tests/                  Unit tests
```

The repository does not include datasets, trained weights, or the raw Colab
files.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Datasets

| Dataset | Images | Input used by the models | Classes | Split |
|---|---:|---|---:|---|
| CIFAR-10 | 60,000 | `32 x 32 x 3` | 10 | 80:20 train/validation; standard test set retained |
| Brain Tumor | 3,064 | `32 x 32 x 1` | 3 | 80:10:10 |
| SVHN | 630,420 | `32 x 32 x 3` | 10 | 80:20 train/validation; supplied test set retained |
| MWD | 1,125 | `32 x 32 x 3` | 4 | 80:10:10 |
| CINIC-10 | 270,000 | `32 x 32 x 3` | 10 | fixed 90,000/90,000/90,000 subsets |

All images are normalized to `[0, 1]`. No data augmentation is applied.
`DATASETS.md` gives the expected files and directory structure.

## Run an experiment

CIFAR-10 is downloaded through Keras:

```bash
python train.py \
  --dataset cifar10 \
  --model nin \
  --activation sb_piplu \
  --k-init 21 \
  --output runs/cifar10_nin_sb_piplu
```

For the other datasets, pass the local dataset directory:

```bash
python train.py \
  --dataset brain_tumor \
  --data-dir data/brain_tumor \
  --model alexnet \
  --activation sb_piplu \
  --output runs/brain_tumor_alexnet_sb_piplu
```

### Settings stated in the paper

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | `0.001` |
| Loss | Categorical cross-entropy |
| Maximum epochs | `50` |
| Early-stopping patience | `10` |
| Learning-rate reduction factor | `0.2` |
| Image size | `32 x 32` |
| Data augmentation | None |
| Initial value of `k` | `21` |



## Activation comparison

```bash
python compare_activations.py \
  --dataset cifar10 \
  --model vgg11 \
  --output runs/cifar10_vgg11_activations
```

Available activation names:

```text
relu, leaky_relu, tanh, tanh_relu, elu, selu, gelu,
softsign, swish, elish, mish, sb_piplu
```

The Tanh-ReLU and ELiSH helpers are algebraically equivalent to the functions
used in the original experiment scripts, but avoid unnecessary temporary
tensors.

## k-initialization experiment

```bash
python sweep_k.py \
  --dataset cifar10 \
  --model nin \
  --k-start 10 \
  --k-stop 25 \
  --output runs/cifar10_nin_k_sweep
```

Every run starts with the requested value of `k`. Because `k` is trainable, the
summary also records the value learned by each Sb-PiPLU layer.

## Files produced by a run

Each run writes:

- `best_model.keras`;
- `config.json` and `classes.json`;
- `model_summary.txt`;
- `history.csv`, including epoch time and learning rate;
- `metrics.json`;
- `learned_k.json`;
- `confusion_matrix.csv` and `confusion_matrix.png`;
- `accuracy.png` and `loss.png`;
- `predictions.csv`.

## Published results

`results/published_sb_piplu_results.csv` contains the Sb-PiPLU rows from
Tables 2–6, with the same two-decimal values as the article.

| Dataset | LeNet-5 | VGG11 | AlexNet | NIN | ResNet18 |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | 62.21 | 83.52 | 77.66 | 87.47 | 79.81 |
| Brain Tumor | 88.70 | 89.57 | 91.74 | 90.43 | 89.13 |
| MWD | 84.38 | 88.39 | 90.18 | 88.84 | 85.53 |
| SVHN | 87.25 | 92.68 | 91.14 | 93.02 | 92.99 |
| CINIC-10 | 54.34 | 68.04 | 57.37 | 77.35 | 58.64 |

The numerical values behind Figures 4 and 5 were obtained manually from the
training logs and are not tabulated in the paper. They are not estimated from
the plots here. New runs record their own epoch times and stopping epochs.

## Tests

```bash
pytest
```

## Citation

```bibtex
@article{mondal2025sbpiplu,
  author  = {Mondal, Ayan and Shrivastava, Vimal K. and
             Chatterjee, Ayan and Ramachandra, Raghavendra},
  title   = {Sb-PiPLU: A Novel Parametric Activation Function for Deep Learning},
  journal = {IEEE Access},
  volume  = {13},
  year    = {2025},
  doi     = {10.1109/ACCESS.2025.3561464}
}
```

## License

The source code is released under the MIT License. The datasets and the
published article remain subject to their own licenses.
