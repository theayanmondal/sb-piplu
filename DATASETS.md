# Datasets

The paper uses five image-classification datasets. Dataset files are not
included in this repository.

## Protocol reported in the paper

| Dataset | Reported size | Original images | Model input | Classes | Split |
|---|---:|---|---|---:|---|
| CIFAR-10 | 60,000 | RGB, `32 x 32` | `32 x 32 x 3` | 10 | 80:20 training/validation |
| Brain Tumor | 3,064 | grayscale, `512 x 512` | `32 x 32 x 1` | 3 | 80:10:10 |
| SVHN | 630,420 | RGB, `32 x 32` | `32 x 32 x 3` | 10 | 80:20 training/validation |
| MWD | 1,125 | RGB, varying size | `32 x 32 x 3` | 4 | 80:10:10 |
| CINIC-10 | 270,000 | RGB, `32 x 32` | `32 x 32 x 3` | 10 | 90,000 images in each supplied subset |

Images are normalized to `[0, 1]`. No augmentation is applied.

The paper specifies dataset sizes and splits, not local filenames. The names
below come from the original experiment scripts or are accepted alternatives.

## CIFAR-10

No local files are required. `tf.keras.datasets.cifar10.load_data()` supplies
50,000 training images and 10,000 test images. The loader divides the 50,000
training images into 40,000 training and 10,000 validation images and keeps the
standard test set unchanged.

## Brain Tumor

The original scripts load:

```text
data/brain_tumor/
    images_32.npy
    label_32.npy
```

The loader also accepts `images.npy` and `labels.npy`.

Images may be stored as `(N, H, W)` or `(N, H, W, C)`. They are converted to
one channel and resized to `32 x 32` when necessary. Labels may be `1, 2, 3`
or `0, 1, 2`; the class order used in the original scripts is:

```text
meningioma
glioma
pituitary_tumor
```

The loader creates a stratified 80:10:10 split.

## SVHN

The original scripts use four NumPy arrays:

```text
data/svhn/
    svhn_x_train.npy
    svhn_y_train.npy
    svhn_x_test.npy
    svhn_y_test.npy
```

The supplied training arrays are split 80:20 for training and validation. The
supplied test arrays remain the test set. A digit label of `10`, used by some
SVHN files to represent zero, is converted to `0`.

## Multi-class Weather Dataset (MWD)

Place images in one directory per class:

```text
data/mwd/
    <class_1>/
        image_001.jpg
        ...
    <class_2>/
        ...
    <class_3>/
        ...
    <class_4>/
        ...
```

Exactly four non-empty class directories are required. Images are decoded as
RGB, resized to `32 x 32`, and divided with a stratified 80:10:10 split.

## CINIC-10

The standard directory form is accepted:

```text
data/cinic10/
    train/
        <class_name>/
            *.png
    valid/
        <class_name>/
            *.png
    test/
        <class_name>/
            *.png
```

The same ten class directories must be present in all three subsets.

The loader also accepts the NumPy files used in the original scripts:

```text
data/cinic10/
    Cinic_dataset_train_data.npy
    Cinic_dataset_train_labels.npy
    Cinic_dataset_valid_data.npy
    Cinic_dataset_valid_labels.npy
    Cinic_dataset_test_data.npy
    Cinic_dataset_test_labels.npy
```

The existing train, validation, and test partitions are preserved. Each subset
is expected to contain 90,000 images.
