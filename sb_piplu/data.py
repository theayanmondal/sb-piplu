from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras


IMAGE_SIZE = (32, 32)


@dataclass
class DatasetBundle:
    train: tf.data.Dataset
    validation: tf.data.Dataset
    test: tf.data.Dataset
    input_shape: tuple[int, int, int]
    num_classes: int
    class_names: list[str]


def _warn_unexpected_count(name: str, actual: int, expected: int) -> None:
    if actual != expected:
        warnings.warn(
            f"{name} contains {actual:,} samples; the published experiment used "
            f"{expected:,}. The loader will continue with the supplied data.",
            stacklevel=2,
        )


def _normalize_images(images):
    images = np.asarray(images, dtype=np.float32)
    if images.size and images.max() > 1.0:
        images = images / 255.0
    return images


def _resize_array_images(images, channels: int, chunk_size: int = 256):
    images = np.asarray(images)
    if images.ndim == 3:
        images = images[..., np.newaxis]
    if images.ndim != 4:
        raise ValueError(
            "Expected image data with shape (N, H, W) or (N, H, W, C); "
            f"received {images.shape}."
        )

    converted = []
    for start in range(0, len(images), chunk_size):
        batch = tf.convert_to_tensor(images[start : start + chunk_size])
        batch = tf.cast(batch, tf.float32)

        current_channels = batch.shape[-1]
        if channels == 1:
            if current_channels == 3:
                batch = tf.image.rgb_to_grayscale(batch)
            elif current_channels != 1:
                raise ValueError(
                    f"Expected one or three input channels, received {current_channels}."
                )
        elif channels == 3:
            if current_channels == 1:
                batch = tf.image.grayscale_to_rgb(batch)
            elif current_channels != 3:
                raise ValueError(
                    f"Expected one or three input channels, received {current_channels}."
                )
        else:
            raise ValueError("channels must be 1 or 3.")

        if tuple(batch.shape[1:3]) != IMAGE_SIZE:
            batch = tf.image.resize(batch, IMAGE_SIZE, method="bilinear")
        converted.append(batch.numpy())

    if not converted:
        return np.empty((0, *IMAGE_SIZE, channels), dtype=np.float32)
    return np.concatenate(converted, axis=0)


def _encode_labels(train_labels, validation_labels, test_labels):
    encoder = LabelEncoder()
    train_int = encoder.fit_transform(np.asarray(train_labels).reshape(-1))
    validation_int = encoder.transform(np.asarray(validation_labels).reshape(-1))
    test_int = encoder.transform(np.asarray(test_labels).reshape(-1))
    num_classes = len(encoder.classes_)
    return (
        keras.utils.to_categorical(train_int, num_classes),
        keras.utils.to_categorical(validation_int, num_classes),
        keras.utils.to_categorical(test_int, num_classes),
        [str(value) for value in encoder.classes_],
    )


def _array_dataset(images, labels, batch_size, training, seed):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        dataset = dataset.shuffle(
            min(len(images), 10000), seed=seed, reshuffle_each_iteration=True
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _bundle_from_arrays(
    x_train,
    y_train,
    x_validation,
    y_validation,
    x_test,
    y_test,
    batch_size,
    seed,
):
    x_train = _normalize_images(x_train)
    x_validation = _normalize_images(x_validation)
    x_test = _normalize_images(x_test)
    y_train, y_validation, y_test, class_names = _encode_labels(
        y_train, y_validation, y_test
    )

    input_shape = tuple(int(value) for value in x_train.shape[1:])
    return DatasetBundle(
        train=_array_dataset(x_train, y_train, batch_size, True, seed),
        validation=_array_dataset(
            x_validation, y_validation, batch_size, False, seed
        ),
        test=_array_dataset(x_test, y_test, batch_size, False, seed),
        input_shape=input_shape,
        num_classes=len(class_names),
        class_names=class_names,
    )


def load_cifar10(batch_size=64, seed=42):
    """Load CIFAR-10 and split its 50,000 training images 80:20."""
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.20,
        random_state=seed,
        stratify=y_train_full.reshape(-1),
    )
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    bundle = _bundle_from_arrays(
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
        batch_size,
        seed,
    )
    bundle.class_names = class_names
    return bundle


def _find_first_existing(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    options = ", ".join(names)
    raise FileNotFoundError(f"Expected one of the following files in {root}: {options}")


def load_brain_tumor(data_dir, batch_size=64, seed=42):
    """Load the 3-class Brain Tumor dataset and create an 80:10:10 split."""
    root = Path(data_dir)
    image_file = _find_first_existing(root, ("images_32.npy", "images.npy"))
    label_file = _find_first_existing(root, ("label_32.npy", "labels.npy"))

    x = np.load(image_file, allow_pickle=True)
    y = np.load(label_file, allow_pickle=True).reshape(-1)
    if len(x) != len(y):
        raise ValueError("The Brain Tumor image and label arrays have different lengths.")
    _warn_unexpected_count("Brain Tumor", len(x), 3064)

    x = _resize_array_images(x, channels=1)
    unique_labels = np.unique(y)
    if len(unique_labels) != 3:
        raise ValueError("The Brain Tumor dataset must contain three classes.")

    if set(unique_labels.tolist()) == {1, 2, 3}:
        y = y.astype(int) - 1
    elif set(unique_labels.tolist()) != {0, 1, 2}:
        raise ValueError(
            "Brain Tumor labels must be encoded as 0/1/2 or 1/2/3."
        )

    x_train, x_remaining, y_train, y_remaining = train_test_split(
        x, y, test_size=0.20, random_state=seed, stratify=y
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_remaining,
        y_remaining,
        test_size=0.50,
        random_state=seed,
        stratify=y_remaining,
    )
    bundle = _bundle_from_arrays(
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
        batch_size,
        seed,
    )
    bundle.class_names = ["meningioma", "glioma", "pituitary_tumor"]
    return bundle


def load_svhn(data_dir, batch_size=64, seed=42):
    """Load SVHN arrays, split the supplied training data 80:20, and retain its test set."""
    root = Path(data_dir)
    x_train_full = np.load(root / "svhn_x_train.npy", allow_pickle=True)
    y_train_full = np.load(root / "svhn_y_train.npy", allow_pickle=True).reshape(-1)
    x_test = np.load(root / "svhn_x_test.npy", allow_pickle=True)
    y_test = np.load(root / "svhn_y_test.npy", allow_pickle=True).reshape(-1)

    x_train_full = _resize_array_images(x_train_full, channels=3)
    x_test = _resize_array_images(x_test, channels=3)
    _warn_unexpected_count("SVHN", len(x_train_full) + len(x_test), 630420)

    y_train_full = np.where(y_train_full == 10, 0, y_train_full)
    y_test = np.where(y_test == 10, 0, y_test)

    if len(np.unique(np.concatenate([y_train_full, y_test]))) != 10:
        raise ValueError("SVHN must contain ten digit classes.")

    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.20,
        random_state=seed,
        stratify=y_train_full,
    )
    bundle = _bundle_from_arrays(
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
        batch_size,
        seed,
    )
    bundle.class_names = [str(index) for index in range(10)]
    return bundle


def _directory_file_split(root, seed):
    root = Path(root)
    class_names = sorted(path.name for path in root.iterdir() if path.is_dir())
    if not class_names:
        raise ValueError(f"No class directories were found in {root}.")

    paths = []
    labels = []
    image_suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
    for label, class_name in enumerate(class_names):
        for path in sorted((root / class_name).glob("*")):
            if path.is_file() and path.suffix.lower() in image_suffixes:
                paths.append(str(path))
                labels.append(label)

    if not paths:
        raise ValueError(f"No supported image files were found in {root}.")

    train_paths, remaining_paths, train_labels, remaining_labels = train_test_split(
        paths,
        labels,
        test_size=0.20,
        random_state=seed,
        stratify=labels,
    )
    validation_paths, test_paths, validation_labels, test_labels = train_test_split(
        remaining_paths,
        remaining_labels,
        test_size=0.50,
        random_state=seed,
        stratify=remaining_labels,
    )
    return (
        train_paths,
        train_labels,
        validation_paths,
        validation_labels,
        test_paths,
        test_labels,
        class_names,
    )


def _path_dataset(paths, labels, num_classes, batch_size, training, seed):
    sample_count = len(paths)
    paths = tf.constant(paths)
    labels = tf.one_hot(labels, depth=num_classes)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            min(sample_count, 10000), seed=seed, reshuffle_each_iteration=True
        )

    def decode(path, label):
        image = tf.io.decode_image(
            tf.io.read_file(path), channels=3, expand_animations=False
        )
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        image.set_shape((*IMAGE_SIZE, 3))
        return image, label

    return dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(
        batch_size
    ).prefetch(tf.data.AUTOTUNE)


def load_mwd(data_dir, batch_size=64, seed=42):
    """Load the four-class MWD folder and create an 80:10:10 split."""
    split = _directory_file_split(data_dir, seed)
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, names = split
    if len(names) != 4:
        raise ValueError("MWD must contain exactly four classes.")
    _warn_unexpected_count("MWD", len(train_paths) + len(val_paths) + len(test_paths), 1125)

    num_classes = len(names)
    return DatasetBundle(
        train=_path_dataset(
            train_paths, train_labels, num_classes, batch_size, True, seed
        ),
        validation=_path_dataset(
            val_paths, val_labels, num_classes, batch_size, False, seed
        ),
        test=_path_dataset(
            test_paths, test_labels, num_classes, batch_size, False, seed
        ),
        input_shape=(*IMAGE_SIZE, 3),
        num_classes=num_classes,
        class_names=names,
    )


def _load_cinic10_arrays(root: Path, batch_size: int, seed: int):
    x_train = np.load(root / "Cinic_dataset_train_data.npy", allow_pickle=True)
    y_train = np.load(root / "Cinic_dataset_train_labels.npy", allow_pickle=True)
    x_validation = np.load(root / "Cinic_dataset_valid_data.npy", allow_pickle=True)
    y_validation = np.load(root / "Cinic_dataset_valid_labels.npy", allow_pickle=True)
    x_test = np.load(root / "Cinic_dataset_test_data.npy", allow_pickle=True)
    y_test = np.load(root / "Cinic_dataset_test_labels.npy", allow_pickle=True)

    x_train = _resize_array_images(x_train, channels=3)
    x_validation = _resize_array_images(x_validation, channels=3)
    x_test = _resize_array_images(x_test, channels=3)
    for name, values in (
        ("train", x_train),
        ("valid", x_validation),
        ("test", x_test),
    ):
        _warn_unexpected_count(f"CINIC-10 {name}", len(values), 90000)

    bundle = _bundle_from_arrays(
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
        batch_size,
        seed,
    )
    if bundle.num_classes != 10:
        raise ValueError("CINIC-10 must contain ten classes.")
    return bundle


def _count_images(directory: Path) -> int:
    image_suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in image_suffixes
    )


def _load_cinic10_directories(root: Path, batch_size: int, seed: int):
    train_dir = root / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing CINIC-10 directory: {train_dir}")

    class_names = sorted(
        path.name for path in train_dir.iterdir() if path.is_dir()
    )
    if len(class_names) != 10:
        raise ValueError("CINIC-10 must contain ten classes.")

    subsets = {}
    for subset_name in ("train", "valid", "test"):
        subset_dir = root / subset_name
        if not subset_dir.is_dir():
            raise FileNotFoundError(f"Missing CINIC-10 directory: {subset_dir}")

        subset_classes = sorted(
            path.name
            for path in subset_dir.iterdir()
            if path.is_dir()
        )
        if subset_classes != class_names:
            raise ValueError(
                "CINIC-10 train, valid, and test directories must use "
                "the same class names."
            )

        _warn_unexpected_count(
            f"CINIC-10 {subset_name}",
            _count_images(subset_dir),
            90000,
        )
        dataset = keras.utils.image_dataset_from_directory(
            subset_dir,
            image_size=IMAGE_SIZE,
            batch_size=batch_size,
            label_mode="categorical",
            class_names=class_names,
            shuffle=subset_name == "train",
            seed=seed,
        )
        subsets[subset_name] = dataset.map(
            lambda image, label: (tf.cast(image, tf.float32) / 255.0, label),
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)

    return DatasetBundle(
        train=subsets["train"],
        validation=subsets["valid"],
        test=subsets["test"],
        input_shape=(*IMAGE_SIZE, 3),
        num_classes=len(class_names),
        class_names=class_names,
    )


def load_cinic10(data_dir, batch_size=64, seed=42):
    """Load the fixed 90k/90k/90k CINIC-10 train, validation, and test splits."""
    root = Path(data_dir)
    array_marker = root / "Cinic_dataset_train_data.npy"
    directory_marker = root / "train"
    if array_marker.exists():
        return _load_cinic10_arrays(root, batch_size, seed)
    if directory_marker.is_dir():
        return _load_cinic10_directories(root, batch_size, seed)
    raise FileNotFoundError(
        "CINIC-10 must be supplied either as the six original NumPy arrays "
        "or as train/valid/test class directories."
    )


def load_dataset(name, data_dir=None, batch_size=64, seed=42):
    key = name.lower().replace("-", "_")
    if key == "cifar10":
        return load_cifar10(batch_size=batch_size, seed=seed)
    if data_dir is None:
        raise ValueError(f"--data-dir is required for dataset '{name}'.")

    loaders = {
        "brain_tumor": load_brain_tumor,
        "svhn": load_svhn,
        "mwd": load_mwd,
        "cinic10": load_cinic10,
    }
    try:
        loader = loaders[key]
    except KeyError as exc:
        valid = "cifar10, brain_tumor, svhn, mwd, cinic10"
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {valid}") from exc
    return loader(data_dir, batch_size=batch_size, seed=seed)
