from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .data import load_dataset
from .metrics import classification_metrics
from .models import build_model


@dataclass
class ExperimentConfig:
    dataset: str
    model: str
    activation: str = "sb_piplu"
    data_dir: str | None = None
    output: str = "runs/experiment"
    k_init: float = 21.0
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 50
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.2
    reduce_lr_min_delta: float = 0.0001
    clipnorm: float | None = None
    seed: int = 42


class EpochLog(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_seconds = []
        self.learning_rates = []

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_seconds.append(time.perf_counter() - self._start)
        learning_rate = keras.backend.get_value(self.model.optimizer.learning_rate)
        self.learning_rates.append(float(learning_rate))


def _save_history(history, epoch_log: EpochLog, path: Path):
    rows = []
    epochs = len(history.history.get("loss", []))

    for index in range(epochs):
        row = {
            "epoch": index + 1,
            "seconds": epoch_log.epoch_seconds[index],
            "learning_rate": epoch_log.learning_rates[index],
        }
        for key, values in history.history.items():
            row[key] = float(values[index])
        rows.append(row)

    fieldnames = list(rows[0]) if rows else ["epoch", "seconds", "learning_rate"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_history(history, output: Path):
    for metric, validation_metric, filename in (
        ("loss", "val_loss", "loss.png"),
        ("accuracy", "val_accuracy", "accuracy.png"),
    ):
        if metric not in history.history:
            continue

        plt.figure(figsize=(7, 5))
        plt.plot(history.history[metric], label=f"training {metric}")
        if validation_metric in history.history:
            plt.plot(
                history.history[validation_metric],
                label=f"validation {metric}",
            )
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / filename, dpi=160)
        plt.close()


def _plot_confusion_matrix(matrix, class_names, path: Path):
    matrix = np.asarray(matrix)

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix)
    plt.colorbar()

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    threshold = matrix.max(initial=0) / 2.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            plt.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _write_model_summary(model, path: Path):
    lines = []
    model.summary(print_fn=lines.append)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def learned_k_values(model):
    values = {}
    for weight in model.trainable_weights:
        name = getattr(weight, "path", weight.name)
        if weight.name == "k" or name.endswith("/k"):
            values[name] = float(weight.numpy())
    return values


def run_experiment(config: ExperimentConfig):
    tf.keras.utils.set_random_seed(config.seed)

    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        config.dataset,
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    model = build_model(
        config.model,
        input_shape=dataset.input_shape,
        num_classes=dataset.num_classes,
        activation=config.activation,
        k_init=config.k_init,
    )
    _write_model_summary(model, output / "model_summary.txt")

    optimizer_options = {"learning_rate": config.learning_rate}
    if config.clipnorm is not None:
        optimizer_options["clipnorm"] = config.clipnorm

    model.compile(
        optimizer=keras.optimizers.Adam(**optimizer_options),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model_path = output / "best_model.keras"
    epoch_log = EpochLog()
    callbacks = [
        epoch_log,
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            restore_best_weights=True,
            mode="min",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_delta=config.reduce_lr_min_delta,
            mode="min",
        ),
    ]

    history = model.fit(
        dataset.train,
        validation_data=dataset.validation,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(dataset.test, verbose=0)
    probabilities = model.predict(dataset.test, verbose=0)
    predictions = probabilities.argmax(axis=1)
    true_labels = np.concatenate(
        [labels.numpy().argmax(axis=1) for _, labels in dataset.test]
    )

    metrics = classification_metrics(
        true_labels,
        predictions,
        num_classes=dataset.num_classes,
    )
    metrics["test_loss"] = float(test_loss)
    metrics["keras_test_accuracy"] = float(test_accuracy)

    k_values = learned_k_values(model)

    (output / "config.json").write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )
    (output / "classes.json").write_text(
        json.dumps(dataset.class_names, indent=2),
        encoding="utf-8",
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    (output / "learned_k.json").write_text(
        json.dumps(k_values, indent=2),
        encoding="utf-8",
    )

    _save_history(history, epoch_log, output / "history.csv")
    _plot_history(history, output)

    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    np.savetxt(
        output / "confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d",
    )
    _plot_confusion_matrix(
        matrix,
        dataset.class_names,
        output / "confusion_matrix.png",
    )

    with (output / "predictions.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fieldnames = ["true_label", "predicted_label"] + [
            f"probability_{name}" for name in dataset.class_names
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for true_label, predicted_label, probability in zip(
            true_labels,
            predictions,
            probabilities,
        ):
            row = {
                "true_label": dataset.class_names[int(true_label)],
                "predicted_label": dataset.class_names[int(predicted_label)],
            }
            row.update(
                {
                    f"probability_{name}": float(value)
                    for name, value in zip(dataset.class_names, probability)
                }
            )
            writer.writerow(row)

    mean_epoch_seconds = (
        float(np.mean(epoch_log.epoch_seconds))
        if epoch_log.epoch_seconds
        else 0.0
    )

    return {
        "metrics": metrics,
        "epochs_completed": len(history.history.get("loss", [])),
        "mean_epoch_seconds": mean_epoch_seconds,
        "learned_k": k_values,
    }
