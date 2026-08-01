from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sb_piplu.activations import activation_names
from sb_piplu.models import model_names
from sb_piplu.training import ExperimentConfig, run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Compare activation functions.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("cifar10", "brain_tumor", "svhn", "mwd", "cinic10"),
    )
    parser.add_argument("--data-dir")
    parser.add_argument("--model", required=True, choices=model_names())
    parser.add_argument(
        "--activations",
        nargs="+",
        default=list(activation_names()),
        choices=activation_names(),
    )
    parser.add_argument("--k-init", type=float, default=21.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    rows = []

    for activation in args.activations:
        result = run_experiment(
            ExperimentConfig(
                dataset=args.dataset,
                data_dir=args.data_dir,
                model=args.model,
                activation=activation,
                k_init=args.k_init,
                output=str(root / activation),
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                epochs=args.epochs,
                seed=args.seed,
            )
        )
        metrics = result["metrics"]
        rows.append(
            {
                "activation": activation,
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "specificity_macro": metrics["specificity_macro"],
                "sensitivity_macro": metrics["sensitivity_macro"],
                "f1_macro": metrics["f1_macro"],
                "epochs_completed": result["epochs_completed"],
                "mean_epoch_seconds": result["mean_epoch_seconds"],
            }
        )

    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
