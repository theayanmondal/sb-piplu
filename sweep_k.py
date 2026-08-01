from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sb_piplu.models import model_names
from sb_piplu.training import ExperimentConfig, run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep Sb-PiPLU k initialization values.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("cifar10", "brain_tumor", "svhn", "mwd", "cinic10"),
    )
    parser.add_argument("--data-dir")
    parser.add_argument("--model", required=True, choices=model_names())
    parser.add_argument("--k-start", type=int, default=10)
    parser.add_argument("--k-stop", type=int, default=25)
    parser.add_argument("--output", required=True)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.k_stop < args.k_start:
        raise ValueError("k-stop must be greater than or equal to k-start.")

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    rows = []

    for k_init in range(args.k_start, args.k_stop + 1):
        result = run_experiment(
            ExperimentConfig(
                dataset=args.dataset,
                data_dir=args.data_dir,
                model=args.model,
                activation="sb_piplu",
                k_init=float(k_init),
                output=str(root / f"k_{k_init}"),
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                epochs=args.epochs,
                seed=args.seed,
            )
        )
        rows.append(
            {
                "k_init": k_init,
                "accuracy": result["metrics"]["accuracy"],
                "epochs_completed": result["epochs_completed"],
                "mean_epoch_seconds": result["mean_epoch_seconds"],
                "learned_k_values": ";".join(
                    f"{name}={value:.8g}" for name, value in result["learned_k"].items()
                ),
            }
        )

    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
