from __future__ import annotations

import argparse
import json

from sb_piplu.activations import activation_names
from sb_piplu.models import model_names
from sb_piplu.training import ExperimentConfig, run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Train one Sb-PiPLU experiment.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("cifar10", "brain_tumor", "svhn", "mwd", "cinic10"),
    )
    parser.add_argument("--data-dir")
    parser.add_argument("--model", required=True, choices=model_names())
    parser.add_argument("--activation", default="sb_piplu", choices=activation_names())
    parser.add_argument("--k-init", type=float, default=21.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--reduce-lr-patience", type=int, default=3)
    parser.add_argument("--reduce-lr-factor", type=float, default=0.2)
    parser.add_argument("--reduce-lr-min-delta", type=float, default=0.0001)
    parser.add_argument("--clipnorm", type=float)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_experiment(ExperimentConfig(**vars(args)))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
