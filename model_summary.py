from __future__ import annotations

import argparse

from sb_piplu.activations import activation_names
from sb_piplu.models import build_model, model_names


def parse_args():
    parser = argparse.ArgumentParser(description="Print a model summary.")
    parser.add_argument("--model", required=True, choices=model_names())
    parser.add_argument("--activation", default="sb_piplu", choices=activation_names())
    parser.add_argument("--k-init", type=float, default=21.0)
    parser.add_argument("--channels", type=int, choices=(1, 3), default=3)
    parser.add_argument("--classes", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    model = build_model(
        args.model,
        input_shape=(32, 32, args.channels),
        num_classes=args.classes,
        activation=args.activation,
        k_init=args.k_init,
    )
    model.summary()


if __name__ == "__main__":
    main()
