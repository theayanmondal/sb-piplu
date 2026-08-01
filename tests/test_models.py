import pytest

from sb_piplu.models import build_model, model_names


@pytest.mark.parametrize("name", model_names())
def test_models_build_with_trainable_k(name):
    model = build_model(
        name,
        input_shape=(32, 32, 3),
        num_classes=10,
        activation="sb_piplu",
        k_init=21.0,
    )

    assert model.output_shape == (None, 10)

    k_weights = [
        weight
        for weight in model.trainable_weights
        if weight.name == "k"
        or getattr(weight, "path", weight.name).endswith("/k")
    ]
    assert k_weights
