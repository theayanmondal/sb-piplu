import numpy as np

from sb_piplu.metrics import classification_metrics


def test_perfect_metrics():
    result = classification_metrics([0, 1, 2], [0, 1, 2], num_classes=3)
    assert result["accuracy"] == 1.0
    assert result["precision_macro"] == 1.0
    assert result["specificity_macro"] == 1.0
    assert result["sensitivity_macro"] == 1.0
    assert result["f1_macro"] == 1.0
    np.testing.assert_array_equal(result["confusion_matrix"], np.eye(3, dtype=int))
