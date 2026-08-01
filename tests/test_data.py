import numpy as np
import pytest

from sb_piplu.data import load_brain_tumor


def test_brain_tumor_loader_maps_labels_and_resizes(tmp_path):
    images = np.arange(30 * 8 * 8, dtype=np.uint8).reshape(30, 8, 8)
    labels = np.repeat([1, 2, 3], 10)

    np.save(tmp_path / "images_32.npy", images)
    np.save(tmp_path / "label_32.npy", labels)

    with pytest.warns(UserWarning, match="published experiment used 3,064"):
        dataset = load_brain_tumor(tmp_path, batch_size=4, seed=7)

    assert dataset.input_shape == (32, 32, 1)
    assert dataset.num_classes == 3
    assert dataset.class_names == [
        "meningioma",
        "glioma",
        "pituitary_tumor",
    ]

    image_batch, label_batch = next(iter(dataset.train))
    assert image_batch.shape[1:] == (32, 32, 1)
    assert label_batch.shape[1] == 3
