import numpy as np
import torch

from dfine_la.preprocess import preprocess_images, resize_and_pad


def test_resize_and_pad_keeps_canvas_shape_and_fill_value():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    canvas = resize_and_pad(image, target_size=960)
    assert canvas.shape == (960, 960, 3)
    assert np.all(canvas[:200] == 114)


def test_batch1_preprocessing_matches_runtime_signature():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    batch, sizes = preprocess_images([image])
    assert batch.shape == (1, 3, 960, 960)
    assert batch.dtype == torch.float32
    assert sizes.tolist() == [[200, 100]]
