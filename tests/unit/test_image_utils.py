import numpy as np

from oxr.utils.image import crop_image


def test_crop_image_rounds_float_coordinates_to_nearest_pixel() -> None:
    image = np.arange(6 * 7, dtype=np.uint8).reshape(6, 7)

    crop = crop_image(image, [1.6, 1.6, 5.6, 4.6])

    np.testing.assert_array_equal(crop, image[2:5, 2:6])


def test_crop_image_clamps_rounded_coordinates_and_returns_copy() -> None:
    image = np.arange(4 * 5, dtype=np.uint8).reshape(4, 5)

    crop = crop_image(image, [-0.4, -0.4, 6.2, 5.2])

    np.testing.assert_array_equal(crop, image)
    assert not np.shares_memory(crop, image)
