import cv2
import numpy as np

# PNG is lossless at every level; level 1 encodes several times faster than
# OpenCV's default (3) for slightly larger payloads sent over localhost.
PNG_ENCODE_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]


def resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    """Resize an image isotropically while keeping both dimensions nonzero."""
    if scale <= 0:
        raise ValueError("Image scale must be greater than zero.")
    height, width = image.shape[:2]
    target_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def crop_image(image: np.ndarray, bbox: list[float]) -> np.ndarray:
    """
    Crop an image using the given bounding box.
    bbox: [x1, y1, x2, y2]
    """
    # D-FINE emits floating-point pixel coordinates. Quantize them to the
    # nearest pixel so recognition crops match the reference pipeline.
    x1, y1, x2, y2 = (round(value) for value in bbox)
    # Ensure coordinates are within image boundaries
    h, w = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    return image[y1:y2, x1:x2].copy()
