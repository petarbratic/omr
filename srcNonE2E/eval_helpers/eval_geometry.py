# Helpers for sorting regions and preparing image crops (crop, resize, normalize).

from typing import List, Tuple

import numpy as np
import cv2

BBox = Tuple[int, int, int, int]


def sort_regions_by_x(bboxes: List[BBox]) -> List[BBox]:
    # Sort bboxes by center x coordinate (left to right).
    def key(bb: BBox):
        x1, y1, x2, y2 = bb
        return (x1 + x2) / 2.0
    return sorted(bboxes, key=key)


def crop_resize_norm(img_bgr: np.ndarray, bb: BBox, out_h: int, out_w: int) -> np.ndarray:
    # Crop region from image, convert to grayscale, resize to (out_h, out_w), normalize to [0,1].
    x1, y1, x2, y2 = bb
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = max(0, int(x2))
    y2 = max(0, int(y2))

    # Ensure bbox is valid and has non-zero area. If not, adjust to a 1x1 box.
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1

    crop = img_bgr[y1:y2, x1:x2]
    # Handle empty crop case (can happen if bb is invalid or very small). Use a 1x1 crop from the top-left pixel.
    if crop.size == 0:
        crop = img_bgr[0:1, 0:1]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    x = resized.astype(np.float32) / 255.0
    # Model expects shape (H, W, 1), so add channel dimension.
    x = np.expand_dims(x, axis=-1)
    return x
