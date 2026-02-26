# I used ChatGPT and Cursor for the development of this project.
# Helpers for running PR/DR model inference on image regions.

from typing import List

import numpy as np
import tensorflow as tf

from srcNonE2E.data.labels import ID_TO_PITCH, ID_TO_DURATION
from srcNonE2E.eval_helpers.eval_geometry import BBox, crop_resize_norm


def infer_tokens_for_image(
    img_bgr: np.ndarray,
    regions: List[BBox],
    pr_model: tf.keras.Model,
    dr_model: tf.keras.Model,
    input_h: int,
    input_w: int,
    batch_size: int,
) -> List[str]:
    # For each region in the image return predicted token 'note.{duration}-{pitch}'.
    if len(regions) == 0:
        return []

    crops = np.zeros((len(regions), input_h, input_w, 1), dtype=np.float32)
    for i, bb in enumerate(regions):
        crops[i] = crop_resize_norm(img_bgr, bb, input_h, input_w)

    pr_logits = pr_model.predict(crops, batch_size=batch_size, verbose=0)
    dr_logits = dr_model.predict(crops, batch_size=batch_size, verbose=0)

    pr_ids = np.argmax(pr_logits, axis=-1).astype(np.int32)
    dr_ids = np.argmax(dr_logits, axis=-1).astype(np.int32)

    tokens: List[str] = []
    for pid, did in zip(pr_ids, dr_ids):
        pitch = ID_TO_PITCH.get(int(pid), f"UNKP{int(pid)}")
        duration = ID_TO_DURATION.get(int(did), f"UNKD{int(did)}")
        tokens.append(f"note.{duration}-{pitch}")

    return tokens
