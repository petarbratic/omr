# python -m srcNonE2E.infer_non_e2e

import os
from typing import List

import numpy as np
import cv2
import tensorflow as tf

from srcNonE2E.data.region_extractor import extract_regions
from srcNonE2E.data.labels import ID_TO_PITCH, ID_TO_DURATION
from srcNonE2E.utils.tf_utils import _enable_gpu_memory_growth
from srcNonE2E.eval_helpers.eval_geometry import sort_regions_by_x, crop_resize_norm


# Configuration (set IMAGE_PATH to your image; path can be absolute or relative to project root)
IMAGE_PATH = "data/primus_raw/package_aa/000100301-1_1_1/000100301-1_1_1.png"
PR_MODEL_PATH = "artifacts/pr_cnn.keras"
DR_MODEL_PATH = "artifacts/dr_cnn.keras"
INPUT_H = 257
INPUT_W = 65
OUT_TXT = None  # e.g. "out/infer_non_e2e.txt"


def infer_one_image(
    image_path: str,
    pr_model: tf.keras.Model,
    dr_model: tf.keras.Model,
    input_h: int,
    input_w: int,
) -> List[str]:
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    regions = extract_regions(img_bgr)
    regions = sort_regions_by_x(regions)

    if len(regions) == 0:
        return []

    crops = np.zeros((len(regions), input_h, input_w, 1), dtype=np.float32)
    for i, bb in enumerate(regions):
        crops[i] = crop_resize_norm(img_bgr, bb, input_h, input_w)

    pr_logits = pr_model.predict(crops, batch_size=256, verbose=0)
    dr_logits = dr_model.predict(crops, batch_size=256, verbose=0)

    pr_ids = np.argmax(pr_logits, axis=-1).astype(np.int32)
    dr_ids = np.argmax(dr_logits, axis=-1).astype(np.int32)

    tokens: List[str] = []
    for pid, did in zip(pr_ids, dr_ids):
        pitch = ID_TO_PITCH.get(int(pid), f"UNKP{int(pid)}")
        duration = ID_TO_DURATION.get(int(did), f"UNKD{int(did)}")
        tokens.append(f"note.{duration}-{pitch}")

    return tokens


def main():
    _enable_gpu_memory_growth()

    if not IMAGE_PATH:
        raise ValueError("Set IMAGE_PATH at the top of infer_non_e2e.py")

    image_path = IMAGE_PATH

    if not os.path.exists(PR_MODEL_PATH):
        raise FileNotFoundError(f"Missing PR model: {PR_MODEL_PATH}")
    if not os.path.exists(DR_MODEL_PATH):
        raise FileNotFoundError(f"Missing DR model: {DR_MODEL_PATH}")

    pr_model = tf.keras.models.load_model(PR_MODEL_PATH)
    dr_model = tf.keras.models.load_model(DR_MODEL_PATH)

    tokens = infer_one_image(
        image_path=image_path,
        pr_model=pr_model,
        dr_model=dr_model,
        input_h=INPUT_H,
        input_w=INPUT_W,
    )

    transcript = " ".join(tokens)

    print("Image:", image_path)
    print("Regions:", len(tokens))
    print("Transcript:")
    print(transcript)

    if OUT_TXT:
        os.makedirs(os.path.dirname(OUT_TXT) or ".", exist_ok=True)
        with open(OUT_TXT, "w", encoding="utf-8") as f:
            f.write(transcript + "\n")
        print("Saved:", OUT_TXT)


if __name__ == "__main__":
    main()