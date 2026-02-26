# python -m srcNonE2E.infer_non_e2e
# Run PR/DR inference on one image and print predicted tokens.
import os
from typing import List

import cv2
import tensorflow as tf

from srcNonE2E.data.region_extractor import extract_regions
from srcNonE2E.utils.tf_utils import _enable_gpu_memory_growth
from srcNonE2E.eval_helpers.eval_geometry import sort_regions_by_x
from srcNonE2E.eval_helpers.eval_inference import infer_tokens_for_image


# Configuration (set IMAGE_PATH to your image; path can be absolute or relative to project root)
IMAGE_PATH = "data/primus_raw/package_aa/000100301-1_1_1/000100301-1_1_1.png"
PR_MODEL_PATH = "artifacts/pr_cnn.keras"
DR_MODEL_PATH = "artifacts/dr_cnn.keras"
INPUT_H = 257
INPUT_W = 65
BATCH_SIZE = 256
OUT_TXT = None  # e.g. "out/infer_non_e2e.txt"


def infer_one_image(
    image_path: str,
    pr_model: tf.keras.Model,
    dr_model: tf.keras.Model,
    input_h: int,
    input_w: int,
    batch_size: int = BATCH_SIZE,
) -> List[str]:
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    regions = extract_regions(img_bgr)
    regions = sort_regions_by_x(regions)

    return infer_tokens_for_image(
        img_bgr=img_bgr,
        regions=regions,
        pr_model=pr_model,
        dr_model=dr_model,
        input_h=input_h,
        input_w=input_w,
        batch_size=batch_size,
    )


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