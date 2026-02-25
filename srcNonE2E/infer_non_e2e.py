import os
import argparse
from typing import List, Tuple

import numpy as np
import cv2
import tensorflow as tf

from srcNonE2E.data.region_extractor import extract_regions as _extract_regions
from srcNonE2E.data.labels import ID_TO_PITCH, ID_TO_DURATION
from srcNonE2E.utils.tf_utils import _enable_gpu_memory_growth


BBox = Tuple[int, int, int, int]


def sort_regions_by_x(bboxes: List[BBox]) -> List[BBox]:
    def key(bb: BBox):
        x1, y1, x2, y2 = bb
        return (x1 + x2) / 2.0
    return sorted(bboxes, key=key)


def _crop_resize_norm(img_bgr: np.ndarray, bb: BBox, out_h: int, out_w: int) -> np.ndarray:
    x1, y1, x2, y2 = bb
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = max(0, int(x2))
    y2 = max(0, int(y2))

    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1

    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = img_bgr[0:1, 0:1]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    x = resized.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=-1)
    return x


def infer_one_image(
    image_path: str,
    pr_model: tf.keras.Model,
    dr_model: tf.keras.Model,
    input_h: int,
    input_w: int,
    batch_size: int = 256,
) -> List[str]:
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    regions = _extract_regions(img_bgr)
    regions = sort_regions_by_x(regions)

    if len(regions) == 0:
        return []

    crops = np.zeros((len(regions), input_h, input_w, 1), dtype=np.float32)
    for i, bb in enumerate(regions):
        crops[i] = _crop_resize_norm(img_bgr, bb, input_h, input_w)

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


def main():
    _enable_gpu_memory_growth()

    ap = argparse.ArgumentParser()
    ap.add_argument("--rel_path", type=str, default=None)
    ap.add_argument("--abs_path", type=str, default=None)
    ap.add_argument("--images_root", type=str, default="data/primus_raw")
    ap.add_argument("--pr_model", type=str, default="artifacts/pr_cnn.keras")
    ap.add_argument("--dr_model", type=str, default="artifacts/dr_cnn.keras")
    ap.add_argument("--h", type=int, default=257)
    ap.add_argument("--w", type=int, default=65)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out_txt", type=str, default=None)
    args = ap.parse_args()

    if args.abs_path is None and args.rel_path is None:
        raise ValueError("Provide --abs_path or --rel_path")

    if args.abs_path is not None:
        image_path = args.abs_path
    else:
        image_path = os.path.join(args.images_root, args.rel_path)

    if not os.path.exists(args.pr_model):
        raise FileNotFoundError(f"Missing PR model: {args.pr_model}")
    if not os.path.exists(args.dr_model):
        raise FileNotFoundError(f"Missing DR model: {args.dr_model}")

    pr_model = tf.keras.models.load_model(args.pr_model)
    dr_model = tf.keras.models.load_model(args.dr_model)

    tokens = infer_one_image(
        image_path=image_path,
        pr_model=pr_model,
        dr_model=dr_model,
        input_h=args.h,
        input_w=args.w,
        batch_size=args.batch_size,
    )

    transcript = " ".join(tokens)

    print("Image:", image_path)
    print("Regions:", len(tokens))
    print("Transcript:")
    print(transcript)

    if args.out_txt:
        os.makedirs(os.path.dirname(args.out_txt) or ".", exist_ok=True)
        with open(args.out_txt, "w", encoding="utf-8") as f:
            f.write(transcript + "\n")
        print("Saved:", args.out_txt)


if __name__ == "__main__":
    main()