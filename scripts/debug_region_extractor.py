# I used ChatGPT and Cursor for the development of this project.
# debug_region_extractor.py
# Manual test for extract_regions(img_bgr).
# Loads image(s), calls extract_regions, draws bboxes and saves debug image to OUT_DIR.
# Set IMAGE_PATHS and OUT_DIR in DEFINES.
#
# python -m scripts.debug_region_extractor
import os
import sys
from pathlib import Path

# Run from project root so srcNonE2E is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List

import cv2

from srcNonE2E.data.region_extractor import extract_regions


# =========================================================
# DEFINES
# =========================================================

IMAGE_PATHS: List[str] = [
    "data/primus_raw/package_ab/200021503-1_9_2/200021503-1_9_2.png",
    "data/primus_raw/package_aa/000105214-1_1_1/000105214-1_1_1.png",
    "data/primus_raw/package_ab/211004799-1_6_1/211004799-1_6_1.png",
    "data/primus_raw/package_ab/211004341-1_2_1/211004341-1_2_1.png",
    "data/primus_raw/package_aa/000120324-3_1_1/000120324-3_1_1.png",
    "data/primus_raw/package_aa/100501023-1_2_3/100501023-1_2_3.png",
    "data/primus_raw/package_ab/230003697-1_20_1/230003697-1_20_1.png",
    "data/primus_raw/package_ab/200021901-1_14_2/200021901-1_14_2.png",
    "data/primus_raw/package_aa/000100301-1_1_1/000100301-1_1_1.png",
    "data/primus_raw/package_ab/220018816-1_1_1/220018816-1_1_1.png",
    "data/primus_raw/package_aa/000100242-1_1_1/000100242-1_1_1.png",
    "data/primus_raw/package_aa/000117511-1_1_1/000117511-1_1_1.png",
    "data/primus_raw/package_aa/000130554-3_3_3/000130554-3_3_3.png",
    "data/primus_raw/package_aa/000113706-1_1_1/000113706-1_1_1.png",
    "data/primus_raw/package_ab/211010628-1_1_1/211010628-1_1_1.png",
    "data/primus_raw/package_ab/212001597-1_1_1/212001597-1_1_1.png",
]

OUT_DIR = "out/debug_regions"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def draw_bboxes(img_bgr, bboxes):
    out = img_bgr.copy()
    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out


def main() -> None:
    ensure_dir(OUT_DIR)

    if not IMAGE_PATHS:
        raise ValueError("Add at least one path to IMAGE_PATHS in debug_region_extractor.py")

    for p in IMAGE_PATHS:
        if not os.path.exists(p):
            print(f"NOT FOUND: {p}")
            continue

        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"FAILED TO LOAD: {p}")
            continue

        bboxes = extract_regions(img)
        print(f"{p} -> regions: {len(bboxes)}")

        vis = draw_bboxes(img, bboxes)

        base = os.path.basename(p).replace(".png", "")
        out_path = os.path.join(OUT_DIR, f"{base}_boxes.png")
        cv2.imwrite(out_path, vis)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
