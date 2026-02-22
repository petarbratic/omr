"""
debug_remove_staff.py

Ručni test za _remove_staff_lines funkciju.

- Učita slike koje uneseš u IMAGE_PATHS
- Prikaže:
    1) grayscale
    2) binarizovanu (invertovanu)
    3) bez staff linija
- Sve slike se čuvaju u OUT_DIR
"""

import os
from typing import List

import cv2
import numpy as np

from srcNonE2E.data.region_extractor import _remove_staff_lines


# =========================================================
# DEFINES
# =========================================================

IMAGE_PATHS: List[str] = [
    "data/primus_raw/package_ab/220012314-1_4_1/220012314-1_4_1.png",
    "data/primus_raw/package_ab/211005446-1_2_1/211005446-1_2_1.png",
    "data/primus_raw/package_aa/000121147-1_1_1/000121147-1_1_1.png",
    "data/primus_raw/package_aa/000103978-1_1_1/000103978-1_1_1.png",
    "data/primus_raw/package_ab/230005182-1_6_1/230005182-1_6_1.png",
    "data/primus_raw/package_aa/000116870-7_1_1/000116870-7_1_1.png",
    "data/primus_raw/package_ab/211006104-1_1_1/211006104-1_1_1.png",
    "data/primus_raw/package_ab/225003166-1_2_1/225003166-1_2_1.png",
    "data/primus_raw/package_aa/000106117-7_1_1/000106117-7_1_1.png",
    "data/primus_raw/package_ab/230003812-1_1_1/230003812-1_1_1.png",
    "data/primus_raw/package_ab/211004821-1_3_1/211004821-1_3_1.png",
    "data/primus_raw/package_ab/220018749-1_2_1/220018749-1_2_1.png",
    "data/primus_raw/package_ab/210000289-1_4_1/210000289-1_4_1.png",
    "data/primus_raw/package_aa/000120630-1_2_1/000120630-1_2_1.png",
    "data/primus_raw/package_aa/000117083-1_1_1/000117083-1_1_1.png",
    "data/primus_raw/package_aa/100501014-1_13_1/100501014-1_13_1.png",
    "data/primus_raw/package_ab/200021503-1_87_1/200021503-1_87_1.png",
    "data/primus_raw/package_aa/000116852-19_1_1/000116852-19_1_1.png",
    "data/primus_raw/package_ab/190004995-1_1_1/190004995-1_1_1.png",
]

OUT_DIR = "out/debug_remove_staff"


# =========================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def preprocess_gray(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def binarize(gray: np.ndarray) -> np.ndarray:
    _, bw = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return bw


def main():
    ensure_dir(OUT_DIR)

    if not IMAGE_PATHS:
        raise ValueError("Dodaj bar jednu sliku u IMAGE_PATHS.")

    for path in IMAGE_PATHS:
        if not os.path.exists(path):
            print(f"Ne postoji: {path}")
            continue

        img = cv2.imread(path)
        if img is None:
            print(f"Ne mogu da učitam: {path}")
            continue

        gray = preprocess_gray(img)
        bw = binarize(gray)
        no_staff = _remove_staff_lines(gray)

        base = os.path.basename(path).replace(".png", "")

        cv2.imwrite(os.path.join(OUT_DIR, f"{base}_gray.png"), gray)
        cv2.imwrite(os.path.join(OUT_DIR, f"{base}_binary.png"), bw)
        cv2.imwrite(os.path.join(OUT_DIR, f"{base}_no_staff.png"), no_staff)

        print(f"Obrađena slika: {path}")

    print("Gotovo.")


if __name__ == "__main__":
    main()