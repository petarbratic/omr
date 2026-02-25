# region_extractor.py
#
# extract_regions(img_bgr) -> List[BBox]
#
# Template matching using multiple notehead templates: filled heads, half heads, whole heads.
# Pipeline: preprocess input (grayscale + inverted Otsu binarization), preprocess templates
# the same way, match templates and apply NMS. Output regions span full image height (y1=0, y2=H-1).

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


# =========================================================
# DEFINES
# =========================================================

# Template folder
TEMPLATES_DIR = "data/templates"

# Match thresholds per group
MATCH_THRESH_FILLED = 0.74
MATCH_THRESH_HALF = 0.70
MATCH_THRESH_WHOLE = 0.74

# NMS overlap threshold
NMS_IOU = 0.35

# Expand bbox in x around matched head
PAD_X = 0.25  # * template_width

NEG_TEMPLATE_PATHS = [
    "data/templates/24.png",
    "data/templates/128.png",
]
MATCH_THRESH_NEG = 0.75
NEG_IOU_1D = 0.35


# =========================================================
# TYPES
# =========================================================

BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class _Det:
    x1: int
    x2: int
    score: float


# =========================================================
# PREPROCESS
# =========================================================

def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def _binarize_inv_otsu(gray: np.ndarray) -> np.ndarray:
    # Foreground = white (255), background = 0
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def _preprocess_input(img_bgr: np.ndarray) -> np.ndarray:
    gray = _to_gray(img_bgr)
    bw = _binarize_inv_otsu(gray)
    return bw


def _preprocess_template(tpl_gray: np.ndarray) -> np.ndarray:
    # Same binarization as input
    return _binarize_inv_otsu(tpl_gray)


# =========================================================
# TEMPLATE LOADING
# =========================================================

def _load_templates(paths: List[str]) -> List[np.ndarray]:
    tpls: List[np.ndarray] = []
    for p in paths:
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        tpls.append(_preprocess_template(g))
    return tpls


def _template_paths() -> tuple[list[str], list[str], list[str]]:
    # Explicit names shown in your folder screenshot
    filled = [
        f"{TEMPLATES_DIR}/note_filled_1.png",
        f"{TEMPLATES_DIR}/note_filled_2.png",
        f"{TEMPLATES_DIR}/note_filled_3.png",
        f"{TEMPLATES_DIR}/note_filled_4.png",
        f"{TEMPLATES_DIR}/note_filled_5.png",
        f"{TEMPLATES_DIR}/note_filled_6.png",
    ]
    half = [
        f"{TEMPLATES_DIR}/note_half_1.png",
        f"{TEMPLATES_DIR}/note_half_2.png",
        f"{TEMPLATES_DIR}/note_half_3.png",
        f"{TEMPLATES_DIR}/note_half_4.png",
    ]
    whole = [
        f"{TEMPLATES_DIR}/note_whole_1.png",
        f"{TEMPLATES_DIR}/note_whole_2.png",
        f"{TEMPLATES_DIR}/note_whole_3.png",
    ]
    return filled, half, whole


# =========================================================
# NMS (1D on x, because y spans full height)
# =========================================================

def _iou_1d(a: tuple[int, int], b: tuple[int, int]) -> float:
    ax1, ax2 = a
    bx1, bx2 = b
    ix1 = max(ax1, bx1)
    ix2 = min(ax2, bx2)
    inter = max(0, ix2 - ix1 + 1)
    area_a = max(0, ax2 - ax1 + 1)
    area_b = max(0, bx2 - bx1 + 1)
    return inter / (area_a + area_b - inter + 1e-6)


def _nms_1d(dets: List[_Det], iou_thresh: float) -> List[_Det]:
    if not dets:
        return []

    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    kept: List[_Det] = []

    for d in dets:
        ok = True
        for k in kept:
            if _iou_1d((d.x1, d.x2), (k.x1, k.x2)) > iou_thresh:
                ok = False
                break
        if ok:
            kept.append(d)

    return kept


# =========================================================
# MATCHING
# =========================================================

def _match_many(
    img_bin: np.ndarray,
    templates: List[np.ndarray],
    thresh: float,
    pad_x: float,
) -> List[_Det]:
    H, W = img_bin.shape[:2]
    dets: List[_Det] = []

    for tpl in templates:
        th, tw = tpl.shape[:2]
        if th > H or tw > W:
            continue

        res = cv2.matchTemplate(img_bin, tpl, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= thresh)

        px = int(round(tw * pad_x))

        for y, x in zip(ys.tolist(), xs.tolist()):
            score = float(res[y, x])

            x1 = max(0, x - px)
            x2 = min(W - 1, x + tw + px)

            dets.append(_Det(x1=x1, x2=x2, score=score))

    return dets


# =========================================================
# NEGATIVE FILTERING
# ==========================================================

def _filter_by_negatives(pos: List[_Det], neg: List[_Det], iou_thresh: float) -> List[_Det]:
    if not neg:
        return pos
    kept: List[_Det] = []
    for p in pos:
        bad = False
        for n in neg:
            if _iou_1d((p.x1, p.x2), (n.x1, n.x2)) >= iou_thresh:
                bad = True
                break
        if not bad:
            kept.append(p)
    return kept

# =========================================================
# PUBLIC API
# =========================================================

def extract_regions(img_bgr: np.ndarray) -> List[BBox]:
    # Preprocess input
    img_bin = _preprocess_input(img_bgr)
    H, W = img_bin.shape[:2]

    # Load and preprocess templates
    filled_paths, half_paths, whole_paths = _template_paths()
    tpls_filled = _load_templates(filled_paths)
    tpls_half = _load_templates(half_paths)
    tpls_whole = _load_templates(whole_paths)

    tpls_neg = _load_templates(NEG_TEMPLATE_PATHS)
    neg_dets = _match_many(img_bin, tpls_neg, MATCH_THRESH_NEG, PAD_X)
    neg_dets = _nms_1d(neg_dets, iou_thresh=NEG_IOU_1D)

    dets: List[_Det] = []
    dets += _match_many(img_bin, tpls_filled, MATCH_THRESH_FILLED, PAD_X)
    dets += _match_many(img_bin, tpls_half, MATCH_THRESH_HALF, PAD_X)
    dets += _match_many(img_bin, tpls_whole, MATCH_THRESH_WHOLE, PAD_X)

    dets = _nms_1d(dets, iou_thresh=NMS_IOU)
    dets = _filter_by_negatives(dets, neg_dets, NEG_IOU_1D)

    # Full-height regions
    bboxes: List[BBox] = [(d.x1, 0, d.x2, H - 1) for d in dets]
    return bboxes