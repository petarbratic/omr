# preprocess_dataset.py
#
# Preprocessing (NO MASTER, direct PR/DR with caps):
# - Loads a split CSV (image_path, transcript)
# - Extracts note tokens from transcript -> (token, duration, pitch)
# - Extracts regions (bbox) from image and sorts them by x-axis (single-stave monophonic)
# - If counts mismatch -> mismatch-<split>.csv
# - If counts match -> writes directly:
#     pr-<split>.csv: (image_path, x1, y1, x2, y2, idx, pitch)
#     dr-<split>.csv: (image_path, x1, y1, x2, y2, idx, duration)
#
# Caps / balancing:
# - Writes to PR only while pitch_count[pitch] < MAX_PER_PITCH
# - Writes to DR only while dur_count[duration] < MAX_PER_DURATION
#
# Optional early stop:
# - If all pitch classes hit cap AND all duration classes hit cap -> stop processing split.
#
# Requirement: implemented extract_regions(img_bgr) function that returns list of bounding boxes:
#     [(x1, y1, x2, y2), ...]

import os
import csv
from typing import List, Tuple
from collections import defaultdict

import cv2

from srcNonE2E.data.string_utils import extract_duration_and_pitch_from_transcript
from srcNonE2E.data.region_extractor import extract_regions
from srcNonE2E.data.labels import PITCH_CLASSES, DURATION_CLASSES
from srcNonE2E.eval_helpers.eval_geometry import sort_regions_by_x
from srcNonE2E.eval_helpers.eval_io import read_split_csv


# =========================================================
# DEFINES
# =========================================================

TRAIN_CSV = "data/splits/trainNonE2E.csv"
VAL_CSV   = "data/splits/valNonE2E.csv"
TEST_CSV  = "data/splits/testNonE2E.csv"

IMAGES_ROOT = "data/primus_raw"
OUT_DIR = "out/region_dataset"

PRINT_MISMATCH_EXAMPLES = 10

# =========================================================
# BALANCING / CAPS
# =========================================================
# Recommended "best" (stable): 1000 per class. Change as needed.
MAX_PER_PITCH = 1000
MAX_PER_DURATION = 1000

EARLY_STOP_WHEN_FULL = True  # set False to always process the full split


def pitch_full(pitch_counts) -> bool:
    return all(pitch_counts.get(p, 0) >= MAX_PER_PITCH for p in PITCH_CLASSES)


def duration_full(dur_counts) -> bool:
    return all(dur_counts.get(d, 0) >= MAX_PER_DURATION for d in DURATION_CLASSES)

# =========================================================
# CSV helpers
# =========================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _notes_preview(notes, max_tokens: int = 5) -> str:
    return " ".join([t for (t, _, _) in notes[:max_tokens]])


def _load_and_validate_image(rel_path: str, transcript: str):
    # Returns (True, regions, notes) on success, or (False, rel_path, n_regions, n_notes, preview, reason) on failure.
    img_path = os.path.join(IMAGES_ROOT, rel_path) if IMAGES_ROOT else rel_path
    notes = extract_duration_and_pitch_from_transcript(transcript)

    if not os.path.exists(img_path):
        return (False, rel_path, -1, len(notes), _notes_preview(notes), "missing_image")

    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return (False, rel_path, -2, len(notes), _notes_preview(notes), "bad_image_read")

    try:
        regions = extract_regions(img_bgr)
    except NotImplementedError:
        raise
    except Exception as e:
        return (False, rel_path, -3, len(notes), _notes_preview(notes), f"region_extract_error:{e}")

    regions = sort_regions_by_x(regions)
    if len(regions) != len(notes):
        return (False, rel_path, len(regions), len(notes), _notes_preview(notes), "count_mismatch")

    return (True, regions, notes)


def _write_pr_dr_rows_for_image(prw, drw, rel_path: str, regions, notes, pitch_counts, dur_counts) -> Tuple[int, int]:
    written_pr = 0
    written_dr = 0
    for idx, (bb, note) in enumerate(zip(regions, notes)):
        x1, y1, x2, y2 = bb
        token, duration, pitch = note
        if pitch_counts[pitch] < MAX_PER_PITCH:
            prw.writerow([rel_path, x1, y1, x2, y2, idx, pitch])
            pitch_counts[pitch] += 1
            written_pr += 1
        if duration in DURATION_CLASSES and dur_counts[duration] < MAX_PER_DURATION:
            drw.writerow([rel_path, x1, y1, x2, y2, idx, duration])
            dur_counts[duration] += 1
            written_dr += 1
    return (written_pr, written_dr)


def _print_split_summary(
    split_name: str,
    pr_path: str,
    dr_path: str,
    mismatch_path: str,
    processed_images: int,
    matched_images: int,
    written_pr: int,
    written_dr: int,
    pitch_counts,
    dur_counts,
) -> None:
    min_pitch = min(pitch_counts.get(p, 0) for p in PITCH_CLASSES) if PITCH_CLASSES else 0
    min_dur = min(dur_counts.get(d, 0) for d in DURATION_CLASSES) if DURATION_CLASSES else 0
    print(f"[{split_name}] Wrote: {pr_path}")
    print(f"[{split_name}] Wrote: {dr_path}")
    print(f"[{split_name}] Wrote: {mismatch_path}")
    print(f"[{split_name}] Stats: processed_images={processed_images} matched_images={matched_images}")
    print(f"[{split_name}] Rows: pr={written_pr} dr={written_dr}")
    print(f"[{split_name}] Coverage(min): pitch={min_pitch} duration={min_dur}")


def process_split(split_name: str, split_csv: str) -> None:
    ensure_dir(OUT_DIR)

    pr_path = os.path.join(OUT_DIR, f"pr-{split_name}.csv")
    dr_path = os.path.join(OUT_DIR, f"dr-{split_name}.csv")
    mismatch_path = os.path.join(OUT_DIR, f"mismatch-{split_name}.csv")

    data = read_split_csv(split_csv)
    pitch_counts = defaultdict(int)
    dur_counts = defaultdict(int)
    mismatch_printed = 0
    written_pr = 0
    written_dr = 0
    processed_images = 0
    matched_images = 0

    with open(pr_path, "w", newline="", encoding="utf-8") as f_pr, \
         open(dr_path, "w", newline="", encoding="utf-8") as f_dr, \
         open(mismatch_path, "w", newline="", encoding="utf-8") as f_mm:

        prw = csv.writer(f_pr)
        drw = csv.writer(f_dr)
        mmw = csv.writer(f_mm)
        prw.writerow(["image_path", "x1", "y1", "x2", "y2", "idx", "pitch"])
        drw.writerow(["image_path", "x1", "y1", "x2", "y2", "idx", "duration"])
        mmw.writerow(["image_path", "n_regions", "n_notes", "notes_kept_preview", "reason"])

        for rel_path, transcript in data:
            processed_images += 1

            if EARLY_STOP_WHEN_FULL and pitch_full(pitch_counts) and duration_full(dur_counts):
                print(
                    f"[{split_name}] STOP: pitch and duration are filled to limit. "
                    f"pr_rows={written_pr} dr_rows={written_dr} processed_images={processed_images}"
                )
                break

            result = _load_and_validate_image(rel_path, transcript)
            if not result[0]:
                _, rp, n_reg, n_notes, preview, reason = result
                mmw.writerow([rp, n_reg, n_notes, preview, reason])
                if mismatch_printed < PRINT_MISMATCH_EXAMPLES:
                    if reason == "missing_image":
                        print(f"[{split_name}] MISSING IMAGE: {rp} (notes={n_notes})")
                    elif reason == "bad_image_read":
                        print(f"[{split_name}] BAD IMAGE READ: {rp} (notes={n_notes})")
                    elif reason.startswith("region_extract_error:"):
                        print(f"[{split_name}] REGION EXTRACT ERROR: {rp} -> {reason.split(':', 1)[1]}")
                    else:
                        print(f"[{split_name}] MISMATCH: {rp} regions={n_reg} notes={n_notes} preview={preview}")
                    mismatch_printed += 1
                continue

            _, regions, notes = result
            matched_images += 1
            delta_pr, delta_dr = _write_pr_dr_rows_for_image(prw, drw, rel_path, regions, notes, pitch_counts, dur_counts)
            written_pr += delta_pr
            written_dr += delta_dr

    _print_split_summary(
        split_name, pr_path, dr_path, mismatch_path,
        processed_images, matched_images, written_pr, written_dr,
        pitch_counts, dur_counts,
    )


def main() -> None:
    ensure_dir(OUT_DIR)

    process_split("train", TRAIN_CSV)
    process_split("val", VAL_CSV)
    process_split("test", TEST_CSV)

    print("Done.")


if __name__ == "__main__":
    main()