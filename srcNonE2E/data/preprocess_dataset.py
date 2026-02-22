"""
preprocess_dataset.py

Preprocessing (NO MASTER, direct PR/DR with caps):
- Loads a split CSV (image_path, transcript)
- Extracts note tokens from transcript -> (token, duration, pitch)
- Extracts regions (bbox) from image and sorts them by x-axis (single-stave monophonic)
- If counts mismatch -> mismatch-<split>.csv
- If counts match -> writes directly:
    pr-<split>.csv: (image_path, x1, y1, x2, y2, idx, pitch)
    dr-<split>.csv: (image_path, x1, y1, x2, y2, idx, duration)

Caps / balancing:
- Writes to PR only while pitch_count[pitch] < MAX_PER_PITCH
- Writes to DR only while dur_count[duration] < MAX_PER_DURATION

Optional early stop:
- If all pitch classes hit cap AND all duration classes hit cap -> stop processing split.

Requirement: implemented extract_regions(img_bgr) function so that it returns a list of bounding boxes:
    [(x1, y1, x2, y2), ...]
"""

import os
import csv
from typing import List, Tuple
from collections import defaultdict

import cv2

from srcNonE2E.data.string_utils import extract_duration_and_pitch_from_transcript
from srcNonE2E.data.region_extractor import extract_regions as _extract_regions


# =========================================================
# DEFINES
# =========================================================

TRAIN_CSV = "srcNonE2E/data/baseCSV/trainNonE2E.csv"
VAL_CSV   = "srcNonE2E/data/baseCSV/valNonE2E.csv"
TEST_CSV  = "srcNonE2E/data/baseCSV/testNonE2E.csv"

IMAGES_ROOT = "data/primus_raw"
OUT_DIR = "out/region_dataset"

PRINT_MISMATCH_EXAMPLES = 10

# =========================================================
# BALANCING / CAPS
# =========================================================
# Preporučeno "najbolje" (stabilno): 1000 po klasi.
# Promeni po potrebi.
MAX_PER_PITCH = 1000
MAX_PER_DURATION = 1000

# 24 pitch klasa: L-3..L8 + S-3..S8
PITCH_CLASSES = [f"L{i}" for i in range(-3, 9)] + [f"S{i}" for i in range(-3, 9)]

# Duration klase koje očekuješ (po tvom master primeru: "quarter", "half", ...)
# Ako neka od ovih ne postoji u dataset-u, early-stop za duration nikad neće okinuti (to je ok).
DURATION_CLASSES = [
    "quadruple_whole",
    "double_whole",
    "whole",
    "half",
    "quarter",
    "eighth",
    "sixteenth",
    "thirty_second",
    "sixty_fourth",
]

EARLY_STOP_WHEN_FULL = True  # ako hoćeš da uvek obradi ceo split, stavi False


def pitch_full(pitch_counts) -> bool:
    return all(pitch_counts.get(p, 0) >= MAX_PER_PITCH for p in PITCH_CLASSES)


def duration_full(dur_counts) -> bool:
    return all(dur_counts.get(d, 0) >= MAX_PER_DURATION for d in DURATION_CLASSES)


# =========================================================
# Region extraction
# =========================================================

BBox = Tuple[int, int, int, int]

def extract_regions(img_bgr) -> List[BBox]:
    return _extract_regions(img_bgr)


def sort_regions_by_x(bboxes: List[BBox]) -> List[BBox]:
    def key(bb: BBox):
        x1, y1, x2, y2 = bb
        xc = (x1 + x2) / 2.0
        return xc
    return sorted(bboxes, key=key)


# =========================================================
# CSV helpers
# =========================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_split_csv(csv_path: str) -> List[Tuple[str, str]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Prazan CSV: {csv_path}")
        if "image_path" not in reader.fieldnames or "transcript" not in reader.fieldnames:
            raise ValueError(f"{csv_path} mora imati kolone: image_path, transcript")
        for r in reader:
            rows.append((r["image_path"].strip(), r["transcript"].strip()))
    return rows


def process_split(split_name: str, split_csv: str) -> None:
    """
    Writes:
      pr-<split>.csv, dr-<split>.csv, mismatch-<split>.csv
    with per-class caps (balanced dataset).
    """
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

            # Optional early-stop when both datasets are "full"
            if EARLY_STOP_WHEN_FULL and pitch_full(pitch_counts) and duration_full(dur_counts):
                print(
                    f"[{split_name}] STOP: pitch i duration su popunjeni do limita. "
                    f"pr_rows={written_pr} dr_rows={written_dr} processed_images={processed_images}"
                )
                break

            img_path = os.path.join(IMAGES_ROOT, rel_path) if IMAGES_ROOT else rel_path

            notes = extract_duration_and_pitch_from_transcript(transcript)  # list[(token, duration, pitch)]

            if not os.path.exists(img_path):
                preview = " ".join([t for (t, _, _) in notes[:5]])
                mmw.writerow([rel_path, -1, len(notes), preview, "missing_image"])
                if mismatch_printed < PRINT_MISMATCH_EXAMPLES:
                    print(f"[{split_name}] MISSING IMAGE: {rel_path} (notes={len(notes)})")
                    mismatch_printed += 1
                continue

            img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                preview = " ".join([t for (t, _, _) in notes[:5]])
                mmw.writerow([rel_path, -2, len(notes), preview, "bad_image_read"])
                if mismatch_printed < PRINT_MISMATCH_EXAMPLES:
                    print(f"[{split_name}] BAD IMAGE READ: {rel_path} (notes={len(notes)})")
                    mismatch_printed += 1
                continue

            try:
                regions = extract_regions(img_bgr)
            except NotImplementedError:
                raise
            except Exception as e:
                preview = " ".join([t for (t, _, _) in notes[:5]])
                mmw.writerow([rel_path, -3, len(notes), preview, f"region_extract_error:{e}"])
                if mismatch_printed < PRINT_MISMATCH_EXAMPLES:
                    print(f"[{split_name}] REGION EXTRACT ERROR: {rel_path} -> {e}")
                    mismatch_printed += 1
                continue

            regions = sort_regions_by_x(regions)

            if len(regions) != len(notes):
                preview = " ".join([t for (t, _, _) in notes[:5]])
                mmw.writerow([rel_path, len(regions), len(notes), preview, "count_mismatch"])
                if mismatch_printed < PRINT_MISMATCH_EXAMPLES:
                    print(f"[{split_name}] MISMATCH: {rel_path} regions={len(regions)} notes={len(notes)} preview={preview}")
                    mismatch_printed += 1
                continue

            matched_images += 1

            # Aligned per-note loop.
            # Sada je dozvoljeno da se pojedinačne note preskoče (zbog cap),
            # jer više ne insistiramo na master fajlu koji mora imati sve.
            for idx, (bb, note) in enumerate(zip(regions, notes)):
                x1, y1, x2, y2 = bb
                token, duration, pitch = note

                # PR (pitch) cap
                if pitch_counts[pitch] < MAX_PER_PITCH:
                    prw.writerow([rel_path, x1, y1, x2, y2, idx, pitch])
                    pitch_counts[pitch] += 1
                    written_pr += 1

                # DR (duration) cap
                # Ako duration nije u listi očekivanih, preskoči (ili dodaj u listu ako želiš da ga treniraš)
                if duration in DURATION_CLASSES and dur_counts[duration] < MAX_PER_DURATION:
                    drw.writerow([rel_path, x1, y1, x2, y2, idx, duration])
                    dur_counts[duration] += 1
                    written_dr += 1

    # Summary
    min_pitch = min(pitch_counts.get(p, 0) for p in PITCH_CLASSES) if PITCH_CLASSES else 0
    min_dur = min(dur_counts.get(d, 0) for d in DURATION_CLASSES) if DURATION_CLASSES else 0

    print(f"[{split_name}] Wrote: {pr_path}")
    print(f"[{split_name}] Wrote: {dr_path}")
    print(f"[{split_name}] Wrote: {mismatch_path}")
    print(f"[{split_name}] Stats: processed_images={processed_images} matched_images={matched_images}")
    print(f"[{split_name}] Rows: pr={written_pr} dr={written_dr}")
    print(f"[{split_name}] Coverage(min): pitch={min_pitch} duration={min_dur}")


def main() -> None:
    ensure_dir(OUT_DIR)

    process_split("train", TRAIN_CSV)
    process_split("val", VAL_CSV)
    process_split("test", TEST_CSV)

    print("Done.")


if __name__ == "__main__":
    main()