# I used ChatGPT and Cursor for the development of this project.
# python -m srcNonE2E.eval_full_images

import os

import cv2
import tensorflow as tf

from srcNonE2E.data.region_extractor import extract_regions
from srcNonE2E.utils.tf_utils import _enable_gpu_memory_growth
from srcNonE2E.eval_helpers.eval_geometry import sort_regions_by_x
from srcNonE2E.eval_helpers.eval_gt import gt_tokens_from_transcript
from srcNonE2E.eval_helpers.eval_metrics import levenshtein_tokens, levenshtein_chars
from srcNonE2E.eval_helpers.eval_inference import infer_tokens_for_image
from srcNonE2E.eval_helpers.eval_io import read_split_csv


CSV_PATH = "data/splits/testNonE2E.csv"
IMAGES_ROOT = "data/primus_raw"
PR_MODEL_PATH = "artifacts/pr_cnn.keras"
DR_MODEL_PATH = "artifacts/dr_cnn.keras"
INPUT_H = 257
INPUT_W = 65
BATCH_SIZE = 256  # crops per batch when calling model.predict (many regions per image)
PROGRESS_EVERY = 100


def main():
    _enable_gpu_memory_growth()

    if not os.path.exists(PR_MODEL_PATH):
        raise FileNotFoundError(f"Missing PR model: {PR_MODEL_PATH}")
    if not os.path.exists(DR_MODEL_PATH):
        raise FileNotFoundError(f"Missing DR model: {DR_MODEL_PATH}")

    pr_model = tf.keras.models.load_model(PR_MODEL_PATH)
    dr_model = tf.keras.models.load_model(DR_MODEL_PATH)

    data = read_split_csv(CSV_PATH)

    processed = 0
    evaluated = 0
    bad_or_missing = 0
    empty_ref = 0

    seq_errors = 0
    total_ref_chars = 0
    total_char_edits = 0

    for rel_path, transcript in data:
        processed += 1

        img_path = os.path.join(IMAGES_ROOT, rel_path) if IMAGES_ROOT else rel_path
        gt_tokens = gt_tokens_from_transcript(transcript)

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            bad_or_missing += 1
            continue

        try:
            regions = extract_regions(img_bgr)
        except Exception:
            bad_or_missing += 1
            continue

        regions = sort_regions_by_x(regions)
        pred_tokens = infer_tokens_for_image(
            img_bgr=img_bgr,
            regions=regions,
            pr_model=pr_model,
            dr_model=dr_model,
            input_h=INPUT_H,
            input_w=INPUT_W,
            batch_size=BATCH_SIZE,
        )

        if len(gt_tokens) == 0:
            empty_ref += 1
            continue

        evaluated += 1

        sym_ed = levenshtein_tokens(gt_tokens, pred_tokens)
        if sym_ed > 0:
            seq_errors += 1

        ref_str = " ".join(gt_tokens)
        hyp_str = " ".join(pred_tokens)
        ref_chars = ref_str.replace(" ", "")
        hyp_chars = hyp_str.replace(" ", "")

        char_ed = levenshtein_chars(ref_chars, hyp_chars)
        total_char_edits += char_ed
        if len(ref_chars) > 0:
            total_ref_chars += len(ref_chars)

        if PROGRESS_EVERY and processed % PROGRESS_EVERY == 0:
            if evaluated:
                ser = (seq_errors / evaluated) * 100.0
            else:
                ser = 0.0
            if total_ref_chars:
                cer = (total_char_edits / total_ref_chars) * 100.0
            else:
                cer = 0.0
            print(
                f"progress: processed={processed} evaluated={evaluated} bad={bad_or_missing} "
                f"SER={ser:.3f}% CER={cer:.3f}%"
            )

    if evaluated:
        ser = (seq_errors / evaluated) * 100.0
    else:
        ser = 0.0

    if total_ref_chars:
        cer = (total_char_edits / total_ref_chars) * 100.0
    else:
        cer = 0.0

    print("CSV:", CSV_PATH)
    print("Processed images:", processed)
    print("Evaluated images:", evaluated)
    print("Bad/missing images:", bad_or_missing)
    print("Empty reference sequences skipped:", empty_ref)
    print(f"SER (%): {ser:.6f}")
    print(f"CER (%): {cer:.6f}")


if __name__ == "__main__":
    main()
