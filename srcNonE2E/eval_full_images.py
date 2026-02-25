# python -m srcNonE2E.eval_full_images

import os

import cv2
import tensorflow as tf

from srcNonE2E.data.region_extractor import extract_regions as _extract_regions
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
BATCH_SIZE = 256
LIMIT = 0  # 0 means no limit
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
    if LIMIT and LIMIT > 0:
        data = data[:LIMIT]

    processed = 0
    evaluated = 0
    bad_or_missing = 0
    empty_ref = 0

    seq_errors = 0

    total_ref_symbols = 0
    total_sym_edits = 0
    sum_norm_sym_ed = 0.0

    total_ref_chars = 0
    total_char_edits = 0
    sum_norm_char_ed = 0.0

    for rel_path, transcript in data:
        processed += 1

        img_path = os.path.join(IMAGES_ROOT, rel_path) if IMAGES_ROOT else rel_path
        gt_tokens = gt_tokens_from_transcript(transcript)

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            bad_or_missing += 1
            continue

        try:
            regions = _extract_regions(img_bgr)
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
        total_sym_edits += sym_ed
        total_ref_symbols += len(gt_tokens)
        norm_sym_ed = sym_ed / len(gt_tokens)
        sum_norm_sym_ed += norm_sym_ed

        ref_str = " ".join(gt_tokens)
        hyp_str = " ".join(pred_tokens)
        ref_chars = ref_str.replace(" ", "")
        hyp_chars = hyp_str.replace(" ", "")

        char_ed = levenshtein_chars(ref_chars, hyp_chars)
        total_char_edits += char_ed
        total_ref_chars += len(ref_chars) if len(ref_chars) > 0 else 0
        norm_char_ed = char_ed / max(1, len(ref_chars))
        sum_norm_char_ed += norm_char_ed

        if sym_ed > 0:
            seq_errors += 1

        if PROGRESS_EVERY and processed % PROGRESS_EVERY == 0:
            ser_seq = (seq_errors / evaluated) * 100.0 if evaluated else 0.0
            ser_sym_micro = (total_sym_edits / total_ref_symbols) * 100.0 if total_ref_symbols else 0.0
            cer_micro = (total_char_edits / total_ref_chars) * 100.0 if total_ref_chars else 0.0
            print(
                f"progress: processed={processed} evaluated={evaluated} bad={bad_or_missing} "
                f"SER_seq={ser_seq:.3f}% SER_sym={ser_sym_micro:.3f}% CER={cer_micro:.3f}%"
            )

    ser_seq = (seq_errors / evaluated) * 100.0 if evaluated else 0.0
    ser_sym_micro = (total_sym_edits / total_ref_symbols) * 100.0 if total_ref_symbols else 0.0
    ser_sym_macro = (sum_norm_sym_ed / evaluated) * 100.0 if evaluated else 0.0

    cer_micro = (total_char_edits / total_ref_chars) * 100.0 if total_ref_chars else 0.0
    cer_macro = (sum_norm_char_ed / evaluated) * 100.0 if evaluated else 0.0

    print("CSV:", CSV_PATH)
    print("Processed images:", processed)
    print("Evaluated images:", evaluated)
    print("Bad/missing images:", bad_or_missing)
    print("Empty reference sequences skipped:", empty_ref)
    print(f"Sequence Error Rate (SER_seq %): {ser_seq:.6f}")
    print(f"Symbol Error Rate (SER_sym % micro): {ser_sym_micro:.6f}")
    print(f"Symbol Error Rate (SER_sym % macro): {ser_sym_macro:.6f}")
    print(f"Character Error Rate (CER % micro): {cer_micro:.6f}")
    print(f"Character Error Rate (CER % macro): {cer_macro:.6f}")


if __name__ == "__main__":
    main()
