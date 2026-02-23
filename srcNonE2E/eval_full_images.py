import os
import csv
import argparse
from typing import List, Tuple

import numpy as np
import cv2
import tensorflow as tf

from srcNonE2E.data.region_extractor import extract_regions as _extract_regions
from srcNonE2E.data.string_utils import extract_duration_and_pitch_from_transcript
from srcNonE2E.data.pr_labels import ID_TO_PITCH
from srcNonE2E.data.dr_labels import ID_TO_DURATION


BBox = Tuple[int, int, int, int]


def _enable_gpu_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("GPUs: []")
        return
    print("GPUs:", gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


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


def _gt_tokens_from_transcript(transcript: str) -> List[str]:
    notes = extract_duration_and_pitch_from_transcript(transcript)
    out: List[str] = []
    for _, duration, pitch in notes:
        if duration is None or pitch is None:
            continue
        out.append(f"note.{duration}-{pitch}")
    return out


def levenshtein_tokens(ref: List[str], hyp: List[str]) -> int:
    n = len(ref)
    m = len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n

    if m < n:
        ref, hyp = hyp, ref
        n, m = m, n

    prev = list(range(m + 1))
    cur = [0] * (m + 1)

    for i in range(1, n + 1):
        cur[0] = i
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev, cur = cur, prev

    return prev[m]


def levenshtein_chars(ref: str, hyp: str) -> int:
    n = len(ref)
    m = len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n

    if m < n:
        ref, hyp = hyp, ref
        n, m = m, n

    prev = list(range(m + 1))
    cur = [0] * (m + 1)

    for i in range(1, n + 1):
        cur[0] = i
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev, cur = cur, prev

    return prev[m]


def infer_tokens_for_image(
    img_bgr: np.ndarray,
    regions: List[BBox],
    pr_model: tf.keras.Model,
    dr_model: tf.keras.Model,
    input_h: int,
    input_w: int,
    batch_size: int,
) -> List[str]:
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


def read_split_csv(csv_path: str) -> List[Tuple[str, str]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {csv_path}")
        if "image_path" not in reader.fieldnames or "transcript" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must have columns: image_path, transcript")
        for r in reader:
            rows.append((r["image_path"].strip(), r["transcript"].strip()))
    return rows


def main():
    _enable_gpu_memory_growth()

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="srcNonE2E/data/baseCSV/testNonE2E.csv")
    ap.add_argument("--images_root", type=str, default="data/primus_raw")
    ap.add_argument("--pr_model", type=str, default="artifacts/pr_cnn.keras")
    ap.add_argument("--dr_model", type=str, default="artifacts/dr_cnn.keras")
    ap.add_argument("--h", type=int, default=257)
    ap.add_argument("--w", type=int, default=65)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress_every", type=int, default=100)
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--out_csv", type=str, default="out/full_eval_worst.csv")
    args = ap.parse_args()

    if not os.path.exists(args.pr_model):
        raise FileNotFoundError(f"Missing PR model: {args.pr_model}")
    if not os.path.exists(args.dr_model):
        raise FileNotFoundError(f"Missing DR model: {args.dr_model}")

    pr_model = tf.keras.models.load_model(args.pr_model)
    dr_model = tf.keras.models.load_model(args.dr_model)

    data = read_split_csv(args.csv)
    if args.limit and args.limit > 0:
        data = data[: args.limit]

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

    worst = []

    for rel_path, transcript in data:
        processed += 1

        img_path = os.path.join(args.images_root, rel_path) if args.images_root else rel_path
        gt_tokens = _gt_tokens_from_transcript(transcript)

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
            input_h=args.h,
            input_w=args.w,
            batch_size=args.batch_size,
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

        worst.append(
            (
                norm_sym_ed,
                rel_path,
                len(gt_tokens),
                len(pred_tokens),
                sym_ed,
                char_ed,
                norm_char_ed,
            )
        )

        if args.progress_every and processed % args.progress_every == 0:
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

    print("CSV:", args.csv)
    print("Processed images:", processed)
    print("Evaluated images:", evaluated)
    print("Bad/missing images:", bad_or_missing)
    print("Empty reference sequences skipped:", empty_ref)
    print(f"Sequence Error Rate (SER_seq %): {ser_seq:.6f}")
    print(f"Symbol Error Rate (SER_sym % micro): {ser_sym_micro:.6f}")
    print(f"Symbol Error Rate (SER_sym % macro): {ser_sym_macro:.6f}")
    print(f"Character Error Rate (CER % micro): {cer_micro:.6f}")
    print(f"Character Error Rate (CER % macro): {cer_macro:.6f}")

    worst.sort(key=lambda x: x[0], reverse=True)
    worst = worst[: max(0, args.top_k)]

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rel_path",
                "ref_len",
                "pred_len",
                "sym_edit_distance",
                "norm_sym_edit_distance",
                "char_edit_distance",
                "norm_char_edit_distance",
            ]
        )
        for norm_sym_ed, rel_path, ref_len, pred_len, sym_ed, char_ed, norm_char_ed in worst:
            w.writerow(
                [
                    rel_path,
                    ref_len,
                    pred_len,
                    sym_ed,
                    f"{norm_sym_ed:.6f}",
                    char_ed,
                    f"{norm_char_ed:.6f}",
                ]
            )

    print("Saved worst cases:", args.out_csv)


if __name__ == "__main__":
    main()