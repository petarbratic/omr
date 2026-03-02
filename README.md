# Non End-to-End Optical Music Recognition (OMR)

Course project for **Soft Computing**, Faculty of Technical Sciences, Novi Sad.

The system performs:
- **Segmentation** of individual note heads (template matching),
- **Pitch classification** (PR) and **duration classification** (DR) via separate CNNs,
- **Reconstruction** of the full note sequence from an image.

---

## Requirements

- **Python** 3.9+
- **Packages:** `pip install -r requirements.txt`  
  (TensorFlow, NumPy, OpenCV, Matplotlib, Pillow)
- **GPU** (optional): recommended for training; CPU is enough for evaluation and inference.

---

## Data (not in this repository)

The **PrIMuS** dataset is not included (see `.gitignore`: `data/primus_raw/`).

1. **Download PrIMuS** from the official source.
2. **Unpack** so that images lie under:
   ```
   data/primus_raw/
     package_aa/
     package_ab/
     ...
   ```
3. **Splits:** CSV files (`data/splits/trainNonE2E.csv`, `valNonE2E.csv`, `testNonE2E.csv`) are in the repo; each row has `image_path` (relative to `data/primus_raw`) and `transcript`.

---

## Trained models (local, not in repo)

Trained PR and DR models are **not** committed to the repository (see `.gitignore`: `artifacts/` is ignored).

By default the code expects:

- `artifacts/pr_cnn.keras` — Pitch Recognition model  
- `artifacts/dr_cnn.keras` — Duration Recognition model  

**Train the models yourself** following the *Training* section below (recommended for reproducing results).  

---

## Repository structure

- **`srcNonE2E/`**
  - `data/` — preprocessing, region extraction, datasets, labels
  - `models/` — PR and DR CNN definitions
  - `eval_helpers/` — metrics (SER/CER), geometry, inference, I/O
  - `train_pr_dr.py` — training script
  - `eval_pr_dr.py` — PR/DR classifier evaluation
  - `eval_full_images.py` — full-system evaluation (SER, CER)
  - `infer_non_e2e.py` — inference on a single image
  - `utils/` — TensorFlow utilities
- **`scripts/`** — e.g. `debug_region_extractor.py`
- **`artifacts/`** — trained models (included)
- **`Poster.pdf`** — project poster

---

## Data preprocessing

From the split CSVs, build region-level PR/DR datasets:

```bash
python -m srcNonE2E.data.preprocess_dataset
```

This writes `out/pr-train.csv`, `out/dr-train.csv`, etc. (paths and limits are set in `preprocess_dataset.py`).

Optional: if you start from raw manifest CSVs, run `python -m srcNonE2E.data.make_manifest` first.

---

## Training

If you want to obtain your own models (or reproduce the reported results), train PR and DR as follows:

**Pitch Recognition:**
```bash
python -m srcNonE2E.train_pr_dr --task pr
```
Saves `artifacts/pr_cnn.keras`.

**Duration Recognition:**
```bash
python -m srcNonE2E.train_pr_dr --task dr
```
Saves `artifacts/dr_cnn.keras`.

Requires preprocessed CSVs in `out/` (see Data preprocessing).

---

## Evaluation

**PR/DR classifiers** (accuracy, per-class stats, confusion matrix):

```bash
python -m srcNonE2E.eval_pr_dr --task pr
python -m srcNonE2E.eval_pr_dr --task dr
```

**Full system** (SER, CER on test images):

```bash
python -m srcNonE2E.eval_full_images
```

Set `CSV_PATH`, `IMAGES_ROOT`, `PR_MODEL_PATH`, `DR_MODEL_PATH` in the script if your paths differ. You need PrIMuS images under `data/primus_raw/`; models are in `artifacts/`.

---

## Inference on one image

1. In `srcNonE2E/infer_non_e2e.py` set:
   - `IMAGE_PATH` — path to the score image
   - `PR_MODEL_PATH`, `DR_MODEL_PATH` (default: `artifacts/...`)
   - `OUT_TXT` (optional) — path to save the transcript

2. Run:
```bash
python -m srcNonE2E.infer_non_e2e
```

Output: number of regions and the predicted token sequence (`note.<duration>-<pitch>`).

---

## Poster

The project poster is provided as **`Poster.pdf`**. It summarizes the problem, dataset, architecture, and evaluation results.

---

## Note

This project was developed using **ChatGPT** and **Cursor**; this is stated in the first line of each Python file.
