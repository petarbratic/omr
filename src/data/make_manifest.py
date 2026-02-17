# One-time script to create a manifest file for the dataset
# The manifest file will contain the paths to the images and their corresponding labels
# Creates train.csv, val.csv, and test.csv in the data/manifest directory

from pathlib import Path
import random
import csv

PRIMUS_ROOT = Path("data/primus_raw")
OUT_DIR = Path("data/manifest")
PACKAGES = ["package_aa", "package_ab"]

TRAIN_RATIO = 0.9
VAL_RATIO = 0.05
SEED = 123

# Helper funcition for collect_pairs
def normalize_ws(s: str) -> str:
    return " ".join(s.strip().split())


def collect_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    for pkg in PACKAGES:
        pkg_dir = PRIMUS_ROOT / pkg
        # Rglob recursively searches for all .png files in the all subdirectories of the package
        for img_path in pkg_dir.rglob("*.png"):
            # Skip files that start with "._" (these are often created by macOS and I will ignore them)
            if img_path.name.startswith("._"):
                continue
            
            # The corresponding ground truth file has the same name but with .agnostic extension
            gt_path = img_path.with_suffix(".agnostic")
            if not gt_path.exists():
                continue

            # Delete leading/trailing whitespace and replace multiple whitespace with a single space
            transcript = normalize_ws(gt_path.read_text(encoding="utf-8", errors="ignore"))
            if not transcript:
                continue

            # Store the relative path to the image (relative to PRIMUS_ROOT) and the transcript
            rel_img = img_path.relative_to(PRIMUS_ROOT).as_posix()
            pairs.append((rel_img, transcript))

    return pairs


def split(pairs: list[tuple[str, str]]):
    rnd = random.Random(SEED)
    rnd.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train = pairs[:n_train]
    val = pairs[n_train:n_train + n_val]
    test = pairs[n_train + n_val:]
    return train, val, test


def write_csv(path: Path, rows: list[tuple[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "transcript"])
        w.writerows(rows)


def main():
    pairs = collect_pairs()
    if not pairs:
        raise RuntimeError("Nema parova (png + .agnostic).")

    train, val, test = split(pairs)

    write_csv(OUT_DIR / "train.csv", train)
    write_csv(OUT_DIR / "val.csv", val)
    write_csv(OUT_DIR / "test.csv", test)

    print("Ukupno:", len(pairs))
    print("Train:", len(train))
    print("Val:", len(val))
    print("Test:", len(test))


if __name__ == "__main__":
    main()