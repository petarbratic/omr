# I used ChatGPT and Cursor for the development of this project.
# Helpers for reading CSV split files (image_path, transcript).

import csv
from typing import List, Tuple


def read_split_csv(csv_path: str) -> List[Tuple[str, str]]:
    # Read CSV with columns image_path and transcript; return list of (rel_path, transcript).
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
