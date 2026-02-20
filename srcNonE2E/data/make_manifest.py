# This script filters out samples with "gracenote." and "dot-" in the transcript
# from the previous manifest CSV files.
import csv

# ====== DEFINES ======
TRAIN_IN = "data/manifest/train.csv"
VAL_IN = "data/manifest/val.csv"
TEST_IN = "data/manifest/test.csv"

TRAIN_OUT = "srcNonE2E/data/trainNonE2E.csv"
VAL_OUT = "srcNonE2E/data/valNonE2E.csv"
TEST_OUT = "srcNonE2E/data/testNonE2E.csv"
# =====================


def filter_file(input_path, output_path):
    kept = 0
    removed_grace = 0
    removed_dot = 0

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8", newline="") as f_out:

        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)

        writer.writeheader()

        for row in reader:
            transcript = row["transcript"]

            if "gracenote." in transcript:
                removed_grace += 1
                continue

            if "dot-" in transcript:
                removed_dot += 1
                continue

            writer.writerow(row)
            kept += 1

    print(f"{input_path} -> {output_path}")
    print(f"  Kept: {kept}")
    print(f"  Removed (gracenote): {removed_grace}")
    print(f"  Removed (dot): {removed_dot}")
    print()


def main():
    filter_file(TRAIN_IN, TRAIN_OUT)
    filter_file(VAL_IN, VAL_OUT)
    filter_file(TEST_IN, TEST_OUT)


if __name__ == "__main__":
    main()