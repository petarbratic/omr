from pathlib import Path
import csv
import json

TRAIN_CSV = Path("data/manifest/train.csv")
VOCAB_DIR = Path("data/vocab")


def main():
    tokens = set()

    with TRAIN_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            transcript = row["transcript"]
            for token in transcript.split():
                tokens.add(token)

    tokens = sorted(tokens)

    token_to_id = {token: i for i, token in enumerate(tokens)}
    id_to_token = {i: token for token, i in token_to_id.items()}

    VOCAB_DIR.mkdir(parents=True, exist_ok=True)

    with (VOCAB_DIR / "token_to_id.json").open("w", encoding="utf-8") as f:
        json.dump(token_to_id, f, indent=2)

    with (VOCAB_DIR / "id_to_token.json").open("w", encoding="utf-8") as f:
        json.dump(id_to_token, f, indent=2)

    print("Broj tokena:", len(tokens))


if __name__ == "__main__":
    main()