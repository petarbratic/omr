import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import tensorflow as tf

from src.data.tf_dataset import make_dataset
from src.model.crnn import build_crnn

VOCAB_PATH = "data/vocab/token_to_id.json"


def main():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        token_to_id = json.load(f)

    vocab_size = len(token_to_id)
    num_classes = vocab_size + 1  # +1 za CTC blank

    # uzmi 1 batch deterministički
    ds = make_dataset(limit=8, batch_size=8, shuffle=False)
    images, labels, label_lengths, img_widths = next(iter(ds))

    model = build_crnn(num_classes=num_classes)

    print("Images:", images.shape, images.dtype,
          "min/max:", float(tf.reduce_min(images).numpy()), float(tf.reduce_max(images).numpy()))
    print("Labels:", labels.shape, labels.dtype)
    print("Label lengths:", label_lengths.shape, label_lengths.dtype)
    print("Img widths:", img_widths.shape, img_widths.dtype)

    # uradi više forward prolaza i proveri numeriku
    for i in range(1, 101):
        logits = model(images, training=False)
        tf.debugging.assert_all_finite(logits, "logits contain NaN/Inf in forward-only")

        if i == 1:
            print("Logits:", logits.shape, logits.dtype)  # (B, T, C)
            print("T (time steps):", logits.shape[1])
            print("C (classes):", logits.shape[2])

        if i % 10 == 0:
            mx = float(tf.reduce_max(tf.abs(logits)).numpy())
            print(f"forward {i}/100 ok | max|logits|={mx:.4f}")


if __name__ == "__main__":
    main()