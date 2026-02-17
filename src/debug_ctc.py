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
    num_classes = vocab_size + 1  # blank

    ds = make_dataset()
    images, labels, label_lengths = next(iter(ds))

    model = build_crnn(num_classes=num_classes)
    logits = model(images, training=False)  # (B, T, C)

    B = tf.shape(logits)[0]
    T = tf.shape(logits)[1]

    # svi imaju isti T (posle padding-a)
    input_lengths = tf.fill([B], T)

    # CTC u TF očekuje: labels int32, lengths int32/int64
    labels = tf.cast(labels, tf.int32)
    label_lengths = tf.cast(label_lengths, tf.int32)

    # tf.nn.ctc_loss očekuje logits time-major ako logits_time_major=True
    # Nama je logits (B, T, C), pa koristimo logits_time_major=False
    loss_per_example = tf.nn.ctc_loss(
        labels=labels,
        logits=logits,
        label_length=label_lengths,
        logit_length=input_lengths,
        logits_time_major=False,
        blank_index=num_classes - 1,
    )

    loss = tf.reduce_mean(loss_per_example)

    print("Logits shape:", logits.shape)
    print("Labels shape:", labels.shape)
    print("Mean CTC loss:", float(loss.numpy()))


if __name__ == "__main__":
    main()