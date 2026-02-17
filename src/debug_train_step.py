# src/debug_train_step.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import argparse
import tensorflow as tf

from src.data.tf_dataset import make_dataset
from src.model.crnn import build_crnn

VOCAB_PATH = "data/vocab/token_to_id.json"


def dense_labels_to_sparse(labels, label_lengths):
    labels = tf.convert_to_tensor(labels)
    label_lengths = tf.convert_to_tensor(label_lengths)

    max_len = tf.shape(labels)[1]
    mask = tf.sequence_mask(label_lengths, maxlen=max_len)

    indices = tf.where(mask)
    values = tf.gather_nd(labels, indices)

    st = tf.SparseTensor(
        indices=tf.cast(indices, tf.int64),
        values=tf.cast(values, tf.int32),
        dense_shape=tf.cast(tf.shape(labels), tf.int64),
    )
    return tf.sparse.reorder(st)


def compute_input_lengths(images, logits, img_widths):
    T = tf.shape(logits)[1]
    Wmax = tf.shape(images)[2]

    input_lengths = tf.cast(
        tf.math.ceil(
            tf.cast(img_widths, tf.float32) * tf.cast(T, tf.float32) / tf.cast(Wmax, tf.float32)
        ),
        tf.int32,
    )
    input_lengths = tf.clip_by_value(input_lengths, 1, T)
    return input_lengths


def ctc_mean_loss(images, logits, labels, label_lengths, img_widths, blank_index):
    tf.debugging.assert_all_finite(images, "images contain NaN/Inf")
    tf.debugging.assert_all_finite(logits, "logits contain NaN/Inf")

    input_lengths = compute_input_lengths(images, logits, img_widths)

    tf.debugging.assert_less_equal(
        tf.cast(label_lengths, tf.int32),
        tf.cast(input_lengths, tf.int32),
        message="CTC invalid: label_length > input_length",
    )

    labels_ctc = labels if isinstance(labels, tf.SparseTensor) else dense_labels_to_sparse(labels, label_lengths)

    loss_per_example = tf.nn.ctc_loss(
        labels=labels_ctc,
        logits=logits,
        label_length=tf.cast(label_lengths, tf.int32),
        logit_length=input_lengths,
        logits_time_major=False,
        blank_index=blank_index,
    )
    loss = tf.reduce_mean(loss_per_example)
    tf.debugging.assert_all_finite(loss, "loss became NaN/Inf")
    return loss, input_lengths


def assert_vars_finite(model):
    for v in model.trainable_variables:
        tf.debugging.assert_all_finite(v, f"variable became NaN/Inf: {v.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clipnorm", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--shuffle", action="store_true", help="ako proslediš, shuffle=True; inače shuffle=False")
    args = ap.parse_args()

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        token_to_id = json.load(f)

    vocab_size = len(token_to_id)
    num_classes = vocab_size + 1
    blank_index = num_classes - 1

    ds = make_dataset(limit=args.limit, batch_size=args.batch_size, shuffle=args.shuffle)

    # uzmi jedan batch i vrti samo njega (najstroži test stabilnosti)
    images, labels, label_lengths, img_widths = next(iter(ds))

    print("images:", images.shape, images.dtype,
          "min/max:", float(tf.reduce_min(images).numpy()), float(tf.reduce_max(images).numpy()))
    print("labels:", labels.shape, labels.dtype)
    print("label_lengths min/max:",
          int(tf.reduce_min(label_lengths).numpy()), int(tf.reduce_max(label_lengths).numpy()))
    print("img_widths min/max:",
          int(tf.reduce_min(img_widths).numpy()), int(tf.reduce_max(img_widths).numpy()))

    model = build_crnn(num_classes=num_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, clipnorm=args.clipnorm)

    # proveri da su početne težine ok
    assert_vars_finite(model)

    for step in range(1, args.steps + 1):
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            loss, input_lengths = ctc_mean_loss(
                images=images,
                logits=logits,
                labels=labels,
                label_lengths=label_lengths,
                img_widths=img_widths,
                blank_index=blank_index,
            )

        grads = tape.gradient(loss, model.trainable_variables)

        # proveri gradijente
        finite_grads = []
        for g in grads:
            if g is None:
                continue
            tf.debugging.assert_all_finite(g, "gradient contains NaN/Inf")
            finite_grads.append(g)

        gn = float(tf.linalg.global_norm(finite_grads).numpy()) if finite_grads else 0.0

        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # proveri težine posle update-a
        assert_vars_finite(model)

        if step == 1 or step % 5 == 0:
            print(
                f"step {step:03d} loss {float(loss.numpy()):.4f} grad_norm {gn:.4f} "
                f"input_len[min,max]=({int(tf.reduce_min(input_lengths).numpy())},{int(tf.reduce_max(input_lengths).numpy())})"
            )

    print("OK: finished without NaNs.")


if __name__ == "__main__":
    main()