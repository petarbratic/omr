# src/debug_train_stream.py
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
        tf.math.ceil(tf.cast(img_widths, tf.float32) * tf.cast(T, tf.float32) / tf.cast(Wmax, tf.float32)),
        tf.int32,
    )
    return tf.clip_by_value(input_lengths, 1, T)

def ctc_mean_loss(images, logits, labels, label_lengths, img_widths, blank_index):
    tf.debugging.assert_all_finite(images, "images contain NaN/Inf")
    tf.debugging.assert_all_finite(logits, "logits contain NaN/Inf")

    input_lengths = compute_input_lengths(images, logits, img_widths)

    tf.debugging.assert_less_equal(
        tf.cast(label_lengths, tf.int32),
        tf.cast(input_lengths, tf.int32),
        message="CTC invalid: label_length > input_length",
    )

    labels_ctc = dense_labels_to_sparse(labels, label_lengths)
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
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clipnorm", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()

    token_to_id = json.loads(open(VOCAB_PATH, "r", encoding="utf-8").read())
    num_classes = len(token_to_id) + 1
    blank_index = num_classes - 1

    ds = make_dataset(limit=args.limit, batch_size=args.batch_size, shuffle=args.shuffle)

    model = build_crnn(num_classes=num_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, clipnorm=args.clipnorm)

    assert_vars_finite(model)

    step = 0
    for batch in ds:
        step += 1
        images, labels, label_lengths, img_widths = batch

        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            loss, input_lengths = ctc_mean_loss(
                images, logits, labels, label_lengths, img_widths, blank_index
            )

        grads = tape.gradient(loss, model.trainable_variables)

        # grad check
        finite_grads = []
        for g in grads:
            if g is None:
                continue
            tf.debugging.assert_all_finite(g, "gradient contains NaN/Inf")
            finite_grads.append(g)
        gn = float(tf.linalg.global_norm(finite_grads).numpy()) if finite_grads else 0.0

        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        assert_vars_finite(model)

        print(
            f"step {step:03d} loss {float(loss.numpy()):.4f} grad_norm {gn:.4f} "
            f"label_len[max]={int(tf.reduce_max(label_lengths).numpy())} "
            f"input_len[min]={int(tf.reduce_min(input_lengths).numpy())}"
        )

        if step >= args.max_steps:
            break

    print("OK: finished stream steps without NaNs.")

if __name__ == "__main__":
    main()