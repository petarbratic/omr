import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import json
from pathlib import Path

import tensorflow as tf

from srcE2E.data.tf_dataset import make_dataset
from srcE2E.model.crnn import build_crnn

VOCAB_PATH = Path("data/vocab/token_to_id.json")


def setup_gpu_print():
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs:", gpus if gpus else "None")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def dense_labels_to_sparse(labels, label_lengths):
    labels = tf.convert_to_tensor(labels)
    label_lengths = tf.convert_to_tensor(label_lengths)

    max_len = tf.shape(labels)[1]
    mask = tf.sequence_mask(label_lengths, maxlen=max_len)

    indices = tf.where(mask)  # (N, 2) -> (batch_idx, time_idx)
    values = tf.gather_nd(labels, indices)

    dense_shape = tf.cast(tf.shape(labels), tf.int64)

    st = tf.SparseTensor(
        indices=tf.cast(indices, tf.int64),
        values=tf.cast(values, tf.int32),
        dense_shape=dense_shape,
    )

    return tf.sparse.reorder(st)


def ctc_mean_loss(images, logits, labels, label_lengths, img_widths, blank_index):
    T = tf.shape(logits)[1]
    Wmax = tf.shape(images)[2]

    input_lengths = tf.cast(
        tf.math.ceil(tf.cast(img_widths, tf.float32) * tf.cast(T, tf.float32) / tf.cast(Wmax, tf.float32)),
        tf.int32,
    )
    input_lengths = tf.clip_by_value(input_lengths, 1, T)

    # --- DEBUG CHECKS ---
    tf.debugging.assert_all_finite(images, "images contain NaN/Inf")
    tf.debugging.assert_all_finite(logits, "logits contain NaN/Inf")

    tf.debugging.assert_less_equal(
        tf.cast(label_lengths, tf.int32),
        tf.cast(input_lengths, tf.int32),
        message="CTC invalid: label_length > input_length"
    )

    if not isinstance(labels, tf.SparseTensor):
        labels_ctc = dense_labels_to_sparse(labels, label_lengths)
    else:
        labels_ctc = labels

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
    return loss


def main():
    t0 = time.time()

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--save_dir", type=str, default="artifacts/crnn_ctc_train.keras")
    args = ap.parse_args()

    setup_gpu_print()

    with VOCAB_PATH.open("r", encoding="utf-8") as f:
        token_to_id = json.load(f)

    vocab_size = len(token_to_id)
    num_classes = vocab_size + 1
    blank_index = num_classes - 1

    ds = make_dataset(limit=args.limit, batch_size=args.batch_size, shuffle=True)

    model = build_crnn(num_classes=num_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, clipnorm=0.5)

    # --- Reduce LR on Plateau ---
    best = float("inf")
    patience = 50
    wait = 0
    factor = 0.5
    min_lr = 1e-6
    eps = 1e-5

    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        epoch_gns = []
        seen = 0

        for images, labels, label_lengths, img_widths in ds:
            seen += int(images.shape[0])

            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                loss = ctc_mean_loss(
                    images=images,
                    logits=logits,
                    labels=labels,
                    label_lengths=label_lengths,
                    img_widths=img_widths,
                    blank_index=blank_index,
                )

            grads = tape.gradient(loss, model.trainable_variables)

            finite_grads = []
            for g in grads:
                if g is None:
                    continue
                tf.debugging.assert_all_finite(g, "gradient contains NaN/Inf")
                finite_grads.append(g)

            gn = tf.linalg.global_norm(finite_grads) if finite_grads else tf.constant(0.0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            for v in model.trainable_variables:
                tf.debugging.assert_all_finite(v, f"variable became NaN/Inf: {v.name}")

            epoch_losses.append(float(loss.numpy()))
            epoch_gns.append(float(gn.numpy()))

        # Prosek po epohi
        cur = sum(epoch_losses) / max(1, len(epoch_losses))
        avg_gn = sum(epoch_gns) / max(1, len(epoch_gns))

        # Reduce LR on Plateau
        if cur + eps < best:
            best = cur
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                old_lr = float(optimizer.learning_rate.numpy())
                new_lr = max(old_lr * factor, min_lr)
                optimizer.learning_rate.assign(new_lr)
                wait = 0
                print(f"LR reduced: {old_lr:.6g} -> {new_lr:.6g} (best {best:.4f})")

        lr_now = float(optimizer.learning_rate.numpy())
        print(
            f"epoch {epoch}/{args.epochs} "
            f"loss {cur:.4f} grad_norm {avg_gn:.4f} lr {lr_now:.6g} seen {seen}"
        )

    t1 = time.time()
    print(f"Training time: {t1 - t0:.2f} s")

    save_path = Path(args.save_dir)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print("Saved model to:", save_path)


if __name__ == "__main__":
    main()