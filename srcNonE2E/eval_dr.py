import numpy as np
import tensorflow as tf

from srcNonE2E.data.dr_dataset import make_dr_dataset
from srcNonE2E.data.dr_labels import ID_TO_DURATION, NUM_DURATION_CLASSES
from srcNonE2E.utils.tf_utils import _enable_gpu_memory_growth


def main():
    _enable_gpu_memory_growth()

    model_path = "artifacts/dr_cnn.keras"
    test_csv = "out/region_dataset/dr-test.csv"
    images_root = "data/primus_raw"

    input_shape = (257, 65, 1)
    batch_size = 64

    ds = make_dr_dataset(
        csv_path=test_csv,
        images_root=images_root,
        input_shape=input_shape,
        batch_size=batch_size,
        shuffle=False,
    )

    model = tf.keras.models.load_model(model_path)

    n_classes = NUM_DURATION_CLASSES
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)

    total = 0
    correct = 0

    per_total = np.zeros((n_classes,), dtype=np.int64)
    per_correct = np.zeros((n_classes,), dtype=np.int64)

    for xb, yb in ds:
        logits = model(xb, training=False)
        pred = tf.argmax(logits, axis=-1, output_type=tf.int32)

        yb_i = tf.cast(yb, tf.int32)

        total += int(tf.size(yb_i))
        correct += int(tf.reduce_sum(tf.cast(tf.equal(pred, yb_i), tf.int64)))

        y_np = yb_i.numpy()
        p_np = pred.numpy()

        per_total += np.bincount(y_np, minlength=n_classes)
        per_correct += np.bincount(y_np[y_np == p_np], minlength=n_classes)

        cm = tf.math.confusion_matrix(
            yb_i, pred, num_classes=n_classes, dtype=tf.int64
        ).numpy()
        conf += cm

    acc = correct / max(1, total)

    print("Model:", model_path)
    print("Test CSV:", test_csv)
    print(f"Total samples: {total}")
    print(f"Accuracy: {acc:.6f}")
    print()

    print("Per-class:")
    for cid in range(n_classes):
        name = ID_TO_DURATION.get(cid, str(cid))
        t = int(per_total[cid])
        c = int(per_correct[cid])
        a = (c / t) if t > 0 else 0.0
        print(f"{cid:2d} {name:>14s}  n={t:6d}  correct={c:6d}  acc={a:.6f}")

    print()
    print("Confusion matrix (rows=true, cols=pred):")
    np.set_printoptions(linewidth=200)
    print(conf)


if __name__ == "__main__":
    main()