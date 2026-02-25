import os
import tensorflow as tf
import matplotlib.pyplot as plt

from srcNonE2E.data.pr_dataset import make_pr_dataset
from srcNonE2E.data.pr_labels import NUM_PITCH_CLASSES
from srcNonE2E.models.pitch_model import build_pr_cnn
from srcNonE2E.utils.tf_utils import _enable_gpu_memory_growth


def _save_training_plots(history, out_dir: str, prefix: str):
    os.makedirs(out_dir, exist_ok=True)

    train_acc = history.history.get("acc", history.history.get("accuracy", []))
    val_acc = history.history.get("val_acc", history.history.get("val_accuracy", []))

    plt.figure()
    if train_acc:
        plt.plot(train_acc)
    if val_acc:
        plt.plot(val_acc)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "val"])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_acc.png"), dpi=200)
    plt.close()

    train_loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    plt.figure()
    if train_loss:
        plt.plot(train_loss)
    if val_loss:
        plt.plot(val_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "val"])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_loss.png"), dpi=200)
    plt.close()


def main():
    _enable_gpu_memory_growth()

    train_csv = "out/region_dataset/pr-train.csv"
    val_csv = "out/region_dataset/pr-val.csv"
    images_root = "data/primus_raw"

    input_shape = (257, 65, 1)
    batch_size = 64
    epochs = 7
    lr = 1e-4

    train_ds = make_pr_dataset(
        csv_path=train_csv,
        images_root=images_root,
        input_shape=input_shape,
        batch_size=batch_size,
        shuffle=True,
        shuffle_buffer=8192,
    )

    val_ds = make_pr_dataset(
        csv_path=val_csv,
        images_root=images_root,
        input_shape=input_shape,
        batch_size=batch_size,
        shuffle=False,
    )

    model = build_pr_cnn(num_classes=NUM_PITCH_CLASSES, input_shape=input_shape)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_acc",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_acc",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    _save_training_plots(history, out_dir="out/plots", prefix="pr")

    os.makedirs("artifacts", exist_ok=True)
    out_path = "artifacts/pr_cnn_delete_later.keras"
    model.save(out_path)
    print("Saved:", out_path)
    print("Plots:", "out/plots/pr_acc.png", "out/plots/pr_loss.png")


if __name__ == "__main__":
    main()