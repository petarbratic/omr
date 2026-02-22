import os
import tensorflow as tf

from srcNonE2E.data.pr_dataset import make_pr_dataset
from srcNonE2E.data.pr_labels import NUM_PITCH_CLASSES
from srcNonE2E.models.pitch_model import build_pr_cnn


def _enable_gpu_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("GPUs: []")
        return
    print("GPUs:", gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


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

    os.makedirs("artifacts", exist_ok=True)
    out_path = "artifacts/pr_cnn.keras"
    model.save(out_path)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()