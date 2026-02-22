import os
import tensorflow as tf

from srcNonE2E.data.dr_dataset import make_dr_dataset
from srcNonE2E.data.dr_labels import NUM_DURATION_CLASSES
from srcNonE2E.models.duration_model import build_dr_cnn


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

    train_csv = "out/region_dataset/dr-train.csv"
    val_csv = "out/region_dataset/dr-val.csv"
    images_root = "data/primus_raw"

    input_shape = (257, 65, 1)
    batch_size = 64
    epochs = 7
    lr = 1e-3

    train_ds = make_dr_dataset(
        csv_path=train_csv,
        images_root=images_root,
        input_shape=input_shape,
        batch_size=batch_size,
        shuffle=True,
        shuffle_buffer=8192,
    )

    val_ds = make_dr_dataset(
        csv_path=val_csv,
        images_root=images_root,
        input_shape=input_shape,
        batch_size=batch_size,
        shuffle=False,
    )

    model = build_dr_cnn(num_classes=NUM_DURATION_CLASSES, input_shape=input_shape)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_acc",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_acc",
            factor=0.5,
            patience=1,
            min_lr=1e-6,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    os.makedirs("artifacts", exist_ok=True)
    out_path = "artifacts/dr_cnn.keras"
    model.save(out_path)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()