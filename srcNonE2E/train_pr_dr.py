"""
python -m srcNonE2E.train_pr_dr --task pr
python -m srcNonE2E.train_pr_dr --task dr
"""

import os
import argparse

import tensorflow as tf

from srcNonE2E.data.pr_dr_dataset import make_pr_dataset, make_dr_dataset
from srcNonE2E.data.labels import NUM_PITCH_CLASSES, NUM_DURATION_CLASSES
from srcNonE2E.models.pitch_model import build_pr_cnn
from srcNonE2E.models.duration_model import build_dr_cnn
from srcNonE2E.utils.tf_utils import _enable_gpu_memory_growth


def _build_pr_config() -> dict:
    # Configuration for PR (pitch recognition) task.
    return {
        "train_csv": "out/region_dataset/pr-train.csv",
        "val_csv": "out/region_dataset/pr-val.csv",
        "images_root": "data/primus_raw",
        "input_shape": (257, 65, 1),
        "batch_size": 64,
        "epochs": 7,
        "lr": 1e-4,
        "num_classes": NUM_PITCH_CLASSES,
        "build_model": build_pr_cnn,
        "artifact_path": "artifacts/pr_cnn.keras",
        "early_stopping_patience": 5,
        "reduce_lr_patience": 2,
    }


def _build_dr_config() -> dict:
    # Configuration for DR (duration recognition) task.
    return {
        "train_csv": "out/region_dataset/dr-train.csv",
        "val_csv": "out/region_dataset/dr-val.csv",
        "images_root": "data/primus_raw",
        "input_shape": (257, 65, 1),
        "batch_size": 64,
        "epochs": 7,
        "lr": 1e-3,
        "num_classes": NUM_DURATION_CLASSES,
        "build_model": build_dr_cnn,
        "artifact_path": "artifacts/dr_cnn.keras",
        "early_stopping_patience": 3,
        "reduce_lr_patience": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        choices=["pr", "dr"],
        required=True
    )

    args = parser.parse_args()

    _enable_gpu_memory_growth()

    if args.task == "pr":
        cfg = _build_pr_config()
        make_ds = make_pr_dataset
        task_name = "PR (pitch)"
    else:
        cfg = _build_dr_config()
        make_ds = make_dr_dataset
        task_name = "DR (duration)"

    print(f"Task: {task_name}")
    print("Konfiguracija:")
    for k, v in cfg.items():
        if k in {"build_model"}:
            continue
        print(f"  {k}: {v}")

    # Datasets
    train_ds = make_ds(
        csv_path=cfg["train_csv"],
        images_root=cfg["images_root"],
        input_shape=cfg["input_shape"],
        batch_size=cfg["batch_size"],
        shuffle=True,
        shuffle_buffer=8192,
    )

    val_ds = make_ds(
        csv_path=cfg["val_csv"],
        images_root=cfg["images_root"],
        input_shape=cfg["input_shape"],
        batch_size=cfg["batch_size"],
        shuffle=False,
    )

    # Model
    model = cfg["build_model"](num_classes=cfg["num_classes"], input_shape=cfg["input_shape"])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["lr"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"],
    )

    os.makedirs(os.path.dirname(cfg["artifact_path"]) or ".", exist_ok=True)
    model.save(cfg["artifact_path"])
    print("Saved model:", cfg["artifact_path"])


if __name__ == "__main__":
    main()