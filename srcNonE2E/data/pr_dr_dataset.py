"""
Single dataset implementation for region crops with a string label (pitch or duration).
Use make_pr_dataset() for pitch recognition, make_dr_dataset() for duration recognition.
"""

import tensorflow as tf

from srcNonE2E.data.labels import PITCH_TO_ID, DURATION_TO_ID


def _parse_csv_line(line: tf.Tensor):
    # CSV: image_path,x1,y1,x2,y2,idx,label (pitch or duration)
    fields = tf.io.decode_csv(
        line,
        record_defaults=[[""], [0], [0], [0], [0], [0], [""]],
    )
    image_path, x1, y1, x2, y2, idx, label = fields
    return image_path, x1, y1, x2, y2, label


def _label_to_id_table(label_to_id: dict) -> tf.lookup.StaticHashTable:
    keys = tf.constant(list(label_to_id.keys()), dtype=tf.string)
    vals = tf.constant(list(label_to_id.values()), dtype=tf.int32)
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, vals),
        default_value=-1,
    )


def _make_region_dataset(
    csv_path: str,
    images_root: str,
    label_to_id: dict,
    input_shape=(257, 65, 1),
    batch_size: int = 64,
    shuffle: bool = False,
    shuffle_buffer: int = 8192,
):
    h, w, c = input_shape
    table = _label_to_id_table(label_to_id)

    ds = tf.data.TextLineDataset(csv_path).skip(1)
    ds = ds.map(_parse_csv_line, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

    def load_crop(image_path, x1, y1, x2, y2, label):
        full_path = tf.strings.join([images_root, "/", image_path])
        img_bytes = tf.io.read_file(full_path)
        img = tf.io.decode_png(img_bytes, channels=1)

        x1i = tf.cast(x1, tf.int32)
        y1i = tf.cast(y1, tf.int32)
        x2i = tf.cast(x2, tf.int32)
        y2i = tf.cast(y2, tf.int32)

        crop_h = tf.maximum(1, y2i - y1i)
        crop_w = tf.maximum(1, x2i - x1i)

        img = tf.image.crop_to_bounding_box(img, y1i, x1i, crop_h, crop_w)
        img = tf.image.resize(img, (h, w), method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0

        label_id = table.lookup(label)
        return img, label_id

    ds = ds.map(load_crop, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(lambda x, y: y >= 0)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_pr_dataset(
    csv_path: str,
    images_root: str,
    input_shape=(257, 65, 1),
    batch_size: int = 32,
    shuffle: bool = False,
    shuffle_buffer: int = 8192,
):
    return _make_region_dataset(
        csv_path=csv_path,
        images_root=images_root,
        label_to_id=PITCH_TO_ID,
        input_shape=input_shape,
        batch_size=batch_size,
        shuffle=shuffle,
        shuffle_buffer=shuffle_buffer,
    )


def make_dr_dataset(
    csv_path: str,
    images_root: str,
    input_shape=(257, 65, 1),
    batch_size: int = 64,
    shuffle: bool = False,
    shuffle_buffer: int = 8192,
):
    return _make_region_dataset(
        csv_path=csv_path,
        images_root=images_root,
        label_to_id=DURATION_TO_ID,
        input_shape=input_shape,
        batch_size=batch_size,
        shuffle=shuffle,
        shuffle_buffer=shuffle_buffer,
    )
