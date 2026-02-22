import tensorflow as tf
from srcNonE2E.data.dr_labels import DURATION_TO_ID


def _parse_csv_line(line: tf.Tensor):
    # CSV: image_path,x1,y1,x2,y2,idx,duration
    fields = tf.io.decode_csv(
        line,
        record_defaults=[[""], [0], [0], [0], [0], [0], [""]],
    )
    image_path, x1, y1, x2, y2, idx, duration = fields
    return image_path, x1, y1, x2, y2, duration


def _duration_to_id(duration: tf.Tensor) -> tf.Tensor:
    keys = tf.constant(list(DURATION_TO_ID.keys()), dtype=tf.string)
    vals = tf.constant(list(DURATION_TO_ID.values()), dtype=tf.int32)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, vals),
        default_value=-1,
    )
    return table.lookup(duration)


def make_dr_dataset(
    csv_path: str,
    images_root: str,
    input_shape=(257, 65, 1),
    batch_size: int = 64,
    shuffle: bool = False,
    shuffle_buffer: int = 8192,
):
    h, w, c = input_shape

    ds = tf.data.TextLineDataset(csv_path).skip(1)
    ds = ds.map(_parse_csv_line, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

    def load_crop(image_path, x1, y1, x2, y2, duration):
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

        label = _duration_to_id(duration)
        return img, label

    ds = ds.map(load_crop, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(lambda x, y: y >= 0)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds