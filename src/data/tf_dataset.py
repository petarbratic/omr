#Creates an efficient TensorFlow dataset that loads images and transcripts from a CSV file, 
# converts them into tensors, and encodes tokens into IDs. Then groups them into batches 
# with padding so that the model can use them for training.
from pathlib import Path
import csv
import json
import tensorflow as tf

TRAIN_CSV = Path("data/manifest/train.csv")
PRIMUS_ROOT = Path("data/primus_raw")
VOCAB_PATH = Path("data/vocab/token_to_id.json")

IMG_HEIGHT = 128
BATCH_SIZE = 8


def load_vocab():
    with VOCAB_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_manifest_lists():
    image_paths = []
    transcripts = []
    with TRAIN_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_paths.append(str((PRIMUS_ROOT / row["image_path"]).as_posix()))
            transcripts.append(row["transcript"])
    return image_paths, transcripts


def make_lookup_table(token_to_id: dict):
    keys = tf.constant(list(token_to_id.keys()), dtype=tf.string)
    vals = tf.constant(list(token_to_id.values()), dtype=tf.int32)
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, vals),
        default_value=tf.constant(-1, tf.int32),
    )


def preprocess(path, transcript, table):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_png(img_bytes, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)

    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    scale = IMG_HEIGHT / tf.cast(h, tf.float32)
    new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)

    img = tf.image.resize(img, size=[IMG_HEIGHT, new_w])
    img_width = tf.shape(img)[1]  # sirina PRE paddinga

    tokens = tf.strings.split(transcript)
    label = table.lookup(tokens)
    tf.debugging.assert_greater_equal(label, 0)

    label = tf.cast(label, tf.int32)
    label_length = tf.shape(label)[0]

    return img, label, label_length, img_width


def make_dataset(limit=0, batch_size=BATCH_SIZE, shuffle=True):
    token_to_id = load_vocab()
    table = make_lookup_table(token_to_id)

    paths, transcripts = read_manifest_lists()
    ds = tf.data.Dataset.from_tensor_slices((paths, transcripts))

    if shuffle:
        ds = ds.shuffle(2048, reshuffle_each_iteration=True)

    if limit and limit > 0:
        ds = ds.take(limit)

    ds = ds.map(lambda p, t: preprocess(p, t, table), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.padded_batch(
        batch_size,
        padded_shapes=([IMG_HEIGHT, None, 1], [None], [], []),
        padding_values=(0.0, 0, 0, 0),
    )

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


if __name__ == "__main__":
    ds = make_dataset()
    images, labels, label_lengths = next(iter(ds))

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Label lengths shape:", label_lengths.shape)