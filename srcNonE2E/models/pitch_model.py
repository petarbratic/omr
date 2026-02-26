# I used ChatGPT and Cursor for the development of this project.
import tensorflow as tf


def build_pr_cnn(num_classes: int, input_shape=(257, 65, 1)) -> tf.keras.Model:
    inp = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(8, 3, padding="same")(inp)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D((2, 2), padding="same")(x)

    x = tf.keras.layers.Conv2D(16, 3, padding="same")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D((2, 2), padding="same")(x)

    x = tf.keras.layers.Conv2D(32, 3, padding="same")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D((2, 2), padding="same")(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D((2, 2), padding="same")(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D((2, 2), padding="same")(x)

    x = tf.keras.layers.Conv2D(256, 3, padding="same")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D((2, 2), padding="same")(x)

    x = tf.keras.layers.Conv2D(512, 3, padding="same")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D((2, 2), padding="same")(x)

    x = tf.keras.layers.Conv2D(1024, 3, padding="same")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D((2, 2), padding="same")(x)

    x = tf.keras.layers.Conv2D(2048, 3, padding="same")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D((2, 2), padding="same")(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.Dense(16)(x)

    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inp, out)