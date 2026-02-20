import tensorflow as tf


def build_crnn(num_classes: int) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(128, None, 1))

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)

    # (B, 8, W', 256) -> (B, W', 8, 256)
    x = tf.keras.layers.Permute((2, 1, 3))(x)

    # (B, W', 8, 256) -> (B, W', 2048)
    x = tf.keras.layers.Reshape((-1, 8 * 256))(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(x)

    logits = tf.keras.layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs=inp, outputs=logits)