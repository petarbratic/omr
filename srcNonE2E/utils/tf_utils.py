# I used ChatGPT and Cursor for the development of this project.
import tensorflow as tf


def _enable_gpu_memory_growth() -> None:
    # Ensure that TensorFlow only allocates GPU memory as needed,
    # rather than pre-allocating all of it.
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("GPUs: []")
        return
    print("GPUs:", gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

