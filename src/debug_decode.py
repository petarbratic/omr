import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import json
import tensorflow as tf

from src.data.tf_dataset import make_dataset


ID_TO_TOKEN_PATH = "data/vocab/id_to_token.json"
TOKEN_TO_ID_PATH = "data/vocab/token_to_id.json"


def load_id_to_token():
    with open(ID_TO_TOKEN_PATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    # JSON ključevi su stringovi, npr. "12": "noteheadBlack"
    return d


def ids_to_tokens(ids, id_to_token):
    out = []
    for x in ids:
        out.append(id_to_token[str(int(x))])
    return out


def estimate_input_lengths(images, T):
    """
    images: (B, 128, Wmax, 1) sa paddingom nulama po širini
    T: broj time-stepova u logits (posle CNN)
    Vrati (B,) input_lengths za CTC decode (aproksimacija po efektivnoj širini slike).
    """
    Wmax = tf.shape(images)[2]

    # kolone koje imaju bar jedan nenulti piksel
    col_has = tf.reduce_any(images > 0.0, axis=[1, 3])  # (B, Wmax) bool

    # nađi poslednju kolonu koja ima sadržaj (zbog padding nula)
    rev = tf.reverse(col_has, axis=[1])
    last_from_end = tf.argmax(tf.cast(rev, tf.int32), axis=1, output_type=tf.int32)  # (B,)
    last_idx = (Wmax - 1) - last_from_end
    w_eff = last_idx + 1  # efektivna širina

    # mapiraj efektivnu širinu na T (proporcionalno)
    input_len = tf.cast(
        tf.math.ceil(tf.cast(w_eff, tf.float32) * tf.cast(T, tf.float32) / tf.cast(Wmax, tf.float32)),
        tf.int32,
    )

    input_len = tf.clip_by_value(input_len, 1, T)
    return input_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="artifacts/crnn_ctc.keras")
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()

    id_to_token = load_id_to_token()

    # učitaj 1 batch
    ds = make_dataset(limit=args.limit, batch_size=args.batch_size, shuffle=False)
    images, labels, label_lengths, img_widths = next(iter(ds))

    # učitaj model
    model = tf.keras.models.load_model(args.model_path, compile=False)

    # forward
    logits = model(images, training=False)  # (B, T, C)
    T = tf.shape(logits)[1]

    # log-softmax i time-major za decoder
    log_probs = tf.nn.log_softmax(logits, axis=-1)        # (B, T, C)
    log_probs_tm = tf.transpose(log_probs, [1, 0, 2])     # (T, B, C)

    input_lengths = estimate_input_lengths(images, T)

    decoded, _ = tf.nn.ctc_greedy_decoder(
        inputs=log_probs_tm,
        sequence_length=input_lengths,
    )

    # SparseTensor -> dense sa -1 kao prazno
    dense = tf.sparse.to_dense(decoded[0], default_value=-1)  # (B, <=something)

    B = images.shape[0]
    show_n = min(args.show, B)

    for i in range(show_n):
        # GT tokeni (koristi label_length da ukloni padding)
        L = int(label_lengths[i].numpy())
        gt_ids = labels[i][:L].numpy()
        gt_tokens = ids_to_tokens(gt_ids, id_to_token)

        # Pred tokeni (ukloni -1)
        pred_ids = dense[i].numpy()
        pred_ids = pred_ids[pred_ids >= 0]
        pred_tokens = ids_to_tokens(pred_ids, id_to_token) if len(pred_ids) else []

        print(f"\n--- Sample {i} ---")
        print("GT  :", " ".join(gt_tokens))
        print("PRED:", " ".join(pred_tokens))


if __name__ == "__main__":
    main()