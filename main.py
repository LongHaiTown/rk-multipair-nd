from train_nets import (
    update_checkpoint_in_callbacks, select_best_delta_key,
    integer_to_binary_array, NDCMultiPairGenerator, make_model_inception, callbacks
)
import argparse
import importlib
import numpy as np
from train_nets import (
    update_checkpoint_in_callbacks, select_best_delta_key,train_by_chunks,
    integer_to_binary_array, NDCMultiPairGenerator, make_model_inception, callbacks
)
import argparse
import importlib
import numpy as np
import tensorflow as tf
import os
import json
import datetime as dt
from eval_nets import evaluate_with_statistics
import csv
from pathlib import Path
from typing import Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Neural distinguisher training")
    parser.add_argument(
        "--cipher",
        type=str,
        default=os.getenv("CIPHER_NAME", "present80"),
        help="Cipher module under 'cipher' package (e.g., present80, simon3264, speck3264, speck64128, simmeck3264, simmeck4896)"
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=int(os.getenv("CIPHER_ROUNDS", 7)),
        help="Number of cipher rounds to use (default: 7)"
    )
    parser.add_argument(
        "--pairs", "-p",
        type=int,
        default=int(os.getenv("PAIRS", 8)),
        help="Number of plaintext-ciphertext pairs per sample (default: 8)"
    )
    parser.add_argument(
        "--input-diff",
        type=str,
        default=os.getenv("INPUT_DIFF", "0x00000080"),
        help="Input difference hex (e.g., 0x80). Ignored when --sweep-csv is provided."
    )
    parser.add_argument(
        "--delta-key-bit",
        type=int,
        default=os.getenv("DELTA_KEY_BIT", None),
        help="Manually specify delta key bit index to use (skip automatic delta-key search)."
    )
    parser.add_argument(
        "--difference",
        type=str,
        default=os.getenv("DIFFERENCE", None),
        help="Optional hex/int difference. If bit-length > plain_bits or --combined-diff set, treated as concatenated (plain_bits + key_bits)."
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("CHUNK_SIZE", 10**4)),
        help="Chunk size for training-by-chunks (default: 10000)"
    )
    parser.add_argument(
        "--num-samples-train",
        type=int,
        default=int(os.getenv("NUM_SAMPLES_TRAIN", 10**6)),
        help="Total number of training samples (default: 1_000_000)"
    )
    args, _ = parser.parse_known_args()
    return args


def import_cipher_module(cipher_name: str):
    try:
        return importlib.import_module(f"cipher.{cipher_name}")
    except Exception as e:
        raise RuntimeError(f"Failed to import cipher module 'cipher.{cipher_name}': {e}")




def _int_to_bits(int_val: int, num_bits: int) -> np.ndarray:
    return np.array([int(b) for b in bin(int_val)[2:].zfill(num_bits)], dtype=np.uint8)


def _split_combined_difference(diff_int: int, plain_bits: int, key_bits: int):
    bits = _int_to_bits(diff_int, plain_bits + key_bits)
    delta_plain_bits = bits[:plain_bits]
    delta_key_bits = bits[plain_bits:]
    plain_int = int(''.join(str(x) for x in delta_plain_bits.tolist()), 2)
    return plain_int, delta_plain_bits, delta_key_bits


def choose_delta_key(encrypt, plain_bits: int, key_bits: int, n_round: int, pairs: int, input_difference: int):
    best_bit, best_score, all_scores = select_best_delta_key(
        encryption_function=encrypt,
        input_difference=input_difference,
        plain_bits=plain_bits,
        key_bits=key_bits,
        n_round=n_round,
        pairs=pairs,
        use_gpu=True,
    )
    delta_plain = integer_to_binary_array(input_difference, plain_bits)
    delta_key = np.zeros(key_bits, dtype=np.uint8)
    delta_key[best_bit] = 1
    return best_bit, best_score, delta_plain, delta_key


def make_generators(encrypt, plain_bits: int, key_bits: int, n_round: int, pairs: int,
                    delta_plain, delta_key, train_samples: int, val_samples: int,
                    batch_size: int, val_batch_size: int):
    gen = NDCMultiPairGenerator(
        encryption_function=encrypt,
        plain_bits=plain_bits, key_bits=key_bits, nr=n_round,
        delta_state=delta_plain,
        delta_key=delta_key,
        pairs=pairs,
        n_samples=train_samples, batch_size=batch_size,
        use_gpu=True,
    )
    gen_val = NDCMultiPairGenerator(
        encryption_function=encrypt,
        plain_bits=plain_bits, key_bits=key_bits, nr=n_round,
        delta_state=delta_plain,
        delta_key=delta_key,
        pairs=pairs,
        n_samples=val_samples, batch_size=val_batch_size,
        use_gpu=True,
    )
    return gen, gen_val


def save_artifacts(model, history, cipher_name: str, n_round: int, run_id: str):
    ckpt_dir = os.path.join("checkpoints", cipher_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    weights_path = os.path.join(ckpt_dir, f"{cipher_name}_final_{n_round}r.weights.h5")
    model.save_weights(weights_path)
    arch_path = os.path.join(ckpt_dir, f"{cipher_name}_final_{n_round}r_architecture.json")
    with open(arch_path, 'w') as f:
        f.write(model.to_json())
    print(f"Saved final model weights: {weights_path}")
    print(f"Saved model architecture: {arch_path}")

    final_path = os.path.join(ckpt_dir, f"{cipher_name}_final_{n_round}r.keras")
    model.save(final_path)
    log_dir = os.path.join("logs", cipher_name, run_id)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"history_{n_round}r.json"), "w", encoding="utf-8") as f:
        json.dump(history.history, f, ensure_ascii=False, indent=2)


def evaluate_model(model, encrypt, plain_bits: int, key_bits: int, input_difference: int, delta_key, pairs: int, n_round: int):
    stats = evaluate_with_statistics(
        model,
        round_number=n_round,
        encryption_function=encrypt,
        plain_bits=plain_bits,
        key_bits=key_bits,
        input_difference=input_difference,
        delta_key=delta_key,
        pairs=pairs,
    )
    print("Evaluation statistics:", stats)
    return stats


def run():
    args = parse_args()
    cipher = import_cipher_module(args.cipher)

    # Rounds / pairs validation
    n_round = int(args.rounds)
    pairs = int(args.pairs)
    if n_round <= 0:
        raise ValueError("--rounds must be a positive integer")
    if pairs <= 0:
        raise ValueError("--pairs must be a positive integer")

    plain_bits = cipher.plain_bits
    key_bits = cipher.key_bits
    encrypt = cipher.encrypt
    cipher_name = getattr(cipher, "cipher_name", args.cipher)
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Constants (tune as needed)
    BATCH_SIZE = 5000
    VAL_BATCH_SIZE = 20000
    EPOCHS = 2
    # `NUM_SAMPLES_TRAIN` and `CHUNK_SIZE` can be overridden by CLI flags
    NUM_SAMPLES_TRAIN = int(args.num_samples_train)
    NUM_SAMPLES_TEST = 10**5

    # Callbacks
    cb = update_checkpoint_in_callbacks(callbacks, rounds=n_round, cipher_name=cipher_name, run_id=run_id)

    delta_plain= args.difference
    mb = int(args.delta_key_bit)
    if mb < 0 or mb >= key_bits:
        raise ValueError(f"--delta-key-bit {mb} out of range (0..{key_bits-1})")
    
    delta_plain = integer_to_binary_array(int(args.input_diff, 16), plain_bits)
    delta_key = np.zeros(key_bits, dtype=np.uint8)
    delta_key[mb] = 1

    CHUNK_SIZE    = int(args.chunk_size)

    model = make_model_inception(pairs=pairs, plain_bits=plain_bits)
    optimizer = tf.keras.optimizers.Adam(amsgrad=True)
    model.compile(optimizer=optimizer, loss='mse', metrics=['acc'])

    history = train_by_chunks(
        model,
        encrypt,
        plain_bits,
        key_bits,
        n_round,
        pairs,
        delta_plain,
        delta_key,
        total_samples=NUM_SAMPLES_TRAIN,
        chunk_size=CHUNK_SIZE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Save artifacts
    save_artifacts(model, history, cipher_name, n_round, run_id)

    # Evaluate
    evaluate_model(model, encrypt, plain_bits, key_bits, delta_plain, delta_key, pairs, n_round)


# python finding_input.py --cipher-module cipher.present80 --nr 7 --pairs 1 --datasize 50000 --clusters 27 --max-bits 64

# python main.py --cipher present80 --rounds 7 --pairs 8 --sweep-csv "differences_findings\logs\present80\20251112-145640\sweep_results.csv" --diff-metric biased_pcs

# python main.py --cipher present80 --rounds 7 --pairs 8 

# python main.py --cipher present80 --rounds 7 --pairs 8 --input-diff 0x00000080

# Manual delta-key bit example (skip automatic search):
# python main.py --cipher present80 --rounds 7 --pairs 8 --input-diff 0x00000080 --delta-key-bit 107

# Multi-round auto selection examples:
# python main.py --cipher present80 --auto-latest-sweep --pairs 8 --diff-metric biased_pcs
# python main.py --cipher present80 --sweep-parent "differences_findings\logs\present80\20251113-101500" --pairs 8 --diff-metric max_diff

if __name__ == "__main__":
    run()