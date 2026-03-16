import argparse
from pathlib import Path
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import tensorflow as tf
from make_data_train import NDCMultiPairGenerator
from train_nets import integer_to_binary_array
import importlib

def count_model_params(model):
    trainable = np.sum([np.prod(w.shape) for w in model.trainable_weights])
    non_trainable = np.sum([np.prod(w.shape) for w in model.non_trainable_weights])
    total = trainable + non_trainable
    return total, trainable, non_trainable

import time

def measure_throughput(model, input_dim, batch_size=10_000, repeats=20):
    X = np.random.randint(0, 2, size=(batch_size, input_dim), dtype=np.uint8)
    X = X.astype(np.float32)

    # warmup
    _ = model.predict(X, batch_size=batch_size, verbose=0)

    start = time.time()
    for _ in range(repeats):
        _ = model.predict(X, batch_size=batch_size, verbose=0)
    end = time.time()

    total_samples = batch_size * repeats
    throughput = total_samples / (end - start)

    return throughput, end - start

def evaluate_with_statistics(
    model,
    round_number,
    n_repeat=20,
    log_path=None,
    encryption_function=None,
    plain_bits=64,
    key_bits=80,
    input_difference=None,
    delta_key=None,
    pairs=8,
    test_samples: int = 1_000_000,
    batch_size: int = 10_000,
    use_gpu: bool = True,
):
    """
    Evaluate the model multiple times and compute statistics, including:
    - accuracy
    - std
    - z-score, p-value
    - runtime throughput
    """

    print(f"\n=== Evaluating model on {n_repeat}× fresh test sets (round {round_number}) ===")

    if encryption_function is None or input_difference is None or delta_key is None:
        raise ValueError("encryption_function, input_difference, and delta_key must be provided.")

    # ----------------------------------------------------
    # 1. Repeat testing for statistical significance
    # ----------------------------------------------------
    test_accs = []
    for i in tqdm(range(n_repeat)):
        test_gen = NDCMultiPairGenerator(
            encryption_function=encryption_function,
            plain_bits=plain_bits,
            key_bits=key_bits,
            nr=round_number,
            delta_state=integer_to_binary_array(input_difference, plain_bits),
            delta_key=delta_key,
            n_samples=test_samples,
            batch_size=batch_size,
            pairs=pairs,
            use_gpu=use_gpu,
        )
        _, acc = model.evaluate(test_gen, verbose=0)
        test_accs.append(acc)

    accs = np.array(test_accs)
    avg_acc = np.mean(accs)
    std_acc = np.std(accs)

    # ----------------------------------------------------
    # 2. Statistical tests vs random guessing (0.5)
    # ----------------------------------------------------
    mean_random = 0.5
    std_random = 0.0005
    std_mean = std_random / np.sqrt(n_repeat)
    z_score = (avg_acc - mean_random) / std_mean
    p_value = 1 - norm.cdf(z_score)

    print(f"\nAccuracy: {avg_acc:.6f} ± {std_acc:.6f}")
    print(f"Z-score: {z_score:.2f}  P-value: {p_value:.3e}")
    print("Significant." if p_value < 0.01 else "Not significant.")


    # ----------------------------------------------------
    # 4. Runtime Throughput
    # ----------------------------------------------------
    input_dim = model.input_shape[1]
    throughput, total_time = measure_throughput(
        model,
        input_dim=input_dim,
        batch_size=batch_size,
        repeats=20
    )

    print(f"\n--- Runtime Efficiency ---")
    print(f"Throughput              : {throughput:,.0f} samples/sec")
    print(f"Time for 1M samples     : {1_000_000/throughput:.3f} sec")
    print(f"Measured over {total_time:.3f} seconds")

    # ----------------------------------------------------
    # 5. Save log (optional)
    # ----------------------------------------------------
    if log_path:
        with open(log_path, "w") as f:
            for i, acc in enumerate(accs):
                f.write(f"Test {i+1}: {acc:.6f}\n")
            f.write(f"\nAverage Accuracy: {avg_acc:.6f}\n")
            f.write(f"Std Accuracy:     {std_acc:.6f}\n")
            f.write(f"Z-score:          {z_score:.2f}\n")
            f.write(f"P-value:          {p_value:.4e}\n")
            f.write(f"Throughput:       {throughput:,.0f} samples/sec\n")

    return {
        "avg_acc": avg_acc,
        "std_acc": std_acc,
        "z_score": z_score,
        "p_value": p_value,
        "throughput": throughput
    }


def _parse_delta_key_from_hex(hex_str: str, key_bits: int) -> np.ndarray:
    mask = int(hex_str, 16)
    arr = np.zeros(key_bits, dtype=np.uint8)
    for i in range(key_bits):
        arr[i] = (mask >> i) & 1
    return arr


def _int_to_bits(int_val: int, num_bits: int) -> np.ndarray:
    return np.array([int(b) for b in bin(int_val)[2:].zfill(num_bits)], dtype=np.uint8)


def _split_combined_difference(diff_int: int, plain_bits: int, key_bits: int):
    """Split a combined (plain_bits + key_bits) difference integer into
    (plain_int, delta_plain_bits, delta_key_bits).
    """
    bits = _int_to_bits(diff_int, plain_bits + key_bits)
    delta_plain_bits = bits[:plain_bits]
    delta_key_bits = bits[plain_bits:]
    plain_int = int(''.join(str(x) for x in delta_plain_bits.tolist()), 2)
    return plain_int, delta_plain_bits, delta_key_bits


def _import_cipher_module(module_path: str):
    try:
        return importlib.import_module(module_path)
    except Exception as e:
        raise RuntimeError(f"Failed to import cipher module '{module_path}': {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained distinguisher with statistical testing")
    parser.add_argument(
        "--cipher-module",
        type=str,
        default="cipher.present80",
        help="Dotted path to cipher module (e.g., cipher.present80)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model: .keras (full model) or .h5/.hdf5. If it's a weights file like *.weights.h5, the script rebuilds the model via RKmcp.make_model_inception(pairs, plain_bits) and loads weights.",
    )
    parser.add_argument("--rounds", "-r", type=int, default=7, help="Number of cipher rounds for evaluation")
    parser.add_argument("--pairs", "-p", type=int, default=8, help="Pairs per sample")
    parser.add_argument("--input-diff", type=str, default="0x00000080", help="Input difference hex (e.g., 0x80)")
    parser.add_argument(
        "--difference",
        type=str,
        default=None,
        help="Optional hex/int difference. If bit-length > plain_bits or --combined-diff set, treated as concatenated (plain_bits + key_bits).",
    )
    parser.add_argument(
        "--combined-diff",
        action="store_true",
        help="Force --difference to be interpreted as combined (plain_bits + key_bits).",
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--delta-key-bit",
        type=int,
        help="Delta key bit index (0-based). If omitted and --delta-key-hex not provided, uses delta_key=0 (no related-key).",
    )
    group.add_argument(
        "--delta-key-hex",
        type=str,
        help="Delta key bitmask as hex (e.g., 0x00000000000000000001). If omitted and --delta-key-bit not provided, uses delta_key=0 (no related-key).",
    )
    parser.add_argument("--n-repeat", type=int, default=10, help="Number of repeated test evaluations")
    parser.add_argument("--test-samples", type=int, default=100_000, help="Samples per test set")
    parser.add_argument("--batch-size", type=int, default=5_000, help="Batch size for test generator")
    parser.add_argument("--no-eca", action="store_true", help="When rebuilding model from weights, use the no-ECA variant (Ablation). Auto-detected from filename if present.")
    # Always use GPU path in generator (if available); no CPU fallback flag
    parser.add_argument("--log-path", type=str, default=None, help="Optional path to save evaluation log")

    args = parser.parse_args()

    # Load cipher
    cipher_mod = _import_cipher_module(args.cipher_module)
    encrypt = cipher_mod.encrypt
    plain_bits = cipher_mod.plain_bits
    key_bits = cipher_mod.key_bits

    # Prepare deltas (supports combined differences with optional override)
    input_difference = None
    delta_key = None

    if args.difference:
        # Direct difference provided (hex or int, auto base)
        try:
            user_diff_int = int(args.difference, 0)
        except Exception:
            user_diff_int = int(args.difference)

        if args.combined_diff or user_diff_int.bit_length() > plain_bits:
            # Treat as combined (plain_bits + key_bits)
            input_difference, d_plain_bits, d_key_bits = _split_combined_difference(
                user_diff_int, plain_bits, key_bits
            )
            delta_key = d_key_bits.astype(np.uint8)
            print(f"[difference] Using combined difference (plain+key): 0x{user_diff_int:X}")
            print(
                f"            delta_plain HW={int(d_plain_bits.sum())} bits={[i for i,v in enumerate(d_plain_bits) if v]}"
            )
            print(
                f"            delta_key  HW={int(d_key_bits.sum())} bits={[i for i,v in enumerate(d_key_bits) if v]}"
            )
            # Allow override by delta-key options
            if args.delta_key_bit is not None:
                mb = int(args.delta_key_bit)
                if mb < 0 or mb >= key_bits:
                    raise ValueError(f"--delta-key-bit out of range (0..{key_bits-1})")
                delta_key = np.zeros(key_bits, dtype=np.uint8)
                delta_key[mb] = 1
                print(f"[override] Replacing combined delta_key with manual bit={mb}")
            elif args.delta_key_hex:
                delta_key = _parse_delta_key_from_hex(args.delta_key_hex, key_bits)
                print("[override] Replacing combined delta_key with provided --delta-key-hex mask")
        else:
            # Plain-only difference
            input_difference = user_diff_int
            print(f"[difference] Using provided plain input difference: 0x{input_difference:X}")
            # Determine delta_key from overrides or default to zeros
            if args.delta_key_bit is not None:
                mb = int(args.delta_key_bit)
                if mb < 0 or mb >= key_bits:
                    raise ValueError(f"--delta-key-bit out of range (0..{key_bits-1})")
                delta_key = np.zeros(key_bits, dtype=np.uint8)
                delta_key[mb] = 1
            elif args.delta_key_hex:
                delta_key = _parse_delta_key_from_hex(args.delta_key_hex, key_bits)
            else:
                delta_key = np.zeros(key_bits, dtype=np.uint8)
                print("[info] No delta-key override provided; using delta_key = 0 (no related-key).")
    else:
        # Backward-compatible path using --input-diff and delta-key overrides
        input_difference = int(args.input_diff, 16)
        if args.delta_key_bit is not None:
            if args.delta_key_bit < 0 or args.delta_key_bit >= key_bits:
                raise ValueError(f"--delta-key-bit out of range (0..{key_bits-1})")
            delta_key = np.zeros(key_bits, dtype=np.uint8)
            delta_key[int(args.delta_key_bit)] = 1
        elif args.delta_key_hex:
            delta_key = _parse_delta_key_from_hex(args.delta_key_hex, key_bits)
        else:
            # Default: no related-key difference
            delta_key = np.zeros(key_bits, dtype=np.uint8)
            print("[info] No --delta-key-bit/--delta-key-hex provided; using delta_key = 0 (no related-key).")

    # Load model
    mp = Path(args.model_path)
    if not mp.exists():
        raise FileNotFoundError(f"Model path not found: {mp}")

    suffix = mp.suffix.lower()
    model = None
    if suffix == ".keras":
        model = tf.keras.models.load_model(str(mp))
    elif suffix in (".h5", ".hdf5"):
        # Case 1: full model saved in H5
        try:
            model = tf.keras.models.load_model(str(mp))
        except Exception:
            # Case 2: weights-only H5 — rebuild model from RKmcp factories
            try:
                from RKmcp import make_model_inception, make_model_inception_no_eca
            except Exception as e:
                raise RuntimeError("Failed to import model factories from RKmcp: " + str(e))

            # Decide whether to use the no-ECA variant (explicit flag or filename hint)
            use_no_eca = bool(args.no_eca) or ('noeca' in mp.stem.lower()) or ('no_eca' in mp.stem.lower())
            factory = make_model_inception_no_eca if use_no_eca else make_model_inception
            model = factory(pairs=int(args.pairs), plain_bits=plain_bits)
            model.load_weights(str(mp))
            if use_no_eca:
                print("[info] Rebuilt model using no-ECA variant (make_model_inception_no_eca)")
    else:
        raise ValueError(f"Unsupported model file type: {suffix}. Use .keras or .h5/.hdf5")
    
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"],
    )
    
    # Run evaluation
    stats = evaluate_with_statistics(
        model,
        round_number=int(args.rounds),
        n_repeat=int(args.n_repeat),
        log_path=args.log_path,
        encryption_function=encrypt,
        plain_bits=plain_bits,
        key_bits=key_bits,
        input_difference=input_difference,
        delta_key=delta_key,
        pairs=int(args.pairs),
        test_samples=int(args.test_samples),
        batch_size=int(args.batch_size),
        use_gpu=True,
    )

    print("\nSummary:", stats)


if __name__ == "__main__":
    main()

# python eval_nets.py --cipher-module cipher.present80 --model-path D:\_Project_RKNDIncECA\Related-key-mcp-attention-Inception-based-ND\checkpoints\present80\with_diff-bit\present80_best_7r.weights.h5 --rounds 7 --pairs 8 --difference 0x800000000000000080000000000000000001 --combined-diff --delta-key-bit 0 --n-repeat 5