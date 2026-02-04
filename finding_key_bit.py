"""Utilities to sweep delta-key bits per round and allow specifying cipher & input difference via flags.

Provides:
- sweep_delta_keys_by_round(...)
- find_best_bits_list(...)
- export_results_to_csv(...)

Run with:
    python finding_key_bit.py --cipher-name present80 --input-diff 0x80 --csv-output results.csv
"""
from typing import Callable, Dict, List, Tuple
import importlib
import csv
import json
import os
import argparse

try:
    import numpy as np
except Exception:
    np = None

try:
    import cupy as cp
except Exception:
    cp = None

from train_nets import select_best_delta_key


def _make_json_safe(o):
    """Convert common numpy/cupy types to Python-native types for JSON serialization."""
    # dict
    if isinstance(o, dict):
        return {str(k): _make_json_safe(v) for k, v in o.items()}
    # numpy array
    if np is not None and isinstance(o, np.ndarray):
        return _make_json_safe(o.tolist())
    # cupy array
    if cp is not None and isinstance(o, cp.ndarray):
        try:
            return _make_json_safe(cp.asnumpy(o).tolist())
        except Exception:
            return str(o)
    # list/tuple
    if isinstance(o, (list, tuple)):
        return [_make_json_safe(x) for x in o]
    # numpy scalar
    if np is not None and isinstance(o, np.generic):
        return _make_json_safe(o.item())
    # cupy scalar
    if cp is not None and hasattr(cp, 'generic') and isinstance(o, cp.generic):
        try:
            return _make_json_safe(o.item())
        except Exception:
            return str(o)
    # primitive
    if isinstance(o, (int, float, str, bool)) or o is None:
        return o
    # fallback
    try:
        return float(o)
    except Exception:
        try:
            return str(o)
        except Exception:
            return None


def find_best_bits_list(
    encryption_function: Callable,
    plain_bits: int,
    key_bits: int,
    input_difference: int = 0,
    pairs: int = 1,
    start_round: int = 1,
    max_rounds: int = 20,
    stop_score_threshold: float = 0.1,
    **kwargs,
) -> List[Tuple[int, float]]:
    rows = sweep_delta_keys_by_round(
        encryption_function=encryption_function,
        plain_bits=plain_bits,
        key_bits=key_bits,
        input_difference=input_difference,
        pairs=pairs,
        start_round=start_round,
        max_rounds=max_rounds,
        stop_score_threshold=stop_score_threshold,
        **kwargs,
    )

    return [(r["best_bit"], r["best_score"]) for r in rows]


def export_results_to_csv(results: List[Dict], filepath: str, include_all_scores: bool = True):
    """Export sweep results to CSV. Returns filepath."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ["round", "best_bit", "best_score", "max_score", "mean_score", "all_scores_json"]
        writer.writerow(header)

        for r in results:
            all_scores = r.get("all_scores", {}) or {}
            # all_scores may be a dict mapping bit->score
            if isinstance(all_scores, dict):
                scores_list = list(all_scores.values())
            else:
                try:
                    scores_list = list(all_scores)
                except Exception:
                    scores_list = []

            # Normalize numeric types to Python floats for CSV/JSON compatibility
            cleaned_scores = []
            for s in scores_list:
                try:
                    cleaned_scores.append(float(s))
                except Exception:
                    try:
                        cleaned_scores.append(float(getattr(s, 'item', lambda: s)()))
                    except Exception:
                        continue

            max_score = max(cleaned_scores) if cleaned_scores else ""
            mean_score = sum(cleaned_scores) / len(cleaned_scores) if cleaned_scores else ""

            all_scores_json = json.dumps(_make_json_safe(all_scores), ensure_ascii=False) if include_all_scores else ""

            writer.writerow([
                r.get("round"),
                r.get("best_bit"),
                r.get("best_score"),
                max_score,
                mean_score,
                all_scores_json,
            ])

    return filepath



def sweep_delta_keys_by_round(
    encryption_function: Callable,
    plain_bits: int,
    key_bits: int,
    input_difference: int = 0,
    pairs: int = 1,
    start_round: int = 1,
    max_rounds: int = 20,
    stop_score_threshold: float = 0.1,
    n_samples: int = 100_000,
    batch_size: int = 5_000,
    use_gpu: bool = True,
    verbose: bool = True,
) -> List[Dict]:
    """Sweep rounds and select best delta_key bit for each round."""

    results: List[Dict] = []

    for r in range(start_round, max_rounds + 1):
        if verbose:
            print(f"🔎 Round {r}: searching best delta-key bit (input_diff=0x{input_difference:X})...")

        best_bit, best_score, all_scores = select_best_delta_key(
            encryption_function=encryption_function,
            input_difference=input_difference,
            plain_bits=plain_bits,
            key_bits=key_bits,
            n_round=r,
            pairs=pairs,
            n_samples=n_samples,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )

        results.append({
            "round": r,
            "best_bit": int(best_bit) if best_bit is not None else best_bit,
            "best_score": float(best_score) if best_score is not None else best_score,
            "all_scores": all_scores,
        })

        if verbose:
            print(f"    → best_bit={best_bit}, score={best_score:.5f}")

        if best_score is None or best_score < stop_score_threshold:
            if verbose:
                print(f"⛔ Stopping: best_score {best_score:.5f} < threshold {stop_score_threshold:.5f}")
            break

    return results


# --- CLI example ---
if __name__ == "__main__":
    DEFAULT_CIPHER = "present80"
    DEFAULT_INPUT_DIFF = 0x00000080
    DEFAULT_CSV_OUTPUT = "delta_key_sweep_results.csv"

    parser = argparse.ArgumentParser(description="Sweep delta-key bits across rounds and export results")
    parser.add_argument("--cipher-name", "-c", default=DEFAULT_CIPHER, help="cipher module name under cipher.<name>")
    parser.add_argument("--input-diff", "-d", default=None, help="input difference as hex (0x..) or int; defaults to %s" % hex(DEFAULT_INPUT_DIFF))
    parser.add_argument("--csv-output", default=DEFAULT_CSV_OUTPUT, help="CSV output path (set empty string to disable)")
    parser.add_argument("--start-round", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--stop-score-threshold", type=float, default=0.1)
    parser.add_argument("--n-samples", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--no-gpu", action="store_true", help="disable GPU usage for generator and selection")

    args = parser.parse_args()

    CIPHER_NAME = args.cipher_name
    if args.input_diff is not None:
        try:
            INPUT_DIFFERENCE = int(str(args.input_diff), 0)
        except Exception as e:
            raise SystemExit(f"Failed to parse --input-diff '{args.input_diff}': {e}")
    else:
        INPUT_DIFFERENCE = DEFAULT_INPUT_DIFF

    CSV_OUTPUT = args.csv_output if args.csv_output != "" else None
    USE_GPU = not args.no_gpu

    try:
        cipher = importlib.import_module(f"cipher.{CIPHER_NAME}")
    except Exception as e:
        raise SystemExit(f"Failed to import cipher module 'cipher.{CIPHER_NAME}': {e}")

    print(f"Running sweep on cipher {CIPHER_NAME} with input difference 0x{INPUT_DIFFERENCE:X}...")
    res = sweep_delta_keys_by_round(
        encryption_function=cipher.encrypt,
        plain_bits=cipher.plain_bits,
        key_bits=cipher.key_bits,
        input_difference=INPUT_DIFFERENCE,
        pairs=args.pairs,
        start_round=args.start_round,
        max_rounds=args.max_rounds,
        stop_score_threshold=args.stop_score_threshold,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        use_gpu=USE_GPU,
        verbose=True,
    )

    print("\nSummary (round, best_bit, score):")
    for row in res:
        print(row["round"], row["best_bit"], f"{row['best_score']:.5f}")

    if CSV_OUTPUT:
        out_path = export_results_to_csv(res, CSV_OUTPUT, include_all_scores=True)
        print(f"Exported sweep results to: {out_path}")

# python finding_key_bit.py --cipher-name present80 --input-diff 0x80   --csv-output results_present.csv
# python finding_key_bit.py --cipher-name speck3264 --input-diff 0x400000 --csv-output results_speck3264.csv
# python finding_key_bit.py --cipher-name simmeck3264 --input-diff 0x40 --csv-output results_simmeck3264.csv