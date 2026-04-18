"""Visualize dataset using PCA and clustering analysis.

Generates PCA projections and KMeans clustering visualizations for neural distinguisher datasets.
Focuses on related-key attacks with multi-pair configuration.
"""
from __future__ import annotations

import argparse
import importlib
import os
import logging
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, adjusted_rand_score

# Core data generation
from make_data_train import NDCMultiPairGenerator
# Cipher utilities
from utils.cipher_utils import integer_to_binary_array, resolve_cipher_module
# Analysis helpers
from analysis.pca_helper import compute_pca
from analysis.clustering_helper import kmeans_cluster, compute_silhouette, elbow_inertia
from analysis.visualization_helper import plot_evr, scatter_2d, plot_elbow_curve, compare_3d_true_vs_pred

def parse_args():
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(
        description="Visualize neural distinguisher dataset using PCA and clustering (related-key multi-pair)"
    )
    ap.add_argument(
        "--cipher",
        default="present80",
        help="Cipher module under cipher/ package (e.g., present80, speck3264)"
    )
    ap.add_argument(
        "--scenario",
        choices=["single-key", "related-key"],
        default="related-key",
        help="Attack scenario: single-key or related-key (affects delta_key handling)"
    )
    ap.add_argument(
        "--rounds", "-r",
        type=int,
        default=7,
        help="Number of cipher rounds"
    )
    ap.add_argument(
        "--pairs", "-p",
        type=int,
        default=8,
        help="Number of plaintext-ciphertext pairs per sample"
    )
    ap.add_argument(
        "--difference",
        default="",
        help="Input difference (hex or int, e.g., 0x80 or 128)"
    )

    ap.add_argument(
        "--key-bit",
        type=int,
        default=-1,
        help="Delta key bit position; -1 for zero delta_key"
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=100000,
        help="Number of samples to generate"
    )
    ap.add_argument(
        "--pca-components",
        type=int,
        default=16,
        help="Number of PCA components"
    )
    ap.add_argument(
        "--kmeans-k",
        type=int,
        default=2,
        help="Number of KMeans clusters"
    )
    ap.add_argument(
        "--out",
        default="",
        help="Output directory for results (default: analysis_results/)"
    )

    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging"
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Save PCA/cluster figures to output directory"
    )
    ap.add_argument(
        "--elbow-kmax",
        type=int,
        default=0,
        help="If >0, compute elbow curve for k=2..kmax"
    )
    return ap.parse_args()


def compute_deltas(diff_int: int, plain_bits: int, key_bits: int):
    """Compute delta_plain and delta_key from input difference.

    Supports two formats for `diff_int`:
    - Plain-only: uses at most `plain_bits` lower bits (delta_key = zeros)
    - Combined: lower `plain_bits` bits for delta_plain; next `key_bits` bits for delta_key
    """
    # Combined format if diff exceeds plaintext width
    if diff_int.bit_length() > plain_bits:
        plain_mask = (1 << plain_bits) - 1
        key_mask = (1 << key_bits) - 1
        plain_part = diff_int & plain_mask
        key_part = (diff_int >> plain_bits) & key_mask
        delta_plain = integer_to_binary_array(plain_part, plain_bits)
        delta_key = integer_to_binary_array(key_part, key_bits)
        try:
            logging.info("Detected combined diff: plain=0x%s key=0x%s",
                         format(plain_part, 'x'), format(key_part, 'x'))
        except Exception:
            pass
    else:
        # Plain-only; mask defensively
        plain_part = diff_int & ((1 << plain_bits) - 1)
        delta_plain = integer_to_binary_array(plain_part, plain_bits)
        delta_key = np.zeros((1, key_bits), dtype=np.uint8)
    return delta_plain, delta_key


def main():
    args = parse_args()
    start_time = time.time()

    # Validate inputs
    if args.rounds <= 0:
        raise ValueError(f"--rounds must be positive, got {args.rounds}")
    if args.pairs <= 0:
        raise ValueError(f"--pairs must be positive, got {args.pairs}")
    if args.samples <= 0:
        raise ValueError(f"--samples must be positive, got {args.samples}")
    if args.pca_components <= 0:
        raise ValueError(f"--pca-components must be positive, got {args.pca_components}")
    if args.kmeans_k < 2:
        raise ValueError(f"--kmeans-k must be at least 2, got {args.kmeans_k}")

    # Load cipher module
    try:
        cipher = resolve_cipher_module(f"cipher.{args.cipher}")
        cipher_name = args.cipher
    except Exception:
        # Fallback to direct import
        cipher = importlib.import_module(f"cipher.{args.cipher}")
        cipher_name = args.cipher
    
    plain_bits = cipher.plain_bits
    key_bits = cipher.key_bits
    encrypt = cipher.encrypt
    diff_int = int(args.difference, 0)
    # Build deltas
    delta_plain, delta_key = compute_deltas(diff_int, plain_bits, key_bits)
    # Apply scenario-specific handling
    if args.scenario == "single-key":
        # Force delta_key = 0 for single-key scenario
        if isinstance(args.key_bit, int) and args.key_bit >= 0:
            logging.info("single-key scenario: ignoring --key-bit=%d (delta_key forced to zeros)", args.key_bit)
        delta_key = np.zeros((1, key_bits), dtype=np.uint8)
    else:
        # related-key: allow explicit key-bit override
        if isinstance(args.key_bit, int) and args.key_bit >= 0:
            bit = min(max(0, args.key_bit), key_bits - 1)
            dk = np.zeros((1, key_bits), dtype=np.uint8)
            dk[0, bit] = 1
            delta_key = dk
            logging.info("Using delta_key with single bit | key_bit=%d", bit)
    logging.debug(
        "Deltas built | delta_plain.shape=%s delta_key.shape=%s",
        getattr(delta_plain, "shape", None),
        getattr(delta_key, "shape", None)
    )

    # Generate dataset using NDCMultiPairGenerator
    logging.info("Generating dataset with %d samples...", args.samples)
    gen_start = time.time()
    
    # Optimize batch size for memory
    batch_size = min(args.samples, 100000)  # Cap at 100k to avoid memory issues
    
    gen = NDCMultiPairGenerator(
        encryption_function=encrypt,
        plain_bits=plain_bits,
        key_bits=key_bits,
        nr=args.rounds,
        delta_state=delta_plain,
        delta_key=delta_key,
        pairs=args.pairs,
        n_samples=args.samples,
        batch_size=batch_size,
        to_float32=True,
    )
    X, y = gen[0]
    gen_elapsed = time.time() - gen_start
    logging.info("Dataset generated in %.2f seconds | X.shape=%s y.shape=%s", gen_elapsed, X.shape, y.shape)

    # PCA on dataset
    proj, evr, *_ = compute_pca(X.astype(np.float32), n_components=args.pca_components)
    logging.info("PCA done | proj.shape=%s", proj.shape)
    logging.debug("Explained variance ratio (first 5): %s", evr[:5])

    # KMeans on PCA space
    labels, inertia, centers, sil = kmeans_cluster(proj, n_clusters=args.kmeans_k)
    if sil is None:
        # fallback: compute silhouette explicitly if not provided
        try:
            sil = compute_silhouette(proj, labels)
        except Exception:
            sil = float("nan")
    
    # Compare with true labels
    accuracy = accuracy_score(y, labels) if args.kmeans_k == 2 else float("nan")
    ari = adjusted_rand_score(y, labels)
    
    logging.info("KMeans done | k=%d inertia=%.6f silhouette=%s",
                 args.kmeans_k, inertia, f"{sil:.6f}" if isinstance(sil, float) else str(sil))
    logging.info("Comparison with true labels | accuracy=%.4f ARI=%.4f", accuracy, ari)

    print(f"Dataset: X={X.shape}, y={y.shape}; PCA proj={proj.shape}")
    print(f"Explained variance ratio (first 5): {evr[:5]}")
    print(f"KMeans(k={args.kmeans_k}) inertia={inertia:.4f}, silhouette={sil:.4f}")
    print(f"vs True labels: accuracy={accuracy:.4f}, ARI={ari:.4f}")

    # Build standardized run folder name: <cipher>_r<rounds>_p<pairs>_<difference>[_timestamp]
    try:
        # diff_token computed above; ensure compact token
        diff_token = (diff_token if isinstance(diff_token, str) else str(diff_token)).lower().strip()
    except Exception:
        diff_token = str(args.difference).lower().strip() or "diff0"
    
    # Include key-bit in folder name for related-key scenario when explicitly provided
    key_suffix = ""
    if args.scenario == "related-key" and isinstance(args.key_bit, int) and args.key_bit >= 0:
        key_suffix = f"_key_bits{args.key_bit}"

    run_folder = f"{cipher_name}_{args.scenario}_r{args.rounds}_p{args.pairs}_{diff_token}{key_suffix}"

    # Decide base output directory
    base_out = args.out if args.out else Path(__file__).parent / "analysis_results"
    run_out_dir = Path(base_out) / run_folder
    run_out_dir.mkdir(parents=True, exist_ok=True)

    # Optional plotting
    if args.plot or args.out:
        img_dir = run_out_dir

        # EVR bar chart
        evr_path = img_dir / "pca_evr.png"
        plot_evr(evr, str(evr_path))
        logging.info("Saved EVR plot: %s", evr_path)

        # PCA scatter colored by dataset labels (cipher vs random)
        pca_labels_path = img_dir / "pca_scatter_labels.png"
        scatter_2d(proj, labels=y, path=str(pca_labels_path), title="PCA Scatter (dataset labels)")
        logging.info("Saved PCA scatter (labels): %s", pca_labels_path)

        # PCA scatter colored by KMeans cluster labels
        pca_kmeans_path = img_dir / "pca_scatter_kmeans.png"
        scatter_2d(proj, labels=labels, path=str(pca_kmeans_path), title=f"PCA Scatter (KMeans k={args.kmeans_k})", cmap="tab10")
        logging.info("Saved PCA scatter (KMeans): %s", pca_kmeans_path)

        # Optional elbow plot
        if isinstance(args.elbow_kmax, int) and args.elbow_kmax and args.elbow_kmax > 2:
            k_vals = list(range(2, args.elbow_kmax + 1))
            inertias = elbow_inertia(proj, k_vals)
            elbow_path = img_dir / "kmeans_elbow.png"
            plot_elbow_curve(k_vals, inertias, str(elbow_path))
            logging.info("Saved KMeans elbow plot: %s", elbow_path)

        # Optional 3D comparison plot if we have at least 3 PCA components
        if args.pca_components >= 3:
            points3d = proj[:, :3]
            try:
                sil_true_3d = compute_silhouette(points3d, y)
            except Exception:
                sil_true_3d = float("nan")
            try:
                sil_pred_3d = compute_silhouette(points3d, labels)
            except Exception:
                sil_pred_3d = float("nan")

            compare_3d_path = img_dir / "pca_compare_3d.png"
            compare_3d_true_vs_pred(
                points3d,
                y_true=y,
                y_pred=labels,
                sil_true=sil_true_3d,
                sil_pred=sil_pred_3d,
                path=str(compare_3d_path),
                title=f"PCA 3D: True vs KMeans (k={args.kmeans_k})",
            )
            logging.info("Saved PCA 3D comparison: %s (sil_true=%.4f, sil_pred=%.4f)", compare_3d_path, sil_true_3d, sil_pred_3d)

    # Optional save
    if base_out:
        # Always save artifacts to the run-specific folder
        np.save(run_out_dir / "projected_data.npy", proj.astype(np.float32))
        np.save(run_out_dir / "eigenvalue_ratios.npy", evr)
        np.save(run_out_dir / "dataset_labels.npy", y.astype(np.int32))
        np.save(run_out_dir / "kmeans_labels.npy", labels.astype(np.int32))
        # Save summary report
        total_elapsed = time.time() - start_time

        
        print(f"\nSaved outputs to: {run_out_dir}")
        print(f"   Total time: {total_elapsed:.1f}s")
        logging.info("Total execution time: %.2f seconds", total_elapsed)


if __name__ == "__main__":
    main()

