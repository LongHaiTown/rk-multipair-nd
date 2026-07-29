"""Regenerate PCA/K-Means visualizations from validated manifest rows.

The script implements the "regenerate from manifest" visualization flow:

    manifest row
        -> locked ProjectionProtocol from geometry config
        -> prepare_projected_context(...)
        -> D1/D0 (or selected control) projected samples
        -> true-label and K-Means-label PCA panels

It intentionally does not read projected coordinates from CSV. Instead, it
rebuilds the exact projection using the same sample-selection and PCA contract
as ``run_projected_geometry.py``. The visualization flow is cipher-agnostic;
cipher identity and data paths are taken from the selected manifest rows.

When ``--geometry-long`` is supplied, the regenerated ``projection_id`` and
``sample_selection_hash`` are checked against the geometry runner output. A
mismatch is fatal by default because it means the visualization does not
represent the same projected analysis. Use ``--allow-projection-mismatch`` only
for debugging.

Typical usage
-------------

python visualize_projected_geometry.py \
  --manifest analysis_data/<cipher_study>/manifest.csv \
  --config configs/<cipher>/projected_geometry_full.json \
  --geometry-long \
      analysis_results/<geometry_study>/projected_geometry_long.csv \
  --out analysis_results \
  --rounds <rounds...> \
  --k-values 1 8 16 32 \
  --seeds 201 \
  --control-types D1_vs_D0

The output contains:

- one two-panel PCA figure per selected configuration/control;
- optional plotted-point CSV files;
- one JSON metadata record per figure;
- a study-level visualization manifest and report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import traceback
import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import matplotlib

# Headless backend for servers and batch runners.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from projected_estimators import (
    COMPARISON_DATASETS,
    fit_kmeans_metrics,
    prepare_projected_context,
    projection_summary,
)
from projected_runner_common import (
    add_common_cli_arguments,
    atomic_write_csv,
    atomic_write_json,
    select_manifest_rows,
)
from run_projected_geometry import load_config


VISUALIZATION_SCHEMA_VERSION = "projected_geometry_visualization_v2"
SUPPORTED_FORMATS = ("png", "pdf", "svg")
DEFAULT_CONTROL_TYPES = ("D1_vs_D0",)

VISUALIZATION_COLUMNS = (
    "visualization_study_id",
    "visualization_schema_version",
    "source_estimator_study_id",
    "source_study_id",
    "study_id",
    "domain",
    "cipher",
    "round",
    "k",
    "seed",
    "control_type",
    "pc_x",
    "pc_y",
    "selected_count_per_dataset",
    "plotted_points_per_side",
    "projection_id",
    "expected_projection_id",
    "projection_id_match",
    "projection_contract_match",
    "sample_selection_hash",
    "expected_sample_selection_hash",
    "sample_selection_hash_match",
    "silhouette_kmeans",
    "expected_silhouette_kmeans",
    "silhouette_abs_diff",
    "kmeans_adjusted_rand",
    "expected_kmeans_adjusted_rand",
    "ari_abs_diff",
    "kmeans_aligned_accuracy",
    "expected_kmeans_aligned_accuracy",
    "aligned_accuracy_abs_diff",
    "metric_contract_match",
    "full_audit_match",
    "audit_source_csv",
    "cluster_labels_inverted_for_display",
    "pair_shuffle_k1_identical",
    "figure_paths",
    "points_csv",
    "metadata_json",
    "status",
    "error",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_") or "unknown"


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def parse_formats(values: Sequence[str]) -> Tuple[str, ...]:
    formats = tuple(dict.fromkeys(value.lower().lstrip(".") for value in values))
    unsupported = sorted(set(formats) - set(SUPPORTED_FORMATS))
    if unsupported:
        raise ValueError(
            f"Unsupported formats: {unsupported}; "
            f"supported={list(SUPPORTED_FORMATS)}"
        )
    if not formats:
        raise ValueError("At least one output format is required")
    return formats


def load_geometry_long(
    paths: Optional[Sequence[Path]],
    *,
    expected_estimator_study_id: str,
    expected_source_study_id: str,
) -> Optional[pd.DataFrame]:
    """Load one or more geometry-long CSVs and select the configured study.

    Full study identity is retained in the audit key so screening, confirmatory,
    and repeated studies for the same cipher cannot be mixed accidentally.
    """

    if not paths:
        return None

    frames: list[pd.DataFrame] = []
    required = {
        "estimator_study_id",
        "source_study_id",
        "study_id",
        "domain",
        "cipher",
        "round",
        "k",
        "seed",
        "control_type",
        "projection_id",
        "sample_selection_hash",
        "silhouette_kmeans",
        "kmeans_adjusted_rand",
        "kmeans_aligned_accuracy",
        "status",
    }

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if frame.empty:
            raise ValueError(f"Geometry long CSV is empty: {path}")
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(
                f"Geometry long CSV {path} is missing columns: {missing}"
            )
        frame = frame.copy()
        frame["audit_source_csv"] = str(path)
        frames.append(frame)

    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame = frame[
        frame["status"].astype(str).str.strip().str.lower().eq("completed")
    ].copy()
    if frame.empty:
        raise ValueError("No completed geometry rows remain in --geometry-long")

    for column in (
        "round",
        "k",
        "seed",
        "silhouette_kmeans",
        "kmeans_adjusted_rand",
        "kmeans_aligned_accuracy",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame[["round", "k", "seed"]].isna().any().any():
        raise ValueError("Geometry long CSV has non-numeric round/k/seed values")
    frame[["round", "k", "seed"]] = frame[["round", "k", "seed"]].astype(int)

    configured = frame[
        frame["estimator_study_id"].astype(str).eq(
            str(expected_estimator_study_id)
        )
        & frame["source_study_id"].astype(str).eq(
            str(expected_source_study_id)
        )
    ].copy()
    if configured.empty:
        available = (
            frame[["estimator_study_id", "source_study_id"]]
            .drop_duplicates()
            .sort_values(["estimator_study_id", "source_study_id"])
        )
        raise ValueError(
            "No geometry-long rows match the configured study pair "
            f"({expected_estimator_study_id!r}, {expected_source_study_id!r}). "
            "Available pairs:\n" + available.to_string(index=False)
        )

    identity_columns = [
        "estimator_study_id",
        "source_study_id",
        "study_id",
        "domain",
        "cipher",
        "round",
        "k",
        "seed",
        "control_type",
    ]
    duplicates = configured.duplicated(identity_columns, keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate geometry-long audit rows across the supplied CSVs:\n"
            + configured.loc[
                duplicates, identity_columns + ["audit_source_csv"]
            ].to_string(index=False)
        )

    return configured.set_index(identity_columns, drop=False).sort_index()


def expected_geometry_row(
    geometry_long: Optional[pd.DataFrame],
    *,
    row: Mapping[str, Any],
    control_type: str,
    estimator_study_id: str,
    source_study_id: str,
) -> Optional[Mapping[str, Any]]:
    if geometry_long is None:
        return None

    key = (
        str(estimator_study_id),
        str(source_study_id),
        str(row["study_id"]),
        str(row["domain"]),
        str(row["cipher"]),
        int(row["round"]),
        int(row["k"]),
        int(row["seed"]),
        str(control_type),
    )
    try:
        value = geometry_long.loc[key]
    except KeyError as exc:
        raise ValueError(
            "No matching row in --geometry-long for "
            f"estimator_study_id={key[0]}, source_study_id={key[1]}, "
            f"study_id={key[2]}, domain={key[3]}, cipher={key[4]}, "
            f"r={key[5]}, k={key[6]}, seed={key[7]}, control={key[8]}"
        ) from exc

    if isinstance(value, pd.DataFrame):
        raise ValueError(f"Geometry-long audit key is not unique: {key}")
    return value.to_dict()


def control_label_names(control_type: str) -> Tuple[str, str]:
    labels = {
        "D1_vs_D0": ("D1 structured", "D0 null"),
        "D0_vs_D0": ("D0 control A", "D0 control B"),
        "D1_vs_D1": ("D1 control A", "D1 control B"),
        "pair_shuffled": ("Pair-shuffled D1", "D0 null"),
    }
    try:
        return labels[control_type]
    except KeyError as exc:
        raise ValueError(f"Unknown control type: {control_type}") from exc


def align_cluster_labels(
    y_true: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """Align binary cluster IDs to class IDs for display only."""

    raw_accuracy = float(accuracy_score(y_true, labels))
    if raw_accuracy >= 0.5:
        return labels.astype(np.int32, copy=True), False
    return (1 - labels).astype(np.int32, copy=False), True


def stable_plot_random_state(
    row: Mapping[str, Any],
    *,
    base_random_state: int,
    control_type: str,
) -> int:
    payload = {
        "study_id": row["study_id"],
        "domain": row["domain"],
        "cipher": row["cipher"],
        "round": int(row["round"]),
        "k": int(row["k"]),
        "seed": int(row["seed"]),
        "control_type": control_type,
        "base_random_state": int(base_random_state),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def plotting_indices(
    n_per_side: int,
    *,
    max_points_per_side: int,
    random_state: int,
) -> np.ndarray:
    """Select the same balanced subset for both visualization panels."""

    count = min(int(n_per_side), int(max_points_per_side))
    if count < 1:
        raise ValueError("max_points_per_side must be positive")

    if count == n_per_side:
        left = np.arange(n_per_side, dtype=np.int64)
        right = np.arange(n_per_side, 2 * n_per_side, dtype=np.int64)
    else:
        rng = np.random.default_rng(random_state)
        left = np.sort(
            rng.choice(n_per_side, size=count, replace=False).astype(np.int64)
        )
        right_local = np.sort(
            rng.choice(n_per_side, size=count, replace=False).astype(np.int64)
        )
        right = n_per_side + right_local
    return np.concatenate([left, right])


def axis_limits(values: np.ndarray) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (-1.0, 1.0)
    low = float(np.min(finite))
    high = float(np.max(finite))
    if math.isclose(low, high):
        margin = max(abs(low) * 0.05, 1e-6)
    else:
        margin = 0.05 * (high - low)
    return low - margin, high + margin


def scatter_by_category(
    axis: plt.Axes,
    *,
    x: np.ndarray,
    y: np.ndarray,
    categories: np.ndarray,
    category_names: Mapping[int, str],
    title: str,
    point_size: float,
    alpha: float,
) -> None:
    for category in sorted(np.unique(categories)):
        mask = categories == category
        axis.scatter(
            x[mask],
            y[mask],
            s=point_size,
            alpha=alpha,
            label=category_names[int(category)],
            linewidths=0,
            rasterized=True,
        )
    axis.set_title(title)
    axis.grid(True, alpha=0.2)
    axis.legend(loc="best", frameon=True)

def build_figure(
    *,
    coordinates: np.ndarray,
    true_labels: np.ndarray,
    display_cluster_labels: np.ndarray,
    explained_variance_ratio: np.ndarray,
    pc_x: int,
    pc_y: int,
    metadata: Mapping[str, Any],
    metrics: Mapping[str, Any],
    projection_audit: Mapping[str, Any],
    point_size: float,
    alpha: float,
) -> plt.Figure:
    x = coordinates[:, pc_x]
    y = coordinates[:, pc_y]
    x_limits = axis_limits(x)
    y_limits = axis_limits(y)

    # Semantic class names corresponding to binary IDs 1 and 0.
    class_one_name, class_zero_name = control_label_names(
        str(metadata["control_type"])
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12.0, 5.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    scatter_by_category(
        axes[0],
        x=x,
        y=y,
        categories=true_labels,
        category_names={
            1: class_one_name,
            0: class_zero_name,
        },
        title="True comparison labels",
        point_size=point_size,
        alpha=alpha,
    )

    scatter_by_category(
        axes[1],
        x=x,
        y=y,
        categories=display_cluster_labels,
        category_names={
            1: f"Cluster → {class_one_name}",
            0: f"Cluster → {class_zero_name}",
        },
        title="K-Means assignments (class-aligned for display)",
        point_size=point_size,
        alpha=alpha,
    )

    x_ratio = 100.0 * float(explained_variance_ratio[pc_x])
    y_ratio = 100.0 * float(explained_variance_ratio[pc_y])

    for axis in axes:
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)
        axis.set_xlabel(
            f"PC{pc_x + 1} ({x_ratio:.2f}% variance)"
        )
        axis.set_ylabel(
            f"PC{pc_y + 1} ({y_ratio:.2f}% variance)"
        )

    audit_text = (
        "runner audit: not supplied"
        if projection_audit.get("full_audit_match") is None
        else (
            "runner audit: matched"
            if projection_audit.get("full_audit_match")
            else "runner audit: MISMATCH"
        )
    )

    figure.suptitle(
        (
            f"{metadata['cipher']} | r={metadata['round']}, "
            f"k={metadata['k']}, seed={metadata['seed']} | "
            f"{metadata['control_type']}\n"
            f"Silhouette={float(metrics['silhouette_kmeans']):.4f}, "
            f"ARI={float(metrics['kmeans_adjusted_rand']):.4f}, "
            f"aligned accuracy="
            f"{float(metrics['kmeans_aligned_accuracy']):.4f} | "
            f"{audit_text}"
        )
    )

    return figure

def per_run_output_dir(
    root: Path,
    visualization_study_id: str,
    row: Mapping[str, Any],
) -> Path:
    return (
        root
        / visualization_study_id
        / "per_run"
        / slug(row["domain"])
        / slug(row["cipher"])
        / f"r{int(row['round'])}"
        / f"k{int(row['k'])}"
        / f"seed{int(row['seed'])}"
    )


def save_figure(
    figure: plt.Figure,
    *,
    base_path: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[str]:
    paths: list[str] = []
    for file_format in formats:
        output_path = base_path.with_suffix(f".{file_format}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs: Dict[str, Any] = {
            "bbox_inches": "tight",
            "format": file_format,
        }
        if file_format == "png":
            save_kwargs["dpi"] = int(dpi)
        figure.savefig(output_path, **save_kwargs)
        paths.append(str(output_path))
    return paths


def write_points_csv(
    path: Path,
    *,
    coordinates: np.ndarray,
    true_labels: np.ndarray,
    raw_cluster_labels: np.ndarray,
    display_cluster_labels: np.ndarray,
    source_indices: np.ndarray,
    pc_x: int,
    pc_y: int,
) -> None:
    frame = pd.DataFrame(
        {
            "source_combined_index": source_indices.astype(int),
            f"pc{pc_x + 1}": coordinates[:, pc_x],
            f"pc{pc_y + 1}": coordinates[:, pc_y],
            "true_label": true_labels.astype(int),
            "kmeans_label_raw": raw_cluster_labels.astype(int),
            "kmeans_label_display_aligned": display_cluster_labels.astype(int),
        }
    )
    atomic_write_dataframe(path, frame)


def audit_metrics(
    *,
    expected: Optional[Mapping[str, Any]],
    context_projection_id: str,
    context_sample_hash: str,
    metrics: Mapping[str, Any],
    metric_atol: float,
) -> Dict[str, Any]:
    if expected is None:
        return {
            "expected_projection_id": None,
            "projection_id_match": None,
            "projection_contract_match": None,
            "expected_sample_selection_hash": None,
            "sample_selection_hash_match": None,
            "expected_silhouette_kmeans": None,
            "silhouette_abs_diff": None,
            "expected_kmeans_adjusted_rand": None,
            "ari_abs_diff": None,
            "expected_kmeans_aligned_accuracy": None,
            "aligned_accuracy_abs_diff": None,
            "metric_contract_match": None,
            "full_audit_match": None,
            "audit_source_csv": None,
        }

    expected_projection_id = str(expected["projection_id"])
    expected_sample_hash = str(expected["sample_selection_hash"])
    expected_silhouette = float(expected["silhouette_kmeans"])
    expected_ari = float(expected["kmeans_adjusted_rand"])
    expected_accuracy = float(expected["kmeans_aligned_accuracy"])

    projection_id_match = bool(
        context_projection_id == expected_projection_id
    )
    sample_hash_match = bool(context_sample_hash == expected_sample_hash)
    silhouette_diff = abs(
        float(metrics["silhouette_kmeans"]) - expected_silhouette
    )
    ari_diff = abs(float(metrics["kmeans_adjusted_rand"]) - expected_ari)
    accuracy_diff = abs(
        float(metrics["kmeans_aligned_accuracy"]) - expected_accuracy
    )
    projection_contract_match = bool(
        projection_id_match and sample_hash_match
    )
    metric_contract_match = bool(
        silhouette_diff <= metric_atol
        and ari_diff <= metric_atol
        and accuracy_diff <= metric_atol
    )

    return {
        "expected_projection_id": expected_projection_id,
        "projection_id_match": projection_id_match,
        "projection_contract_match": projection_contract_match,
        "expected_sample_selection_hash": expected_sample_hash,
        "sample_selection_hash_match": sample_hash_match,
        "expected_silhouette_kmeans": expected_silhouette,
        "silhouette_abs_diff": silhouette_diff,
        "expected_kmeans_adjusted_rand": expected_ari,
        "ari_abs_diff": ari_diff,
        "expected_kmeans_aligned_accuracy": expected_accuracy,
        "aligned_accuracy_abs_diff": accuracy_diff,
        "metric_contract_match": metric_contract_match,
        "full_audit_match": bool(
            projection_contract_match and metric_contract_match
        ),
        "audit_source_csv": expected.get("audit_source_csv"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate PCA/K-Means visualizations from manifest rows using "
            "the locked geometry projection protocol"
        )
    )
    add_common_cli_arguments(parser)
    parser.add_argument(
        "--geometry-long",
        nargs="+",
        help=(
            "Optional one or more projected_geometry_long.csv files used to "
            "audit regenerated projection IDs, sample hashes, and geometry "
            "metrics. Rows are filtered to the study IDs in --config."
        ),
    )
    parser.add_argument(
        "--control-types",
        nargs="+",
        default=list(DEFAULT_CONTROL_TYPES),
        choices=list(COMPARISON_DATASETS),
    )
    parser.add_argument(
        "--plot-pcs",
        type=int,
        nargs=2,
        metavar=("PC_X", "PC_Y"),
        default=(1, 2),
        help="One-based principal-component indices used for plotting.",
    )
    parser.add_argument(
        "--max-points-per-side",
        type=int,
        default=3000,
        help="Balanced plotting cap per side; estimator metrics still use the full selected context.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help="Any of: png pdf svg.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--point-size", type=float, default=8.0)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument(
        "--export-points",
        action="store_true",
        help="Export the plotted PCA coordinates and labels to CSV.",
    )
    parser.add_argument(
        "--metric-atol",
        type=float,
        default=1e-10,
        help="Absolute tolerance for regenerated geometry metric audit.",
    )
    parser.add_argument(
        "--allow-projection-mismatch",
        action="store_true",
        help="Continue even when regenerated projection IDs do not match --geometry-long.",
    )
    parser.add_argument(
        "--allow-metric-mismatch",
        action="store_true",
        help="Continue even when regenerated geometry metrics exceed --metric-atol.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = parse_formats(args.formats)

    if args.max_points_per_side <= 0:
        raise ValueError("--max-points-per-side must be positive")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    if args.point_size <= 0:
        raise ValueError("--point-size must be positive")
    if not (0 < args.alpha <= 1):
        raise ValueError("--alpha must lie in (0, 1]")
    if args.metric_atol < 0:
        raise ValueError("--metric-atol must be non-negative")

    config_path = Path(args.config).resolve()
    config, projection_protocol, geometry_protocol = load_config(config_path)
    _, rows = select_manifest_rows(args, str(config["source_study_id"]))

    pc_x_one_based, pc_y_one_based = map(int, args.plot_pcs)
    if pc_x_one_based == pc_y_one_based:
        raise ValueError("--plot-pcs must contain two different components")
    if pc_x_one_based < 1 or pc_y_one_based < 1:
        raise ValueError("--plot-pcs uses one-based positive indices")
    if max(pc_x_one_based, pc_y_one_based) > projection_protocol.pca_components:
        raise ValueError(
            f"Requested PC exceeds locked PCA dimension "
            f"{projection_protocol.pca_components}"
        )
    pc_x = pc_x_one_based - 1
    pc_y = pc_y_one_based - 1

    geometry_long_paths = (
        [Path(value).resolve() for value in args.geometry_long]
        if args.geometry_long
        else None
    )
    geometry_long = load_geometry_long(
        geometry_long_paths,
        expected_estimator_study_id=str(config["estimator_study_id"]),
        expected_source_study_id=str(config["source_study_id"]),
    )

    visualization_study_id = (
        f"{config['estimator_study_id']}_pca_visualizations"
    )
    output_root = Path(args.out).resolve()
    study_root = output_root / visualization_study_id
    study_root.mkdir(parents=True, exist_ok=True)

    config_snapshot = {
        "visualization_study_id": visualization_study_id,
        "visualization_schema_version": VISUALIZATION_SCHEMA_VERSION,
        "source_estimator_study_id": config["estimator_study_id"],
        "source_study_id": config["source_study_id"],
        "source_geometry_config": str(config_path),
        "projection_protocol": asdict(projection_protocol),
        "geometry_protocol": asdict(geometry_protocol),
        "plot_pcs": [pc_x_one_based, pc_y_one_based],
        "max_points_per_side": args.max_points_per_side,
        "formats": list(formats),
        "dpi": args.dpi,
        "point_size": args.point_size,
        "alpha": args.alpha,
        "control_types": list(args.control_types),
        "geometry_long": (
            [str(path) for path in geometry_long_paths]
            if geometry_long_paths
            else None
        ),
        "metric_atol": args.metric_atol,
        "allow_projection_mismatch": args.allow_projection_mismatch,
        "allow_metric_mismatch": args.allow_metric_mismatch,
        "generated_at_utc": utc_now(),
    }
    atomic_write_json(
        study_root / "visualization_config.json",
        config_snapshot,
    )

    result_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    total = len(rows) * len(args.control_types)
    completed = 0

    for manifest_row in rows:
        if args.dry_run:
            for control_type in args.control_types:
                completed += 1
                output_dir = per_run_output_dir(
                    output_root,
                    visualization_study_id,
                    manifest_row,
                )
                stem = (
                    f"pca_{slug(control_type)}_"
                    f"pc{pc_x_one_based}_pc{pc_y_one_based}"
                )
                print(
                    f"[DRY {completed}/{total}] "
                    f"{manifest_row['cipher']} r={manifest_row['round']} "
                    f"k={manifest_row['k']} seed={manifest_row['seed']} "
                    f"{control_type} -> {output_dir / stem}"
                )
            continue

        try:
            context = prepare_projected_context(
                d1_path=Path(manifest_row["d1_path"]),
                d0_path=Path(manifest_row["d0_path"]),
                d0_control_a_path=Path(
                    manifest_row["d0_control_a_path"]
                ),
                d0_control_b_path=Path(
                    manifest_row["d0_control_b_path"]
                ),
                d1_control_a_path=Path(
                    manifest_row["d1_control_a_path"]
                ),
                d1_control_b_path=Path(
                    manifest_row["d1_control_b_path"]
                ),
                pair_shuffled_path=Path(
                    manifest_row["pair_shuffled_path"]
                ),
                projection_protocol=projection_protocol,
                run_metadata={
                    "study_id": manifest_row["study_id"],
                    "domain": manifest_row["domain"],
                    "cipher": manifest_row["cipher"],
                    "round": manifest_row["round"],
                    "k": manifest_row["k"],
                    "seed": manifest_row["seed"],
                },
            )

            for control_type in args.control_types:
                completed += 1
                output_dir = per_run_output_dir(
                    output_root,
                    visualization_study_id,
                    manifest_row,
                )
                stem = (
                    f"pca_{slug(control_type)}_"
                    f"pc{pc_x_one_based}_pc{pc_y_one_based}"
                )
                base_path = output_dir / stem
                metadata_path = output_dir / f"{stem}.json"
                points_path = output_dir / f"{stem}_points.csv"

                existing_paths = [
                    base_path.with_suffix(f".{file_format}")
                    for file_format in formats
                ]
                if (
                    not args.overwrite
                    and metadata_path.exists()
                    and all(path.exists() for path in existing_paths)
                ):
                    metadata = json.loads(
                        metadata_path.read_text(encoding="utf-8")
                    )
                    if (
                        metadata.get("visualization_schema_version")
                        == VISUALIZATION_SCHEMA_VERSION
                    ):
                        print(
                            f"[SKIP {completed}/{total}] "
                            f"{manifest_row['cipher']} "
                            f"r={manifest_row['round']} "
                            f"k={manifest_row['k']} "
                            f"seed={manifest_row['seed']} "
                            f"{control_type}"
                        )
                        result_rows.append(metadata["manifest_row"])
                        continue
                    print(
                        f"[REBUILD {completed}/{total}] incompatible "
                        f"visualization metadata schema for "
                        f"{manifest_row['cipher']} "
                        f"r={manifest_row['round']} "
                        f"k={manifest_row['k']} "
                        f"seed={manifest_row['seed']} "
                        f"{control_type}"
                    )

                left, right = context.comparison(control_type)
                metrics = fit_kmeans_metrics(
                    left,
                    right,
                    geometry_protocol,
                    random_state=geometry_protocol.random_state,
                )
                raw_labels = np.asarray(metrics.pop("labels"), dtype=np.int32)
                n_per_side = int(metrics["n_samples_per_side"])
                combined = np.concatenate(
                    [left[:n_per_side], right[:n_per_side]], axis=0
                )
                true_labels_full = np.concatenate(
                    [
                        np.ones(n_per_side, dtype=np.int32),
                        np.zeros(n_per_side, dtype=np.int32),
                    ]
                )
                display_labels_full, labels_inverted = align_cluster_labels(
                    true_labels_full,
                    raw_labels,
                )

                plot_indices = plotting_indices(
                    n_per_side,
                    max_points_per_side=args.max_points_per_side,
                    random_state=stable_plot_random_state(
                        manifest_row,
                        base_random_state=geometry_protocol.random_state,
                        control_type=control_type,
                    ),
                )
                plot_coordinates = combined[plot_indices]
                plot_true_labels = true_labels_full[plot_indices]
                plot_raw_labels = raw_labels[plot_indices]
                plot_display_labels = display_labels_full[plot_indices]

                expected = expected_geometry_row(
                    geometry_long,
                    row=manifest_row,
                    control_type=control_type,
                    estimator_study_id=str(config["estimator_study_id"]),
                    source_study_id=str(config["source_study_id"]),
                )
                audit = audit_metrics(
                    expected=expected,
                    context_projection_id=context.projection_id,
                    context_sample_hash=context.sample_selection_hash,
                    metrics=metrics,
                    metric_atol=args.metric_atol,
                )

                if expected is not None:
                    if (
                        not audit["projection_contract_match"]
                        and not args.allow_projection_mismatch
                    ):
                        raise ValueError(
                            "Regenerated visualization does not match the "
                            "geometry runner projection contract: "
                            f"projection_id_match="
                            f"{audit['projection_id_match']}, "
                            f"sample_selection_hash_match="
                            f"{audit['sample_selection_hash_match']}"
                        )
                    if (
                        not audit["metric_contract_match"]
                        and not args.allow_metric_mismatch
                    ):
                        raise ValueError(
                            "Regenerated geometry metrics do not match the "
                            "runner output within metric_atol="
                            f"{args.metric_atol}: "
                            f"silhouette_abs_diff="
                            f"{audit['silhouette_abs_diff']}, "
                            f"ari_abs_diff={audit['ari_abs_diff']}, "
                            f"aligned_accuracy_abs_diff="
                            f"{audit['aligned_accuracy_abs_diff']}"
                        )

                figure_metadata = {
                    "domain": manifest_row["domain"],
                    "cipher": manifest_row["cipher"],
                    "round": int(manifest_row["round"]),
                    "k": int(manifest_row["k"]),
                    "seed": int(manifest_row["seed"]),
                    "control_type": control_type,
                }
                figure = build_figure(
                    coordinates=plot_coordinates,
                    true_labels=plot_true_labels,
                    display_cluster_labels=plot_display_labels,
                    explained_variance_ratio=np.asarray(
                        context.pca.explained_variance_ratio_,
                        dtype=np.float64,
                    ),
                    pc_x=pc_x,
                    pc_y=pc_y,
                    metadata=figure_metadata,
                    metrics=metrics,
                    projection_audit=audit,
                    point_size=args.point_size,
                    alpha=args.alpha,
                )
                figure_paths = save_figure(
                    figure,
                    base_path=base_path,
                    formats=formats,
                    dpi=args.dpi,
                )
                plt.close(figure)

                points_csv_value: Optional[str] = None
                if args.export_points:
                    write_points_csv(
                        points_path,
                        coordinates=plot_coordinates,
                        true_labels=plot_true_labels,
                        raw_cluster_labels=plot_raw_labels,
                        display_cluster_labels=plot_display_labels,
                        source_indices=plot_indices,
                        pc_x=pc_x,
                        pc_y=pc_y,
                    )
                    points_csv_value = str(points_path)

                manifest_output_row = {
                    "visualization_study_id": visualization_study_id,
                    "visualization_schema_version": (
                        VISUALIZATION_SCHEMA_VERSION
                    ),
                    "source_estimator_study_id": config[
                        "estimator_study_id"
                    ],
                    "source_study_id": config["source_study_id"],
                    "study_id": manifest_row["study_id"],
                    "domain": manifest_row["domain"],
                    "cipher": manifest_row["cipher"],
                    "round": int(manifest_row["round"]),
                    "k": int(manifest_row["k"]),
                    "seed": int(manifest_row["seed"]),
                    "control_type": control_type,
                    "pc_x": pc_x_one_based,
                    "pc_y": pc_y_one_based,
                    "selected_count_per_dataset": int(
                        context.selected_count_per_dataset
                    ),
                    "plotted_points_per_side": int(len(plot_indices) // 2),
                    "projection_id": context.projection_id,
                    "expected_projection_id": audit[
                        "expected_projection_id"
                    ],
                    "projection_id_match": audit["projection_id_match"],
                    "projection_contract_match": audit[
                        "projection_contract_match"
                    ],
                    "sample_selection_hash": context.sample_selection_hash,
                    "expected_sample_selection_hash": audit[
                        "expected_sample_selection_hash"
                    ],
                    "sample_selection_hash_match": audit[
                        "sample_selection_hash_match"
                    ],
                    "silhouette_kmeans": float(
                        metrics["silhouette_kmeans"]
                    ),
                    "expected_silhouette_kmeans": audit[
                        "expected_silhouette_kmeans"
                    ],
                    "silhouette_abs_diff": audit["silhouette_abs_diff"],
                    "kmeans_adjusted_rand": float(
                        metrics["kmeans_adjusted_rand"]
                    ),
                    "expected_kmeans_adjusted_rand": audit[
                        "expected_kmeans_adjusted_rand"
                    ],
                    "ari_abs_diff": audit["ari_abs_diff"],
                    "kmeans_aligned_accuracy": float(
                        metrics["kmeans_aligned_accuracy"]
                    ),
                    "expected_kmeans_aligned_accuracy": audit[
                        "expected_kmeans_aligned_accuracy"
                    ],
                    "aligned_accuracy_abs_diff": audit[
                        "aligned_accuracy_abs_diff"
                    ],
                    "metric_contract_match": audit[
                        "metric_contract_match"
                    ],
                    "full_audit_match": audit["full_audit_match"],
                    "audit_source_csv": audit["audit_source_csv"],
                    "cluster_labels_inverted_for_display": labels_inverted,
                    "pair_shuffle_k1_identical": (
                        context.pair_shuffle_k1_identical
                    ),
                    "figure_paths": "|".join(figure_paths),
                    "points_csv": points_csv_value,
                    "metadata_json": str(metadata_path),
                    "status": "completed",
                    "error": "",
                }

                metadata_payload = {
                    "visualization_schema_version": (
                        VISUALIZATION_SCHEMA_VERSION
                    ),
                    "generated_at_utc": utc_now(),
                    "manifest_source_row": json_safe(manifest_row),
                    "visualization": {
                        "control_type": control_type,
                        "plot_pcs": [
                            pc_x_one_based,
                            pc_y_one_based,
                        ],
                        "max_points_per_side": args.max_points_per_side,
                        "plotted_points_per_side": int(
                            len(plot_indices) // 2
                        ),
                        "formats": list(formats),
                        "figure_paths": figure_paths,
                        "points_csv": points_csv_value,
                        "cluster_labels_inverted_for_display": labels_inverted,
                        "note": (
                            "Cluster IDs are aligned to class IDs for display "
                            "only. ARI and K-Means fitting remain permutation-"
                            "invariant and unchanged."
                        ),
                    },
                    "projection": projection_summary(context),
                    "projection_protocol": asdict(projection_protocol),
                    "geometry_protocol": asdict(geometry_protocol),
                    "regenerated_metrics": json_safe(metrics),
                    "geometry_long_audit": json_safe(audit),
                    "manifest_row": json_safe(manifest_output_row),
                }
                atomic_write_json(metadata_path, metadata_payload)
                result_rows.append(manifest_output_row)

                print(
                    f"[OK {completed}/{total}] "
                    f"{manifest_row['cipher']} r={manifest_row['round']} "
                    f"k={manifest_row['k']} seed={manifest_row['seed']} "
                    f"{control_type} | "
                    f"S={float(metrics['silhouette_kmeans']):.4f} "
                    f"ARI={float(metrics['kmeans_adjusted_rand']):.4f}"
                )

        except Exception as exc:
            failure = {
                "study_id": manifest_row.get("study_id"),
                "domain": manifest_row.get("domain"),
                "cipher": manifest_row.get("cipher"),
                "round": manifest_row.get("round"),
                "k": manifest_row.get("k"),
                "seed": manifest_row.get("seed"),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            print(
                f"[FAIL] {manifest_row.get('cipher')} "
                f"r={manifest_row.get('round')} "
                f"k={manifest_row.get('k')} "
                f"seed={manifest_row.get('seed')} | {failure['error']}"
            )
            if not args.continue_on_error:
                raise

    manifest_path = study_root / "projected_geometry_visualization_manifest.csv"
    atomic_write_csv(
        manifest_path,
        result_rows,
        VISUALIZATION_COLUMNS,
    )
    atomic_write_json(
        study_root / "projected_geometry_visualization_report.json",
        {
            "visualization_study_id": visualization_study_id,
            "visualization_schema_version": VISUALIZATION_SCHEMA_VERSION,
            "generated_at_utc": utc_now(),
            "selected_manifest_rows": len(rows),
            "selected_control_types": list(args.control_types),
            "expected_visualizations": total,
            "completed_visualizations": len(
                [row for row in result_rows if row["status"] == "completed"]
            ),
            "failure_count": len(failures),
            "full_audit_match_count": int(
                sum(row.get("full_audit_match") is True for row in result_rows)
            ),
            "audit_mismatch_count": int(
                sum(row.get("full_audit_match") is False for row in result_rows)
            ),
            "pair_shuffle_k1_warning_count": int(
                sum(
                    int(row.get("k", -1)) == 1
                    and row.get("pair_shuffle_k1_identical") is False
                    for row in result_rows
                )
            ),
            "failures": failures,
            "projection_audit_enabled": geometry_long is not None,
            "geometry_long_inputs": (
                [str(path) for path in geometry_long_paths]
                if geometry_long_paths
                else []
            ),
            "metric_atol": args.metric_atol,
            "allow_projection_mismatch": args.allow_projection_mismatch,
            "allow_metric_mismatch": args.allow_metric_mismatch,
            "manifest_csv": str(manifest_path),
            "study_root": str(study_root),
        },
    )

    print("=" * 88)
    print("PROJECTED-GEOMETRY VISUALIZATION")
    print("=" * 88)
    print(f"Study root              : {study_root}")
    print(f"Selected manifest rows  : {len(rows)}")
    print(f"Control types           : {','.join(args.control_types)}")
    print(f"Completed visualizations: {len(result_rows)}")
    print(f"Failures                : {len(failures)}")
    print(f"Visualization manifest  : {manifest_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()


# PRESENT-80 example:
# python visualize_projected_geometry.py \
#   --manifest analysis_data/present80_rkmp_controls_v1/manifest.csv \
#   --config configs/present80/projected_geometry_full.json \
#   --geometry-long \
#     analysis_results/present80_projected_geometry_full/projected_geometry_long.csv \
#   --out analysis_results \
#   --rounds 7 8 9 \
#   --k-values 1 8 16 32 \
#   --seeds 201 \
#   --control-types D1_vs_D0 \
#   --formats png  \

# SIMECK-32/64 example:
# 
# python visualize_projected_geometry.py \
#   --manifest analysis_data/simeck3264_rkmp_controls_v1/manifest.csv \
#   --config configs/simeck3264_projected_geometry_full.json \
#   --geometry-long \
#     analysis_results/simeck3264_projected_geometry_full/projected_geometry_long.csv \
#   --out analysis_results \
#   --rounds 8 \
#   --k-values 1 \
#   --seeds 201 \
#   --control-types D1_vs_D0 \
#   --formats png \

# python visualize_projected_geometry.py \
#   --manifest analysis_data/simeck3264_rkmp_controls_v1/manifest.csv \
#   --config configs/simeck3264_projected_geometry_full.json \
#   --geometry-long \
#     analysis_results/simeck3264_projected_geometry_full/projected_geometry_long.csv \
#   --out analysis_results \
#   --rounds 12 \
#   --k-values 4 32 \
#   --seeds 201 \
#   --control-types D1_vs_D0 \
#   --formats png \
