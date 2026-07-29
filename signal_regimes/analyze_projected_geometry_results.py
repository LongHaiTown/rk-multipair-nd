"""Analyze cipher-agnostic projected PCA/K-Means geometry results.

This analyzer is the projected-geometric flow. It summarizes PCA variance,
K-Means silhouette, same-class-adjusted silhouette, class alignment (ARI),
initialization stability, and pair-shuffled mechanism controls.

Geometry qualifiers are descriptive and control-relative. They do not assign
the primary Signal Regime and are not used as hard gates for MMD evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


CONTROL_TYPES = ("D1_vs_D0", "D0_vs_D0", "D1_vs_D1", "pair_shuffled")
IDENTITY_CANDIDATES = (
    "estimator_study_id",
    "source_study_id",
    "domain",
    "cipher",
)

REQUIRED_COLUMNS = {
    "cipher",
    "round",
    "k",
    "seed",
    "control_type",
    "n_samples_total",
    "pca_components",
    "explained_variance_sum",
    "projection_id",
    "sample_selection_hash",
    "silhouette_kmeans",
    "silhouette_control_median",
    "silhouette_excess_null",
    "kmeans_adjusted_rand",
    "kmeans_aligned_accuracy",
    "kmeans_stability_ari",
    "status",
}

AGGREGATE_METRICS = (
    "explained_variance_sum",
    "silhouette_kmeans",
    "silhouette_control_median",
    "silhouette_null_median",
    "silhouette_excess_null",
    "kmeans_inertia",
    "kmeans_adjusted_rand",
    "kmeans_aligned_accuracy",
    "kmeans_n_iter",
    "kmeans_stability_ari",
    "kmeans_stability_ari_median",
    "kmeans_stability_ari_min",
    "stability_pair_count",
)

TRAJECTORY_METRICS = (
    "silhouette_excess_null",
    "kmeans_adjusted_rand",
    "kmeans_aligned_accuracy",
    "kmeans_stability_ari",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_json(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.replace({np.nan: None}).to_dict(orient="records")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    return value


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(safe_json(dict(payload)), handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_write_df(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def parse_nullable_bool(value: Any) -> Optional[bool]:
    """Parse booleans robustly after CSV round-trips."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    if text in {"", "nan", "none", "null"}:
        return None
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def load_many_results(paths: Sequence[Path]) -> pd.DataFrame:
    """Load and combine one or more compatible long-form result CSVs."""

    if not paths:
        raise ValueError("At least one input CSV is required")

    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = load_results(path)
        frame = frame.copy()
        frame["input_csv"] = str(path)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    duplicate_keys = job_columns(combined) + ["control_type"]
    duplicates = combined.duplicated(duplicate_keys, keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate comparison rows across input CSVs:\n"
            + combined.loc[
                duplicates, duplicate_keys + ["input_csv"]
            ].to_string(index=False)
        )

    return combined.sort_values(duplicate_keys).reset_index(drop=True)


def identity_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in IDENTITY_CANDIDATES if column in frame.columns]


def job_columns(frame: pd.DataFrame) -> list[str]:
    return identity_columns(frame) + ["round", "k", "seed"]


def config_columns(frame: pd.DataFrame) -> list[str]:
    return identity_columns(frame) + ["round", "k"]


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("Input geometry long-form CSV is empty")

    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Input geometry CSV is missing columns: {missing}")

    numeric_columns = {
        "round",
        "k",
        "seed",
        "n_samples_total",
        "pca_components",
        *AGGREGATE_METRICS,
    }
    for column in sorted(numeric_columns.intersection(frame.columns)):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame[["round", "k", "seed"]].isna().any().any():
        raise ValueError("round, k, and seed must be numeric")
    frame[["round", "k", "seed"]] = frame[["round", "k", "seed"]].astype(int)

    if (frame["k"] <= 0).any():
        raise ValueError("k must be positive")
    if (frame["round"] < 0).any():
        raise ValueError("round must be non-negative")
    if frame["cipher"].astype(str).str.strip().eq("").any():
        raise ValueError("cipher values must be non-empty")

    frame["control_type"] = frame["control_type"].astype(str)
    unknown = sorted(set(frame["control_type"]) - set(CONTROL_TYPES))
    if unknown:
        raise ValueError(f"Unknown control types: {unknown}")

    completed = frame["status"].astype(str).str.lower().eq("completed")
    if not completed.all():
        frame = frame.loc[completed].copy()
    if frame.empty:
        raise ValueError("No completed geometry rows remain")

    duplicate_keys = job_columns(frame) + ["control_type"]
    duplicates = frame.duplicated(duplicate_keys, keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate geometry comparison rows:\n"
            + frame.loc[duplicates, duplicate_keys].to_string(index=False)
        )

    return frame.sort_values(duplicate_keys).reset_index(drop=True)


def filter_results(
    frame: pd.DataFrame,
    *,
    rounds: Optional[Sequence[int]],
    k_values: Optional[Sequence[int]],
    seeds: Optional[Sequence[int]],
    domains: Optional[Sequence[str]],
    ciphers: Optional[Sequence[str]],
) -> pd.DataFrame:
    output = frame.copy()
    if rounds:
        output = output[output["round"].isin(rounds)]
    if k_values:
        output = output[output["k"].isin(k_values)]
    if seeds:
        output = output[output["seed"].isin(seeds)]
    if domains and "domain" in output.columns:
        output = output[output["domain"].isin(domains)]
    if ciphers and "cipher" in output.columns:
        output = output[output["cipher"].isin(ciphers)]
    if output.empty:
        raise ValueError("No geometry rows remain after filters")
    return output.reset_index(drop=True)



def projection_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = job_columns(frame)

    for key, group in frame.groupby(keys, sort=True, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key_values))

        observed_controls = set(group["control_type"].astype(str))
        missing_controls = sorted(set(CONTROL_TYPES) - observed_controls)
        unexpected_controls = sorted(observed_controls - set(CONTROL_TYPES))

        projection_valid = bool(
            group["projection_id"].nunique(dropna=False) == 1
            and group["sample_selection_hash"].nunique(dropna=False) == 1
                    )
        control_valid = not missing_controls and not unexpected_controls

        row.update(
            {
                "comparison_rows": int(len(group)),
                "control_types": ",".join(sorted(observed_controls)),
                "missing_control_types": ",".join(missing_controls),
                "unexpected_control_types": ",".join(unexpected_controls),
                "control_contract_valid": bool(control_valid),
                "projection_id_count": int(
                    group["projection_id"].nunique(dropna=False)
                ),
                "sample_selection_hash_count": int(
                    group["sample_selection_hash"].nunique(dropna=False)
                ),
                "projection_contract_valid": projection_valid,
                "job_contract_valid": bool(projection_valid and control_valid),
            }
        )

        if "pair_shuffle_k1_identical" in group.columns:
            parsed_flags = [
                parse_nullable_bool(value)
                for value in group["pair_shuffle_k1_identical"].tolist()
            ]
            parsed_flags = [value for value in parsed_flags if value is not None]
            unique_flags = set(parsed_flags)
            row["pair_shuffle_k1_identical"] = (
                parsed_flags[0] if len(unique_flags) == 1 else None
            )
            row["pair_shuffle_k1_flag_consistent"] = len(unique_flags) <= 1

        rows.append(row)

    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def completeness_matrix(
    frame: pd.DataFrame,
    *,
    expected_rounds: Optional[Sequence[int]],
    expected_k_values: Optional[Sequence[int]],
    expected_seeds: Optional[Sequence[int]],
) -> pd.DataFrame:
    """Build completeness per identity.

    When expected axes are omitted, each cipher/study identity is validated
    against its own observed round, aggregation, and seed grid. This avoids
    false missing-cell reports when a combined CSV contains ciphers with
    different reduced-round ranges.
    """

    ids = identity_columns(frame)
    grouped = (
        frame.groupby(ids, sort=True, dropna=False)
        if ids
        else [((), frame)]
    )

    rows: list[dict[str, Any]] = []
    for identity_key, identity_group in grouped:
        identity_values = (
            identity_key if isinstance(identity_key, tuple) else (identity_key,)
        )
        identity = dict(zip(ids, identity_values))

        rounds = sorted(
            set(map(int, expected_rounds))
            if expected_rounds
            else set(identity_group["round"].astype(int))
        )
        k_values = sorted(
            set(map(int, expected_k_values))
            if expected_k_values
            else set(identity_group["k"].astype(int))
        )
        seeds = sorted(
            set(map(int, expected_seeds))
            if expected_seeds
            else set(identity_group["seed"].astype(int))
        )

        present = {
            (
                int(row["round"]),
                int(row["k"]),
                int(row["seed"]),
                str(row["control_type"]),
            )
            for _, row in identity_group.iterrows()
        }

        for round_number in rounds:
            for k in k_values:
                for control_type in CONTROL_TYPES:
                    completed = sorted(
                        seed
                        for seed in seeds
                        if (
                            int(round_number),
                            int(k),
                            int(seed),
                            control_type,
                        )
                        in present
                    )
                    missing = sorted(set(seeds) - set(completed))
                    rows.append(
                        {
                            **identity,
                            "round": int(round_number),
                            "k": int(k),
                            "control_type": control_type,
                            "expected_seed_count": len(seeds),
                            "completed_seed_count": len(completed),
                            "completed_seeds": ",".join(map(str, completed)),
                            "missing_seeds": ",".join(map(str, missing)),
                            "complete": not missing,
                        }
                    )

    columns = ids + [
        "round",
        "k",
        "control_type",
        "expected_seed_count",
        "completed_seed_count",
        "completed_seeds",
        "missing_seeds",
        "complete",
    ]
    return pd.DataFrame(rows, columns=columns)

def aggregate_by_control(frame: pd.DataFrame) -> pd.DataFrame:
    group_keys = identity_columns(frame) + ["control_type", "round", "k"]
    rows: list[dict[str, Any]] = []

    for key, group in frame.groupby(group_keys, sort=True, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        row: dict[str, Any] = dict(zip(group_keys, key_values))
        row.update(
            {
                "n_seeds": int(group["seed"].nunique()),
                "seeds": ",".join(map(str, sorted(group["seed"].unique()))),
                "n_samples_total_median": float(group["n_samples_total"].median()),
                "pca_components_median": float(group["pca_components"].median()),
            }
        )
        for metric in AGGREGATE_METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else math.nan
            row[f"{metric}_median"] = (
                float(values.median()) if len(values) else math.nan
            )
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            row[f"{metric}_min"] = float(values.min()) if len(values) else math.nan
            row[f"{metric}_max"] = float(values.max()) if len(values) else math.nan
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_keys).reset_index(drop=True)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return math.nan
    x_rank = pd.Series(x[mask]).rank().to_numpy()
    y_rank = pd.Series(y[mask]).rank().to_numpy()
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return math.nan
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def aggregation_response(summary: pd.DataFrame) -> pd.DataFrame:
    group_keys = identity_columns(summary) + ["control_type", "round"]
    rows: list[dict[str, Any]] = []

    for key, group in summary.groupby(group_keys, sort=True, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_keys, key_values))
        group = group.sort_values("k")
        x = np.log2(group["k"].to_numpy(dtype=float))

        for metric in TRAJECTORY_METRICS:
            column = f"{metric}_median"
            if column not in group.columns:
                continue
            y = group[column].to_numpy(dtype=float)
            mask = np.isfinite(y)
            if int(mask.sum()) >= 2:
                slope, _ = np.polyfit(x[mask], y[mask], 1)
                gain = float(y[mask][-1] - y[mask][0])
                monotone = bool(np.all(np.diff(y[mask]) >= -1e-12))
            else:
                slope = gain = math.nan
                monotone = False
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "n_k": int(mask.sum()),
                    "spearman_rho_log2k": spearman(x, y),
                    "slope_per_doubling_k": float(slope),
                    "gain_kmax_minus_kmin": gain,
                    "monotone_non_decreasing": monotone,
                }
            )

    return pd.DataFrame(rows)


def geometry_qualifiers(
    frame: pd.DataFrame,
    *,
    stability_warning_threshold: float,
    ari_min_effect: float,
    silhouette_min_effect: float,
) -> pd.DataFrame:
    keys = config_columns(frame)
    main = frame[frame["control_type"] == "D1_vs_D0"]
    controls = frame[frame["control_type"].isin(["D0_vs_D0", "D1_vs_D1"])]
    rows: list[dict[str, Any]] = []

    for key, group in main.groupby(keys, sort=True, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key_values))
        mask = pd.Series(True, index=controls.index)
        for column, value in row.items():
            mask &= controls[column].eq(value)
        control_group = controls.loc[mask]

        ari_main = float(group["kmeans_adjusted_rand"].median())
        silhouette_excess = float(group["silhouette_excess_null"].median())
        stability = float(group["kmeans_stability_ari"].median())

        control_ari_values = pd.to_numeric(
            control_group["kmeans_adjusted_rand"], errors="coerce"
        ).dropna()
        control_ari_q95 = (
            float(control_ari_values.quantile(0.95))
            if len(control_ari_values)
            else math.nan
        )
        ari_effect_threshold = max(
            float(ari_min_effect),
            control_ari_q95 if math.isfinite(control_ari_q95) else -math.inf,
        )
        ari_above_control = bool(ari_main >= ari_effect_threshold)
        silhouette_above_control = bool(
            silhouette_excess >= float(silhouette_min_effect)
        )

        if ari_above_control and silhouette_above_control:
            qualifier = "compact_class_aligned"
        elif ari_above_control:
            qualifier = "class_aligned_diffuse"
        else:
            qualifier = "no_resolved_kmeans_geometry"

        row.update(
            {
                "n_seeds": int(group["seed"].nunique()),
                "silhouette_kmeans_median": float(
                    group["silhouette_kmeans"].median()
                ),
                "silhouette_control_median": float(
                    group["silhouette_control_median"].median()
                ),
                "silhouette_excess_null_median": silhouette_excess,
                "kmeans_adjusted_rand_median": ari_main,
                "kmeans_aligned_accuracy_median": float(
                    group["kmeans_aligned_accuracy"].median()
                ),
                "kmeans_stability_ari_median": stability,
                "same_class_ari_q95": control_ari_q95,
                "ari_min_effect": float(ari_min_effect),
                "ari_effect_threshold": ari_effect_threshold,
                "ari_above_effect_threshold": ari_above_control,
                "silhouette_min_effect": float(silhouette_min_effect),
                "silhouette_above_effect_threshold": silhouette_above_control,
                "geometry_qualifier": qualifier,
                "stability_warning": bool(stability < stability_warning_threshold),
                "stability_warning_threshold": float(stability_warning_threshold),
                "qualifier_scope": "descriptive_control_relative",
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def pair_shuffled_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    keys = config_columns(frame)
    metrics = (
        "silhouette_kmeans",
        "silhouette_excess_null",
        "kmeans_adjusted_rand",
        "kmeans_aligned_accuracy",
        "kmeans_stability_ari",
    )
    medians = (
        frame[frame["control_type"].isin(["D1_vs_D0", "pair_shuffled"])]
        .groupby(keys + ["control_type"], sort=True)[list(metrics)]
        .median()
        .reset_index()
    )

    main = medians[medians["control_type"] == "D1_vs_D0"].drop(
        columns="control_type"
    )
    shuffled = medians[medians["control_type"] == "pair_shuffled"].drop(
        columns="control_type"
    )
    merged = main.merge(shuffled, on=keys, suffixes=("_main", "_pair_shuffled"))
    for metric in metrics:
        merged[f"main_minus_pair_shuffled_{metric}"] = (
            merged[f"{metric}_main"] - merged[f"{metric}_pair_shuffled"]
        )
    return merged.sort_values(keys).reset_index(drop=True)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one or more cipher-agnostic projected geometry "
            "long-form CSVs"
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help=(
            "One or more projected_geometry_long.csv files. Multiple cipher "
            "studies can be analyzed together."
        ),
    )
    parser.add_argument(
        "--out",
        default="analysis_results/projected_geometry_analysis",
    )
    parser.add_argument("--rounds", type=int, nargs="+")
    parser.add_argument("--k-values", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--domains", nargs="+")
    parser.add_argument("--ciphers", nargs="+")
    parser.add_argument(
        "--expected-rounds",
        type=int,
        nargs="+",
        help=(
            "Optional global expected rounds. Omit for combined multi-cipher "
            "inputs so each identity uses its own observed round grid."
        ),
    )
    parser.add_argument("--expected-k-values", type=int, nargs="+")
    parser.add_argument("--expected-seeds", type=int, nargs="+")
    parser.add_argument("--stability-warning-threshold", type=float, default=0.5)
    parser.add_argument(
        "--ari-min-effect",
        type=float,
        default=0.01,
        help="Minimum practical ARI used only for descriptive qualifiers.",
    )
    parser.add_argument(
        "--silhouette-min-effect",
        type=float,
        default=0.001,
        help=(
            "Minimum practical excess silhouette used only for descriptive "
            "qualifiers."
        ),
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if any expected round/k/control cell is missing a seed.",
    )
    parser.add_argument(
        "--fail-on-audit-error",
        action="store_true",
        help=(
            "Fail on projection/control-contract errors or a failed k=1 "
            "pair-shuffle invariant."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0 <= args.stability_warning_threshold <= 1):
        raise ValueError("stability-warning-threshold must lie in [0, 1]")
    if args.ari_min_effect < 0:
        raise ValueError("ari-min-effect must be non-negative")
    if args.silhouette_min_effect < 0:
        raise ValueError("silhouette-min-effect must be non-negative")
    if args.k_values and any(value <= 0 for value in args.k_values):
        raise ValueError("All k-values must be positive")

    input_paths = [Path(value).resolve() for value in args.input]
    output_dir = Path(args.out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = filter_results(
        load_many_results(input_paths),
        rounds=args.rounds,
        k_values=args.k_values,
        seeds=args.seeds,
        domains=args.domains,
        ciphers=args.ciphers,
    )

    if (frame["k"] <= 0).any():
        raise ValueError("Input CSV contains non-positive k values")

    audit = projection_audit(frame)
    completeness = completeness_matrix(
        frame,
        expected_rounds=args.expected_rounds,
        expected_k_values=args.expected_k_values,
        expected_seeds=args.expected_seeds,
    )
    summary = aggregate_by_control(frame)
    response = aggregation_response(summary)
    qualifiers = geometry_qualifiers(
        frame,
        stability_warning_threshold=args.stability_warning_threshold,
        ari_min_effect=args.ari_min_effect,
        silhouette_min_effect=args.silhouette_min_effect,
    )
    shuffled = pair_shuffled_contrasts(frame)

    outputs = {
        "seed_level": output_dir / "projected_geometry_seed_level.csv",
        "projection_audit": output_dir
        / "projected_geometry_projection_audit.csv",
        "completeness": output_dir / "projected_geometry_completeness.csv",
        "summary": output_dir
        / "projected_geometry_summary_by_round_k_control.csv",
        "aggregation_response": output_dir
        / "projected_geometry_aggregation_response.csv",
        "qualifiers": output_dir
        / "projected_geometry_configuration_qualifiers.csv",
        "pair_shuffled": output_dir
        / "projected_geometry_pair_shuffled_contrasts.csv",
        "report": output_dir / "projected_geometry_analysis_report.json",
    }

    atomic_write_df(outputs["seed_level"], frame)
    atomic_write_df(outputs["projection_audit"], audit)
    atomic_write_df(outputs["completeness"], completeness)
    atomic_write_df(outputs["summary"], summary)
    atomic_write_df(outputs["aggregation_response"], response)
    atomic_write_df(outputs["qualifiers"], qualifiers)
    atomic_write_df(outputs["pair_shuffled"], shuffled)

    k1_warning_jobs = 0
    if "pair_shuffle_k1_identical" in audit.columns:
        k1_warning_jobs = int(
            (
                (audit["k"] == 1)
                & audit["pair_shuffle_k1_identical"].eq(False)
            ).sum()
        )

    audit_failure_column = (
        "job_contract_valid"
        if "job_contract_valid" in audit.columns
        else "projection_contract_valid"
    )
    audit_failures = int((~audit[audit_failure_column].astype(bool)).sum())
    incomplete_cells = int((~completeness["complete"].astype(bool)).sum())
    expected_rows = int(completeness["expected_seed_count"].sum())

    if args.require_complete and incomplete_cells:
        raise ValueError(
            f"Completeness check failed for {incomplete_cells} cells; "
            f"see {outputs['completeness']}"
        )
    if args.fail_on_audit_error and (audit_failures or k1_warning_jobs):
        raise ValueError(
            "Audit failed: "
            f"job_contract_failures={audit_failures}, "
            f"k1_pair_shuffle_warnings={k1_warning_jobs}"
        )

    atomic_write_json(
        outputs["report"],
        {
            "inputs": [str(path) for path in input_paths],
            "generated_at_utc": utc_now(),
            "identity_columns": identity_columns(frame),
            "identities": frame[identity_columns(frame)]
            .drop_duplicates()
            .reset_index(drop=True)
            if identity_columns(frame)
            else [],
            "seed_control_rows": len(frame),
            "expected_rows": expected_rows,
            "complete_cells": int(completeness["complete"].sum()),
            "incomplete_cells": incomplete_cells,
            "expected_cells": len(completeness),
            "job_contract_failures": audit_failures,
            "pair_shuffle_k1_warning_jobs": k1_warning_jobs,
            "ari_min_effect": args.ari_min_effect,
            "silhouette_min_effect": args.silhouette_min_effect,
            "geometry_qualifiers": qualifiers,
            "outputs": {key: str(value) for key, value in outputs.items()},
        },
    )

    print("=" * 82)
    print("CIPHER-AGNOSTIC PROJECTED GEOMETRY ANALYSIS")
    print("=" * 82)
    print(f"Input CSVs              : {len(input_paths)}")
    print(f"Cipher identities       : {frame['cipher'].nunique()}")
    print(f"Seed/control rows       : {len(frame)}")
    print(
        f"Complete cells          : "
        f"{int(completeness['complete'].sum())}/{len(completeness)}"
    )
    print(f"Job-contract failures   : {audit_failures}")
    print(f"k=1 shuffle warnings    : {k1_warning_jobs}")
    print(f"Geometry qualifiers     : {outputs['qualifiers']}")
    print(f"Aggregation response    : {outputs['aggregation_response']}")
    print("=" * 82)

if __name__ == "__main__":
    main()
# Single-cipher example:
#
# python analyze_projected_geometry_results.py \
#   --input analysis_results/lea128_projected_geometry_full/projected_geometry_long.csv \
#   --out analysis_results/lea128_projected_geometry_analysis 

# Combined cross-cipher example:
#
# python analyze_projected_geometry_results.py \
#   --input \
#     analysis_results/present80_projected_geometry_full/projected_geometry_long.csv \
#     analysis_results/simeck3264_projected_geometry_full/projected_geometry_long.csv \
#   --out analysis_results/cross_cipher_projected_geometry_analysis
