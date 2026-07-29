"""Run the projected MMD analysis over validated cipher manifest rows.

This runner is cipher-agnostic. Cipher-specific choices belong to the source
analysis-data config and to the MMD estimator config, not to this module.

Expected workflow
-----------------
1. ``run_analysis_data.py`` creates a validated manifest for a cipher study.
2. This runner selects rows whose ``study_id`` matches ``source_study_id``.
3. ``projected_estimators.py`` regenerates the locked PCA representation and
   evaluates projected MMD for the main and control comparisons.
4. ``projected_runner_common.py`` handles hashing, resume, failure markers,
   long-form CSV rebuilding, and completeness reporting.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from projected_estimators import (
    MMDProtocol,
    ProjectionProtocol,
    analyze_projected_mmd_bundle,
    flatten_mmd_bundle_for_csv,
)
from projected_runner_common import (
    RunnerSpec,
    add_common_cli_arguments,
    run_study,
    select_manifest_rows,
)


MMD_SCHEMA_VERSION = "projected_mmd_v1"
MMD_CONTROL_REFERENCE = "median_observed_D0_vs_D0_and_D1_vs_D1"

MMD_LONG_COLUMNS = (
    "estimator_study_id",
    "source_study_id",
    "study_id",
    "domain",
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
    "pair_shuffle_k1_identical",
    "mmd2",
    "mmd_permutation_p",
    "mmd_permutation_null_median",
    "mmd_permutation_null_q95",
    "mmd_control_null_median",
    "mmd_excess_null",
    "mmd_kernel",
    "mmd_bandwidth",
    "mmd_bandwidth_scope",
    "mmd_samples_per_distribution",
    "source_config_hash",
    "estimator_protocol_hash",
    "estimator_job_hash",
    "status",
)

SPEC = RunnerSpec(
    analysis_name="projected_mmd",
    result_stem="projected_mmd",
    complete_marker="mmd_complete.json",
    failed_marker="mmd_failed.json",
    config_snapshot_name="mmd_estimator_config.json",
    long_filename="projected_mmd_long.csv",
    run_manifest_filename="projected_mmd_run_manifest.csv",
    completeness_filename="projected_mmd_completeness.json",
    long_columns=MMD_LONG_COLUMNS,
    expected_rows_per_job=4,
)


def _require_non_empty_string(config: Mapping[str, Any], key: str) -> str:
    value = str(config.get(key, "")).strip()
    if not value:
        raise ValueError(f"MMD config field {key!r} must be a non-empty string")
    return value


def load_config(
    path: Path,
) -> Tuple[Dict[str, Any], ProjectionProtocol, MMDProtocol]:
    """Load and validate a cipher-specific projected-MMD config."""

    if not path.exists():
        raise FileNotFoundError(path)

    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON config {path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError("MMD config root must be a JSON object")

    required = {
        "estimator_study_id",
        "source_study_id",
        "metric_schema_version",
        "projection",
        "mmd",
        "mmd_control_reference",
        "status",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"MMD config is missing: {sorted(missing)}")

    _require_non_empty_string(config, "estimator_study_id")
    _require_non_empty_string(config, "source_study_id")

    if str(config["status"]).strip().lower() != "locked":
        raise ValueError("MMD config status must be 'locked'")

    if config["metric_schema_version"] != MMD_SCHEMA_VERSION:
        raise ValueError(
            f"metric_schema_version must be {MMD_SCHEMA_VERSION!r}"
        )

    if config["mmd_control_reference"] != MMD_CONTROL_REFERENCE:
        raise ValueError(
            "Unsupported MMD control reference: "
            f"{config['mmd_control_reference']!r}"
        )

    if not isinstance(config["projection"], dict):
        raise ValueError("MMD config field 'projection' must be an object")
    if not isinstance(config["mmd"], dict):
        raise ValueError("MMD config field 'mmd' must be an object")

    projection = ProjectionProtocol(**config["projection"])
    mmd = MMDProtocol(**config["mmd"])
    projection.validate()
    mmd.validate()
    return config, projection, mmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the cipher-agnostic projected MMD view over a validated "
            "analysis-data manifest"
        )
    )
    add_common_cli_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config, projection_protocol, mmd_protocol = load_config(config_path)

    manifest_path, rows = select_manifest_rows(
        args,
        str(config["source_study_id"]),
    )

    protocol_payload = {
        "metric_schema_version": config["metric_schema_version"],
        "projection": asdict(projection_protocol),
        "mmd": asdict(mmd_protocol),
        "mmd_control_reference": config["mmd_control_reference"],
    }

    def analyze_row(
        row: Mapping[str, Any],
    ) -> Tuple[Mapping[str, Any], list[Dict[str, Any]]]:
        bundle = analyze_projected_mmd_bundle(
            d1_path=Path(row["d1_path"]),
            d0_path=Path(row["d0_path"]),
            d0_control_a_path=Path(row["d0_control_a_path"]),
            d0_control_b_path=Path(row["d0_control_b_path"]),
            d1_control_a_path=Path(row["d1_control_a_path"]),
            d1_control_b_path=Path(row["d1_control_b_path"]),
            pair_shuffled_path=Path(row["pair_shuffled_path"]),
            projection_protocol=projection_protocol,
            mmd_protocol=mmd_protocol,
            run_metadata={
                "study_id": row["study_id"],
                "domain": row["domain"],
                "cipher": row["cipher"],
                "round": row["round"],
                "k": row["k"],
                "seed": row["seed"],
            },
        )
        return bundle, flatten_mmd_bundle_for_csv(bundle)

    def format_main_row(row: Mapping[str, Any]) -> str:
        return (
            f"MMD2={float(row['mmd2']):.6f} | "
            f"MMD_excess={float(row['mmd_excess_null']):.6f} | "
            f"p={float(row['mmd_permutation_p']):.6g} | "
            f"bw={row['mmd_bandwidth']}"
        )

    run_study(
        args=args,
        spec=SPEC,
        config=config,
        protocol_payload=protocol_payload,
        manifest_path=manifest_path,
        rows=rows,
        analyze_callback=analyze_row,
        format_main_row=format_main_row,
    )


if __name__ == "__main__":
    main()


# PRESENT-80 example:
# python signal_regimes/run_projected_mmd.py \
#   --manifest signal_regimes/analysis_data/present80_rkmp_controls_v1/manifest.csv \
#   --config signal_regimes/configs/present80_projected_mmd_full.json \
#   --out signal_regimes/analysis_results

# SIMECK-32/64 example:
# python signal_regimes/run_projected_mmd.py \
#   --manifest signal_regimes/analysis_data/simeck3264_rkmp_controls_v1/manifest.csv \
#   --config signal_regimes/configs/simeck3264/projected_mmd_full.json \
#   --out signal_regimes/analysis_results
# HIGHT-64 
# python signal_regimes/run_projected_mmd.py \
#   --manifest signal_regimes/analysis_data/hight64_rkmp_controls_v1/manifest.csv \
#   --config signal_regimes/configs/hight64_projected_mmd_full.json \
#   --out signal_regimes/analysis_results
# HIGHT-64 
# python signal_regimes/run_projected_mmd.py \
#   --manifest signal_regimes/analysis_data/hight64_rkmp_controls_v1/manifest.csv \
#   --config signal_regimes/configs/hight64_projected_mmd_full.json \
#   --out signal_regimes/analysis_results
# LEA-128
# python signal_regimes/run_projected_mmd.py \
#   --manifest signal_regimes/analysis_data/hight64_rkmp_controls_v1/manifest.csv \
#   --config signal_regimes/configs/hight64_projected_mmd_full.json \
#   --out signal_regimes/analysis_results



