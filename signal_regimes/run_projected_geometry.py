"""Run projected PCA/K-Means geometry over validated cipher manifest rows.

This runner is cipher-agnostic. Cipher-specific parameters are supplied by the
source analysis-data study and the geometry config. The module only orchestrates
locked projection and geometry estimators over manifest rows.

Expected workflow
-----------------
1. ``run_analysis_data.py`` creates a validated manifest for a cipher study.
2. This runner selects rows whose ``study_id`` matches ``source_study_id``.
3. ``projected_estimators.py`` regenerates one locked PCA representation and
   evaluates K-Means silhouette, ARI, aligned accuracy, and stability.
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
    GeometryProtocol,
    ProjectionProtocol,
    analyze_projected_geometry_bundle,
    flatten_geometry_bundle_for_csv,
)
from projected_runner_common import (
    RunnerSpec,
    add_common_cli_arguments,
    run_study,
    select_manifest_rows,
)


GEOMETRY_SCHEMA_VERSION = "projected_geometry_v1"
SILHOUETTE_CONTROL_REFERENCE = (
    "median_observed_D0_vs_D0_and_D1_vs_D1"
)

GEOMETRY_LONG_COLUMNS = (
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
    "source_config_hash",
    "estimator_protocol_hash",
    "estimator_job_hash",
    "status",
)

SPEC = RunnerSpec(
    analysis_name="projected_geometry",
    result_stem="projected_geometry",
    complete_marker="geometry_complete.json",
    failed_marker="geometry_failed.json",
    config_snapshot_name="geometry_estimator_config.json",
    long_filename="projected_geometry_long.csv",
    run_manifest_filename="projected_geometry_run_manifest.csv",
    completeness_filename="projected_geometry_completeness.json",
    long_columns=GEOMETRY_LONG_COLUMNS,
    expected_rows_per_job=4,
)


def _require_non_empty_string(config: Mapping[str, Any], key: str) -> str:
    value = str(config.get(key, "")).strip()
    if not value:
        raise ValueError(
            f"Geometry config field {key!r} must be a non-empty string"
        )
    return value


def load_config(
    path: Path,
) -> Tuple[Dict[str, Any], ProjectionProtocol, GeometryProtocol]:
    """Load and validate a cipher-specific projected-geometry config."""

    if not path.exists():
        raise FileNotFoundError(path)

    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON config {path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError("Geometry config root must be a JSON object")

    required = {
        "estimator_study_id",
        "source_study_id",
        "metric_schema_version",
        "projection",
        "geometry",
        "silhouette_control_reference",
        "status",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Geometry config is missing: {sorted(missing)}")

    _require_non_empty_string(config, "estimator_study_id")
    _require_non_empty_string(config, "source_study_id")

    if str(config["status"]).strip().lower() != "locked":
        raise ValueError("Geometry config status must be 'locked'")

    if config["metric_schema_version"] != GEOMETRY_SCHEMA_VERSION:
        raise ValueError(
            f"metric_schema_version must be {GEOMETRY_SCHEMA_VERSION!r}"
        )

    if (
        config["silhouette_control_reference"]
        != SILHOUETTE_CONTROL_REFERENCE
    ):
        raise ValueError(
            "Unsupported silhouette control reference: "
            f"{config['silhouette_control_reference']!r}"
        )

    if not isinstance(config["projection"], dict):
        raise ValueError("Geometry config field 'projection' must be an object")
    if not isinstance(config["geometry"], dict):
        raise ValueError("Geometry config field 'geometry' must be an object")

    projection = ProjectionProtocol(**config["projection"])
    geometry = GeometryProtocol(**config["geometry"])
    projection.validate()
    geometry.validate()
    return config, projection, geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the cipher-agnostic projected PCA/K-Means geometry view "
            "over a validated analysis-data manifest"
        )
    )
    add_common_cli_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config, projection_protocol, geometry_protocol = load_config(config_path)

    manifest_path, rows = select_manifest_rows(
        args,
        str(config["source_study_id"]),
    )

    protocol_payload = {
        "metric_schema_version": config["metric_schema_version"],
        "projection": asdict(projection_protocol),
        "geometry": asdict(geometry_protocol),
        "silhouette_control_reference": config[
            "silhouette_control_reference"
        ],
    }

    def analyze_row(
        row: Mapping[str, Any],
    ) -> Tuple[Mapping[str, Any], list[Dict[str, Any]]]:
        bundle = analyze_projected_geometry_bundle(
            d1_path=Path(row["d1_path"]),
            d0_path=Path(row["d0_path"]),
            d0_control_a_path=Path(row["d0_control_a_path"]),
            d0_control_b_path=Path(row["d0_control_b_path"]),
            d1_control_a_path=Path(row["d1_control_a_path"]),
            d1_control_b_path=Path(row["d1_control_b_path"]),
            pair_shuffled_path=Path(row["pair_shuffled_path"]),
            projection_protocol=projection_protocol,
            geometry_protocol=geometry_protocol,
            run_metadata={
                "study_id": row["study_id"],
                "domain": row["domain"],
                "cipher": row["cipher"],
                "round": row["round"],
                "k": row["k"],
                "seed": row["seed"],
            },
        )
        return bundle, flatten_geometry_bundle_for_csv(bundle)

    def format_main_row(row: Mapping[str, Any]) -> str:
        return (
            f"S={float(row['silhouette_kmeans']):.6f} | "
            f"S_excess={float(row['silhouette_excess_null']):.6f} | "
            f"ARI={float(row['kmeans_adjusted_rand']):.6f} | "
            f"stability={float(row['kmeans_stability_ari']):.6f}"
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
# python signal_regimes/run_projected_geometry.py \
#   --manifest signal_regimes/analysis_data/present80_rkmp_controls_v1/manifest.csv \
#   --config signal_regimes/configs/present80_projected_geometry_full.json \
#   --out signal_regimes/analysis_results
#
# SIMECK-32/64 example:
# python signal_regimes/run_projected_geometry.py \
#   --manifest signal_regimes/analysis_data/simeck3264_rkmp_controls_v1/manifest.csv \
#   --config signal_regimes/configs/simeck3264_projected_geometry_full.json \
#   --out signal_regimes/analysis_results
#
# hight64 example:
# python signal_regimes/run_projected_geometry.py \
#   --manifest signal_regimes/analysis_data/hight64_rkmp_controls_v1/manifest.csv \
#   --config signal_regimes/configs/hight64_projected_geometry_full.json \
#   --out signal_regimes/analysis_results
#
# lea128 example:
# python signal_regimes/run_projected_geometry.py \
#   --manifest signal_regimes/analysis_data/lea_rkmp_controls_v1/manifest.csv \
#   --config signal_regimes/configs/lea128_projected_geometry_full.json \
#   --out signal_regimes/analysis_results

