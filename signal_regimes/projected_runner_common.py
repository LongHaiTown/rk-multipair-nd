"""Shared orchestration utilities for projected-analysis runners.

This module contains workflow concerns only: manifest loading, validation,
filtering, hashing, resumable per-job execution, output rebuilding, and
completeness reporting.  Estimator computation remains in
``projected_estimators.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REQUIRED_MANIFEST_COLUMNS = {
    "study_id",
    "domain",
    "cipher",
    "round",
    "k",
    "seed",
    "samples_per_class",
    "plain_bits",
    "output_layout",
    "d1_path",
    "d0_path",
    "d0_control_a_path",
    "d0_control_b_path",
    "d1_control_a_path",
    "d1_control_b_path",
    "pair_shuffled_path",
    "protocol_valid",
    "config_hash",
    "status",
}

PATH_COLUMNS = (
    "d1_path",
    "d0_path",
    "d0_control_a_path",
    "d0_control_b_path",
    "d1_control_a_path",
    "d1_control_b_path",
    "pair_shuffled_path",
)

TRUE_VALUES = {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class RunnerSpec:
    analysis_name: str
    result_stem: str
    complete_marker: str
    failed_marker: str
    config_snapshot_name: str
    long_filename: str
    run_manifest_filename: str
    completeness_filename: str
    long_columns: Tuple[str, ...]
    expected_rows_per_job: int = 4


AnalyzeCallback = Callable[
    [Mapping[str, Any]], Tuple[Mapping[str, Any], List[Dict[str, Any]]]
]
FormatCallback = Callable[[Mapping[str, Any]], str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in TRUE_VALUES


def canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json_safe(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(dict(payload)), handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows([dict(row) for row in rows])
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def resolve_path(raw: str, manifest_path: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def load_manifest(path: Path, source_study_id: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_MANIFEST_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing: {sorted(missing)}")

        for row_number, raw in enumerate(reader, start=2):
            if str(raw["status"]).strip().lower() != "completed":
                continue
            if not parse_bool(raw["protocol_valid"]):
                continue
            if str(raw["study_id"]).strip() != source_study_id:
                raise ValueError(
                    f"Source study mismatch at manifest row {row_number}"
                )

            row: Dict[str, Any] = {
                "source_row_number": row_number,
                "study_id": str(raw["study_id"]).strip(),
                "domain": str(raw["domain"]).strip(),
                "cipher": str(raw["cipher"]).strip(),
                "round": int(raw["round"]),
                "k": int(raw["k"]),
                "seed": int(raw["seed"]),
                "samples_per_class": int(raw["samples_per_class"]),
                "plain_bits": int(raw["plain_bits"]),
                "output_layout": str(raw["output_layout"]).strip(),
                "source_config_hash": str(raw["config_hash"]).strip(),
            }
            for column in PATH_COLUMNS:
                row[column] = resolve_path(str(raw[column]), path)
            rows.append(row)
    return rows


def validate_unique(rows: Sequence[Mapping[str, Any]]) -> None:
    seen: Dict[Tuple[Any, ...], Any] = {}
    for row in rows:
        key = (
            row["domain"],
            row["cipher"],
            row["round"],
            row["k"],
            row["seed"],
        )
        if key in seen:
            raise ValueError(
                f"Duplicate configuration {key} at rows "
                f"{seen[key]} and {row['source_row_number']}"
            )
        seen[key] = row["source_row_number"]


def validate_paths(row: Mapping[str, Any]) -> Dict[str, Any]:
    arrays: Dict[str, np.ndarray] = {}
    info: Dict[str, Any] = {}
    for column in PATH_COLUMNS:
        path = Path(row[column])
        if not path.exists():
            raise FileNotFoundError(path)
        array = np.load(path, mmap_mode="r")
        if array.ndim not in {2, 3}:
            raise ValueError(f"Unsupported shape at {path}: {array.shape}")
        arrays[column] = array
        info[column] = {
            "path": str(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "size_bytes": path.stat().st_size,
        }

    reference = arrays["d1_path"]
    for column, array in arrays.items():
        if array.shape != reference.shape:
            raise ValueError(
                f"Control shape mismatch: {column}={array.shape}, "
                f"reference={reference.shape}"
            )
        if array.dtype != reference.dtype:
            raise ValueError(f"Control dtype mismatch at {column}")

    if reference.shape[0] != int(row["samples_per_class"]):
        raise ValueError("samples_per_class does not match stored arrays")
    if reference.ndim == 3:
        if reference.shape[1] != int(row["k"]):
            raise ValueError("Grouped k does not match manifest")
        if reference.shape[2] != 3 * int(row["plain_bits"]):
            raise ValueError("Pair feature dimension does not match plain_bits")
    return info


def filter_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    rounds: Optional[Sequence[int]],
    k_values: Optional[Sequence[int]],
    seeds: Optional[Sequence[int]],
    domains: Optional[Sequence[str]],
) -> List[Dict[str, Any]]:
    round_set = set(rounds) if rounds else None
    k_set = set(k_values) if k_values else None
    seed_set = set(seeds) if seeds else None
    domain_set = set(domains) if domains else None
    return [
        row
        for row in rows
        if (round_set is None or row["round"] in round_set)
        and (k_set is None or row["k"] in k_set)
        and (seed_set is None or row["seed"] in seed_set)
        and (domain_set is None or row["domain"] in domain_set)
    ]


def output_run_dir(
    root: Path,
    study_id: str,
    row: Mapping[str, Any],
) -> Path:
    return (
        root
        / study_id
        / "per_run"
        / str(row["domain"])
        / str(row["cipher"])
        / f"r{row['round']}"
        / f"k{row['k']}"
        / f"seed{row['seed']}"
    )


def add_common_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="analysis_results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--rounds", type=int, nargs="+")
    parser.add_argument("--k-values", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--domains", nargs="+")
    parser.add_argument("--limit", type=int, default=0)


def select_manifest_rows(
    args: argparse.Namespace,
    source_study_id: str,
) -> Tuple[Path, List[Dict[str, Any]]]:
    manifest_path = Path(args.manifest).resolve()
    rows = load_manifest(manifest_path, source_study_id)
    validate_unique(rows)
    rows = filter_rows(
        rows,
        rounds=args.rounds,
        k_values=args.k_values,
        seeds=args.seeds,
        domains=args.domains,
    )
    rows.sort(
        key=lambda row: (
            row["domain"],
            row["cipher"],
            row["round"],
            row["k"],
            row["seed"],
        )
    )
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No jobs remain after filters")
    for row in rows:
        validate_paths(row)
    return manifest_path, rows


def _build_job_hash(
    *,
    spec: RunnerSpec,
    config: Mapping[str, Any],
    protocol_payload: Mapping[str, Any],
    row: Mapping[str, Any],
    path_info: Mapping[str, Any],
    manifest_path: Path,
) -> Tuple[str, str]:
    protocol_hash = canonical_hash(
        {
            "analysis_name": spec.analysis_name,
            "metric_schema_version": config["metric_schema_version"],
            "protocol": protocol_payload,
        }
    )
    job_hash = canonical_hash(
        {
            "analysis_name": spec.analysis_name,
            "estimator_study_id": config["estimator_study_id"],
            "source_study_id": config["source_study_id"],
            "metric_schema_version": config["metric_schema_version"],
            "source_manifest": str(manifest_path),
            "source_row_number": row["source_row_number"],
            "source_config_hash": row["source_config_hash"],
            "control_paths": path_info,
            "protocol_hash": protocol_hash,
        }
    )
    return protocol_hash, job_hash


def _run_one_job(
    *,
    spec: RunnerSpec,
    config: Mapping[str, Any],
    protocol_payload: Mapping[str, Any],
    manifest_path: Path,
    row: Mapping[str, Any],
    output_root: Path,
    overwrite: bool,
    analyze_callback: AnalyzeCallback,
) -> Dict[str, Any]:
    path_info = validate_paths(row)
    protocol_hash, job_hash = _build_job_hash(
        spec=spec,
        config=config,
        protocol_payload=protocol_payload,
        row=row,
        path_info=path_info,
        manifest_path=manifest_path,
    )
    run_dir = output_run_dir(
        output_root, str(config["estimator_study_id"]), row
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    result_json = run_dir / f"{spec.result_stem}.json"
    result_csv = run_dir / f"{spec.result_stem}.csv"
    config_json = run_dir / spec.config_snapshot_name
    complete_json = run_dir / spec.complete_marker
    failed_json = run_dir / spec.failed_marker

    if complete_json.exists() and not overwrite:
        record = json.loads(complete_json.read_text(encoding="utf-8"))
        if (
            record.get("status") == "completed"
            and record.get("estimator_job_hash") == job_hash
            and result_json.exists()
            and result_csv.exists()
            and config_json.exists()
        ):
            record["_runner_status"] = "skipped_existing"
            return record
        raise RuntimeError(f"Existing estimator hash mismatch at {run_dir}")

    if overwrite:
        for path in (
            result_json,
            result_csv,
            config_json,
            complete_json,
            failed_json,
        ):
            if path.exists():
                path.unlink()

    atomic_write_json(
        config_json,
        {
            "analysis_name": spec.analysis_name,
            "estimator_study_id": config["estimator_study_id"],
            "source_study_id": config["source_study_id"],
            "source_manifest": str(manifest_path),
            "source_row_number": row["source_row_number"],
            "source_config_hash": row["source_config_hash"],
            "estimator_protocol_hash": protocol_hash,
            "estimator_job_hash": job_hash,
            "metric_schema_version": config["metric_schema_version"],
            "protocol": protocol_payload,
            "control_paths": path_info,
            "created_at_utc": utc_now(),
        },
    )

    started = time.perf_counter()
    try:
        bundle, result_rows = analyze_callback(row)
        for result_row in result_rows:
            result_row.update(
                {
                    "estimator_study_id": config["estimator_study_id"],
                    "source_study_id": config["source_study_id"],
                    "source_config_hash": row["source_config_hash"],
                    "estimator_protocol_hash": protocol_hash,
                    "estimator_job_hash": job_hash,
                    "status": "completed",
                }
            )

        atomic_write_json(result_json, bundle)
        atomic_write_csv(result_csv, result_rows, spec.long_columns)
        completion = {
            "analysis_name": spec.analysis_name,
            "estimator_study_id": config["estimator_study_id"],
            "source_study_id": config["source_study_id"],
            "domain": row["domain"],
            "cipher": row["cipher"],
            "round": row["round"],
            "k": row["k"],
            "seed": row["seed"],
            "source_config_hash": row["source_config_hash"],
            "estimator_protocol_hash": protocol_hash,
            "estimator_job_hash": job_hash,
            "result_json_path": str(result_json),
            "result_csv_path": str(result_csv),
            "result_rows": result_rows,
            "runner_seconds": round(time.perf_counter() - started, 6),
            "completed_at_utc": utc_now(),
            "status": "completed",
        }
        atomic_write_json(complete_json, completion)
        if failed_json.exists():
            failed_json.unlink()
        completion["_runner_status"] = "completed"
        return completion
    except Exception as exc:
        atomic_write_json(
            failed_json,
            {
                "analysis_name": spec.analysis_name,
                "round": row["round"],
                "k": row["k"],
                "seed": row["seed"],
                "source_config_hash": row["source_config_hash"],
                "estimator_protocol_hash": protocol_hash,
                "estimator_job_hash": job_hash,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "runner_seconds": round(time.perf_counter() - started, 6),
                "failed_at_utc": utc_now(),
                "status": "failed",
            },
        )
        raise


def rebuild_outputs(
    study_root: Path,
    spec: RunnerSpec,
) -> Tuple[int, Path, Path]:
    long_rows: List[Dict[str, Any]] = []
    run_rows: List[Dict[str, Any]] = []
    for marker in study_root.rglob(spec.complete_marker):
        try:
            record = json.loads(marker.read_text(encoding="utf-8"))
            if record.get("status") != "completed":
                continue
            long_rows.extend(record["result_rows"])
            run_rows.append(
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"result_rows", "_runner_status"}
                }
            )
        except Exception:
            continue

    long_rows.sort(
        key=lambda row: (
            int(row.get("round", 0)),
            int(row.get("k", 0)),
            int(row.get("seed", 0)),
            str(row.get("control_type", "")),
        )
    )
    run_rows.sort(
        key=lambda row: (
            int(row.get("round", 0)),
            int(row.get("k", 0)),
            int(row.get("seed", 0)),
        )
    )

    long_csv = study_root / spec.long_filename
    run_manifest = study_root / spec.run_manifest_filename
    atomic_write_csv(long_csv, long_rows, spec.long_columns)
    run_fields = (
        sorted({key for row in run_rows for key in row})
        if run_rows
        else ["round", "k", "seed", "estimator_job_hash", "status"]
    )
    atomic_write_csv(run_manifest, run_rows, run_fields)
    return len(long_rows), long_csv, run_manifest


def build_completeness(
    selected_rows: Sequence[Mapping[str, Any]],
    study_root: Path,
    spec: RunnerSpec,
) -> Dict[str, Any]:
    expected = {
        (int(row["round"]), int(row["k"]), int(row["seed"]))
        for row in selected_rows
    }
    completed = set()
    for marker in study_root.rglob(spec.complete_marker):
        try:
            record = json.loads(marker.read_text(encoding="utf-8"))
            if record.get("status") == "completed":
                completed.add(
                    (
                        int(record["round"]),
                        int(record["k"]),
                        int(record["seed"]),
                    )
                )
        except Exception:
            continue
    missing = sorted(expected - completed)
    return {
        "analysis_name": spec.analysis_name,
        "expected_jobs": len(expected),
        "completed_selected_jobs": len(expected & completed),
        "expected_long_rows": spec.expected_rows_per_job * len(expected),
        "missing_jobs": [
            {"round": r, "k": k, "seed": seed} for r, k, seed in missing
        ],
        "complete": not missing,
        "generated_at_utc": utc_now(),
    }


def run_study(
    *,
    args: argparse.Namespace,
    spec: RunnerSpec,
    config: Mapping[str, Any],
    protocol_payload: Mapping[str, Any],
    manifest_path: Path,
    rows: Sequence[Mapping[str, Any]],
    analyze_callback: AnalyzeCallback,
    format_main_row: FormatCallback,
) -> None:
    output_root = Path(args.out).resolve()
    study_root = output_root / str(config["estimator_study_id"])
    study_root.mkdir(parents=True, exist_ok=True)

    protocol_hash = canonical_hash(
        {
            "analysis_name": spec.analysis_name,
            "metric_schema_version": config["metric_schema_version"],
            "protocol": protocol_payload,
        }
    )
    atomic_write_json(
        study_root / "locked_estimator_config.json",
        {
            **dict(config),
            "protocol": protocol_payload,
            "estimator_protocol_hash": protocol_hash,
            "source_manifest": str(manifest_path),
            "copied_at_utc": utc_now(),
        },
    )

    print("=" * 82)
    print(f"PRESENT {spec.analysis_name.upper()} RUNNER")
    print("=" * 82)
    print(f"Estimator study        : {config['estimator_study_id']}")
    print(f"Metric schema          : {config['metric_schema_version']}")
    print(f"Selected source jobs   : {len(rows)}")
    print(
        f"Long rows expected     : "
        f"{spec.expected_rows_per_job * len(rows)}"
    )
    print("=" * 82)

    if args.dry_run:
        for index, row in enumerate(rows, start=1):
            print(
                f"[{index:03d}/{len(rows):03d}] "
                f"r={row['round']} k={row['k']} seed={row['seed']}"
            )
        return

    completed = skipped = failed = 0
    for index, row in enumerate(rows, start=1):
        label = f"r={row['round']} k={row['k']} seed={row['seed']}"
        print(f"\n[{index:03d}/{len(rows):03d}] START {label}")
        try:
            outcome = _run_one_job(
                spec=spec,
                config=config,
                protocol_payload=protocol_payload,
                manifest_path=manifest_path,
                row=row,
                output_root=output_root,
                overwrite=args.overwrite,
                analyze_callback=analyze_callback,
            )
            if outcome["_runner_status"] == "skipped_existing":
                skipped += 1
                print(f"[SKIP] {label}")
            else:
                completed += 1
                main_row = next(
                    item
                    for item in outcome["result_rows"]
                    if item["control_type"] == "D1_vs_D0"
                )
                print(f"[DONE] {label} | {format_main_row(main_row)}")
        except Exception as exc:
            failed += 1
            print(
                f"[FAIL] {label}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            rebuild_outputs(study_root, spec)
            atomic_write_json(
                study_root / spec.completeness_filename,
                build_completeness(rows, study_root, spec),
            )
            if not args.continue_on_error:
                raise

    long_count, long_csv, run_manifest = rebuild_outputs(study_root, spec)
    completeness = build_completeness(rows, study_root, spec)
    completeness.update(
        {
            "completed_this_execution": completed,
            "skipped_this_execution": skipped,
            "failed_this_execution": failed,
            "long_rows_available": long_count,
        }
    )
    atomic_write_json(study_root / spec.completeness_filename, completeness)

    print("\n" + "=" * 82)
    print(f"Completed              : {completed}")
    print(f"Skipped                : {skipped}")
    print(f"Failed                 : {failed}")
    print(f"Long rows              : {long_count}")
    print(f"Long CSV               : {long_csv}")
    print(f"Run manifest           : {run_manifest}")
    print("=" * 82)
    if failed:
        raise SystemExit(1)
