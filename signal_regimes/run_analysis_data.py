"""Generate RKMP analysis datasets for any cipher adapter in ``cipher/``.

The runner is the cipher-agnostic replacement for
``run_present_analysis_data.py``.  It consumes one locked JSON configuration
and generates, for every (domain, round, k, seed) job:

* D1_structured and D0_null for the main D1_vs_D0 comparison;
* two independent D0 datasets for D0_vs_D0;
* two independent D1 datasets for D1_vs_D1;
* a pair-shuffled D1 control that preserves the pair-level marginal multiset
  while breaking the original multi-pair grouping when k > 1.

The data-generation semantics remain in ``make_data_analysis.py``.  This file
only resolves a cipher adapter, orchestrates the grid, streams arrays to .npy
memmaps, validates the protocol, manages per-control resume hashes, and rebuilds
one consolidated manifest.csv.

Supported cipher configuration forms
------------------------------------
Legacy string form::

    "cipher": "present80"

Extended adapter form::

    "cipher": {
      "slug": "simeck3264",
      "name": "SIMECK-32/64",
      "module": "cipher.simeck3264",
      "encrypt_function": "encrypt",
      "expected_plain_bits": 32,
      "expected_key_bits": 64
    }

A cipher module must expose ``plain_bits``, ``key_bits``, and a vectorized
function compatible with ``encrypt(plaintext_bits, key_bits, rounds)``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.lib.format import open_memmap

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from make_data_analysis import (
    NDCMultiPairAnalysisGenerator,
    build_control_generators,
    derive_control_seed,
    single_bit_difference,
    validate_protocol_batch,
)


MANIFEST_FIELDS = [
    "study_id",
    "domain",
    "cipher",
    "cipher_slug",
    "cipher_module",
    "round",
    "k",
    "seed",
    "source_seed",
    "samples_per_class",
    "plain_bits",
    "key_bits",
    "delta_p_hex",
    "delta_k_hex",
    "delta_k_bit",
    "delta_k_indexing",
    "representation",
    "output_layout",
    "key_mode",
    "dtype",
    "feature_shape_per_class",
    "d1_path",
    "d0_path",
    "d1_labels_path",
    "d0_labels_path",
    "d0_control_a_path",
    "d0_control_b_path",
    "d1_control_a_path",
    "d1_control_b_path",
    "pair_shuffled_path",
    "main_config_hash",
    "d0_control_config_hash",
    "d1_control_config_hash",
    "pair_shuffled_config_hash",
    "run_config_path",
    "validation_path",
    "control_metadata_path",
    "protocol_valid",
    "generation_seconds",
    "actual_bytes",
    "config_hash",
    "completed_at_utc",
    "status",
]


@dataclass(frozen=True)
class CipherAdapter:
    slug: str
    name: str
    module_name: str
    encrypt_function_name: str
    module: ModuleType
    encrypt: Callable[..., Any]
    plain_bits: int
    key_bits: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        json_safe(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(dict(payload)), handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=MANIFEST_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows([json_safe(dict(row)) for row in rows])
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def parse_int(value: Any, *, name: str) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        return int(str(value), 0)
    except Exception as exc:
        raise ValueError(f"{name} must be an integer-compatible value") from exc


def safe_slug(value: str, *, name: str) -> str:
    slug = str(value).strip()
    if not slug:
        raise ValueError(f"{name} must not be empty")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", slug):
        raise ValueError(
            f"{name}={slug!r} is unsafe for paths; use letters, digits, _, -, or ."
        )
    return slug


def normalize_cipher_spec(config: Mapping[str, Any]) -> Dict[str, Any]:
    raw = config["cipher"]
    if isinstance(raw, Mapping):
        spec = dict(raw)
        slug = safe_slug(spec.get("slug", spec.get("name", "")), name="cipher.slug")
        module_name = str(spec.get("module", f"cipher.{slug}"))
        name = str(spec.get("name", slug))
        encrypt_name = str(spec.get("encrypt_function", "encrypt"))
        expected_plain_bits = spec.get("expected_plain_bits")
        expected_key_bits = spec.get("expected_key_bits")
    else:
        slug = safe_slug(str(raw), name="cipher")
        module_name = str(config.get("cipher_module", f"cipher.{slug}"))
        name = str(config.get("cipher_name", slug))
        encrypt_name = str(config.get("encrypt_function", "encrypt"))
        expected_plain_bits = config.get("expected_plain_bits")
        expected_key_bits = config.get("expected_key_bits")

    return {
        "slug": slug,
        "name": name,
        "module": module_name,
        "encrypt_function": encrypt_name,
        "expected_plain_bits": (
            int(expected_plain_bits) if expected_plain_bits is not None else None
        ),
        "expected_key_bits": (
            int(expected_key_bits) if expected_key_bits is not None else None
        ),
    }


def load_cipher_adapter(config: Mapping[str, Any]) -> CipherAdapter:
    spec = normalize_cipher_spec(config)
    module = importlib.import_module(spec["module"])

    for attr in ("plain_bits", "key_bits", spec["encrypt_function"]):
        if not hasattr(module, attr):
            raise AttributeError(
                f"Cipher module {spec['module']!r} does not expose {attr!r}"
            )

    plain_bits = int(getattr(module, "plain_bits"))
    key_bits = int(getattr(module, "key_bits"))
    encrypt = getattr(module, spec["encrypt_function"])
    if plain_bits <= 0 or key_bits <= 0:
        raise ValueError("Cipher plain_bits and key_bits must be positive")
    if not callable(encrypt):
        raise TypeError(
            f"{spec['module']}.{spec['encrypt_function']} must be callable"
        )

    if (
        spec["expected_plain_bits"] is not None
        and plain_bits != spec["expected_plain_bits"]
    ):
        raise ValueError(
            f"Cipher plain_bits={plain_bits}, expected {spec['expected_plain_bits']}"
        )
    if (
        spec["expected_key_bits"] is not None
        and key_bits != spec["expected_key_bits"]
    ):
        raise ValueError(
            f"Cipher key_bits={key_bits}, expected {spec['expected_key_bits']}"
        )

    return CipherAdapter(
        slug=spec["slug"],
        name=spec["name"],
        module_name=spec["module"],
        encrypt_function_name=spec["encrypt_function"],
        module=module,
        encrypt=encrypt,
        plain_bits=plain_bits,
        key_bits=key_bits,
    )


def load_locked_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    required = {
        "study_id",
        "cipher",
        "delta_p_hex",
        "delta_k",
        "rounds",
        "k_values",
        "analysis_seeds",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Locked config is missing: {sorted(missing)}")
    if str(config.get("status", "")).lower() != "locked":
        raise ValueError("Config status must be 'locked'")

    config["study_id"] = safe_slug(config["study_id"], name="study_id")
    config["rounds"] = [int(value) for value in config["rounds"]]
    config["k_values"] = [int(value) for value in config["k_values"]]
    config["analysis_seeds"] = [int(value) for value in config["analysis_seeds"]]
    if not config["rounds"] or not config["k_values"] or not config["analysis_seeds"]:
        raise ValueError("rounds, k_values, and analysis_seeds must not be empty")
    if any(value <= 0 for value in config["rounds"] + config["k_values"]):
        raise ValueError("rounds and k_values must be positive")
    if len(set(config["rounds"])) != len(config["rounds"]):
        raise ValueError("rounds contains duplicates")
    if len(set(config["k_values"])) != len(config["k_values"]):
        raise ValueError("k_values contains duplicates")
    if len(set(config["analysis_seeds"])) != len(config["analysis_seeds"]):
        raise ValueError("analysis_seeds contains duplicates")

    domains = config.get("domains") or [
        {
            "name": "analysis",
            "samples_per_class": int(config.get("samples_per_class", 10_000)),
            "seed_offset": 0,
        }
    ]
    normalized_domains = []
    for domain in domains:
        name = safe_slug(domain["name"], name="domain.name")
        samples = int(domain["samples_per_class"])
        seed_offset = int(domain.get("seed_offset", 0))
        if samples <= 0:
            raise ValueError("samples_per_class must be positive")
        normalized_domains.append(
            {
                "name": name,
                "samples_per_class": samples,
                "seed_offset": seed_offset,
            }
        )
    if len({item["name"] for item in normalized_domains}) != len(normalized_domains):
        raise ValueError("domain names must be unique")
    config["domains"] = normalized_domains

    config["batch_size"] = int(config.get("batch_size", 5_000))
    config["output_layout"] = str(config.get("output_layout", "grouped"))
    config["key_mode"] = str(config.get("key_mode", "shared"))
    config["encrypt_backend"] = str(config.get("encrypt_backend", "numpy"))
    config["dtype"] = str(config.get("dtype", "uint8"))
    config["validate_protocol"] = bool(config.get("validate_protocol", True))
    config["generate_controls"] = bool(config.get("generate_controls", True))
    config["representation"] = str(
        config.get("representation", "delta_C||C||C_star")
    )
    config["adapter_check"] = bool(config.get("adapter_check", True))

    if config["batch_size"] <= 0:
        raise ValueError("batch_size must be positive")
    if config["output_layout"] not in {"flat", "grouped"}:
        raise ValueError("output_layout must be flat or grouped")
    if config["key_mode"] not in {"shared", "independent"}:
        raise ValueError("key_mode must be shared or independent")
    if config["encrypt_backend"] not in {"numpy", "cupy", "auto"}:
        raise ValueError("encrypt_backend must be numpy, cupy, or auto")
    if config["dtype"] not in {"uint8", "float32"}:
        raise ValueError("dtype must be uint8 or float32")
    if not config["generate_controls"]:
        raise ValueError(
            "This runner requires generate_controls=true because downstream "
            "MMD and projected-geometry runners consume the control paths."
        )

    return config


def resolve_delta_k(
    spec: Mapping[str, Any], key_bits: int
) -> Tuple[int, Dict[str, Any]]:
    mode = str(spec.get("mode", "integer"))
    if mode == "integer":
        raw = spec.get("hex", spec.get("value"))
        if raw is None:
            raise ValueError("delta_k integer mode requires hex or value")
        value = parse_int(raw, name="delta_k")
        resolved: Dict[str, Any] = {
            "mode": mode,
            "input": raw,
            "bit_index": None,
            "indexing": None,
            "array_index_msb0": None,
        }
    elif mode == "single_bit":
        bit_index = int(spec["bit_index"])
        indexing = str(spec.get("indexing", "msb0"))
        value, vector = single_bit_difference(
            bit_index,
            key_bits,
            indexing=indexing,
        )
        resolved = {
            "mode": mode,
            "bit_index": bit_index,
            "indexing": indexing,
            "array_index_msb0": int(np.flatnonzero(vector)[0]),
        }
    else:
        raise ValueError("delta_k mode must be integer or single_bit")

    if value < 0 or value >= (1 << key_bits):
        raise ValueError(f"delta_k does not fit in {key_bits} bits")
    resolved.update(
        {
            "integer": int(value),
            "hex": f"0x{value:0{(key_bits + 3) // 4}x}",
            "hamming_weight": int(value.bit_count()),
            "origin": spec.get("origin", "unspecified"),
            "selection_note": spec.get("selection_note"),
        }
    )
    return int(value), resolved


def iter_jobs(config: Mapping[str, Any]) -> Iterator[Dict[str, Any]]:
    for domain in config["domains"]:
        for round_number in config["rounds"]:
            for k in config["k_values"]:
                for seed in config["analysis_seeds"]:
                    yield {
                        "domain": domain["name"],
                        "samples_per_class": domain["samples_per_class"],
                        "domain_seed_offset": domain["seed_offset"],
                        "round": int(round_number),
                        "k": int(k),
                        "seed": int(seed),
                    }


def filter_jobs(
    jobs: Sequence[Dict[str, Any]],
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
    filtered = [
        job
        for job in jobs
        if (round_set is None or job["round"] in round_set)
        and (k_set is None or job["k"] in k_set)
        and (seed_set is None or job["seed"] in seed_set)
        and (domain_set is None or job["domain"] in domain_set)
    ]
    if not filtered:
        raise ValueError("No jobs remain after CLI filters")
    return filtered


def run_directory(
    root: Path,
    config: Mapping[str, Any],
    adapter: CipherAdapter,
    job: Mapping[str, Any],
) -> Path:
    return (
        root
        / str(config["study_id"])
        / str(job["domain"])
        / adapter.slug
        / f"r{job['round']}"
        / f"k{job['k']}"
        / f"seed{job['seed']}"
    )


def feature_shape(generator: NDCMultiPairAnalysisGenerator) -> Tuple[int, ...]:
    if generator.output_layout == "grouped":
        return (generator.n, generator.pairs, generator.pair_feature_dim)
    return (generator.n, generator.input_dim)


def stream_generator_to_npy(
    generator: NDCMultiPairAnalysisGenerator,
    feature_path: Path,
    label_path: Optional[Path],
    *,
    validate: bool,
) -> Dict[str, Any]:
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_temp = feature_path.with_name(feature_path.name + ".tmp")
    label_temp = label_path.with_name(label_path.name + ".tmp") if label_path else None
    for path in (feature_temp, label_temp):
        if path is not None and path.exists():
            path.unlink()

    dtype = np.float32 if generator.to_float32 else np.uint8
    feature_mm = open_memmap(
        feature_temp,
        mode="w+",
        dtype=dtype,
        shape=feature_shape(generator),
    )
    label_mm = (
        open_memmap(label_temp, mode="w+", dtype=np.uint8, shape=(generator.n,))
        if label_temp is not None
        else None
    )

    offset = 0
    validation_reports: List[Dict[str, Any]] = []
    try:
        for batch_index in range(len(generator)):
            batch = generator[batch_index]
            if validate:
                features, labels, metadata = batch
            else:
                features, labels = batch
                metadata = None

            features_np = np.asarray(features)
            labels_np = np.asarray(labels, dtype=np.uint8)
            end = offset + len(labels_np)
            feature_mm[offset:end] = features_np
            if label_mm is not None:
                label_mm[offset:end] = labels_np

            if validate:
                report = validate_protocol_batch(
                    labels_np,
                    metadata,
                    plain_bits=generator.plain_bits,
                    key_bits=generator.key_bits,
                )
                report.update(
                    {
                        "batch_index": batch_index,
                        "control_type": metadata["control_type"],
                        "control_replica": metadata["control_replica"],
                        "source_round": metadata["source_round"],
                        "source_k": metadata["source_k"],
                        "source_seed": metadata["source_seed"],
                    }
                )
                validation_reports.append(report)
            offset = end

        if offset != generator.n:
            raise RuntimeError(f"Generated {offset} rows; expected {generator.n}")

        feature_mm.flush()
        if label_mm is not None:
            label_mm.flush()
        del feature_mm
        if label_mm is not None:
            del label_mm
        os.replace(feature_temp, feature_path)
        if label_temp is not None and label_path is not None:
            os.replace(label_temp, label_path)
    except Exception:
        try:
            del feature_mm
            if label_mm is not None:
                del label_mm
        except Exception:
            pass
        for path in (feature_temp, label_temp):
            if path is not None and path.exists():
                path.unlink()
        raise

    return {
        "feature_path": str(feature_path),
        "label_path": str(label_path) if label_path else None,
        "shape": list(feature_shape(generator)),
        "dtype": np.dtype(dtype).name,
        "samples_written": int(offset),
        "validation_batches": validation_reports,
        "all_passed": (
            all(report["all_passed"] for report in validation_reports)
            if validate
            else True
        ),
        "metadata": {
            "control_type": generator.control_type,
            "control_replica": generator.control_replica,
            "source_round": generator.source_round,
            "source_k": generator.source_k,
            "source_seed": generator.source_seed,
        },
    }


def ensure_dataset_group(
    *,
    marker_path: Path,
    config_hash: str,
    outputs: Sequence[Path],
    overwrite: bool,
    generate_fn: Callable[[], Dict[str, Any]],
) -> Dict[str, Any]:
    if marker_path.exists() and not overwrite:
        record = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            record.get("status") == "completed"
            and record.get("config_hash") == config_hash
            and all(path.exists() for path in outputs)
        ):
            record["_status"] = "skipped_existing"
            return record
        raise RuntimeError(f"Existing marker mismatch at {marker_path.parent}")

    if not marker_path.exists() and not overwrite:
        orphaned = [str(path) for path in outputs if path.exists()]
        if orphaned:
            raise RuntimeError(
                "Found output files without a completion marker. Verify or remove "
                f"them before resuming: {orphaned}"
            )

    if overwrite:
        for path in [marker_path, *outputs]:
            if path.exists():
                path.unlink()

    started = time.perf_counter()
    result = generate_fn()
    record = {
        "status": "completed",
        "config_hash": config_hash,
        "outputs": [str(path) for path in outputs],
        "result": result,
        "seconds": round(time.perf_counter() - started, 6),
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(marker_path, record)
    record["_status"] = "completed"
    return record


def create_pair_shuffled_file(
    source_path: Path,
    destination_path: Path,
    *,
    pairs: int,
    pair_feature_dim: int,
    output_layout: str,
    seed: int,
    chunk_groups: int = 2_000,
) -> Dict[str, Any]:
    source = np.load(source_path, mmap_mode="r")
    if source.ndim not in {2, 3}:
        raise ValueError(f"Unsupported source shape for pair shuffling: {source.shape}")

    n = int(source.shape[0])
    grouped = source.reshape(n, pairs, pair_feature_dim)
    flat_pairs = grouped.reshape(n * pairs, pair_feature_dim)
    permutation = np.random.default_rng(seed).permutation(n * pairs)

    expected_shape = (
        (n, pairs, pair_feature_dim)
        if output_layout == "grouped"
        else (n, pairs * pair_feature_dim)
    )
    temp = destination_path.with_name(destination_path.name + ".tmp")
    if temp.exists():
        temp.unlink()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    output = open_memmap(
        temp,
        mode="w+",
        dtype=source.dtype,
        shape=expected_shape,
    )

    source_sum = np.zeros(pair_feature_dim, dtype=np.int64)
    destination_sum = np.zeros(pair_feature_dim, dtype=np.int64)
    try:
        pair_chunk = max(1, int(chunk_groups)) * pairs
        for start_pair in range(0, n * pairs, pair_chunk):
            end_pair = min(n * pairs, start_pair + pair_chunk)
            source_sum += np.asarray(flat_pairs[start_pair:end_pair]).sum(
                axis=0,
                dtype=np.int64,
            )

        for start_group in range(0, n, max(1, int(chunk_groups))):
            end_group = min(n, start_group + max(1, int(chunk_groups)))
            pair_start = start_group * pairs
            pair_end = end_group * pairs
            shuffled_pairs = np.asarray(flat_pairs[permutation[pair_start:pair_end]])
            shuffled = shuffled_pairs.reshape(
                end_group - start_group,
                pairs,
                pair_feature_dim,
            )
            if output_layout == "grouped":
                output[start_group:end_group] = shuffled
            else:
                output[start_group:end_group] = shuffled.reshape(
                    end_group - start_group,
                    pairs * pair_feature_dim,
                )
            destination_sum += shuffled_pairs.sum(axis=0, dtype=np.int64)

        output.flush()
        del output
        os.replace(temp, destination_path)
    except Exception:
        try:
            del output
        except Exception:
            pass
        if temp.exists():
            temp.unlink()
        raise

    return {
        "control_type": "pair_shuffled",
        "control_replica": 0,
        "shuffle_seed": int(seed),
        "shape": list(expected_shape),
        "marginal_bit_sum_preserved": bool(
            np.array_equal(source_sum, destination_sum)
        ),
        "grouping_effective": bool(pairs > 1),
    }


def adapter_smoke_check(
    adapter: CipherAdapter,
    config: Mapping[str, Any],
    *,
    delta_p: int,
    delta_k: int,
) -> Dict[str, Any]:
    round_number = min(int(value) for value in config["rounds"])
    generator = NDCMultiPairAnalysisGenerator(
        encryption_function=adapter.encrypt,
        plain_bits=adapter.plain_bits,
        key_bits=adapter.key_bits,
        nr=round_number,
        delta_state=delta_p,
        delta_key=delta_k,
        n_samples=2,
        batch_size=2,
        pairs=1,
        seed=91_337,
        class_mode="structured",
        key_mode="shared",
        output_layout="grouped",
        encrypt_backend=config["encrypt_backend"],
        to_float32=False,
        return_metadata=False,
    )
    features, labels = generator[0]
    expected_shape = (2, 1, 3 * adapter.plain_bits)
    if np.asarray(features).shape != expected_shape:
        raise ValueError(
            f"Cipher adapter smoke check returned {np.asarray(features).shape}; "
            f"expected {expected_shape}"
        )
    if np.asarray(labels).shape != (2,):
        raise ValueError("Cipher adapter smoke check returned invalid labels")
    if not np.all(np.isin(np.asarray(features), [0, 1])):
        raise ValueError("Cipher adapter output is not a binary bit array")
    return {
        "cipher": adapter.name,
        "cipher_slug": adapter.slug,
        "cipher_module": adapter.module_name,
        "encrypt_function": adapter.encrypt_function_name,
        "plain_bits": adapter.plain_bits,
        "key_bits": adapter.key_bits,
        "round": round_number,
        "feature_shape": list(expected_shape),
        "passed": True,
    }


def nested_all_passed(record: Mapping[str, Any]) -> bool:
    result = record.get("result", record)
    if isinstance(result, Mapping) and "all_passed" in result:
        return bool(result["all_passed"])
    values: List[bool] = []
    if isinstance(result, Mapping):
        for item in result.values():
            if isinstance(item, Mapping) and "all_passed" in item:
                values.append(bool(item["all_passed"]))
    return all(values) if values else True


def run_one_job(
    *,
    config: Mapping[str, Any],
    job: Mapping[str, Any],
    adapter: CipherAdapter,
    delta_p: int,
    delta_k: int,
    delta_k_resolved: Mapping[str, Any],
    output_root: Path,
    locked_config_path: Path,
    overwrite: bool,
) -> Dict[str, Any]:
    run_dir = run_directory(output_root, config, adapter, job)
    run_dir.mkdir(parents=True, exist_ok=True)
    controls_dir = run_dir / "controls"

    samples = int(job["samples_per_class"])
    source_seed = int(job["seed"]) + int(job["domain_seed_offset"])
    batch_size = min(int(config["batch_size"]), samples)
    validate = bool(config["validate_protocol"])
    pair_dim = 3 * adapter.plain_bits

    paths = {
        "d1": run_dir / "D1_structured.npy",
        "d0": run_dir / "D0_null.npy",
        "d1_labels": run_dir / "D1_labels.npy",
        "d0_labels": run_dir / "D0_labels.npy",
        "d0_a": controls_dir / "D0_vs_D0" / "D0_control_a.npy",
        "d0_b": controls_dir / "D0_vs_D0" / "D0_control_b.npy",
        "d1_a": controls_dir / "D1_vs_D1" / "D1_control_a.npy",
        "d1_b": controls_dir / "D1_vs_D1" / "D1_control_b.npy",
        "pair": controls_dir / "pair_shuffled" / "D1_pair_shuffled.npy",
    }

    base_payload = {
        "config_schema_version": config.get("config_schema_version"),
        "study_id": config["study_id"],
        "locked_config_path": str(locked_config_path),
        "domain": job["domain"],
        "cipher": adapter.name,
        "cipher_slug": adapter.slug,
        "cipher_module": adapter.module_name,
        "encrypt_function": adapter.encrypt_function_name,
        "round": int(job["round"]),
        "k": int(job["k"]),
        "seed": int(job["seed"]),
        "source_seed": source_seed,
        "samples_per_class": samples,
        "batch_size": batch_size,
        "plain_bits": adapter.plain_bits,
        "key_bits": adapter.key_bits,
        "delta_p": int(delta_p),
        "delta_p_hex": f"0x{delta_p:0{(adapter.plain_bits + 3) // 4}x}",
        "delta_k": dict(delta_k_resolved),
        "representation": config["representation"],
        "output_layout": config["output_layout"],
        "key_mode": config["key_mode"],
        "dtype": config["dtype"],
        "encrypt_backend": config["encrypt_backend"],
        "validate_protocol": validate,
    }

    main_hash = canonical_hash({**base_payload, "control_type": "D1_vs_D0"})
    d0_hash = canonical_hash(
        {
            **base_payload,
            "control_type": "D0_vs_D0",
            "seeds": [
                derive_control_seed(source_seed, "D0_control_a"),
                derive_control_seed(source_seed, "D0_control_b"),
            ],
        }
    )
    d1_hash = canonical_hash(
        {
            **base_payload,
            "control_type": "D1_vs_D1",
            "seeds": [
                derive_control_seed(source_seed, "D1_control_a"),
                derive_control_seed(source_seed, "D1_control_b"),
            ],
        }
    )
    pair_hash = canonical_hash(
        {
            **base_payload,
            "control_type": "pair_shuffled",
            "source_path": str(paths["d1"]),
            "shuffle_seed": derive_control_seed(source_seed, "pair_shuffled"),
        }
    )
    overall_hash = canonical_hash(
        {
            "main": main_hash,
            "D0_vs_D0": d0_hash,
            "D1_vs_D1": d1_hash,
            "pair_shuffled": pair_hash,
        }
    )

    all_outputs = [
        paths["d1"],
        paths["d0"],
        paths["d1_labels"],
        paths["d0_labels"],
        paths["d0_a"],
        paths["d0_b"],
        paths["d1_a"],
        paths["d1_b"],
        paths["pair"],
    ]
    completion_path = run_dir / "run_complete.json"
    if completion_path.exists() and not overwrite:
        completed = json.loads(completion_path.read_text(encoding="utf-8"))
        if (
            completed.get("status") == "completed"
            and completed.get("config_hash") == overall_hash
            and all(path.exists() for path in all_outputs)
        ):
            completed["status"] = "skipped_existing"
            return completed
        raise RuntimeError(
            f"Existing completed run does not match the locked job at {run_dir}. "
            "Use --overwrite only after reviewing the configuration change."
        )

    if overwrite:
        for metadata_path in (
            completion_path,
            run_dir / "run_failed.json",
            run_dir / "run_config.json",
            run_dir / "protocol_validation.json",
            run_dir / "control_metadata.json",
        ):
            if metadata_path.exists():
                metadata_path.unlink()

    run_config_path = run_dir / "run_config.json"
    validation_path = run_dir / "protocol_validation.json"
    metadata_path = run_dir / "control_metadata.json"
    atomic_write_json(
        run_config_path,
        {
            **base_payload,
            "main_config_hash": main_hash,
            "d0_control_config_hash": d0_hash,
            "d1_control_config_hash": d1_hash,
            "pair_shuffled_config_hash": pair_hash,
            "config_hash": overall_hash,
        },
    )

    common = dict(
        encryption_function=adapter.encrypt,
        plain_bits=adapter.plain_bits,
        key_bits=adapter.key_bits,
        nr=int(job["round"]),
        delta_state=delta_p,
        delta_key=delta_k,
        n_samples=samples,
        batch_size=batch_size,
        pairs=int(job["k"]),
        key_mode=config["key_mode"],
        output_layout=config["output_layout"],
        encrypt_backend=config["encrypt_backend"],
        to_float32=(config["dtype"] == "float32"),
        return_metadata=validate,
        metadata_full=validate,
    )

    started = time.perf_counter()
    try:
        def generate_main() -> Dict[str, Any]:
            d1_gen = NDCMultiPairAnalysisGenerator(
                **common,
                seed=derive_control_seed(source_seed, "main_D1"),
                class_mode="structured",
                control_type="D1_vs_D0",
                control_replica=0,
                source_round=int(job["round"]),
                source_k=int(job["k"]),
                source_seed=source_seed,
            )
            d0_gen = NDCMultiPairAnalysisGenerator(
                **common,
                seed=derive_control_seed(source_seed, "main_D0"),
                class_mode="null",
                control_type="D1_vs_D0",
                control_replica=1,
                source_round=int(job["round"]),
                source_k=int(job["k"]),
                source_seed=source_seed,
            )
            return {
                "D1": stream_generator_to_npy(
                    d1_gen,
                    paths["d1"],
                    paths["d1_labels"],
                    validate=validate,
                ),
                "D0": stream_generator_to_npy(
                    d0_gen,
                    paths["d0"],
                    paths["d0_labels"],
                    validate=validate,
                ),
            }

        main_record = ensure_dataset_group(
            marker_path=run_dir / "main_complete.json",
            config_hash=main_hash,
            outputs=[
                paths["d1"],
                paths["d0"],
                paths["d1_labels"],
                paths["d0_labels"],
            ],
            overwrite=overwrite,
            generate_fn=generate_main,
        )

        def generate_d0_control() -> Dict[str, Any]:
            gen_a, gen_b = build_control_generators(
                "D0_vs_D0",
                common_kwargs=common,
                source_seed=source_seed,
            )
            return {
                "A": stream_generator_to_npy(
                    gen_a,
                    paths["d0_a"],
                    None,
                    validate=validate,
                ),
                "B": stream_generator_to_npy(
                    gen_b,
                    paths["d0_b"],
                    None,
                    validate=validate,
                ),
            }

        d0_record = ensure_dataset_group(
            marker_path=controls_dir / "D0_vs_D0" / "control_complete.json",
            config_hash=d0_hash,
            outputs=[paths["d0_a"], paths["d0_b"]],
            overwrite=overwrite,
            generate_fn=generate_d0_control,
        )

        def generate_d1_control() -> Dict[str, Any]:
            gen_a, gen_b = build_control_generators(
                "D1_vs_D1",
                common_kwargs=common,
                source_seed=source_seed,
            )
            return {
                "A": stream_generator_to_npy(
                    gen_a,
                    paths["d1_a"],
                    None,
                    validate=validate,
                ),
                "B": stream_generator_to_npy(
                    gen_b,
                    paths["d1_b"],
                    None,
                    validate=validate,
                ),
            }

        d1_record = ensure_dataset_group(
            marker_path=controls_dir / "D1_vs_D1" / "control_complete.json",
            config_hash=d1_hash,
            outputs=[paths["d1_a"], paths["d1_b"]],
            overwrite=overwrite,
            generate_fn=generate_d1_control,
        )

        def generate_pair_control() -> Dict[str, Any]:
            metadata = create_pair_shuffled_file(
                paths["d1"],
                paths["pair"],
                pairs=int(job["k"]),
                pair_feature_dim=pair_dim,
                output_layout=config["output_layout"],
                seed=derive_control_seed(source_seed, "pair_shuffled"),
            )
            metadata.update(
                {
                    "source_round": int(job["round"]),
                    "source_k": int(job["k"]),
                    "source_seed": source_seed,
                }
            )
            if not metadata["marginal_bit_sum_preserved"]:
                raise AssertionError("Pair-shuffled marginal check failed")
            return metadata

        pair_record = ensure_dataset_group(
            marker_path=controls_dir / "pair_shuffled" / "control_complete.json",
            config_hash=pair_hash,
            outputs=[paths["pair"]],
            overwrite=overwrite,
            generate_fn=generate_pair_control,
        )

        protocol_valid = bool(
            nested_all_passed(main_record)
            and nested_all_passed(d0_record)
            and nested_all_passed(d1_record)
            and pair_record["result"]["marginal_bit_sum_preserved"]
        )

        validation_report = {
            "study_id": config["study_id"],
            "domain": job["domain"],
            "cipher": adapter.name,
            "round": int(job["round"]),
            "k": int(job["k"]),
            "seed": int(job["seed"]),
            "source_seed": source_seed,
            "main": main_record["result"],
            "D0_vs_D0": d0_record["result"],
            "D1_vs_D1": d1_record["result"],
            "pair_shuffled": pair_record["result"],
            "all_passed": protocol_valid,
            "validated_at_utc": utc_now(),
        }
        atomic_write_json(validation_path, validation_report)
        atomic_write_json(
            metadata_path,
            {
                "main": main_record,
                "D0_vs_D0": d0_record,
                "D1_vs_D1": d1_record,
                "pair_shuffled": pair_record,
            },
        )

        if validate and not protocol_valid:
            raise AssertionError("One or more main/control validations failed")

        actual_bytes = sum(path.stat().st_size for path in all_outputs)
        completion = {
            "study_id": config["study_id"],
            "domain": job["domain"],
            "cipher": adapter.name,
            "cipher_slug": adapter.slug,
            "cipher_module": adapter.module_name,
            "round": int(job["round"]),
            "k": int(job["k"]),
            "seed": int(job["seed"]),
            "source_seed": source_seed,
            "samples_per_class": samples,
            "plain_bits": adapter.plain_bits,
            "key_bits": adapter.key_bits,
            "delta_p_hex": base_payload["delta_p_hex"],
            "delta_k_hex": delta_k_resolved["hex"],
            "delta_k_bit": delta_k_resolved.get("bit_index"),
            "delta_k_indexing": delta_k_resolved.get("indexing"),
            "representation": config["representation"],
            "output_layout": config["output_layout"],
            "key_mode": config["key_mode"],
            "dtype": config["dtype"],
            "feature_shape_per_class": json.dumps(
                main_record["result"]["D1"]["shape"]
            ),
            "d1_path": str(paths["d1"]),
            "d0_path": str(paths["d0"]),
            "d1_labels_path": str(paths["d1_labels"]),
            "d0_labels_path": str(paths["d0_labels"]),
            "d0_control_a_path": str(paths["d0_a"]),
            "d0_control_b_path": str(paths["d0_b"]),
            "d1_control_a_path": str(paths["d1_a"]),
            "d1_control_b_path": str(paths["d1_b"]),
            "pair_shuffled_path": str(paths["pair"]),
            "main_config_hash": main_hash,
            "d0_control_config_hash": d0_hash,
            "d1_control_config_hash": d1_hash,
            "pair_shuffled_config_hash": pair_hash,
            "run_config_path": str(run_config_path),
            "validation_path": str(validation_path),
            "control_metadata_path": str(metadata_path),
            "protocol_valid": protocol_valid,
            "generation_seconds": round(time.perf_counter() - started, 6),
            "actual_bytes": int(actual_bytes),
            "config_hash": overall_hash,
            "completed_at_utc": utc_now(),
            "status": "completed",
        }
        atomic_write_json(completion_path, completion)
        failed_path = run_dir / "run_failed.json"
        if failed_path.exists():
            failed_path.unlink()
        return completion

    except Exception as exc:
        atomic_write_json(
            run_dir / "run_failed.json",
            {
                "study_id": config["study_id"],
                "domain": job["domain"],
                "cipher": adapter.name,
                "round": int(job["round"]),
                "k": int(job["k"]),
                "seed": int(job["seed"]),
                "source_seed": source_seed,
                "config_hash": overall_hash,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at_utc": utc_now(),
                "status": "failed",
            },
        )
        raise


def rebuild_manifest(study_root: Path) -> Tuple[int, Path]:
    rows = []
    for marker in study_root.rglob("run_complete.json"):
        try:
            row = json.loads(marker.read_text(encoding="utf-8"))
            if row.get("status") == "completed":
                rows.append(row)
        except Exception:
            continue
    rows.sort(
        key=lambda row: (
            row.get("domain", ""),
            row.get("cipher_slug", row.get("cipher", "")),
            int(row.get("round", 0)),
            int(row.get("k", 0)),
            int(row.get("seed", 0)),
        )
    )
    path = study_root / "manifest.csv"
    atomic_write_csv(path, rows)
    return len(rows), path


def format_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{value} B"


def estimate_job_bytes(
    *,
    samples_per_class: int,
    k: int,
    plain_bits: int,
    dtype: str,
) -> int:
    itemsize = np.dtype(dtype).itemsize
    feature_elements = samples_per_class * k * 3 * plain_bits
    # Seven feature arrays: D1, D0, D0a, D0b, D1a, D1b, pair-shuffled D1.
    feature_bytes = 7 * feature_elements * itemsize
    label_bytes = 2 * samples_per_class * np.dtype(np.uint8).itemsize
    return int(feature_bytes + label_bytes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cipher-agnostic RKMP main data and controls"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a locked analysis-data JSON configuration",
    )
    parser.add_argument("--out", default="analysis_data")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-adapter-check", action="store_true")
    parser.add_argument("--rounds", type=int, nargs="+")
    parser.add_argument("--k-values", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--domains", nargs="+")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    output_root = Path(args.out).resolve()
    config = load_locked_config(config_path)
    adapter = load_cipher_adapter(config)

    delta_p = parse_int(config["delta_p_hex"], name="delta_p")
    if delta_p < 0 or delta_p >= (1 << adapter.plain_bits):
        raise ValueError(f"delta_p does not fit in {adapter.plain_bits} bits")
    delta_k, delta_k_resolved = resolve_delta_k(config["delta_k"], adapter.key_bits)

    if config["adapter_check"] and not args.skip_adapter_check:
        check = adapter_smoke_check(
            adapter,
            config,
            delta_p=delta_p,
            delta_k=delta_k,
        )
        print(
            "[ADAPTER OK] "
            f"{check['cipher']} | module={check['cipher_module']} | "
            f"block={check['plain_bits']} | key={check['key_bits']}"
        )

    jobs = filter_jobs(
        list(iter_jobs(config)),
        rounds=args.rounds,
        k_values=args.k_values,
        seeds=args.seeds,
        domains=args.domains,
    )
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.limit:
        jobs = jobs[: args.limit]

    print("=" * 78)
    print("RKMP ANALYSIS-DATA GRID")
    print("=" * 78)
    print(f"Study          : {config['study_id']}")
    print(f"Cipher         : {adapter.name} ({adapter.slug})")
    print(f"Module         : {adapter.module_name}")
    print(f"Block/key bits : {adapter.plain_bits}/{adapter.key_bits}")
    print(f"Delta P        : 0x{delta_p:0{(adapter.plain_bits + 3) // 4}x}")
    print(f"Delta K        : {delta_k_resolved['hex']}")
    print(f"Jobs           : {len(jobs)}")
    print(f"Output root    : {output_root}")
    print("=" * 78)

    if args.dry_run:
        total_estimate = 0
        for index, job in enumerate(jobs, start=1):
            estimate = estimate_job_bytes(
                samples_per_class=int(job["samples_per_class"]),
                k=int(job["k"]),
                plain_bits=adapter.plain_bits,
                dtype=config["dtype"],
            )
            total_estimate += estimate
            print(
                f"[{index:03d}] domain={job['domain']} r={job['round']} "
                f"k={job['k']} seed={job['seed']} "
                f"samples/class={job['samples_per_class']} "
                f"estimated={format_bytes(estimate)}"
            )
        print(f"Estimated grid bytes: {format_bytes(total_estimate)}")
        return

    completed = 0
    skipped = 0
    failed = 0

    for index, job in enumerate(jobs, start=1):
        label = (
            f"cipher={adapter.slug} domain={job['domain']} "
            f"r={job['round']} k={job['k']} seed={job['seed']}"
        )
        print(f"\n[{index:03d}/{len(jobs):03d}] START {label}")
        try:
            result = run_one_job(
                config=config,
                job=job,
                adapter=adapter,
                delta_p=delta_p,
                delta_k=delta_k,
                delta_k_resolved=delta_k_resolved,
                output_root=output_root,
                locked_config_path=config_path,
                overwrite=args.overwrite,
            )
            if result["status"] == "skipped_existing":
                skipped += 1
                print(f"[SKIP] Existing verified run: {label}")
            else:
                completed += 1
                print(
                    f"[DONE] {label} | "
                    f"{result['generation_seconds']:.2f}s | "
                    f"{format_bytes(int(result['actual_bytes']))}"
                )
        except Exception as exc:
            failed += 1
            print(
                f"[FAIL] {label} | {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            if not args.continue_on_error:
                study_root = output_root / config["study_id"]
                rebuild_manifest(study_root)
                raise

    study_root = output_root / config["study_id"]
    manifest_rows, manifest_path = rebuild_manifest(study_root)

    print("\n" + "=" * 78)
    print("RUN SUMMARY")
    print("=" * 78)
    print(f"Completed jobs : {completed}")
    print(f"Skipped jobs   : {skipped}")
    print(f"Failed jobs    : {failed}")
    print(f"Manifest rows  : {manifest_rows}")
    print(f"Manifest       : {manifest_path}")
    print(f"Study root     : {study_root}")
    print("=" * 78)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

# python signal_regimes/run_analysis_data.py \
#   --config signal_regimes/configs/present80_rkmp_controls_full.json \
#   --out signal_regimes/analysis_data \
#   --dry-run


# python signal_regimes/run_analysis_data.py \
#   --config signal_regimes/configs/present80_rkmp_controls_v1.json \
#   --out signal_regimes/analysis_data 
