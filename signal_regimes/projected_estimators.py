"""View-specific projected estimators for RKMP ciphertext analysis.

The module intentionally separates two complementary statistical views:

1. Distributional view
   - projected MMD²;
   - permutation calibration;
   - same-class control floor;
   - control-adjusted MMD excess.

2. Projected geometry view
   - PCA visualization coordinates (through ``prepare_projected_context``);
   - K-Means silhouette;
   - cluster/class alignment through ARI;
   - K-Means initialization stability;
   - same-class control-adjusted silhouette excess.

Both views use the same representation contract:

    pooled main D1/D0 reference -> one locked PCA fit -> transform every
    main/control dataset with the same fitted projection.

This is important because comparisons should not be subtracted or interpreted
as belonging to one trajectory when they were computed in unrelated PCA bases.
The public view-specific APIs are:

    analyze_projected_mmd_bundle(...)
    flatten_mmd_bundle_for_csv(...)

    analyze_projected_geometry_bundle(...)
    flatten_geometry_bundle_for_csv(...)

No Signal-Regime or manuscript-level interpretation is performed here.  Those
belong in downstream analysis scripts.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    pairwise_distances,
    silhouette_score,
)


FloatArray = NDArray[np.floating]
UInt8Array = NDArray[np.uint8]
IntArray = NDArray[np.integer]


DATASET_KEYS = (
    "d1",
    "d0",
    "d0_control_a",
    "d0_control_b",
    "d1_control_a",
    "d1_control_b",
    "pair_shuffled",
)

COMPARISON_DATASETS = {
    "D1_vs_D0": ("d1", "d0"),
    "D0_vs_D0": ("d0_control_a", "d0_control_b"),
    "D1_vs_D1": ("d1_control_a", "d1_control_b"),
    "pair_shuffled": ("pair_shuffled", "d0"),
}


@dataclass(frozen=True)
class ProjectionProtocol:
    """Locked representation and balanced sampling protocol."""

    max_samples_total: int = 100_000
    pca_components: int = 16
    pca_solver: str = "auto"
    pca_whiten: bool = False
    random_state: int = 42
    sample_mode: str = "head"

    def validate(self) -> None:
        if self.max_samples_total < 4 or self.max_samples_total % 2:
            raise ValueError("max_samples_total must be an even integer >= 4")
        if self.pca_components <= 0:
            raise ValueError("pca_components must be positive")
        if self.pca_solver not in {"auto", "full", "arpack", "randomized"}:
            raise ValueError("Unsupported PCA solver")
        if self.sample_mode not in {"head", "deterministic_random"}:
            raise ValueError(
                "sample_mode must be 'head' or 'deterministic_random'"
            )


@dataclass(frozen=True)
class MMDProtocol:
    """Projected MMD and permutation-test settings."""

    kernel: str = "rbf"
    bandwidth_mode: str = "median_main_reference"
    bandwidth: Optional[float] = None
    permutations: int = 200
    max_samples_per_distribution: int = 500
    random_state: int = 42
    bandwidth_reference_max_points: int = 1_000

    def validate(self) -> None:
        if self.kernel not in {"rbf", "linear"}:
            raise ValueError("kernel must be 'rbf' or 'linear'")
        if self.bandwidth_mode not in {
            "median_main_reference",
            "fixed",
        }:
            raise ValueError(
                "bandwidth_mode must be 'median_main_reference' or 'fixed'"
            )
        if self.bandwidth_mode == "fixed":
            if self.bandwidth is None or self.bandwidth <= 0:
                raise ValueError("fixed bandwidth requires bandwidth > 0")
        if self.permutations <= 0:
            raise ValueError("permutations must be positive")
        if self.max_samples_per_distribution < 2:
            raise ValueError("max_samples_per_distribution must be >= 2")
        if self.bandwidth_reference_max_points < 2:
            raise ValueError("bandwidth_reference_max_points must be >= 2")


@dataclass(frozen=True)
class GeometryProtocol:
    """PCA/K-Means projected-geometry settings."""

    kmeans_clusters: int = 2
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300
    stability_repetitions: int = 10
    random_state: int = 42

    def validate(self) -> None:
        if self.kmeans_clusters != 2:
            raise ValueError("The binary geometry analysis requires 2 clusters")
        if self.kmeans_n_init <= 0:
            raise ValueError("kmeans_n_init must be positive")
        if self.kmeans_max_iter <= 0:
            raise ValueError("kmeans_max_iter must be positive")
        if self.stability_repetitions < 2:
            raise ValueError("stability_repetitions must be at least 2")


@dataclass
class ProjectedContext:
    """In-memory datasets transformed through one shared PCA model."""

    projected: Dict[str, FloatArray]
    selected_count_per_dataset: int
    raw_feature_dim: int
    stored_shapes: Dict[str, Sequence[int]]
    stored_dtypes: Dict[str, str]
    sample_indices: Dict[str, NDArray[np.int64]]
    pca: PCA
    projection_id: str
    sample_selection_hash: str
    pair_shuffle_k1_identical: Optional[bool]
    metadata: Dict[str, Any]

    def comparison(self, control_type: str) -> Tuple[FloatArray, FloatArray]:
        try:
            left_key, right_key = COMPARISON_DATASETS[control_type]
        except KeyError as exc:
            raise ValueError(f"Unknown comparison type: {control_type}") from exc
        return self.projected[left_key], self.projected[right_key]


# ---------------------------------------------------------------------------
# Shared representation
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_feature_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    array = np.load(path, mmap_mode="r")
    if array.ndim not in {2, 3}:
        raise ValueError(f"Unsupported feature shape {array.shape} at {path}")
    if array.shape[0] < 2:
        raise ValueError(f"At least two observations are required at {path}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"Feature array must be numeric: {path}")
    return array


def _validate_loaded_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    missing = set(DATASET_KEYS).difference(arrays)
    if missing:
        raise ValueError(f"Missing datasets: {sorted(missing)}")

    reference = arrays["d1"]
    for key, array in arrays.items():
        if array.shape[1:] != reference.shape[1:]:
            raise ValueError(
                f"Feature-shape mismatch: {key}={array.shape}, "
                f"reference={reference.shape}"
            )
        if array.dtype != reference.dtype:
            raise ValueError(
                f"Feature dtype mismatch: {key}={array.dtype}, "
                f"reference={reference.dtype}"
            )


def _build_sample_indices(
    arrays: Mapping[str, np.ndarray],
    protocol: ProjectionProtocol,
) -> Tuple[Dict[str, NDArray[np.int64]], int]:
    max_per_dataset = protocol.max_samples_total // 2
    n = min(
        min(int(array.shape[0]) for array in arrays.values()),
        int(max_per_dataset),
    )
    if n < 2:
        raise ValueError("At least two selected samples per dataset are required")

    indices: Dict[str, NDArray[np.int64]] = {}
    for offset, key in enumerate(DATASET_KEYS):
        count = int(arrays[key].shape[0])
        if n == count or protocol.sample_mode == "head":
            indices[key] = np.arange(n, dtype=np.int64)
        else:
            rng = np.random.default_rng(protocol.random_state + 10_000 + offset)
            indices[key] = np.sort(
                rng.choice(count, size=n, replace=False).astype(np.int64)
            )
    return indices, n


def _flatten_selected(
    array: np.ndarray,
    indices: NDArray[np.int64],
) -> FloatArray:
    selected = np.asarray(array[indices], dtype=np.float32)
    return selected.reshape(len(indices), -1)


def prepare_projected_context(
    *,
    d1_path: Path,
    d0_path: Path,
    d0_control_a_path: Path,
    d0_control_b_path: Path,
    d1_control_a_path: Path,
    d1_control_b_path: Path,
    pair_shuffled_path: Path,
    projection_protocol: ProjectionProtocol,
    run_metadata: Optional[Mapping[str, Any]] = None,
) -> ProjectedContext:
    """Fit one PCA on pooled main D1/D0 data and transform every dataset.

    This function is public so visualization code can use the exact same
    representation contract as both statistical views.
    """

    projection_protocol.validate()
    paths = {
        "d1": Path(d1_path),
        "d0": Path(d0_path),
        "d0_control_a": Path(d0_control_a_path),
        "d0_control_b": Path(d0_control_b_path),
        "d1_control_a": Path(d1_control_a_path),
        "d1_control_b": Path(d1_control_b_path),
        "pair_shuffled": Path(pair_shuffled_path),
    }
    arrays = {key: load_feature_array(path) for key, path in paths.items()}
    _validate_loaded_arrays(arrays)

    sample_indices, n = _build_sample_indices(arrays, projection_protocol)
    flattened = {
        key: _flatten_selected(arrays[key], sample_indices[key])
        for key in DATASET_KEYS
    }
    raw_feature_dim = int(flattened["d1"].shape[1])
    if raw_feature_dim < projection_protocol.pca_components:
        raise ValueError(
            f"Raw dimension {raw_feature_dim} is smaller than PCA dimension "
            f"{projection_protocol.pca_components}"
        )

    reference_matrix = np.concatenate(
        [flattened["d1"], flattened["d0"]], axis=0
    )
    if reference_matrix.shape[0] <= projection_protocol.pca_components:
        raise ValueError("Insufficient pooled observations for requested PCA")

    pca = PCA(
        n_components=projection_protocol.pca_components,
        whiten=projection_protocol.pca_whiten,
        svd_solver=projection_protocol.pca_solver,
        random_state=projection_protocol.random_state,
    )
    pca.fit(reference_matrix)
    projected = {
        key: np.asarray(pca.transform(matrix), dtype=np.float32)
        for key, matrix in flattened.items()
    }

    index_payload = {
        key: sample_indices[key].tolist() for key in DATASET_KEYS
    }
    sample_selection_hash = _canonical_hash(index_payload)
    projection_digest = hashlib.sha256()
    projection_digest.update(
        json.dumps(
            asdict(projection_protocol),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    projection_digest.update(np.asarray(pca.mean_, dtype=np.float64).tobytes())
    projection_digest.update(
        np.asarray(pca.components_, dtype=np.float64).tobytes()
    )
    projection_digest.update(sample_selection_hash.encode("ascii"))
    projection_id = projection_digest.hexdigest()

    metadata = dict(run_metadata or {})
    k_value = metadata.get("k")
    pair_shuffle_k1_identical: Optional[bool] = None
    if k_value is not None and int(k_value) == 1:
        pair_shuffle_k1_identical = bool(
            np.array_equal(
                np.asarray(arrays["d1"][sample_indices["d1"]]),
                np.asarray(
                    arrays["pair_shuffled"][sample_indices["pair_shuffled"]]
                ),
            )
        )

    return ProjectedContext(
        projected=projected,
        selected_count_per_dataset=n,
        raw_feature_dim=raw_feature_dim,
        stored_shapes={key: list(array.shape) for key, array in arrays.items()},
        stored_dtypes={key: str(array.dtype) for key, array in arrays.items()},
        sample_indices=sample_indices,
        pca=pca,
        projection_id=projection_id,
        sample_selection_hash=sample_selection_hash,
        pair_shuffle_k1_identical=pair_shuffle_k1_identical,
        metadata=metadata,
    )


def projection_summary(context: ProjectedContext) -> Dict[str, Any]:
    return {
        "projection_id": context.projection_id,
        "sample_selection_hash": context.sample_selection_hash,
        "fit_scope": "pooled_main_D1_D0",
        "reuse_for_controls": True,
        "pca_components": int(context.pca.n_components_),
        "explained_variance_ratio": (
            context.pca.explained_variance_ratio_.tolist()
        ),
        "explained_variance_sum": float(
            np.sum(context.pca.explained_variance_ratio_)
        ),
        "selected_count_per_dataset": context.selected_count_per_dataset,
        "raw_feature_dim": context.raw_feature_dim,
        "stored_shapes": context.stored_shapes,
        "stored_dtypes": context.stored_dtypes,
        "pair_shuffle_k1_identical": context.pair_shuffle_k1_identical,
    }


# ---------------------------------------------------------------------------
# Distributional view: projected MMD
# ---------------------------------------------------------------------------


def _median_bandwidth_from_main_reference(
    context: ProjectedContext,
    protocol: MMDProtocol,
) -> float:
    reference = np.concatenate(
        [context.projected["d1"], context.projected["d0"]], axis=0
    )
    n = len(reference)
    max_points = min(n, protocol.bandwidth_reference_max_points)
    if max_points == n:
        indices = np.arange(n, dtype=np.int64)
    else:
        rng = np.random.default_rng(protocol.random_state + 70_003)
        indices = rng.choice(n, size=max_points, replace=False)
    distances = pairwise_distances(
        np.asarray(reference[indices], dtype=np.float32), metric="euclidean"
    )
    upper = distances[np.triu_indices_from(distances, k=1)]
    positive = upper[upper > 0]
    return 1.0 if positive.size == 0 else float(np.median(positive))


def _select_mmd_samples(
    left: FloatArray,
    right: FloatArray,
    protocol: MMDProtocol,
    *,
    seed_offset: int,
) -> Tuple[FloatArray, FloatArray]:
    n = min(
        len(left),
        len(right),
        protocol.max_samples_per_distribution,
    )
    if n < 2:
        raise ValueError("At least two MMD samples per distribution are required")
    rng = np.random.default_rng(protocol.random_state + seed_offset)
    left_indices = (
        np.arange(n, dtype=np.int64)
        if len(left) == n
        else rng.choice(len(left), size=n, replace=False)
    )
    right_indices = (
        np.arange(n, dtype=np.int64)
        if len(right) == n
        else rng.choice(len(right), size=n, replace=False)
    )
    return (
        np.asarray(left[left_indices], dtype=np.float32),
        np.asarray(right[right_indices], dtype=np.float32),
    )


def mmd2_permutation_test(
    left: FloatArray,
    right: FloatArray,
    protocol: MMDProtocol,
    *,
    bandwidth: Optional[float],
    seed_offset: int,
) -> Dict[str, Any]:
    """Biased non-negative MMD² with a balanced permutation test."""

    protocol.validate()
    left_selected, right_selected = _select_mmd_samples(
        left, right, protocol, seed_offset=seed_offset
    )
    n = len(left_selected)
    z = np.concatenate([left_selected, right_selected], axis=0).astype(
        np.float32, copy=False
    )

    if protocol.kernel == "linear":
        gram = z @ z.T
        used_bandwidth = None
    else:
        if bandwidth is None or bandwidth <= 0:
            raise ValueError("RBF MMD requires a positive shared bandwidth")
        used_bandwidth = float(bandwidth)
        squared = pairwise_distances(z, metric="sqeuclidean")
        gram = np.exp(
            -squared / (2.0 * used_bandwidth * used_bandwidth)
        ).astype(np.float32)

    signs = np.concatenate(
        [
            np.full(n, 1.0 / n, dtype=np.float32),
            np.full(n, -1.0 / n, dtype=np.float32),
        ]
    )
    observed = max(float(signs @ gram @ signs), 0.0)

    rng = np.random.default_rng(protocol.random_state + seed_offset + 1)
    base = np.concatenate(
        [np.ones(n, dtype=np.float32), -np.ones(n, dtype=np.float32)]
    ) / n
    permutation_signs = np.empty(
        (protocol.permutations, 2 * n), dtype=np.float32
    )
    for repetition in range(protocol.permutations):
        permutation_signs[repetition] = rng.permutation(base)
    null_values = np.sum((permutation_signs @ gram) * permutation_signs, axis=1)
    null_values = np.maximum(null_values.astype(np.float64), 0.0)
    p_value = float(
        (1 + np.sum(null_values >= observed)) / (protocol.permutations + 1)
    )

    return {
        "mmd2": observed,
        "mmd_permutation_p": p_value,
        "mmd_permutation_null_median": float(np.median(null_values)),
        "mmd_permutation_null_q95": float(np.quantile(null_values, 0.95)),
        "mmd_kernel": protocol.kernel,
        "mmd_bandwidth": used_bandwidth,
        "mmd_bandwidth_scope": (
            "shared_main_reference"
            if protocol.kernel == "rbf"
            else "not_applicable"
        ),
        "mmd_samples_per_distribution": int(n),
    }


def analyze_projected_mmd_context(
    context: ProjectedContext,
    mmd_protocol: MMDProtocol,
) -> Dict[str, Any]:
    """Run every MMD comparison inside one already-fitted projection."""

    mmd_protocol.validate()
    started = time.perf_counter()
    bandwidth: Optional[float]
    if mmd_protocol.kernel == "linear":
        bandwidth = None
    elif mmd_protocol.bandwidth_mode == "fixed":
        bandwidth = float(mmd_protocol.bandwidth)
    else:
        bandwidth = _median_bandwidth_from_main_reference(context, mmd_protocol)

    comparisons: Dict[str, Dict[str, Any]] = {}
    for offset, control_type in enumerate(COMPARISON_DATASETS, start=1):
        left, right = context.comparison(control_type)
        comparisons[control_type] = {
            "metadata": {
                **context.metadata,
                "control_type": control_type,
            },
            "mmd": mmd2_permutation_test(
                left,
                right,
                mmd_protocol,
                bandwidth=bandwidth,
                seed_offset=80_000 + offset * 1_000,
            ),
        }

    control_values = [
        float(comparisons["D0_vs_D0"]["mmd"]["mmd2"]),
        float(comparisons["D1_vs_D1"]["mmd"]["mmd2"]),
    ]
    control_median = float(np.median(control_values))
    for comparison in comparisons.values():
        observed = float(comparison["mmd"]["mmd2"])
        comparison["mmd"]["mmd_control_null_median"] = control_median
        comparison["mmd"]["mmd_excess_null"] = observed - control_median

    return {
        "metadata": dict(context.metadata),
        "projection": projection_summary(context),
        "mmd_protocol": asdict(mmd_protocol),
        "control_reference": {
            "mmd_control_values": control_values,
            "mmd_control_null_median": control_median,
            "mmd_control_reference": (
                "median_observed_D0_vs_D0_and_D1_vs_D1"
            ),
        },
        "comparisons": comparisons,
        "runtime": {
            "analysis_seconds": round(time.perf_counter() - started, 6),
            "completed_at_utc": utc_now(),
        },
    }


def analyze_projected_mmd_bundle(
    *,
    d1_path: Path,
    d0_path: Path,
    d0_control_a_path: Path,
    d0_control_b_path: Path,
    d1_control_a_path: Path,
    d1_control_b_path: Path,
    pair_shuffled_path: Path,
    projection_protocol: ProjectionProtocol,
    mmd_protocol: MMDProtocol,
    run_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    context = prepare_projected_context(
        d1_path=d1_path,
        d0_path=d0_path,
        d0_control_a_path=d0_control_a_path,
        d0_control_b_path=d0_control_b_path,
        d1_control_a_path=d1_control_a_path,
        d1_control_b_path=d1_control_b_path,
        pair_shuffled_path=pair_shuffled_path,
        projection_protocol=projection_protocol,
        run_metadata=run_metadata,
    )
    bundle = analyze_projected_mmd_context(context, mmd_protocol)
    bundle["projection_protocol"] = asdict(projection_protocol)
    return bundle


def flatten_mmd_bundle_for_csv(bundle: Mapping[str, Any]) -> list[Dict[str, Any]]:
    projection = bundle["projection"]
    rows: list[Dict[str, Any]] = []
    for control_type, comparison in bundle["comparisons"].items():
        metadata = comparison["metadata"]
        mmd = comparison["mmd"]
        rows.append(
            {
                "study_id": metadata.get("study_id"),
                "domain": metadata.get("domain"),
                "cipher": metadata.get("cipher"),
                "round": metadata.get("round"),
                "k": metadata.get("k"),
                "seed": metadata.get("seed"),
                "control_type": control_type,
                "n_samples_total": 2
                * int(projection["selected_count_per_dataset"]),
                "pca_components": projection["pca_components"],
                "explained_variance_sum": projection[
                    "explained_variance_sum"
                ],
                "projection_id": projection["projection_id"],
                "sample_selection_hash": projection[
                    "sample_selection_hash"
                ],
                "pair_shuffle_k1_identical": projection[
                    "pair_shuffle_k1_identical"
                ],
                "mmd2": mmd["mmd2"],
                "mmd_permutation_p": mmd["mmd_permutation_p"],
                "mmd_permutation_null_median": mmd[
                    "mmd_permutation_null_median"
                ],
                "mmd_permutation_null_q95": mmd[
                    "mmd_permutation_null_q95"
                ],
                "mmd_control_null_median": mmd[
                    "mmd_control_null_median"
                ],
                "mmd_excess_null": mmd["mmd_excess_null"],
                "mmd_kernel": mmd["mmd_kernel"],
                "mmd_bandwidth": mmd["mmd_bandwidth"],
                "mmd_bandwidth_scope": mmd["mmd_bandwidth_scope"],
                "mmd_samples_per_distribution": mmd[
                    "mmd_samples_per_distribution"
                ],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Projected geometry view
# ---------------------------------------------------------------------------


def aligned_binary_accuracy(y_true: UInt8Array, labels: IntArray) -> float:
    raw = float(accuracy_score(y_true, labels))
    return max(raw, 1.0 - raw)


def _fit_kmeans_labels(
    projected: FloatArray,
    protocol: GeometryProtocol,
    *,
    random_state: int,
) -> Tuple[KMeans, NDArray[np.int32]]:
    model = KMeans(
        n_clusters=protocol.kmeans_clusters,
        n_init=protocol.kmeans_n_init,
        max_iter=protocol.kmeans_max_iter,
        random_state=int(random_state),
    )
    labels = np.asarray(model.fit_predict(projected), dtype=np.int32)
    return model, labels


def fit_kmeans_metrics(
    left: FloatArray,
    right: FloatArray,
    protocol: GeometryProtocol,
    *,
    random_state: int,
) -> Dict[str, Any]:
    n = min(len(left), len(right))
    if n < 2:
        raise ValueError("At least two geometry samples per side are required")
    projected = np.concatenate([left[:n], right[:n]], axis=0)
    y = np.concatenate(
        [np.ones(n, dtype=np.uint8), np.zeros(n, dtype=np.uint8)]
    )
    model, labels = _fit_kmeans_labels(
        projected, protocol, random_state=random_state
    )
    silhouette = (
        float(silhouette_score(projected, labels, metric="euclidean"))
        if np.unique(labels).size > 1
        else math.nan
    )
    raw_accuracy = float(accuracy_score(y, labels))
    return {
        "labels": labels,
        "silhouette_kmeans": silhouette,
        "kmeans_inertia": float(model.inertia_),
        "kmeans_accuracy_raw": raw_accuracy,
        "kmeans_aligned_accuracy": aligned_binary_accuracy(y, labels),
        "kmeans_adjusted_rand": float(adjusted_rand_score(y, labels)),
        "kmeans_n_iter": int(model.n_iter_),
        "cluster_counts": {
            str(cluster): int(np.sum(labels == cluster))
            for cluster in range(protocol.kmeans_clusters)
        },
        "n_samples_per_side": int(n),
    }


def kmeans_stability_ari(
    left: FloatArray,
    right: FloatArray,
    protocol: GeometryProtocol,
) -> Dict[str, Any]:
    n = min(len(left), len(right))
    projected = np.concatenate([left[:n], right[:n]], axis=0)
    assignments = []
    for repetition in range(protocol.stability_repetitions):
        _, labels = _fit_kmeans_labels(
            projected,
            protocol,
            random_state=protocol.random_state + repetition,
        )
        assignments.append(labels)

    pairwise = [
        float(adjusted_rand_score(assignments[i], assignments[j]))
        for i, j in itertools.combinations(range(len(assignments)), 2)
    ]
    return {
        "kmeans_stability_ari": float(np.mean(pairwise)),
        "kmeans_stability_ari_median": float(np.median(pairwise)),
        "kmeans_stability_ari_min": float(np.min(pairwise)),
        "stability_pair_count": len(pairwise),
    }


def analyze_projected_geometry_context(
    context: ProjectedContext,
    geometry_protocol: GeometryProtocol,
) -> Dict[str, Any]:
    """Run K-Means geometry comparisons in one shared PCA representation."""

    geometry_protocol.validate()
    started = time.perf_counter()
    comparisons: Dict[str, Dict[str, Any]] = {}
    for control_type in COMPARISON_DATASETS:
        left, right = context.comparison(control_type)
        metrics = fit_kmeans_metrics(
            left,
            right,
            geometry_protocol,
            random_state=geometry_protocol.random_state,
        )
        stability = kmeans_stability_ari(left, right, geometry_protocol)
        comparisons[control_type] = {
            "metadata": {
                **context.metadata,
                "control_type": control_type,
            },
            "kmeans": {
                key: value for key, value in metrics.items() if key != "labels"
            },
            "stability": stability,
        }

    control_values = [
        float(comparisons["D0_vs_D0"]["kmeans"]["silhouette_kmeans"]),
        float(comparisons["D1_vs_D1"]["kmeans"]["silhouette_kmeans"]),
    ]
    control_median = float(np.median(control_values))
    for comparison in comparisons.values():
        silhouette = float(comparison["kmeans"]["silhouette_kmeans"])
        comparison["kmeans"]["silhouette_control_median"] = control_median
        # Backward-compatible alias retained in JSON/CSV.
        comparison["kmeans"]["silhouette_null_median"] = control_median
        comparison["kmeans"]["silhouette_excess_null"] = (
            silhouette - control_median
        )

    return {
        "metadata": dict(context.metadata),
        "projection": projection_summary(context),
        "geometry_protocol": asdict(geometry_protocol),
        "control_reference": {
            "silhouette_control_values": control_values,
            "silhouette_control_median": control_median,
            "silhouette_control_reference": (
                "median_observed_D0_vs_D0_and_D1_vs_D1"
            ),
            "note": (
                "K-Means initialization repetitions are used only for "
                "stability, not as an empirical null distribution."
            ),
        },
        "comparisons": comparisons,
        "runtime": {
            "analysis_seconds": round(time.perf_counter() - started, 6),
            "completed_at_utc": utc_now(),
        },
    }


def analyze_projected_geometry_bundle(
    *,
    d1_path: Path,
    d0_path: Path,
    d0_control_a_path: Path,
    d0_control_b_path: Path,
    d1_control_a_path: Path,
    d1_control_b_path: Path,
    pair_shuffled_path: Path,
    projection_protocol: ProjectionProtocol,
    geometry_protocol: GeometryProtocol,
    run_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    context = prepare_projected_context(
        d1_path=d1_path,
        d0_path=d0_path,
        d0_control_a_path=d0_control_a_path,
        d0_control_b_path=d0_control_b_path,
        d1_control_a_path=d1_control_a_path,
        d1_control_b_path=d1_control_b_path,
        pair_shuffled_path=pair_shuffled_path,
        projection_protocol=projection_protocol,
        run_metadata=run_metadata,
    )
    bundle = analyze_projected_geometry_context(context, geometry_protocol)
    bundle["projection_protocol"] = asdict(projection_protocol)
    return bundle


def flatten_geometry_bundle_for_csv(
    bundle: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    projection = bundle["projection"]
    rows: list[Dict[str, Any]] = []
    for control_type, comparison in bundle["comparisons"].items():
        metadata = comparison["metadata"]
        kmeans = comparison["kmeans"]
        stability = comparison["stability"]
        rows.append(
            {
                "study_id": metadata.get("study_id"),
                "domain": metadata.get("domain"),
                "cipher": metadata.get("cipher"),
                "round": metadata.get("round"),
                "k": metadata.get("k"),
                "seed": metadata.get("seed"),
                "control_type": control_type,
                "n_samples_total": 2
                * int(projection["selected_count_per_dataset"]),
                "pca_components": projection["pca_components"],
                "explained_variance_sum": projection[
                    "explained_variance_sum"
                ],
                "projection_id": projection["projection_id"],
                "sample_selection_hash": projection[
                    "sample_selection_hash"
                ],
                "pair_shuffle_k1_identical": projection[
                    "pair_shuffle_k1_identical"
                ],
                "silhouette_kmeans": kmeans["silhouette_kmeans"],
                "silhouette_control_median": kmeans[
                    "silhouette_control_median"
                ],
                "silhouette_null_median": kmeans[
                    "silhouette_null_median"
                ],
                "silhouette_excess_null": kmeans[
                    "silhouette_excess_null"
                ],
                "kmeans_inertia": kmeans["kmeans_inertia"],
                "kmeans_adjusted_rand": kmeans[
                    "kmeans_adjusted_rand"
                ],
                "kmeans_aligned_accuracy": kmeans[
                    "kmeans_aligned_accuracy"
                ],
                "kmeans_n_iter": kmeans["kmeans_n_iter"],
                "kmeans_stability_ari": stability[
                    "kmeans_stability_ari"
                ],
                "kmeans_stability_ari_median": stability[
                    "kmeans_stability_ari_median"
                ],
                "kmeans_stability_ari_min": stability[
                    "kmeans_stability_ari_min"
                ],
                "stability_pair_count": stability[
                    "stability_pair_count"
                ],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Optional backward-compatible combined API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmpiricalDetectabilityProtocol:
    """Compatibility protocol for older imports.

    New code should instantiate ``ProjectionProtocol``, ``MMDProtocol``, and
    ``GeometryProtocol`` separately.
    """

    max_samples_total: int = 100_000
    pca_components: int = 16
    pca_solver: str = "auto"
    pca_whiten: bool = False
    kmeans_clusters: int = 2
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300
    random_state: int = 42
    null_repetitions: int = 5
    stability_repetitions: int = 10
    mmd_kernel: str = "rbf"
    mmd_bandwidth_mode: str = "median"
    mmd_bandwidth: Optional[float] = None
    mmd_permutations: int = 200
    mmd_max_samples_per_distribution: int = 500

    def projection(self) -> ProjectionProtocol:
        return ProjectionProtocol(
            max_samples_total=self.max_samples_total,
            pca_components=self.pca_components,
            pca_solver=self.pca_solver,
            pca_whiten=self.pca_whiten,
            random_state=self.random_state,
        )

    def mmd(self) -> MMDProtocol:
        mode = (
            "median_main_reference"
            if self.mmd_bandwidth_mode == "median"
            else self.mmd_bandwidth_mode
        )
        return MMDProtocol(
            kernel=self.mmd_kernel,
            bandwidth_mode=mode,
            bandwidth=self.mmd_bandwidth,
            permutations=self.mmd_permutations,
            max_samples_per_distribution=self.mmd_max_samples_per_distribution,
            random_state=self.random_state,
        )

    def geometry(self) -> GeometryProtocol:
        return GeometryProtocol(
            kmeans_clusters=self.kmeans_clusters,
            kmeans_n_init=self.kmeans_n_init,
            kmeans_max_iter=self.kmeans_max_iter,
            stability_repetitions=self.stability_repetitions,
            random_state=self.random_state,
        )

    def validate(self) -> None:
        self.projection().validate()
        self.mmd().validate()
        self.geometry().validate()


LegacyKMeansProtocol = EmpiricalDetectabilityProtocol


def analyze_empirical_detectability_bundle(
    *,
    d1_path: Path,
    d0_path: Path,
    d0_control_a_path: Path,
    d0_control_b_path: Path,
    d1_control_a_path: Path,
    d1_control_b_path: Path,
    pair_shuffled_path: Path,
    protocol: EmpiricalDetectabilityProtocol,
    run_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compatibility wrapper returning both views from one shared context."""

    protocol.validate()
    context = prepare_projected_context(
        d1_path=d1_path,
        d0_path=d0_path,
        d0_control_a_path=d0_control_a_path,
        d0_control_b_path=d0_control_b_path,
        d1_control_a_path=d1_control_a_path,
        d1_control_b_path=d1_control_b_path,
        pair_shuffled_path=pair_shuffled_path,
        projection_protocol=protocol.projection(),
        run_metadata=run_metadata,
    )
    mmd_bundle = analyze_projected_mmd_context(context, protocol.mmd())
    geometry_bundle = analyze_projected_geometry_context(
        context, protocol.geometry()
    )
    comparisons: Dict[str, Dict[str, Any]] = {}
    for control_type in COMPARISON_DATASETS:
        comparisons[control_type] = {
            "metadata": mmd_bundle["comparisons"][control_type]["metadata"],
            "mmd": mmd_bundle["comparisons"][control_type]["mmd"],
            "kmeans": geometry_bundle["comparisons"][control_type]["kmeans"],
            "stability": geometry_bundle["comparisons"][control_type][
                "stability"
            ],
        }
    return {
        "metadata": dict(run_metadata or {}),
        "protocol": asdict(protocol),
        "projection": projection_summary(context),
        "mmd_control_reference": mmd_bundle["control_reference"],
        "geometry_control_reference": geometry_bundle["control_reference"],
        "comparisons": comparisons,
    }


def flatten_bundle_for_csv(bundle: Mapping[str, Any]) -> list[Dict[str, Any]]:
    """Compatibility flattener for the old combined runner."""

    projection = bundle["projection"]
    rows: list[Dict[str, Any]] = []
    for control_type, comparison in bundle["comparisons"].items():
        metadata = comparison["metadata"]
        mmd = comparison["mmd"]
        kmeans = comparison["kmeans"]
        stability = comparison["stability"]
        rows.append(
            {
                "study_id": metadata.get("study_id"),
                "domain": metadata.get("domain"),
                "cipher": metadata.get("cipher"),
                "round": metadata.get("round"),
                "k": metadata.get("k"),
                "seed": metadata.get("seed"),
                "control_type": control_type,
                "n_samples_total": 2
                * int(projection["selected_count_per_dataset"]),
                "pca_components": projection["pca_components"],
                "explained_variance_sum": projection[
                    "explained_variance_sum"
                ],
                "silhouette_kmeans": kmeans["silhouette_kmeans"],
                "silhouette_null_median": kmeans[
                    "silhouette_null_median"
                ],
                "silhouette_excess_null": kmeans[
                    "silhouette_excess_null"
                ],
                "kmeans_inertia": kmeans["kmeans_inertia"],
                "kmeans_adjusted_rand": kmeans[
                    "kmeans_adjusted_rand"
                ],
                "kmeans_aligned_accuracy": kmeans[
                    "kmeans_aligned_accuracy"
                ],
                "kmeans_stability_ari": stability[
                    "kmeans_stability_ari"
                ],
                "mmd2": mmd["mmd2"],
                "mmd_permutation_p": mmd["mmd_permutation_p"],
                "mmd_permutation_null_median": mmd[
                    "mmd_permutation_null_median"
                ],
                "mmd_permutation_null_q95": mmd[
                    "mmd_permutation_null_q95"
                ],
                "mmd_control_null_median": mmd[
                    "mmd_control_null_median"
                ],
                "mmd_excess_null": mmd["mmd_excess_null"],
                "mmd_bandwidth": mmd["mmd_bandwidth"],
                "mmd_samples_per_distribution": mmd[
                    "mmd_samples_per_distribution"
                ],
            }
        )
    return rows
