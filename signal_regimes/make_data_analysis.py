"""Data generation for RKMP-SDF statistical analysis.

This module is the analysis counterpart of ``make_data_train.py``.  It keeps
the same ciphertext representation

    [delta_C || C || C_star]

but adds controls required for pre-analysis configuration lock-in and for the
formal two-distribution problem

    D1^(r,k)(delta_P, delta_K)  versus  D0^(r,k)(delta_P, delta_K).

Key properties
--------------
* Deterministic, batch-order-independent generation through ``seed``.
* Explicit class modes: ``structured`` (D1), ``null`` (D0), or ``balanced``.
* Shared-key multi-pair groups by default.
* Explicit MSB/LSB bit-index convention for single-bit delta_K lock-in.
* Flat output for the existing PCA/neural code, or grouped output with shape
  ``(N, k, 3 * block_bits)`` for pair-level and group-level analysis.
* Optional full metadata for protocol validation.
* No neural training dependency is required.  TensorFlow's ``Sequence`` is
  used when available only for interface compatibility.

The null class preserves the related-key relation K_star = K xor delta_K, but
breaks the plaintext-difference relation by sampling P_star independently.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Literal, Optional, Tuple, Union

import numpy as np

try:
    from tensorflow.keras.utils import Sequence
except Exception:  # Analysis generation should not require TensorFlow.
    class Sequence:  # type: ignore[override]
        pass

try:
    import cupy as cp
except Exception:
    cp = None


ArrayLike = Union[int, np.integer, np.ndarray]
ClassMode = Literal["balanced", "structured", "null"]
ControlType = Literal["D1_vs_D0", "D0_vs_D0", "D1_vs_D1", "pair_shuffled"]
OutputLayout = Literal["flat", "grouped"]
KeyMode = Literal["shared", "independent"]
BitIndexing = Literal["msb0", "lsb0"]

DEFAULT_USE_GPU = cp is not None


@dataclass(frozen=True)
class LockedAnalysisConfig:
    """Serializable configuration for one locked analysis run."""

    cipher: str
    rounds: int
    pairs: int
    samples_per_class: int
    plain_bits: int
    key_bits: int
    delta_p_int: int
    delta_p_hex: str
    delta_k_int: int
    delta_k_hex: str
    delta_k_bit: Optional[int]
    delta_k_indexing: Optional[str]
    seed: int
    output_layout: str
    key_mode: str
    control_type: str = "D1_vs_D0"
    control_replica: int = 0
    source_round: Optional[int] = None
    source_k: Optional[int] = None
    source_seed: Optional[int] = None
    representation: str = "delta_C||C||C_star"


def _to_numpy(x: Any) -> np.ndarray:
    if cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def _int_to_bitarray(
    value: int,
    nbits: int,
    lib: Any = np,
) -> Any:
    """Convert an integer to an MSB-first uint8 bit vector of length ``nbits``."""

    if nbits <= 0:
        raise ValueError(f"nbits must be positive, got {nbits}")

    value = int(value)
    if value < 0 or value >= (1 << nbits):
        raise ValueError(
            f"value={value} does not fit in an unsigned {nbits}-bit vector"
        )

    shifts = np.arange(nbits - 1, -1, -1, dtype=np.int64)
    bits = ((value >> shifts) & 1).astype(np.uint8)
    return lib.asarray(bits) if (cp is not None and lib is cp) else bits


def _bitarray_to_int(value: ArrayLike, nbits: int) -> int:
    """Convert an MSB-first bit vector to an integer."""

    arr = _to_numpy(value).astype(np.uint8, copy=False).reshape(-1)
    if arr.size != nbits:
        raise ValueError(f"Expected {nbits} bits, received {arr.size}")
    if np.any((arr != 0) & (arr != 1)):
        raise ValueError("Bit vectors must contain only 0 and 1")

    result = 0
    for bit in arr:
        result = (result << 1) | int(bit)
    return result


def _normalize_bit_vector(
    value: ArrayLike,
    nbits: int,
    *,
    name: str,
    lib: Any,
) -> Any:
    """Normalize an integer or array into a one-dimensional MSB-first vector."""

    if isinstance(value, (int, np.integer)):
        return _int_to_bitarray(int(value), nbits, lib)

    arr = _to_numpy(value).astype(np.uint8, copy=False).reshape(-1)
    if arr.size != nbits:
        raise ValueError(f"{name} must contain {nbits} bits, got {arr.size}")
    if np.any((arr != 0) & (arr != 1)):
        raise ValueError(f"{name} must contain only 0 and 1")
    return lib.asarray(arr) if (cp is not None and lib is cp) else arr


def single_bit_difference(
    bit_index: int,
    nbits: int,
    *,
    indexing: BitIndexing = "msb0",
) -> Tuple[int, np.ndarray]:
    """Build a locked single-bit difference.

    Parameters
    ----------
    bit_index:
        Zero-based bit index.
    nbits:
        Vector width.
    indexing:
        ``"msb0"`` means array index 0 is the most-significant bit.
        ``"lsb0"`` means bit index 0 is the least-significant bit.

    Returns
    -------
    integer_value, msb_first_vector
    """

    if not 0 <= int(bit_index) < nbits:
        raise ValueError(f"bit_index must be in [0, {nbits - 1}]")

    bit_index = int(bit_index)
    if indexing == "msb0":
        integer_value = 1 << (nbits - 1 - bit_index)
        array_index = bit_index
    elif indexing == "lsb0":
        integer_value = 1 << bit_index
        array_index = nbits - 1 - bit_index
    else:
        raise ValueError("indexing must be 'msb0' or 'lsb0'")

    vector = np.zeros(nbits, dtype=np.uint8)
    vector[array_index] = 1

    reconstructed = _bitarray_to_int(vector, nbits)
    if reconstructed != integer_value or int(vector.sum()) != 1:
        raise AssertionError("Internal single-bit conversion error")

    return integer_value, vector


def _safe_encrypt(enc_fn: Any, plaintexts: Any, keys: Any, rounds: int) -> Any:
    """Call the cipher and attempt one CPU/GPU backend fallback.

    When both attempts fail, the original failure is preserved as the cause so
    cipher-shape and cipher-logic errors are not silently hidden.
    """

    try:
        return enc_fn(plaintexts, keys, rounds)
    except Exception as first_error:
        if cp is None:
            raise

        try:
            if isinstance(plaintexts, cp.ndarray):
                return enc_fn(
                    cp.asnumpy(plaintexts),
                    cp.asnumpy(keys),
                    rounds,
                )
            return enc_fn(
                cp.asarray(plaintexts),
                cp.asarray(keys),
                rounds,
            )
        except Exception as fallback_error:
            raise RuntimeError(
                "Cipher encryption failed on both the requested and fallback "
                "array backends."
            ) from first_error


class NDCMultiPairAnalysisGenerator(Sequence):
    """Deterministic generator for D1/D0 related-key multi-pair observations."""

    def __init__(
        self,
        encryption_function: Any,
        plain_bits: int,
        key_bits: int,
        nr: int,
        *,
        delta_state: ArrayLike = 0,
        delta_key: ArrayLike = 0,
        n_samples: int = 100_000,
        batch_size: int = 100_000,
        pairs: int = 1,
        seed: int = 0,
        start_idx: int = 0,
        class_mode: ClassMode = "balanced",
        key_mode: KeyMode = "shared",
        output_layout: OutputLayout = "flat",
        use_gpu: Optional[bool] = None,
        encrypt_backend: Literal["numpy", "cupy", "auto"] = "numpy",
        to_float32: bool = False,
        return_metadata: bool = False,
        metadata_full: bool = False,
        control_type: ControlType = "D1_vs_D0",
        control_replica: int = 0,
        source_round: Optional[int] = None,
        source_k: Optional[int] = None,
        source_seed: Optional[int] = None,
    ) -> None:
        if plain_bits <= 0 or key_bits <= 0:
            raise ValueError("plain_bits and key_bits must be positive")
        if nr <= 0:
            raise ValueError("nr must be positive")
        if n_samples <= 0 or batch_size <= 0 or pairs <= 0:
            raise ValueError("n_samples, batch_size, and pairs must be positive")
        if class_mode not in {"balanced", "structured", "null"}:
            raise ValueError(
                "class_mode must be 'balanced', 'structured', or 'null'"
            )
        if key_mode not in {"shared", "independent"}:
            raise ValueError("key_mode must be 'shared' or 'independent'")
        if output_layout not in {"flat", "grouped"}:
            raise ValueError("output_layout must be 'flat' or 'grouped'")
        if encrypt_backend not in {"numpy", "cupy", "auto"}:
            raise ValueError("encrypt_backend must be numpy, cupy, or auto")
        if control_type not in {"D1_vs_D0", "D0_vs_D0", "D1_vs_D1", "pair_shuffled"}:
            raise ValueError("Unsupported control_type")
        if int(control_replica) < 0:
            raise ValueError("control_replica must be non-negative")

        self.encryption_function = encryption_function
        self.plain_bits = int(plain_bits)
        self.key_bits = int(key_bits)
        self.nr = int(nr)
        self.delta_state = delta_state
        self.delta_key = delta_key
        self.n = int(n_samples)
        self.batch_size = int(batch_size)
        self.pairs = int(pairs)
        self.seed = int(seed)
        self.start_idx = int(start_idx)
        self.class_mode = class_mode
        self.key_mode = key_mode
        self.output_layout = output_layout
        self.to_float32 = bool(to_float32)
        self.return_metadata = bool(return_metadata)
        self.metadata_full = bool(metadata_full)
        self.control_type = str(control_type)
        self.control_replica = int(control_replica)
        self.source_round = int(source_round) if source_round is not None else self.nr
        self.source_k = int(source_k) if source_k is not None else self.pairs
        self.source_seed = int(source_seed) if source_seed is not None else self.seed

        requested_gpu = DEFAULT_USE_GPU if use_gpu is None else bool(use_gpu)
        self.use_gpu = requested_gpu and cp is not None

        if encrypt_backend == "auto":
            self.encrypt_use_gpu = self.use_gpu
        elif encrypt_backend == "cupy":
            self.encrypt_use_gpu = cp is not None
        else:
            self.encrypt_use_gpu = False

        self.steps = math.ceil(self.n / self.batch_size)
        self.pair_feature_dim = 3 * self.plain_bits
        self.input_dim = self.pairs * self.pair_feature_dim

        # Validate deltas eagerly using NumPy.
        self.delta_state_int = _bitarray_to_int(
            _normalize_bit_vector(
                self.delta_state,
                self.plain_bits,
                name="delta_state",
                lib=np,
            ),
            self.plain_bits,
        )
        self.delta_key_int = _bitarray_to_int(
            _normalize_bit_vector(
                self.delta_key,
                self.key_bits,
                name="delta_key",
                lib=np,
            ),
            self.key_bits,
        )

    def __len__(self) -> int:
        return self.steps

    def _rng_for_batch(self, idx: int) -> np.random.Generator:
        if idx < 0 or idx >= self.steps:
            raise IndexError(idx)

        # Stateless per-batch RNG: requesting batches in a different order yields
        # the same data for each batch.
        sequence = np.random.SeedSequence(
            [self.seed, self.start_idx, int(idx), self.nr, self.pairs]
        )
        return np.random.default_rng(sequence)

    def _make_labels(
        self,
        curr_n: int,
        global_start: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if self.class_mode == "structured":
            return np.ones(curr_n, dtype=np.uint8)
        if self.class_mode == "null":
            return np.zeros(curr_n, dtype=np.uint8)

        # Alternating global labels provide exact balance for even N.  Shuffle
        # within each batch so labels are not ordered.
        global_ids = np.arange(global_start, global_start + curr_n)
        labels = (global_ids % 2 == 0).astype(np.uint8)
        rng.shuffle(labels)
        return labels

    def __getitem__(self, idx: int):
        curr_n = min(self.batch_size, self.n - idx * self.batch_size)
        if curr_n <= 0:
            raise IndexError(idx)

        rng = self._rng_for_batch(idx)
        global_start = self.start_idx + idx * self.batch_size
        sample_ids = np.arange(
            global_start,
            global_start + curr_n,
            dtype=np.int64,
        )

        labels_np = self._make_labels(curr_n, global_start, rng)

        delta_state_np = _normalize_bit_vector(
            self.delta_state,
            self.plain_bits,
            name="delta_state",
            lib=np,
        )
        delta_key_np = _normalize_bit_vector(
            self.delta_key,
            self.key_bits,
            name="delta_key",
            lib=np,
        )

        if self.key_mode == "shared":
            key0_grouped_np = rng.integers(
                0,
                2,
                size=(curr_n, 1, self.key_bits),
                dtype=np.uint8,
            )
            key0_grouped_np = np.repeat(key0_grouped_np, self.pairs, axis=1)
            key_group_ids = np.repeat(sample_ids[:, None], self.pairs, axis=1)
        else:
            key0_grouped_np = rng.integers(
                0,
                2,
                size=(curr_n, self.pairs, self.key_bits),
                dtype=np.uint8,
            )
            key_group_ids = (
                sample_ids[:, None] * self.pairs
                + np.arange(self.pairs, dtype=np.int64)[None, :]
            )

        key1_grouped_np = key0_grouped_np ^ delta_key_np.reshape(1, 1, -1)

        plaintext_grouped_np = rng.integers(
            0,
            2,
            size=(curr_n, self.pairs, self.plain_bits),
            dtype=np.uint8,
        )
        plaintext_star_grouped_np = np.empty_like(plaintext_grouped_np)

        structured_mask = labels_np == 1
        if np.any(structured_mask):
            plaintext_star_grouped_np[structured_mask] = (
                plaintext_grouped_np[structured_mask]
                ^ delta_state_np.reshape(1, 1, -1)
            )

        null_mask = ~structured_mask
        if np.any(null_mask):
            plaintext_star_grouped_np[null_mask] = rng.integers(
                0,
                2,
                size=(
                    int(null_mask.sum()),
                    self.pairs,
                    self.plain_bits,
                ),
                dtype=np.uint8,
            )

        plaintext_np = plaintext_grouped_np.reshape(
            curr_n * self.pairs,
            self.plain_bits,
        )
        plaintext_star_np = plaintext_star_grouped_np.reshape(
            curr_n * self.pairs,
            self.plain_bits,
        )
        key0_np = key0_grouped_np.reshape(
            curr_n * self.pairs,
            self.key_bits,
        )
        key1_np = key1_grouped_np.reshape(
            curr_n * self.pairs,
            self.key_bits,
        )

        if self.encrypt_use_gpu and cp is not None:
            plaintext_in = cp.asarray(plaintext_np)
            plaintext_star_in = cp.asarray(plaintext_star_np)
            key0_in = cp.asarray(key0_np)
            key1_in = cp.asarray(key1_np)
        else:
            plaintext_in = plaintext_np
            plaintext_star_in = plaintext_star_np
            key0_in = key0_np
            key1_in = key1_np

        ciphertext = _safe_encrypt(
            self.encryption_function,
            plaintext_in,
            key0_in,
            self.nr,
        )
        ciphertext_star = _safe_encrypt(
            self.encryption_function,
            plaintext_star_in,
            key1_in,
            self.nr,
        )

        ciphertext_np = _to_numpy(ciphertext).astype(np.uint8, copy=False)
        ciphertext_star_np = _to_numpy(ciphertext_star).astype(
            np.uint8,
            copy=False,
        )

        expected_shape = (curr_n * self.pairs, self.plain_bits)
        if ciphertext_np.shape != expected_shape:
            raise ValueError(
                f"Cipher returned C shape {ciphertext_np.shape}; "
                f"expected {expected_shape}"
            )
        if ciphertext_star_np.shape != expected_shape:
            raise ValueError(
                f"Cipher returned C_star shape {ciphertext_star_np.shape}; "
                f"expected {expected_shape}"
            )

        delta_ciphertext_np = ciphertext_np ^ ciphertext_star_np
        pair_features_np = np.concatenate(
            [delta_ciphertext_np, ciphertext_np, ciphertext_star_np],
            axis=1,
        ).reshape(curr_n, self.pairs, self.pair_feature_dim)

        if self.output_layout == "flat":
            features_np = pair_features_np.reshape(curr_n, self.input_dim)
        else:
            features_np = pair_features_np

        if self.to_float32:
            features_np = features_np.astype(np.float32, copy=False)
        else:
            features_np = features_np.astype(np.uint8, copy=False)

        labels_np = labels_np.astype(np.uint8, copy=False)

        if not self.return_metadata:
            return features_np, labels_np

        metadata: Dict[str, Any] = {
            "sample_ids": sample_ids,
            "key_group_ids": key_group_ids,
            "pair_indices": np.broadcast_to(
                np.arange(self.pairs, dtype=np.int64),
                (curr_n, self.pairs),
            ).copy(),
            "class_mode": self.class_mode,
            "key_mode": self.key_mode,
            "rounds": self.nr,
            "pairs": self.pairs,
            "delta_state_int": self.delta_state_int,
            "delta_key_int": self.delta_key_int,
            "control_type": self.control_type,
            "control_replica": self.control_replica,
            "source_round": self.source_round,
            "source_k": self.source_k,
            "source_seed": self.source_seed,
        }

        if self.metadata_full:
            metadata.update(
                {
                    "plaintexts": plaintext_grouped_np,
                    "plaintexts_star": plaintext_star_grouped_np,
                    "keys": key0_grouped_np,
                    "keys_star": key1_grouped_np,
                    "ciphertexts": ciphertext_np.reshape(
                        curr_n,
                        self.pairs,
                        self.plain_bits,
                    ),
                    "ciphertexts_star": ciphertext_star_np.reshape(
                        curr_n,
                        self.pairs,
                        self.plain_bits,
                    ),
                }
            )

        return features_np, labels_np, metadata


# Compact alias for analysis scripts.
NDCAnalysisGenerator = NDCMultiPairAnalysisGenerator


CONTROL_SEED_OFFSETS = {
    "main_D1": 0,
    "main_D0": 1_000_003,
    "D0_control_a": 2_000_003,
    "D0_control_b": 3_000_003,
    "D1_control_a": 4_000_003,
    "D1_control_b": 5_000_003,
    "pair_shuffled": 6_000_003,
}


def derive_control_seed(source_seed: int, role: str, replica: int = 0) -> int:
    """Derive a deterministic seed namespace for a generated control."""

    if role not in CONTROL_SEED_OFFSETS:
        raise ValueError(f"Unknown control seed role: {role}")
    if replica < 0:
        raise ValueError("replica must be non-negative")
    return int(source_seed) + CONTROL_SEED_OFFSETS[role] + int(replica) * 10_000_019


def build_control_generators(
    control_type: Literal["D0_vs_D0", "D1_vs_D1"],
    *,
    common_kwargs: Dict[str, Any],
    source_seed: int,
) -> Tuple[NDCMultiPairAnalysisGenerator, NDCMultiPairAnalysisGenerator]:
    """Construct two independent same-class generators for nuisance controls."""

    if control_type == "D0_vs_D0":
        class_mode = "null"
        roles = ("D0_control_a", "D0_control_b")
    elif control_type == "D1_vs_D1":
        class_mode = "structured"
        roles = ("D1_control_a", "D1_control_b")
    else:
        raise ValueError("control_type must be D0_vs_D0 or D1_vs_D1")

    generators = []
    for replica, role in enumerate(roles):
        generators.append(
            NDCMultiPairAnalysisGenerator(
                **common_kwargs,
                seed=derive_control_seed(source_seed, role),
                class_mode=class_mode,
                control_type=control_type,
                control_replica=replica,
                source_round=int(common_kwargs["nr"]),
                source_k=int(common_kwargs["pairs"]),
                source_seed=int(source_seed),
            )
        )
    return generators[0], generators[1]


def pair_shuffle_groups(
    features: np.ndarray,
    *,
    pairs: int,
    pair_feature_dim: int,
    seed: int,
    output_layout: OutputLayout,
    source_round: Optional[int] = None,
    source_k: Optional[int] = None,
    source_seed: Optional[int] = None,
    control_replica: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Destroy shared-key grouping while preserving the exact pair multiset.

    The input is interpreted as N groups containing ``pairs`` pair features.
    All N*k pair observations are globally permuted and regrouped into N new
    groups. Thus every marginal pair observation is retained exactly once, but
    the original within-group shared-key association is broken for k > 1.
    """

    array = np.asarray(features)
    if array.ndim not in {2, 3}:
        raise ValueError("features must be flat or grouped")
    n = int(array.shape[0])
    expected_flat = int(pairs) * int(pair_feature_dim)

    if array.ndim == 3:
        if array.shape[1:] != (pairs, pair_feature_dim):
            raise ValueError(
                f"Expected grouped shape (N,{pairs},{pair_feature_dim}), got {array.shape}"
            )
        grouped = array.reshape(n, pairs, pair_feature_dim)
    else:
        if array.shape[1] != expected_flat:
            raise ValueError(
                f"Expected flat dimension {expected_flat}, got {array.shape[1]}"
            )
        grouped = array.reshape(n, pairs, pair_feature_dim)

    flat_pairs = grouped.reshape(n * pairs, pair_feature_dim)
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(n * pairs)
    shuffled_grouped = flat_pairs[permutation].reshape(
        n, pairs, pair_feature_dim
    )

    if output_layout == "grouped":
        output = shuffled_grouped
    elif output_layout == "flat":
        output = shuffled_grouped.reshape(n, expected_flat)
    else:
        raise ValueError("output_layout must be flat or grouped")

    metadata = {
        "control_type": "pair_shuffled",
        "control_replica": int(control_replica),
        "source_round": int(source_round) if source_round is not None else None,
        "source_k": int(source_k) if source_k is not None else int(pairs),
        "source_seed": int(source_seed) if source_seed is not None else None,
        "shuffle_seed": int(seed),
        "n_groups": n,
        "n_pairs_total": int(n * pairs),
        "marginal_bit_sum_preserved": bool(
            np.array_equal(
                flat_pairs.sum(axis=0, dtype=np.int64),
                shuffled_grouped.reshape(-1, pair_feature_dim).sum(
                    axis=0, dtype=np.int64
                ),
            )
        ),
        "grouping_effective": bool(pairs > 1),
    }
    return output, metadata


def materialize_analysis_data(
    generator: NDCMultiPairAnalysisGenerator,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]]]:
    """Materialize all batches instead of accidentally analysing only batch 0."""

    feature_batches = []
    label_batches = []
    metadata_batches = []

    for idx in range(len(generator)):
        batch = generator[idx]
        if len(batch) == 2:
            features, labels = batch
            metadata = None
        else:
            features, labels, metadata = batch

        feature_batches.append(np.asarray(features))
        label_batches.append(np.asarray(labels))
        if metadata is not None:
            metadata_batches.append(metadata)

    features_all = np.concatenate(feature_batches, axis=0)
    labels_all = np.concatenate(label_batches, axis=0)

    if not metadata_batches:
        return features_all, labels_all, None

    # Concatenate array-valued metadata; preserve scalar protocol fields.
    metadata_all: Dict[str, Any] = {}
    keys = metadata_batches[0].keys()
    for key in keys:
        values = [batch[key] for batch in metadata_batches]
        if isinstance(values[0], np.ndarray):
            metadata_all[key] = np.concatenate(values, axis=0)
        else:
            metadata_all[key] = values[0]

    return features_all, labels_all, metadata_all


def validate_protocol_batch(
    labels: np.ndarray,
    metadata: Dict[str, Any],
    *,
    plain_bits: int,
    key_bits: int,
) -> Dict[str, bool]:
    """Validate D1/D0 and related-key relations for a full-metadata batch."""

    required = {
        "plaintexts",
        "plaintexts_star",
        "keys",
        "keys_star",
        "key_mode",
        "delta_state_int",
        "delta_key_int",
    }
    missing = required.difference(metadata)
    if missing:
        raise ValueError(
            "Full metadata is required for validation; missing "
            + ", ".join(sorted(missing))
        )

    labels = np.asarray(labels).reshape(-1)
    plaintexts = np.asarray(metadata["plaintexts"], dtype=np.uint8)
    plaintexts_star = np.asarray(
        metadata["plaintexts_star"],
        dtype=np.uint8,
    )
    keys = np.asarray(metadata["keys"], dtype=np.uint8)
    keys_star = np.asarray(metadata["keys_star"], dtype=np.uint8)

    delta_state = _int_to_bitarray(
        int(metadata["delta_state_int"]),
        plain_bits,
        np,
    )
    delta_key = _int_to_bitarray(
        int(metadata["delta_key_int"]),
        key_bits,
        np,
    )

    structured_mask = labels == 1
    structured_relation = True
    if np.any(structured_mask):
        structured_relation = bool(
            np.all(
                plaintexts_star[structured_mask]
                == (
                    plaintexts[structured_mask]
                    ^ delta_state.reshape(1, 1, -1)
                )
            )
        )

    related_key_relation = bool(
        np.all(keys_star == (keys ^ delta_key.reshape(1, 1, -1)))
    )

    shared_key_relation = True
    if metadata["key_mode"] == "shared" and keys.shape[1] > 1:
        shared_key_relation = bool(
            np.all(keys == keys[:, :1, :])
            and np.all(keys_star == keys_star[:, :1, :])
        )

    output = {
        "structured_plaintext_relation": structured_relation,
        "related_key_relation": related_key_relation,
        "shared_key_within_group": shared_key_relation,
        "labels_binary": bool(np.all(np.isin(labels, [0, 1]))),
        "delta_key_width_valid": delta_key.size == key_bits,
        "delta_state_width_valid": delta_state.size == plain_bits,
    }
    output["all_passed"] = all(output.values())
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate locked D1/D0 datasets for related-key multi-pair "
            "statistical analysis."
        )
    )
    parser.add_argument(
        "--cipher",
        default="present80",
        help="Module name under cipher/, e.g. present80",
    )
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=50_000,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
    )
    parser.add_argument(
        "--delta-p",
        required=True,
        help="Plaintext difference, e.g. 0x80",
    )

    delta_group = parser.add_mutually_exclusive_group(required=True)
    delta_group.add_argument(
        "--delta-k",
        help="Related-key difference as an integer/hex value",
    )
    delta_group.add_argument(
        "--delta-k-bit",
        type=int,
        help="Single-bit related-key difference index",
    )

    parser.add_argument(
        "--bit-indexing",
        choices=["msb0", "lsb0"],
        default="msb0",
        help="Convention used by --delta-k-bit",
    )
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument(
        "--output-layout",
        choices=["flat", "grouped"],
        default="flat",
    )
    parser.add_argument(
        "--key-mode",
        choices=["shared", "independent"],
        default="shared",
    )
    parser.add_argument(
        "--encrypt-backend",
        choices=["numpy", "cupy", "auto"],
        default="numpy",
    )
    parser.add_argument(
        "--float32",
        action="store_true",
        help="Store feature arrays as float32 instead of uint8",
    )
    parser.add_argument(
        "--out",
        default="analysis_data",
        help="Output root directory",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run full-metadata protocol validation before saving",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    if args.pairs <= 0:
        raise ValueError("--pairs must be positive")
    if args.samples_per_class <= 0:
        raise ValueError("--samples-per-class must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    cipher = importlib.import_module(f"cipher.{args.cipher}")
    encrypt = cipher.encrypt
    plain_bits = int(cipher.plain_bits)
    key_bits = int(cipher.key_bits)

    delta_p_int = int(args.delta_p, 0)
    if delta_p_int < 0 or delta_p_int >= (1 << plain_bits):
        raise ValueError(
            f"delta_p does not fit in {plain_bits} bits"
        )

    delta_k_bit: Optional[int]
    delta_k_indexing: Optional[str]

    if args.delta_k_bit is not None:
        delta_k_int, delta_k_vector = single_bit_difference(
            args.delta_k_bit,
            key_bits,
            indexing=args.bit_indexing,
        )
        delta_k: ArrayLike = delta_k_vector
        delta_k_bit = int(args.delta_k_bit)
        delta_k_indexing = args.bit_indexing
    else:
        delta_k_int = int(args.delta_k, 0)
        if delta_k_int < 0 or delta_k_int >= (1 << key_bits):
            raise ValueError(
                f"delta_k does not fit in {key_bits} bits"
            )
        delta_k = delta_k_int
        delta_k_bit = None
        delta_k_indexing = None

    common = dict(
        encryption_function=encrypt,
        plain_bits=plain_bits,
        key_bits=key_bits,
        nr=args.rounds,
        delta_state=delta_p_int,
        delta_key=delta_k,
        n_samples=args.samples_per_class,
        batch_size=args.batch_size,
        pairs=args.pairs,
        key_mode=args.key_mode,
        output_layout=args.output_layout,
        encrypt_backend=args.encrypt_backend,
        to_float32=args.float32,
    )

    # Different seed namespaces prevent accidental D1/D0 sample reuse.
    structured_generator = NDCMultiPairAnalysisGenerator(
        **common,
        seed=args.seed,
        class_mode="structured",
        return_metadata=args.validate,
        metadata_full=args.validate,
    )
    null_generator = NDCMultiPairAnalysisGenerator(
        **common,
        seed=args.seed + 1_000_003,
        class_mode="null",
        return_metadata=args.validate,
        metadata_full=args.validate,
    )

    x1, y1, metadata1 = materialize_analysis_data(
        structured_generator
    )
    x0, y0, metadata0 = materialize_analysis_data(null_generator)

    validation_report: Dict[str, Any] = {}
    if args.validate:
        assert metadata1 is not None and metadata0 is not None
        validation_report["D1"] = validate_protocol_batch(
            y1,
            metadata1,
            plain_bits=plain_bits,
            key_bits=key_bits,
        )
        validation_report["D0"] = validate_protocol_batch(
            y0,
            metadata0,
            plain_bits=plain_bits,
            key_bits=key_bits,
        )
        validation_report["all_passed"] = bool(
            validation_report["D1"]["all_passed"]
            and validation_report["D0"]["all_passed"]
        )
        if not validation_report["all_passed"]:
            raise AssertionError(
                f"Protocol validation failed: {validation_report}"
            )

    config = LockedAnalysisConfig(
        cipher=args.cipher,
        rounds=args.rounds,
        pairs=args.pairs,
        samples_per_class=args.samples_per_class,
        plain_bits=plain_bits,
        key_bits=key_bits,
        delta_p_int=delta_p_int,
        delta_p_hex=f"0x{delta_p_int:0{(plain_bits + 3) // 4}x}",
        delta_k_int=delta_k_int,
        delta_k_hex=f"0x{delta_k_int:0{(key_bits + 3) // 4}x}",
        delta_k_bit=delta_k_bit,
        delta_k_indexing=delta_k_indexing,
        seed=args.seed,
        output_layout=args.output_layout,
        key_mode=args.key_mode,
        control_type="D1_vs_D0",
        control_replica=0,
        source_round=args.rounds,
        source_k=args.pairs,
        source_seed=args.seed,
    )

    run_name = (
        f"{args.cipher}_r{args.rounds}_k{args.pairs}"
        f"_dp{delta_p_int:x}_dk{delta_k_int:x}_seed{args.seed}"
    )
    out_dir = Path(args.out) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "D1_structured.npy", x1)
    np.save(out_dir / "D0_null.npy", x0)
    np.save(out_dir / "D1_labels.npy", y1)
    np.save(out_dir / "D0_labels.npy", y0)

    with (out_dir / "locked_config.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(asdict(config), handle, indent=2)

    if validation_report:
        with (out_dir / "protocol_validation.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(validation_report, handle, indent=2)

    print("Analysis datasets generated successfully.")
    print(f"D1 shape: {x1.shape}, dtype={x1.dtype}")
    print(f"D0 shape: {x0.shape}, dtype={x0.dtype}")
    print(f"Locked configuration: {out_dir / 'locked_config.json'}")
    if validation_report:
        print(
            "Protocol validation: "
            f"{out_dir / 'protocol_validation.json'}"
        )
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()

    # value_msb0, bits_msb0 = single_bit_difference(
    #     bit_index=56,
    #     nbits=80,
    #     indexing="msb0",
    # )

    # value_lsb0, bits_lsb0 = single_bit_difference(
    #     bit_index=56,
    #     nbits=80,
    #     indexing="lsb0",
    # )

    # print("msb0:", hex(value_msb0))
    # print("lsb0:", hex(value_lsb0))
    # print("legacy:", hex(1 << 56))

# python make_data_analysis.py \
#   --cipher present80 \
#   --rounds 7 \
#   --pairs 1 \
#   --samples-per-class 50000 \
#   --batch-size 10000 \
#   --delta-p 0x80 \
#   --delta-k-bit 56 \
#   --bit-indexing msb0 \
#   --seed 201 \
#   --validate \
#   --out analysis_data

# python make_data_analysis.py \
#   --cipher present80 \
#   --rounds 7 \
#   --pairs 8 \
#   --samples-per-class 50000 \
#   --delta-p 0x80 \
#   --delta-k-bit 56 \
#   --bit-indexing msb0 \
#   --output-layout grouped \
#   --seed 201 \
#   --validate

# python make_data_analysis.py \
#   --cipher present80 \
#   --rounds 7 \
#   --pairs 8 \
#   --samples-per-class 1000 \
#   --batch-size 500 \
#   --delta-p 0x80 \
#   --delta-k 0x100000000000000 \
#   --seed 201 \
#   --output-layout grouped \
#   --validate \
#   --out analysis_data_smoke