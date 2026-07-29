"""Common utilities for downstream partial-key ranking.

This module contains the cipher-independent parts of the experiment:
model construction/loading, fixed-key attack-data generation, candidate
scoring, ranking metrics, and result summaries.

Cipher-specific inverse-round logic belongs in a CandidateAdapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence as TypingSequence

import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None

from make_data_train import NDCMultiPairGenerator, _int_to_bitarray, _safe_encrypt

DEFAULT_LOGIT_EPS = 1e-7


# ---------------------------------------------------------------------------
# Neural distinguisher construction / loading
# ---------------------------------------------------------------------------

def build_neural_distinguisher(
    architecture: str,
    *,
    plain_bits: int,
    pairs: int,
    compile_model: bool = False,
    learning_rate: float = 1e-3,
    builder_kwargs: Optional[dict] = None,
):
    """Build one of the neural distinguishers used by the training pipeline."""
    import tensorflow as tf

    name = architecture.strip().lower()
    kwargs = dict(builder_kwargs or {})

    if name in {"inception", "inception_eca", "rk_inception"}:
        from RKmcp import make_model_inception

        kwargs.setdefault("plain_bits", plain_bits)
        kwargs.setdefault("pairs", pairs)
        model = make_model_inception(**kwargs)

    elif name in {"resnet", "multipair_resnet"}:
        from Resnet import make_multipair_resnet

        kwargs.setdefault("plain_bits", plain_bits)
        kwargs.setdefault("pairs", pairs)
        model = make_multipair_resnet(**kwargs)

    elif name in {"dbitnet", "multipair_dbitnet"}:
        from dbitnet import make_multipair_dbitnet

        kwargs.setdefault("plain_bits", plain_bits)
        kwargs.setdefault("pairs", pairs)
        model = make_multipair_dbitnet(**kwargs)

    elif name in {"senet", "multipair_senet"}:
        from YuanWang_RK_SENet import make_multipair_senet

        kwargs.setdefault("plain_bits", plain_bits)
        kwargs.setdefault("pairs", pairs)
        kwargs.setdefault("num_filters", 64)
        kwargs.setdefault("residual_blocks", 2)
        kwargs.setdefault("dropout_rate", 0.3)
        kwargs.setdefault("reg_param", 1e-5)
        model = make_multipair_senet(**kwargs)

    else:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. "
            "Use inception, resnet, dbitnet, or senet."
        )

    if compile_model:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            amsgrad=True,
        )
        model.compile(optimizer=optimizer, loss="mse", metrics=["acc"])

    return model


def load_neural_distinguisher(
    *,
    architecture: Optional[str] = None,
    plain_bits: Optional[int] = None,
    pairs: Optional[int] = None,
    full_model_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    architecture_json_path: Optional[str] = None,
    builder_kwargs: Optional[dict] = None,
    custom_objects: Optional[dict] = None,
    compile_model: bool = False,
):
    """Load a full model, JSON+weights pair, or rebuilt weights-only model."""
    from tensorflow.keras.models import load_model, model_from_json

    supplied_modes = sum(
        [
            full_model_path is not None,
            architecture_json_path is not None,
            architecture is not None,
        ]
    )
    if supplied_modes != 1:
        raise ValueError(
            "Choose exactly one loading mode: full_model_path, "
            "architecture_json_path, or architecture."
        )

    if full_model_path is not None:
        return load_model(
            full_model_path,
            custom_objects=custom_objects,
            compile=compile_model,
        )

    if architecture_json_path is not None:
        if weights_path is None:
            raise ValueError("weights_path is required with architecture_json_path.")
        with open(architecture_json_path, "r", encoding="utf-8") as handle:
            model = model_from_json(handle.read(), custom_objects=custom_objects)
        model.load_weights(weights_path)
        return model

    if plain_bits is None or pairs is None:
        raise ValueError("plain_bits and pairs are required when rebuilding a model.")
    if weights_path is None:
        raise ValueError("weights_path is required when rebuilding a model.")

    model = build_neural_distinguisher(
        architecture,
        plain_bits=plain_bits,
        pairs=pairs,
        compile_model=compile_model,
        builder_kwargs=builder_kwargs,
    )
    model.load_weights(weights_path)
    return model


def validate_model_input_shape(model, *, plain_bits: int, pairs: int) -> None:
    """Verify compatibility with the flattened ΔC || C || C* representation."""
    expected = int(pairs) * 3 * int(plain_bits)
    input_shape = getattr(model, "input_shape", None)

    if input_shape is None:
        return
    if isinstance(input_shape, list):
        if len(input_shape) != 1:
            raise ValueError("The ranking code expects a single-input model.")
        input_shape = input_shape[0]

    actual = input_shape[-1]
    if actual is not None and int(actual) != expected:
        raise ValueError(
            f"Model input mismatch: model expects {actual} features, "
            f"but pairs={pairs}, plain_bits={plain_bits} require {expected}."
        )


def predict_structured_probability(
    model,
    X: np.ndarray,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    """Return sigmoid structured-class probabilities as a finite 1-D array."""
    pred = model.predict(
        np.asarray(X, dtype=np.float32),
        batch_size=batch_size,
        verbose=0,
    )
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)

    if not np.all(np.isfinite(pred)):
        raise FloatingPointError("The model returned NaN or infinite predictions.")
    if np.any(pred < -1e-6) or np.any(pred > 1.0 + 1e-6):
        raise ValueError(
            "The ranking code expects sigmoid probabilities in [0, 1], "
            "but the loaded model appears to return unbounded logits."
        )
    return np.clip(pred, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Attack data and cipher adapter interface
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AttackBatch:
    """Structured observations generated under one fixed related-key pair."""

    ciphertexts: np.ndarray
    ciphertexts_star: np.ndarray
    base_keys: np.ndarray
    related_keys: np.ndarray

    @property
    def num_groups(self) -> int:
        return int(self.ciphertexts.shape[0])

    @property
    def pairs(self) -> int:
        return int(self.ciphertexts.shape[1])


class CandidateAdapter(ABC):
    """Cipher-specific candidate extraction and inverse-round reconstruction."""

    candidate_component: str = "partial round-key component"
    guessed_bits_per_branch: int = 0
    oracle_known_bits_per_branch: int = 0

    @property
    @abstractmethod
    def num_candidates(self) -> int:
        """Number of partial-key-pair hypotheses."""

    def prepare_trial(self, attack_batch: AttackBatch, nr_target: int) -> None:
        """Optional hook for extracting trial-specific true round-key material."""

    @abstractmethod
    def true_candidate_indices(
        self,
        base_keys: np.ndarray,
        related_keys: np.ndarray,
        nr_target: int,
    ) -> np.ndarray:
        """Return one true candidate index per grouped observation."""

    @abstractmethod
    def reconstruct_features(
        self,
        ciphertexts: np.ndarray,
        ciphertexts_star: np.ndarray,
        candidate_indices: np.ndarray,
        pairs: int,
    ) -> np.ndarray:
        """Return group-major candidate-dependent neural inputs."""

    def candidate_components(self, candidate_index: int) -> dict[str, int]:
        """Return auditable candidate components for CSV logging."""
        return {"candidate_index": int(candidate_index)}


def make_shared_key_attack_batch(
    generator: NDCMultiPairGenerator,
    num_groups: int,
    *,
    seed: Optional[int] = None,
) -> AttackBatch:
    """Generate one trial under one base/related-key pair shared by all groups."""
    if num_groups <= 0:
        raise ValueError("num_groups must be positive.")

    g = generator
    rng = np.random.default_rng(seed)

    base_key = rng.integers(0, 2, size=(1, g.key_bits), dtype=np.uint8)
    delta_key = _int_to_bitarray(g.delta_key, g.key_bits, np)
    related_key = base_key ^ delta_key

    plaintexts = rng.integers(
        0,
        2,
        size=(num_groups * g.pairs, g.plain_bits),
        dtype=np.uint8,
    )
    delta_state = _int_to_bitarray(g.delta_state, g.plain_bits, np)
    plaintexts_star = plaintexts ^ delta_state

    base_keys = np.repeat(base_key, num_groups * g.pairs, axis=0)
    related_keys = np.repeat(related_key, num_groups * g.pairs, axis=0)

    if g.encrypt_use_gpu and cp is not None:
        ciphertexts = _safe_encrypt(
            g.encryption_function,
            cp.asarray(plaintexts),
            cp.asarray(base_keys),
            g.nr,
        )
        ciphertexts_star = _safe_encrypt(
            g.encryption_function,
            cp.asarray(plaintexts_star),
            cp.asarray(related_keys),
            g.nr,
        )
    else:
        ciphertexts = _safe_encrypt(
            g.encryption_function,
            plaintexts,
            base_keys,
            g.nr,
        )
        ciphertexts_star = _safe_encrypt(
            g.encryption_function,
            plaintexts_star,
            related_keys,
            g.nr,
        )

    if cp is not None and isinstance(ciphertexts, cp.ndarray):
        ciphertexts = cp.asnumpy(ciphertexts)
    if cp is not None and isinstance(ciphertexts_star, cp.ndarray):
        ciphertexts_star = cp.asnumpy(ciphertexts_star)

    ciphertexts = np.asarray(ciphertexts, dtype=np.uint8).reshape(
        num_groups,
        g.pairs,
        g.plain_bits,
    )
    ciphertexts_star = np.asarray(ciphertexts_star, dtype=np.uint8).reshape(
        num_groups,
        g.pairs,
        g.plain_bits,
    )

    return AttackBatch(
        ciphertexts=ciphertexts,
        ciphertexts_star=ciphertexts_star,
        base_keys=np.repeat(base_key, num_groups, axis=0),
        related_keys=np.repeat(related_key, num_groups, axis=0),
    )


# ---------------------------------------------------------------------------
# Candidate scoring and ranking
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RankResult:
    true_index: int
    best_wrong_index: int
    true_rank: int
    true_score: float
    median_wrong_score: float
    best_wrong_score: float
    median_wrong_margin: float
    best_wrong_margin: float
    top1: bool
    top5: bool
    top10: bool
    scores: np.ndarray

    @property
    def score_margin(self) -> float:
        """Backward-compatible alias for the median-wrong margin."""
        return self.median_wrong_margin


def stable_logit(
    probabilities: np.ndarray,
    eps: float = DEFAULT_LOGIT_EPS,
) -> np.ndarray:
    """Numerically stable natural-log odds."""
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must lie in (0, 0.5).")
    p = np.asarray(probabilities, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def deterministic_midrank(scores: np.ndarray, true_index: int) -> int:
    """Return a deterministic one-based midpoint rank under exact ties."""
    true_score = float(scores[true_index])
    greater = int(np.sum(scores > true_score))
    equal = int(np.sum(scores == true_score))
    return 1 + greater + (equal - 1) // 2


def score_candidates(
    model,
    attack_batch: AttackBatch,
    adapter: CandidateAdapter,
    *,
    model_batch_size: int = 8192,
    candidate_chunk_size: int = 32,
    logit_eps: float = DEFAULT_LOGIT_EPS,
) -> np.ndarray:
    """Score every candidate by accumulated neural log-odds over groups."""
    if candidate_chunk_size <= 0:
        raise ValueError("candidate_chunk_size must be positive.")

    num_groups = attack_batch.num_groups
    validate_model_input_shape(
        model,
        plain_bits=attack_batch.ciphertexts.shape[-1],
        pairs=attack_batch.pairs,
    )

    all_scores = np.zeros(adapter.num_candidates, dtype=np.float64)

    for start in range(0, adapter.num_candidates, candidate_chunk_size):
        stop = min(start + candidate_chunk_size, adapter.num_candidates)
        candidates = np.arange(start, stop, dtype=np.int64)

        X = adapter.reconstruct_features(
            attack_batch.ciphertexts,
            attack_batch.ciphertexts_star,
            candidates,
            attack_batch.pairs,
        )
        X = np.asarray(X, dtype=np.float32)

        expected_rows = num_groups * len(candidates)
        if X.ndim != 2 or X.shape[0] != expected_rows:
            raise ValueError(
                "reconstruct_features returned an invalid shape: "
                f"{X.shape}; expected ({expected_rows}, input_dim)."
            )

        pred = predict_structured_probability(
            model,
            X,
            batch_size=model_batch_size,
        ).reshape(num_groups, len(candidates))

        all_scores[start:stop] = stable_logit(pred, eps=logit_eps).sum(axis=0)

    return all_scores


def evaluate_one_trial(
    model,
    attack_batch: AttackBatch,
    adapter: CandidateAdapter,
    *,
    nr_target: int,
    model_batch_size: int = 8192,
    candidate_chunk_size: int = 32,
    logit_eps: float = DEFAULT_LOGIT_EPS,
) -> RankResult:
    """Evaluate one fixed-key partial-key-ranking trial."""
    adapter.prepare_trial(attack_batch, nr_target)

    true_indices = np.asarray(
        adapter.true_candidate_indices(
            attack_batch.base_keys,
            attack_batch.related_keys,
            nr_target,
        ),
        dtype=np.int64,
    ).reshape(-1)

    if true_indices.shape[0] != attack_batch.num_groups:
        raise ValueError("One true candidate index is required per group.")
    if np.any(true_indices < 0) or np.any(true_indices >= adapter.num_candidates):
        raise ValueError("A true candidate index is outside the candidate space.")
    if not np.all(true_indices == true_indices[0]):
        raise ValueError(
            "A ranking trial must keep one target key pair across all groups."
        )

    true_index = int(true_indices[0])
    scores = score_candidates(
        model,
        attack_batch,
        adapter,
        model_batch_size=model_batch_size,
        candidate_chunk_size=candidate_chunk_size,
        logit_eps=logit_eps,
    )

    true_score = float(scores[true_index])
    wrong_mask = np.ones(adapter.num_candidates, dtype=bool)
    wrong_mask[true_index] = False
    wrong_indices = np.arange(adapter.num_candidates, dtype=np.int64)[wrong_mask]
    wrong_scores = scores[wrong_mask]

    best_wrong_pos = int(np.argmax(wrong_scores))
    best_wrong_index = int(wrong_indices[best_wrong_pos])
    best_wrong_score = float(wrong_scores[best_wrong_pos])
    median_wrong_score = float(np.median(wrong_scores))
    rank = deterministic_midrank(scores, true_index)

    return RankResult(
        true_index=true_index,
        best_wrong_index=best_wrong_index,
        true_rank=rank,
        true_score=true_score,
        median_wrong_score=median_wrong_score,
        best_wrong_score=best_wrong_score,
        median_wrong_margin=true_score - median_wrong_score,
        best_wrong_margin=true_score - best_wrong_score,
        top1=rank <= 1,
        top5=rank <= 5,
        top10=rank <= 10,
        scores=scores,
    )


def summarize_trials(results: TypingSequence[RankResult]) -> dict[str, float]:
    """Aggregate ranking and margin statistics across independent trials."""
    if not results:
        raise ValueError("At least one result is required.")

    ranks = np.asarray([result.true_rank for result in results], dtype=np.float64)
    median_margins = np.asarray(
        [result.median_wrong_margin for result in results],
        dtype=np.float64,
    )
    best_margins = np.asarray(
        [result.best_wrong_margin for result in results],
        dtype=np.float64,
    )

    return {
        "trials": float(len(results)),
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "top1_success": float(np.mean([result.top1 for result in results])),
        "top5_success": float(np.mean([result.top5 for result in results])),
        "top10_success": float(np.mean([result.top10 for result in results])),
        "mean_median_wrong_margin": float(median_margins.mean()),
        "median_median_wrong_margin": float(np.median(median_margins)),
        "mean_best_wrong_margin": float(best_margins.mean()),
        "median_best_wrong_margin": float(np.median(best_margins)),
        # Backward-compatible names used by the earlier runner.
        "mean_score_margin": float(median_margins.mean()),
        "median_score_margin": float(np.median(median_margins)),
    }
