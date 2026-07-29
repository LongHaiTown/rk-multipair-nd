"""
Minimal downstream partial-key ranking for related-key multi-pair NDC.

Purpose
-------
This module does NOT implement a full key-recovery attack. It validates whether
a trained related-key multi-pair neural distinguisher assigns systematically
better scores to the correct partial-key hypothesis than to wrong hypotheses.

The implementation reuses the configuration and encryption function of
NDCMultiPairGenerator, but generates structured attack observations only
(Y = 1) and retains the ciphertexts and sampled keys needed for ranking.

Expected neural input
---------------------
For each candidate h and each grouped observation i:

    Z_{i,j}^{(h)} = ΔS_{i,j}^{(h)} || S_{i,j}^{(h)} || S*_{i,j}^{(h)}
    X_i^{(h)}     = Z_{i,1}^{(h)} || ... || Z_{i,k}^{(h)}

where S and S* are candidate-dependent partially decrypted states.

The cipher-specific part is isolated in CandidateAdapter.reconstruct_features()
and CandidateAdapter.true_candidate_indices().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence as TypingSequence

import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None

# Rename make_data_train(2).py to make_data_train.py before importing.
from make_data_train import NDCMultiPairGenerator, _int_to_bitarray, _safe_encrypt


# ---------------------------------------------------------------------------
# NEURAL DISTINGUISHER CONSTRUCTION / LOADING
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
    """
    Build one of the neural distinguishers used by main.py.

    Supported names
    ---------------
    inception
        RKmcp.make_model_inception
    resnet
        Resnet.make_multipair_resnet
    dbitnet
        dbitnet.make_multipair_dbitnet
    senet
        YuanWang_RK_SENet.make_multipair_senet

    `builder_kwargs` is forwarded to the selected model builder. This is useful
    because the exact optional arguments differ among the architecture files.

    The default SENet settings reproduce the model call currently used in
    main.py, except that `plain_bits` and `pairs` are not hard-coded.
    """
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
        # Matches main.py. Compilation is not required for model.predict().
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
    """
    Load a trained neural distinguisher saved by main.py / train_nets.py.

    Supported artifact formats
    --------------------------
    1. Full `.keras` model:
           full_model_path="checkpoints/present80/present80_final_8r.keras"

    2. JSON architecture + weights:
           architecture_json_path="..._architecture.json"
           weights_path="...weights.h5"

    3. Rebuild through the original model factory + load weights:
           architecture="senet"
           plain_bits=64
           pairs=4
           weights_path="...weights.h5"

    Option 3 is often the safest when older JSON serialization contains custom
    objects or when you want the architecture to be created exactly as in
    main.py.
    """
    import tensorflow as tf
    from tensorflow.keras.models import load_model, model_from_json

    supplied_modes = sum([
        full_model_path is not None,
        architecture_json_path is not None,
        architecture is not None,
    ])
    if supplied_modes != 1:
        raise ValueError(
            "Choose exactly one loading mode: full_model_path, "
            "architecture_json_path, or architecture."
        )

    if full_model_path is not None:
        model = load_model(
            full_model_path,
            custom_objects=custom_objects,
            compile=compile_model,
        )

    elif architecture_json_path is not None:
        if weights_path is None:
            raise ValueError(
                "weights_path is required with architecture_json_path."
            )
        with open(architecture_json_path, "r", encoding="utf-8") as f:
            model = model_from_json(
                f.read(),
                custom_objects=custom_objects,
            )
        model.load_weights(weights_path)

    else:
        if plain_bits is None or pairs is None:
            raise ValueError(
                "plain_bits and pairs are required when rebuilding a model."
            )
        if weights_path is None:
            raise ValueError(
                "weights_path is required when rebuilding a model."
            )
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
    """
    Verify compatibility with the NDC representation ΔC || C || C*.

    Expected flattened dimension:
        pairs * 3 * plain_bits
    """
    expected = int(pairs) * 3 * int(plain_bits)
    input_shape = getattr(model, "input_shape", None)

    if input_shape is None:
        return

    if isinstance(input_shape, list):
        if len(input_shape) != 1:
            raise ValueError(
                "The ranking code expects a single-input neural distinguisher."
            )
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
    """
    Call the neural distinguisher and normalize its output to shape (N,).

    The models in main.py use a sigmoid binary prediction head and are trained
    with labels 0/1. Therefore output is interpreted as P(Y=1 | X), where Y=1
    is the structured related-key class.
    """
    pred = model.predict(
        np.asarray(X, dtype=np.float32),
        batch_size=batch_size,
        verbose=0,
    )
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)

    if not np.all(np.isfinite(pred)):
        raise FloatingPointError(
            "The neural distinguisher returned NaN or infinite predictions."
        )

    return pred


@dataclass(frozen=True)
class AttackBatch:
    """Structured related-key observations retained before feature conversion."""

    ciphertexts: np.ndarray          # shape: (num_groups, pairs, block_bits)
    ciphertexts_star: np.ndarray     # shape: (num_groups, pairs, block_bits)
    base_keys: np.ndarray            # shape: (num_groups, key_bits)
    related_keys: np.ndarray         # shape: (num_groups, key_bits)

    @property
    def num_groups(self) -> int:
        return int(self.ciphertexts.shape[0])

    @property
    def pairs(self) -> int:
        return int(self.ciphertexts.shape[1])


class CandidateAdapter(ABC):
    """
    Cipher-specific bridge between ciphertexts and candidate-dependent features.

    A candidate may represent:
      * a partial round key;
      * a pair (u, u*) of related partial round keys; or
      * any restricted key-hypothesis identifier.

    The recommended MVP for PRESENT is a 256-candidate space, e.g. two 4-bit
    partial round-key hypotheses combined into one 8-bit candidate identifier.
    """

    @property
    @abstractmethod
    def num_candidates(self) -> int:
        """Number of candidate hypotheses."""

    def prepare_trial(self, attack_batch: "AttackBatch", nr_target: int) -> None:
        """Optional cipher-specific trial preparation hook."""
        return None

    @abstractmethod
    def true_candidate_indices(
        self,
        base_keys: np.ndarray,
        related_keys: np.ndarray,
        nr_target: int,
    ) -> np.ndarray:
        """
        Return one true candidate index per grouped observation.

        Shape: (num_groups,)
        Values must lie in [0, num_candidates).
        """

    @abstractmethod
    def reconstruct_features(
        self,
        ciphertexts: np.ndarray,
        ciphertexts_star: np.ndarray,
        candidate_indices: np.ndarray,
        pairs: int,
    ) -> np.ndarray:
        """
        Build candidate-dependent neural inputs.

        Parameters
        ----------
        ciphertexts, ciphertexts_star:
            Arrays with shape (num_groups, pairs, block_bits).
        candidate_indices:
            A one-dimensional candidate chunk.
        pairs:
            Number of ciphertext pairs per grouped observation.

        Returns
        -------
        X:
            Float32 array with shape
            (num_groups * len(candidate_indices), pairs * 3 * block_bits).

        Ordering contract
        -----------------
        Candidate-major within each group is NOT used. The required layout is:

            group 0: candidate 0, candidate 1, ...
            group 1: candidate 0, candidate 1, ...
            ...

        so predictions can be reshaped as:
            (num_groups, len(candidate_indices)).
        """


class RankingAttackGenerator:
    """
    Structured-only attack-data generator based on NDCMultiPairGenerator.

    It reuses:
      * encryption_function
      * block/key sizes
      * round count
      * ΔP and ΔK
      * pairs per grouped observation
      * CPU/GPU backend policy

    Unlike NDCMultiPairGenerator.__getitem__, this class retains C, C*, K, K*
    instead of immediately converting them to labels and flattened features.
    """

    def __init__(
        self,
        ndc_generator: NDCMultiPairGenerator,
        seed: Optional[int] = None,
    ) -> None:
        self.gen = ndc_generator
        self.rng = np.random.default_rng(seed)

    def generate(self, num_groups: int) -> AttackBatch:
        if num_groups <= 0:
            raise ValueError("num_groups must be positive.")

        g = self.gen
        use_cp = bool(g.use_gpu and cp is not None)
        lib = cp if use_cp else np

        # Local deterministic NumPy generation is preferred for reproducibility.
        base_keys_np = self.rng.integers(
            0, 2, size=(num_groups, g.key_bits), dtype=np.uint8
        )
        plaintexts_np = self.rng.integers(
            0, 2, size=(num_groups * g.pairs, g.plain_bits), dtype=np.uint8
        )

        delta_key_np = _int_to_bitarray(g.delta_key, g.key_bits, np)
        delta_state_np = _int_to_bitarray(g.delta_state, g.plain_bits, np)

        related_keys_np = base_keys_np ^ delta_key_np
        plaintexts_star_np = plaintexts_np ^ delta_state_np

        base_keys_repeated_np = np.repeat(base_keys_np, g.pairs, axis=0)
        related_keys_repeated_np = np.repeat(related_keys_np, g.pairs, axis=0)

        if g.encrypt_use_gpu and cp is not None:
            P_in = cp.asarray(plaintexts_np)
            P_star_in = cp.asarray(plaintexts_star_np)
            K_in = cp.asarray(base_keys_repeated_np)
            K_star_in = cp.asarray(related_keys_repeated_np)
        else:
            P_in = plaintexts_np
            P_star_in = plaintexts_star_np
            K_in = base_keys_repeated_np
            K_star_in = related_keys_repeated_np

        C = _safe_encrypt(g.encryption_function, P_in, K_in, g.nr)
        C_star = _safe_encrypt(g.encryption_function, P_star_in, K_star_in, g.nr)

        if cp is not None and isinstance(C, cp.ndarray):
            C = cp.asnumpy(C)
        if cp is not None and isinstance(C_star, cp.ndarray):
            C_star = cp.asnumpy(C_star)

        C = np.asarray(C, dtype=np.uint8).reshape(
            num_groups, g.pairs, g.plain_bits
        )
        C_star = np.asarray(C_star, dtype=np.uint8).reshape(
            num_groups, g.pairs, g.plain_bits
        )

        return AttackBatch(
            ciphertexts=C,
            ciphertexts_star=C_star,
            base_keys=base_keys_np,
            related_keys=related_keys_np,
        )


@dataclass(frozen=True)
class RankResult:
    true_rank: int
    true_score: float
    median_wrong_score: float
    score_margin: float
    top1: bool
    top5: bool
    top10: bool
    scores: np.ndarray


def stable_logit(probabilities: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Numerically stable natural-log odds."""
    p = np.asarray(probabilities, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def deterministic_midrank(scores: np.ndarray, true_index: int) -> int:
    """
    One-based deterministic midrank.

    For ties, this returns the rounded midpoint of the tied rank interval.
    """
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
    use_mean_probability_baseline: bool = False,
) -> np.ndarray:
    """
    Score all candidates by summing group-level neural evidence.

    Primary score:
        S(h) = Σ_i logit(f_theta(X_i^(h)))

    Optional baseline:
        S_prob(h) = mean_i f_theta(X_i^(h))
    """
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

        if use_mean_probability_baseline:
            chunk_scores = pred.mean(axis=0)
        else:
            chunk_scores = stable_logit(pred).sum(axis=0)

        all_scores[start:stop] = chunk_scores

    return all_scores


def evaluate_one_trial(
    model,
    attack_batch: AttackBatch,
    adapter: CandidateAdapter,
    *,
    nr_target: int,
    model_batch_size: int = 8192,
    candidate_chunk_size: int = 32,
) -> RankResult:
    """
    Evaluate one key-ranking trial.

    MVP assumption
    --------------
    One trial should use one base/related-key pair shared across all grouped
    observations. Therefore, all rows in true_candidate_indices must agree.
    """
    adapter.prepare_trial(attack_batch, nr_target)

    true_indices = adapter.true_candidate_indices(
        attack_batch.base_keys,
        attack_batch.related_keys,
        nr_target,
    )
    true_indices = np.asarray(true_indices, dtype=np.int64).reshape(-1)

    if true_indices.shape[0] != attack_batch.num_groups:
        raise ValueError("One true candidate index is required per group.")
    if np.any(true_indices < 0) or np.any(true_indices >= adapter.num_candidates):
        raise ValueError("A true candidate index is outside the candidate space.")
    if not np.all(true_indices == true_indices[0]):
        raise ValueError(
            "A ranking trial must keep one target key pair across all groups. "
            "Generate all groups under one shared key pair, or evaluate each "
            "key pair as a separate trial."
        )

    true_index = int(true_indices[0])
    scores = score_candidates(
        model,
        attack_batch,
        adapter,
        model_batch_size=model_batch_size,
        candidate_chunk_size=candidate_chunk_size,
    )

    true_score = float(scores[true_index])
    wrong_scores = np.delete(scores, true_index)
    median_wrong = float(np.median(wrong_scores))
    rank = deterministic_midrank(scores, true_index)

    return RankResult(
        true_rank=rank,
        true_score=true_score,
        median_wrong_score=median_wrong,
        score_margin=true_score - median_wrong,
        top1=rank <= 1,
        top5=rank <= 5,
        top10=rank <= 10,
        scores=scores,
    )


def summarize_trials(results: TypingSequence[RankResult]) -> dict[str, float]:
    if not results:
        raise ValueError("At least one result is required.")

    ranks = np.asarray([r.true_rank for r in results], dtype=np.float64)
    margins = np.asarray([r.score_margin for r in results], dtype=np.float64)

    return {
        "trials": float(len(results)),
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "top1_success": float(np.mean([r.top1 for r in results])),
        "top5_success": float(np.mean([r.top5 for r in results])),
        "top10_success": float(np.mean([r.top10 for r in results])),
        "mean_score_margin": float(margins.mean()),
        "median_score_margin": float(np.median(margins)),
    }


# ---------------------------------------------------------------------------
# PRESENT ADAPTER TEMPLATE
# ---------------------------------------------------------------------------

class PresentPartialKeyPairAdapter(CandidateAdapter):
    """
    Oracle-assisted 8-bit local ranking adapter for PRESENT-80.

    Candidate encoding
    ------------------
    high nibble : guessed 4-bit nibble u of the base-branch final whitening key
    low nibble  : guessed 4-bit nibble u* of the related-branch final whitening key

    The other 60 bits of each final whitening key are fixed to their true values
    in the simulated trial. This makes the experiment a *local partial-key
    ranking validity check*, not a practical end-to-end key-recovery attack.

    Why this design is needed
    -------------------------
    The existing neural distinguisher expects the complete r-round input
    representation ΔC || C || C*. Guessing only four key bits cannot reconstruct
    all 64 bits after inverse-round processing unless the remaining key bits are
    supplied. The oracle-assisted restriction allows the existing full-input
    model to be reused while testing whether its scores prefer the correct local
    related-key hypothesis.

    PRESENT round convention in present80.py
    ----------------------------------------
    encrypt(..., r) performs r-1 full SP rounds and then XORs ks[r-1].
    Therefore, to transform an (r+1)-round ciphertext into the representation
    consumed by an r-round distinguisher, we compute:

        state_r = InvSBox(InvPBox(ciphertext XOR guessed_ks[r]))

    which equals the r-round ciphertext when the complete guessed whitening key
    is correct.
    """

    def __init__(self, present_module, target_nibble: int = 0):
        if not 0 <= target_nibble < 16:
            raise ValueError("target_nibble must lie in [0, 15].")
        self.present = present_module
        self.target_nibble = int(target_nibble)
        self._base_round_key: Optional[np.ndarray] = None
        self._related_round_key: Optional[np.ndarray] = None

        sbox = self._to_numpy(self.present.Sbox).astype(np.uint8)
        inv_sbox = np.empty_like(sbox)
        inv_sbox[sbox] = np.arange(16, dtype=np.uint8)
        self._inv_sbox = inv_sbox

        pbox = self._to_numpy(self.present.PBox).astype(np.int64)
        inv_pbox = np.empty_like(pbox)
        inv_pbox[pbox] = np.arange(64, dtype=np.int64)
        self._inv_pbox = inv_pbox

    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        if cp is not None and isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
        return np.asarray(x)

    @property
    def num_candidates(self) -> int:
        return 256

    @staticmethod
    def decode_candidates(candidate_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        candidates = np.asarray(candidate_indices, dtype=np.uint16)
        u = ((candidates >> 4) & 0xF).astype(np.uint8)
        u_star = (candidates & 0xF).astype(np.uint8)
        return u, u_star

    @staticmethod
    def _nibble_values(bits: np.ndarray, nibble: int) -> np.ndarray:
        block = bits[..., 4 * nibble:4 * (nibble + 1)]
        weights = np.array([8, 4, 2, 1], dtype=np.uint8)
        return np.sum(block * weights, axis=-1, dtype=np.uint16).astype(np.uint8)

    @staticmethod
    def _write_nibble(bits: np.ndarray, nibble: int, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.uint8)
        target = bits[..., 4 * nibble:4 * (nibble + 1)]
        target[..., 0] = (values >> 3) & 1
        target[..., 1] = (values >> 2) & 1
        target[..., 2] = (values >> 1) & 1
        target[..., 3] = values & 1

    def _expand_last_key(self, master_keys: np.ndarray, nr_target: int) -> np.ndarray:
        keys_cp = cp.asarray(master_keys, dtype=cp.uint8)
        round_keys = self.present.expand_key(keys_cp, nr_target)
        return cp.asnumpy(round_keys[nr_target - 1]).astype(np.uint8)

    def prepare_trial(self, attack_batch: AttackBatch, nr_target: int) -> None:
        # The trial uses one key pair repeated over all grouped observations.
        base_unique = attack_batch.base_keys[:1]
        related_unique = attack_batch.related_keys[:1]
        self._base_round_key = self._expand_last_key(base_unique, nr_target)[0]
        self._related_round_key = self._expand_last_key(related_unique, nr_target)[0]

    def true_candidate_indices(
        self,
        base_keys: np.ndarray,
        related_keys: np.ndarray,
        nr_target: int,
    ) -> np.ndarray:
        base_rk = self._expand_last_key(base_keys, nr_target)
        related_rk = self._expand_last_key(related_keys, nr_target)
        u = self._nibble_values(base_rk, self.target_nibble).astype(np.int64)
        u_star = self._nibble_values(related_rk, self.target_nibble).astype(np.int64)
        return (u << 4) | u_star

    def _inverse_p(self, x: np.ndarray) -> np.ndarray:
        # present80.P performs out[:, PBox] = in[:, arange(64)].
        # Thus inverse is out[:, i] = in[:, PBox[i]].
        return x[..., self._to_numpy(self.present.PBox).astype(np.int64)]

    def _inverse_sbox(self, x: np.ndarray) -> np.ndarray:
        shape = x.shape
        nibbles = x.reshape(*shape[:-1], 16, 4)
        values = (
            8 * nibbles[..., 0]
            + 4 * nibbles[..., 1]
            + 2 * nibbles[..., 2]
            + nibbles[..., 3]
        ).astype(np.uint8)
        inv_values = self._inv_sbox[values]
        out = np.empty_like(nibbles, dtype=np.uint8)
        out[..., 0] = (inv_values >> 3) & 1
        out[..., 1] = (inv_values >> 2) & 1
        out[..., 2] = (inv_values >> 1) & 1
        out[..., 3] = inv_values & 1
        return out.reshape(shape)

    def _decrypt_one_full_round(
        self,
        ciphertexts: np.ndarray,
        candidate_round_keys: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorized inverse of the last full SP round plus whitening XOR.

        ciphertexts shape       : (G, P, 64)
        candidate_round_keys    : (H, 64)
        output                  : (G, H, P, 64)
        """
        c = np.asarray(ciphertexts, dtype=np.uint8)
        rk = np.asarray(candidate_round_keys, dtype=np.uint8)
        x = c[:, None, :, :] ^ rk[None, :, None, :]
        x = self._inverse_p(x)
        return self._inverse_sbox(x)

    def reconstruct_features(
        self,
        ciphertexts: np.ndarray,
        ciphertexts_star: np.ndarray,
        candidate_indices: np.ndarray,
        pairs: int,
    ) -> np.ndarray:
        if self._base_round_key is None or self._related_round_key is None:
            raise RuntimeError(
                "prepare_trial() must be called before reconstruct_features()."
            )

        u, u_star = self.decode_candidates(candidate_indices)
        h = len(candidate_indices)

        base_keys = np.repeat(self._base_round_key[None, :], h, axis=0)
        related_keys = np.repeat(self._related_round_key[None, :], h, axis=0)
        self._write_nibble(base_keys, self.target_nibble, u)
        self._write_nibble(related_keys, self.target_nibble, u_star)

        state = self._decrypt_one_full_round(ciphertexts, base_keys)
        state_star = self._decrypt_one_full_round(ciphertexts_star, related_keys)
        delta_state = state ^ state_star

        # (G, H, P, 192), preserving ΔS || S || S* exactly as in training.
        triple = np.concatenate(
            [delta_state, state, state_star],
            axis=-1,
        )

        # Group-major, candidate-minor ordering required by score_candidates().
        return triple.reshape(
            ciphertexts.shape[0] * h,
            pairs * 3 * ciphertexts.shape[-1],
        ).astype(np.float32)


def make_shared_key_attack_batch(
    generator: NDCMultiPairGenerator,
    num_groups: int,
    *,
    seed: Optional[int] = None,
) -> AttackBatch:
    """
    Generate one ranking trial with ONE key pair shared across all groups.

    NDCMultiPairGenerator normally samples a new key pair per grouped
    observation. That is appropriate for classifier training but not for a
    standard key-ranking trial, which accumulates evidence about one fixed
    unknown key pair.

    This helper first generates one key pair and then encrypts all groups under
    that same pair.
    """
    if num_groups <= 0:
        raise ValueError("num_groups must be positive.")

    g = generator
    rng = np.random.default_rng(seed)

    base_key = rng.integers(0, 2, size=(1, g.key_bits), dtype=np.uint8)
    delta_key = _int_to_bitarray(g.delta_key, g.key_bits, np)
    related_key = base_key ^ delta_key

    P = rng.integers(
        0, 2, size=(num_groups * g.pairs, g.plain_bits), dtype=np.uint8
    )
    delta_state = _int_to_bitarray(g.delta_state, g.plain_bits, np)
    P_star = P ^ delta_state

    K = np.repeat(base_key, num_groups * g.pairs, axis=0)
    K_star = np.repeat(related_key, num_groups * g.pairs, axis=0)

    if g.encrypt_use_gpu and cp is not None:
        C = _safe_encrypt(
            g.encryption_function, cp.asarray(P), cp.asarray(K), g.nr
        )
        C_star = _safe_encrypt(
            g.encryption_function, cp.asarray(P_star), cp.asarray(K_star), g.nr
        )
    else:
        C = _safe_encrypt(g.encryption_function, P, K, g.nr)
        C_star = _safe_encrypt(g.encryption_function, P_star, K_star, g.nr)

    if cp is not None and isinstance(C, cp.ndarray):
        C = cp.asnumpy(C)
    if cp is not None and isinstance(C_star, cp.ndarray):
        C_star = cp.asnumpy(C_star)

    C = np.asarray(C, dtype=np.uint8).reshape(
        num_groups, g.pairs, g.plain_bits
    )
    C_star = np.asarray(C_star, dtype=np.uint8).reshape(
        num_groups, g.pairs, g.plain_bits
    )

    return AttackBatch(
        ciphertexts=C,
        ciphertexts_star=C_star,
        base_keys=np.repeat(base_key, num_groups, axis=0),
        related_keys=np.repeat(related_key, num_groups, axis=0),
    )


# ---------------------------------------------------------------------------
# EXAMPLE: LOAD THE SAME MODEL FAMILY USED BY main.py
# ---------------------------------------------------------------------------

# # Mode B: rebuild the SENet exactly through the model factory used in main.py
# # and then load a weights-only checkpoint.
# model = load_neural_distinguisher(
#     architecture="senet",
#     plain_bits=pr.plain_bits,
#     pairs=4,
#     weights_path=(
#         "checkpoints/present80/"
#         "present80_best_8r.weights.h5"
#     ),
#     builder_kwargs={
#         "num_filters": 64,
#         "residual_blocks": 2,
#         "dropout_rate": 0.3,
#         "reg_param": 1e-5,
#     },
# )
#
# # Mode C: JSON architecture + weights, also generated by save_artifacts().
# model = load_neural_distinguisher(
#     architecture_json_path=(
#         "checkpoints/present80/"
#         "present80_final_8r_architecture.json"
#     ),
#     weights_path=(
#         "checkpoints/present80/"
#         "present80_final_8r.weights.h5"
#     ),
# )
#
# generator = NDCMultiPairGenerator(
#     encryption_function=pr.encrypt,
#     plain_bits=pr.plain_bits,
#     key_bits=pr.key_bits,
#     nr=9,                 # attack target: one round deeper
#     delta_state=0x0000000000000080,
#     delta_key=(1 << 56),  # or the exact bit-array convention used in training
#     n_samples=1,
#     batch_size=1,
#     pairs=4,              # MUST match the loaded model
#     use_gpu=False,
#     encrypt_backend="numpy",
# )
#
# adapter = PresentPartialKeyPairAdapter(
#     pr,
#     target_nibble=0,
# )
#
# results = []
# for trial_seed in range(50):
#     attack_batch = make_shared_key_attack_batch(
#         generator,
#         num_groups=100,
#         seed=trial_seed,
#     )
#     result = evaluate_one_trial(
#         model,
#         attack_batch,
#         adapter,
#         nr_target=9,
#         model_batch_size=8192,
#         candidate_chunk_size=16,
#     )
#     results.append(result)
#
# print(summarize_trials(results))


