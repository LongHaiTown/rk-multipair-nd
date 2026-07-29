"""PRESENT-80 adapter for downstream partial-key-pair ranking."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover - PRESENT implementation is GPU-oriented
    cp = None

from key_rank_common import AttackBatch, CandidateAdapter

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class PresentPartialKeyPairAdapter(CandidateAdapter):
    """Oracle-assisted ranking of one whitening-key nibble per branch."""

    candidate_component = "final whitening-key nibble"
    guessed_bits_per_branch = 4
    oracle_known_bits_per_branch = 60

    def __init__(self, present_module, target_nibble: int = 0):
        if not 0 <= target_nibble < 16:
            raise ValueError("PRESENT target_nibble must lie in [0, 15].")
        if cp is None:
            raise RuntimeError("CuPy is required by the supplied PRESENT module.")

        self.present = present_module
        self.target_nibble = int(target_nibble)
        self._base_round_key: Optional[np.ndarray] = None
        self._related_round_key: Optional[np.ndarray] = None

        sbox = self._to_numpy(self.present.Sbox).astype(np.uint8)
        inv_sbox = np.empty_like(sbox)
        inv_sbox[sbox] = np.arange(16, dtype=np.uint8)
        self._inv_sbox = inv_sbox

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        if cp is not None and isinstance(value, cp.ndarray):
            return cp.asnumpy(value)
        return np.asarray(value)

    @property
    def num_candidates(self) -> int:
        return 256

    @staticmethod
    def decode_candidates(
        candidate_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        candidates = np.asarray(candidate_indices, dtype=np.uint16)
        base = ((candidates >> 4) & 0xF).astype(np.uint8)
        related = (candidates & 0xF).astype(np.uint8)
        return base, related

    def candidate_components(self, candidate_index: int) -> dict[str, int]:
        base, related = self.decode_candidates(np.asarray([candidate_index]))
        return {
            "candidate_index": int(candidate_index),
            "base_component": int(base[0]),
            "related_component": int(related[0]),
        }

    @staticmethod
    def _nibble_values(bits: np.ndarray, nibble: int) -> np.ndarray:
        block = bits[..., 4 * nibble : 4 * (nibble + 1)]
        weights = np.asarray([8, 4, 2, 1], dtype=np.uint8)
        return np.sum(block * weights, axis=-1, dtype=np.uint16).astype(np.uint8)

    @staticmethod
    def _write_nibble(bits: np.ndarray, nibble: int, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.uint8)
        target = bits[..., 4 * nibble : 4 * (nibble + 1)]
        target[..., 0] = (values >> 3) & 1
        target[..., 1] = (values >> 2) & 1
        target[..., 2] = (values >> 1) & 1
        target[..., 3] = values & 1

    def _expand_last_key(self, master_keys: np.ndarray, nr_target: int) -> np.ndarray:
        keys_cp = cp.asarray(master_keys, dtype=cp.uint8)
        round_keys = self.present.expand_key(keys_cp, nr_target)
        return cp.asnumpy(round_keys[nr_target - 1]).astype(np.uint8)

    def prepare_trial(self, attack_batch: AttackBatch, nr_target: int) -> None:
        self._base_round_key = self._expand_last_key(
            attack_batch.base_keys[:1],
            nr_target,
        )[0]
        self._related_round_key = self._expand_last_key(
            attack_batch.related_keys[:1],
            nr_target,
        )[0]

    def true_candidate_indices(
        self,
        base_keys: np.ndarray,
        related_keys: np.ndarray,
        nr_target: int,
    ) -> np.ndarray:
        base_rk = self._expand_last_key(base_keys, nr_target)
        related_rk = self._expand_last_key(related_keys, nr_target)
        base = self._nibble_values(base_rk, self.target_nibble).astype(np.int64)
        related = self._nibble_values(related_rk, self.target_nibble).astype(np.int64)
        return (base << 4) | related

    def _inverse_permutation(self, state: np.ndarray) -> np.ndarray:
        # present.P performs out[:, PBox] = in[:, arange(64)].
        pbox = self._to_numpy(self.present.PBox).astype(np.int64)
        return state[..., pbox]

    def _inverse_sbox(self, state: np.ndarray) -> np.ndarray:
        shape = state.shape
        nibbles = state.reshape(*shape[:-1], 16, 4)
        values = (
            8 * nibbles[..., 0]
            + 4 * nibbles[..., 1]
            + 2 * nibbles[..., 2]
            + nibbles[..., 3]
        ).astype(np.uint8)
        inverse_values = self._inv_sbox[values]

        output = np.empty_like(nibbles, dtype=np.uint8)
        output[..., 0] = (inverse_values >> 3) & 1
        output[..., 1] = (inverse_values >> 2) & 1
        output[..., 2] = (inverse_values >> 1) & 1
        output[..., 3] = inverse_values & 1
        return output.reshape(shape)

    def _decrypt_one_extra_round(
        self,
        ciphertexts: np.ndarray,
        candidate_round_keys: np.ndarray,
    ) -> np.ndarray:
        ciphertexts = np.asarray(ciphertexts, dtype=np.uint8)
        candidate_round_keys = np.asarray(candidate_round_keys, dtype=np.uint8)
        state = (
            ciphertexts[:, None, :, :]
            ^ candidate_round_keys[None, :, None, :]
        )
        state = self._inverse_permutation(state)
        return self._inverse_sbox(state)

    def reconstruct_features(
        self,
        ciphertexts: np.ndarray,
        ciphertexts_star: np.ndarray,
        candidate_indices: np.ndarray,
        pairs: int,
    ) -> np.ndarray:
        if self._base_round_key is None or self._related_round_key is None:
            raise RuntimeError("prepare_trial() must be called first.")

        base_values, related_values = self.decode_candidates(candidate_indices)
        candidate_count = len(candidate_indices)

        base_keys = np.repeat(
            self._base_round_key[None, :],
            candidate_count,
            axis=0,
        )
        related_keys = np.repeat(
            self._related_round_key[None, :],
            candidate_count,
            axis=0,
        )
        self._write_nibble(base_keys, self.target_nibble, base_values)
        self._write_nibble(related_keys, self.target_nibble, related_values)

        state = self._decrypt_one_extra_round(ciphertexts, base_keys)
        state_star = self._decrypt_one_extra_round(ciphertexts_star, related_keys)
        delta_state = state ^ state_star

        triple = np.concatenate([delta_state, state, state_star], axis=-1)
        return triple.reshape(
            ciphertexts.shape[0] * candidate_count,
            pairs * 3 * ciphertexts.shape[-1],
        ).astype(np.float32)


# python key_rank/run_key_rank_present80.py \
#   --weights checkpoints/present80/present80_best_8r.weights.h5 \
#   --architecture senet \
#   --base-rounds 8 \
#   --pairs 4 \
#   --input-diff 0x80 \
#   --delta-key-bit 56 \
#   --num-groups 100 \
#   --trials 50 \
#   --target-nibble 0 \
#   --output-dir results/key_rank_r9_k4