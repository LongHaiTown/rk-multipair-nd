"""SIMECK-32/64 adapter for downstream partial-key-pair ranking."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover - supplied cipher module is GPU-oriented
    cp = None

from key_rank_common import AttackBatch, CandidateAdapter


class SimeckPartialKeyPairAdapter(CandidateAdapter):
    """Oracle-assisted ranking of one 4-bit round-key nibble per branch.

    The supplied SIMECK implementation uses 16-bit words and the round mapping

        (x_{i+1}, y_{i+1}) = (y_i XOR F(x_i) XOR k_i, x_i),

    where F(x) = (x AND ROL_5(x)) XOR ROL_1(x).

    A correct candidate for the final attacked round reconstructs the exact
    ciphertext representation produced after one fewer round.
    """

    candidate_component = "final attacked round-key nibble"
    guessed_bits_per_branch = 4
    oracle_known_bits_per_branch = 12

    WORD_BITS = 16
    WORD_MASK = np.uint16(0xFFFF)

    def __init__(self, simeck_module, target_nibble: int = 0):
        if not 0 <= target_nibble < 4:
            raise ValueError("SIMECK target_nibble must lie in [0, 3].")
        if cp is None:
            raise RuntimeError("CuPy is required by the supplied SIMECK module.")

        self.simeck = simeck_module
        self.target_nibble = int(target_nibble)
        self._base_round_key: Optional[np.uint16] = None
        self._related_round_key: Optional[np.uint16] = None

    @property
    def num_candidates(self) -> int:
        return 256

    @staticmethod
    def decode_candidates(
        candidate_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        candidates = np.asarray(candidate_indices, dtype=np.uint16)
        base = ((candidates >> 4) & 0xF).astype(np.uint16)
        related = (candidates & 0xF).astype(np.uint16)
        return base, related

    def candidate_components(self, candidate_index: int) -> dict[str, int]:
        base, related = self.decode_candidates(np.asarray([candidate_index]))
        return {
            "candidate_index": int(candidate_index),
            "base_component": int(base[0]),
            "related_component": int(related[0]),
        }

    @classmethod
    def bits_to_words(cls, bits: np.ndarray) -> np.ndarray:
        """Convert MSB-first bit arrays into 16-bit words."""
        bits = np.asarray(bits, dtype=np.uint8)
        if bits.shape[-1] % cls.WORD_BITS != 0:
            raise ValueError("The bit dimension must be a multiple of 16.")

        num_words = bits.shape[-1] // cls.WORD_BITS
        grouped = bits.reshape(*bits.shape[:-1], num_words, cls.WORD_BITS)
        weights = (1 << np.arange(15, -1, -1, dtype=np.uint32)).reshape(
            *((1,) * (grouped.ndim - 1)),
            cls.WORD_BITS,
        )
        words = np.sum(
            grouped.astype(np.uint32) * weights,
            axis=-1,
            dtype=np.uint32,
        )
        return words.astype(np.uint16)

    @classmethod
    def words_to_bits(cls, words: np.ndarray) -> np.ndarray:
        """Convert 16-bit words into the implementation's MSB-first bits."""
        words = np.asarray(words, dtype=np.uint16)
        shifts = np.arange(15, -1, -1, dtype=np.uint16)
        bits = ((words[..., None] >> shifts) & 1).astype(np.uint8)
        return bits.reshape(*words.shape[:-1], words.shape[-1] * cls.WORD_BITS)

    @classmethod
    def _rol16(cls, values: np.ndarray, amount: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.uint16)
        amount %= cls.WORD_BITS
        if amount == 0:
            return values.copy()
        return (
            ((values << amount) & cls.WORD_MASK)
            | (values >> (cls.WORD_BITS - amount))
        ).astype(np.uint16)

    @classmethod
    def round_function(cls, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.uint16)
        return (
            (values & cls._rol16(values, 5))
            ^ cls._rol16(values, 1)
        ).astype(np.uint16)

    def _nibble_shift(self) -> int:
        # target_nibble=0 denotes the most significant nibble, matching the
        # MSB-first external bit-array convention.
        return 12 - 4 * self.target_nibble

    def _extract_nibble(self, round_keys: np.ndarray) -> np.ndarray:
        shift = self._nibble_shift()
        return ((np.asarray(round_keys, dtype=np.uint16) >> shift) & 0xF).astype(
            np.uint16
        )

    def _replace_nibble(
        self,
        round_keys: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        shift = self._nibble_shift()
        nibble_mask = np.uint16(0xF << shift)
        clear_mask = np.uint16(0xFFFF ^ int(nibble_mask))
        keys = np.asarray(round_keys, dtype=np.uint16)
        values = np.asarray(values, dtype=np.uint16)
        return ((keys & clear_mask) | ((values & 0xF) << shift)).astype(np.uint16)

    def _expand_last_key(self, master_keys: np.ndarray, nr_target: int) -> np.ndarray:
        master_keys_cp = cp.asarray(master_keys, dtype=cp.uint8)
        key_words = self.simeck.convert_from_binary(
            master_keys_cp,
            _dtype=cp.uint16,
        )
        round_keys = self.simeck.expand_key(key_words, nr_target)
        return cp.asnumpy(round_keys[nr_target - 1]).astype(np.uint16)

    def prepare_trial(self, attack_batch: AttackBatch, nr_target: int) -> None:
        self._base_round_key = np.uint16(
            self._expand_last_key(attack_batch.base_keys[:1], nr_target)[0]
        )
        self._related_round_key = np.uint16(
            self._expand_last_key(attack_batch.related_keys[:1], nr_target)[0]
        )

    def true_candidate_indices(
        self,
        base_keys: np.ndarray,
        related_keys: np.ndarray,
        nr_target: int,
    ) -> np.ndarray:
        base_round_keys = self._expand_last_key(base_keys, nr_target)
        related_round_keys = self._expand_last_key(related_keys, nr_target)
        base = self._extract_nibble(base_round_keys).astype(np.int64)
        related = self._extract_nibble(related_round_keys).astype(np.int64)
        return (base << 4) | related

    def inverse_one_round_words(
        self,
        ciphertext_words: np.ndarray,
        candidate_round_keys: np.ndarray,
    ) -> np.ndarray:
        """Vectorized inverse round.

        ciphertext_words shape: (..., 2)
        candidate_round_keys shape: (H,)
        output shape: (G, H, P, 2) for input (G, P, 2)
        """
        words = np.asarray(ciphertext_words, dtype=np.uint16)
        keys = np.asarray(candidate_round_keys, dtype=np.uint16)
        if words.ndim != 3 or words.shape[-1] != 2:
            raise ValueError("Expected ciphertext words with shape (G, P, 2).")

        out_x = words[..., 0]
        out_y = words[..., 1]

        previous_x = np.broadcast_to(
            out_y[:, None, :],
            (words.shape[0], len(keys), words.shape[1]),
        )
        previous_y = (
            out_x[:, None, :]
            ^ self.round_function(out_y[:, None, :])
            ^ keys[None, :, None]
        ).astype(np.uint16)

        return np.stack([previous_x, previous_y], axis=-1).astype(np.uint16)

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

        base_true = np.full(candidate_count, self._base_round_key, dtype=np.uint16)
        related_true = np.full(
            candidate_count,
            self._related_round_key,
            dtype=np.uint16,
        )
        base_candidate_keys = self._replace_nibble(base_true, base_values)
        related_candidate_keys = self._replace_nibble(related_true, related_values)

        ciphertext_words = self.bits_to_words(ciphertexts)
        ciphertext_words_star = self.bits_to_words(ciphertexts_star)

        state_words = self.inverse_one_round_words(
            ciphertext_words,
            base_candidate_keys,
        )
        state_words_star = self.inverse_one_round_words(
            ciphertext_words_star,
            related_candidate_keys,
        )

        state = self.words_to_bits(state_words)
        state_star = self.words_to_bits(state_words_star)
        delta_state = state ^ state_star

        triple = np.concatenate([delta_state, state, state_star], axis=-1)
        return triple.reshape(
            ciphertexts.shape[0] * candidate_count,
            pairs * 3 * ciphertexts.shape[-1],
        ).astype(np.float32)
