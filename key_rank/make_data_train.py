from tensorflow.keras.utils import Sequence
import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None

DEFAULT_USE_GPU = cp is not None


def _int_to_bitarray(val, nbits, lib):
    """
    Convert val (int or numpy/cupy array) -> bit array length nbits
    in backend lib (np or cp). Returns uint8 array.
    """
    if isinstance(val, (int, np.integer)):
        bits = np.zeros(nbits, dtype=np.uint8)
        v = int(val)
        for i in range(nbits):
            bits[nbits - 1 - i] = (v >> i) & 1
        return lib.asarray(bits) if (cp is not None and lib is cp) else bits

    if cp is not None and isinstance(val, cp.ndarray) and lib is np:
        return cp.asnumpy(val).astype(np.uint8)

    if cp is not None and isinstance(val, np.ndarray) and lib is cp:
        return cp.asarray(val.astype(np.uint8))

    if isinstance(val, (np.ndarray,)) or (cp is not None and isinstance(val, cp.ndarray)):
        return val.astype(np.uint8)

    arr = np.asarray(val, dtype=np.uint8)
    return lib.asarray(arr) if (cp is not None and lib is cp) else arr


def _safe_encrypt(enc_fn, P, K, nr):
    """
    Call encryption function with automatic CPU <-> GPU fallback.
    Tries once, then falls back once. Cipher logic errors will still raise.
    """
    try:
        return enc_fn(P, K, nr)

    except Exception:
        if cp is None:
            raise

        if isinstance(P, cp.ndarray):
            # GPU -> CPU fallback
            return enc_fn(cp.asnumpy(P), cp.asnumpy(K), nr)
        else:
            # CPU -> GPU fallback
            return enc_fn(cp.asarray(P), cp.asarray(K), nr)


class NDCMultiPairGenerator(Sequence):
    def __init__(self, encryption_function, plain_bits, key_bits, nr,
                 delta_state=0, delta_key=0,
                 n_samples=10**7, batch_size=10**5,
                 pairs=2, use_gpu=None, to_float32=True,
                 start_idx=0, encrypt_backend='numpy'):

        self.encryption_function = encryption_function
        self.plain_bits = plain_bits
        self.key_bits = key_bits
        self.nr = nr

        self.delta_state = delta_state
        self.delta_key = delta_key

        self.n = int(n_samples)
        self.batch_size = int(batch_size)
        self.start_idx = int(start_idx)
        self.pairs = int(pairs)

        if use_gpu is None:
            self.use_gpu = DEFAULT_USE_GPU
        else:
            self.use_gpu = bool(use_gpu)

        if self.use_gpu and cp is None:
            self.use_gpu = False

        self.to_float32 = bool(to_float32)

        if encrypt_backend == 'auto':
            self.encrypt_use_gpu = self.use_gpu
        elif encrypt_backend == 'cupy':
            self.encrypt_use_gpu = True
        else:
            self.encrypt_use_gpu = False

        if self.encrypt_use_gpu and cp is None:
            self.encrypt_use_gpu = False

        self.steps = (self.n + self.batch_size - 1) // self.batch_size
        self.input_dim = self.pairs * 3 * self.plain_bits

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        curr_n = min(self.batch_size, self.n - idx * self.batch_size)
        if curr_n <= 0:
            raise IndexError

        lib = cp if (self.use_gpu and cp is not None) else np

        delta_key_vec = _int_to_bitarray(self.delta_key, self.key_bits, lib)
        delta_state_vec = _int_to_bitarray(self.delta_state, self.plain_bits, lib)

        Y = lib.zeros(curr_n, dtype=lib.uint8)
        Y[:curr_n // 2] = 1
        lib.random.shuffle(Y)

        K0 = lib.random.randint(0, 2, (curr_n, self.key_bits), dtype=lib.uint8)
        K1 = K0 ^ delta_key_vec

        K = lib.repeat(K0, self.pairs, axis=0)
        K_star = lib.repeat(K1, self.pairs, axis=0)

        P = lib.random.randint(
            0, 2, (curr_n * self.pairs, self.plain_bits), dtype=lib.uint8
        )
        P_star = lib.empty_like(P)

        mask = lib.repeat(Y == 1, self.pairs, axis=0)
        P_star[mask] = P[mask] ^ delta_state_vec

        n_false = int((~mask).sum())
        if n_false > 0:
            P_star[~mask] = lib.random.randint(
                0, 2, (n_false, self.plain_bits), dtype=lib.uint8
            )

        # --- ENCRYPTION PREP ---
        if (self.use_gpu and cp is not None) and not self.encrypt_use_gpu:
            P_in, K_in = cp.asnumpy(P), cp.asnumpy(K)
            P_star_in, K_star_in = cp.asnumpy(P_star), cp.asnumpy(K_star)

        elif (not self.use_gpu) and self.encrypt_use_gpu and cp is not None:
            P_in, K_in = cp.asarray(P), cp.asarray(K)
            P_star_in, K_star_in = cp.asarray(P_star), cp.asarray(K_star)

        else:
            P_in, K_in = P, K
            P_star_in, K_star_in = P_star, K_star

        # --- SAFE ENCRYPT ---
        C = _safe_encrypt(self.encryption_function, P_in, K_in, self.nr)
        C_star = _safe_encrypt(self.encryption_function, P_star_in, K_star_in, self.nr)

        # --- NORMALIZE BACKEND ---
        if self.use_gpu and cp is not None:
            C = cp.asarray(C) if not isinstance(C, cp.ndarray) else C
            C_star = cp.asarray(C_star) if not isinstance(C_star, cp.ndarray) else C_star
        else:
            C = cp.asnumpy(C) if (cp is not None and isinstance(C, cp.ndarray)) else C
            C_star = cp.asnumpy(C_star) if (cp is not None and isinstance(C_star, cp.ndarray)) else C_star

        delta_C = C ^ C_star
        triple = lib.concatenate([delta_C, C, C_star], axis=1)

        X = triple.reshape(curr_n, -1)
        if self.to_float32:
            X = X.astype(lib.float32)

        if self.use_gpu and cp is not None:
            return cp.asnumpy(X), cp.asnumpy(Y)

        return X.astype(np.float32), Y.astype(np.uint8)
