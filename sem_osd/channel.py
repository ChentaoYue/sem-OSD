"""BPSK + AWGN / Gilbert-Elliott channels and symbol-level posteriors."""
from __future__ import annotations
import numpy as np


def ebn0_db_to_sigma(ebn0_db: float, rate: float) -> float:
    snr_lin = 10 ** (ebn0_db / 10.0)
    return float(np.sqrt(1.0 / (2.0 * rate * snr_lin)))


def bytes_to_bits(bs: np.ndarray, bitorder: str = "big") -> np.ndarray:
    bs = np.asarray(bs, dtype=np.uint8)
    return np.unpackbits(bs, bitorder=bitorder)


def bits_to_bytes(bits: np.ndarray, bitorder: str = "big") -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    return np.packbits(bits.reshape(-1, 8), axis=-1, bitorder=bitorder).flatten()


def bpsk_modulate(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    return (1.0 - 2.0 * bits.astype(np.float64))


def awgn(signal: np.ndarray, sigma, rng=None) -> np.ndarray:
    """AWGN with scalar or per-sample sigma."""
    rng = rng if rng is not None else np.random.default_rng()
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.ndim == 0:
        return signal + rng.normal(0.0, float(sigma), size=signal.shape)
    return signal + rng.standard_normal(size=signal.shape) * sigma


def gilbert_elliott_channel(signal: np.ndarray, ebn0_db: float, rate: float,
                              pi_B: float = 0.10,
                              mean_burst_len: float = 16.0,
                              sigma_ratio_sq: float = 100.0,
                              rng=None):
    """Gilbert-Elliott channel with state-aware per-bit noise.

    Returns (y, sigma_per_sample). The composite noise variance is calibrated
    so that the average Eb/N0 equals `ebn0_db` (matching AWGN convention).
    """
    rng = rng if rng is not None else np.random.default_rng()
    pi_G = 1.0 - pi_B
    sigma_avg_sq = 1.0 / (2.0 * rate * 10.0 ** (ebn0_db / 10.0))
    sigma_G_sq = sigma_avg_sq / (pi_G + pi_B * sigma_ratio_sq)
    sigma_B_sq = sigma_ratio_sq * sigma_G_sq
    sigma_G = float(np.sqrt(sigma_G_sq))
    sigma_B = float(np.sqrt(sigma_B_sq))

    p_BG = 1.0 / float(mean_burst_len)
    p_GB = (pi_B / pi_G) * p_BG

    n = int(signal.size)
    states = np.empty(n, dtype=np.uint8)
    state = 1 if rng.random() < pi_B else 0
    for i in range(n):
        states[i] = state
        if state == 0:
            if rng.random() < p_GB:
                state = 1
        else:
            if rng.random() < p_BG:
                state = 0
    sigmas = np.where(states == 0, sigma_G, sigma_B).astype(np.float64)
    noise = rng.standard_normal(n) * sigmas
    y = signal + noise.reshape(signal.shape)
    return y, sigmas.reshape(signal.shape)


def bit_llr_from_received(y: np.ndarray, sigma) -> np.ndarray:
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    if sigma_arr.ndim == 0:
        return 2.0 * y / (float(sigma_arr) ** 2)
    return 2.0 * y / (sigma_arr ** 2)


def hard_decisions_from_y(y: np.ndarray) -> np.ndarray:
    return (y < 0).astype(np.uint8)


def symbol_logposterior(y: np.ndarray, sigma, n_symbols: int,
                         bitorder: str = "big") -> np.ndarray:
    """Per-symbol log-posterior over GF(2^8). sigma scalar or per-sample array."""
    y = np.asarray(y, dtype=np.float64)
    if y.size != 8 * n_symbols:
        raise ValueError(f"y has {y.size} samples, expected {8*n_symbols}")
    y_mat = y.reshape(n_symbols, 8)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    if sigma_arr.ndim == 0:
        sigma_mat = sigma_arr
    else:
        if sigma_arr.size != y.size:
            raise ValueError(f"sigma has {sigma_arr.size} entries, expected {y.size}")
        sigma_mat = sigma_arr.reshape(n_symbols, 8)
    L = 2.0 * y_mat / (sigma_mat ** 2)
    log_p0 = -np.logaddexp(0.0, -L)
    log_p1 = -np.logaddexp(0.0,  L)

    vals = np.arange(256, dtype=np.uint8)
    bit_pat = np.unpackbits(vals[:, None], axis=1, bitorder=bitorder)
    log_pj = np.stack([log_p0, log_p1], axis=-1)

    out = np.zeros((n_symbols, 256), dtype=np.float64)
    for j in range(8):
        out += log_pj[:, j, bit_pat[:, j]]
    return out


def hard_decision_symbols(symbol_logp: np.ndarray) -> np.ndarray:
    return np.argmax(symbol_logp, axis=1).astype(np.uint8)


__all__ = [
    "ebn0_db_to_sigma",
    "bytes_to_bits", "bits_to_bytes",
    "bpsk_modulate", "awgn", "gilbert_elliott_channel",
    "bit_llr_from_received", "hard_decisions_from_y",
    "symbol_logposterior", "hard_decision_symbols",
]
