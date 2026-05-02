"""
GF(2^8) arithmetic with primitive polynomial 0x11D (same as RS used by reedsolo / RS DVB).
Provides:
- log/antilog tables for fast multiply / divide
- gf_mul, gf_div, gf_inv (scalar)
- vector / matrix helpers used by OSD over RS
- Gaussian elimination over GF(256)

Conventions:
- Elements of GF(256) are stored as np.uint8 (0..255).
- Vectors are 1-D np.uint8 arrays; matrices are 2-D np.uint8 arrays of shape (rows, cols).
"""
from __future__ import annotations

import numpy as np

PRIM_POLY = 0x11D  # x^8 + x^4 + x^3 + x^2 + 1 (matches reedsolo default)


def _build_tables(prim_poly: int = PRIM_POLY):
    log_t = np.zeros(256, dtype=np.int32)
    exp_t = np.zeros(512, dtype=np.uint8)  # double-length for easy mod
    x = 1
    for i in range(255):
        exp_t[i] = x
        log_t[x] = i
        x <<= 1
        if x & 0x100:
            x ^= prim_poly
    # extend exp table for fast (a+b) mod 255 indexing
    for i in range(255, 512):
        exp_t[i] = exp_t[i - 255]
    log_t[0] = -1  # undefined, sentinel
    return log_t, exp_t


LOG, EXP = _build_tables()


def gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return int(EXP[(LOG[a] + LOG[b]) % 255])


def gf_div(a: int, b: int) -> int:
    if a == 0:
        return 0
    if b == 0:
        raise ZeroDivisionError("GF(256) division by zero")
    return int(EXP[(LOG[a] - LOG[b]) % 255])


def gf_inv(a: int) -> int:
    if a == 0:
        raise ZeroDivisionError("GF(256) inverse of zero")
    return int(EXP[(255 - LOG[a]) % 255])


# ---------------- vectorized helpers ----------------

def vec_mul_scalar(v: np.ndarray, s: int) -> np.ndarray:
    """Multiply each element of vector v (uint8) by scalar s (int) over GF(256)."""
    if s == 0:
        return np.zeros_like(v)
    out = np.zeros_like(v)
    nz = v != 0
    out[nz] = EXP[(LOG[v[nz].astype(np.int32)] + LOG[s]) % 255]
    return out


def matmul_gf256(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix multiply over GF(256). A: (m, k) uint8, B: (k, n) uint8 -> (m, n) uint8.

    Uses log-tables; handles zeros explicitly.
    """
    A = np.asarray(A, dtype=np.uint8)
    B = np.asarray(B, dtype=np.uint8)
    m, k = A.shape
    k2, n = B.shape
    assert k == k2
    out = np.zeros((m, n), dtype=np.uint8)
    # Precompute logs (use -1 sentinel for zero)
    lA = np.where(A == 0, -1, LOG[A.astype(np.int32)])  # (m,k)
    lB = np.where(B == 0, -1, LOG[B.astype(np.int32)])  # (k,n)
    for i in range(m):
        for j in range(n):
            acc = 0
            for t in range(k):
                la = lA[i, t]
                lb = lB[t, j]
                if la < 0 or lb < 0:
                    continue
                acc ^= int(EXP[(la + lb) % 255])
            out[i, j] = acc
    return out


def vecmat_gf256(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Row-vector v (k,) times matrix M (k, n) over GF(256) -> (n,)."""
    v = np.asarray(v, dtype=np.uint8)
    M = np.asarray(M, dtype=np.uint8)
    k = v.size
    n = M.shape[1]
    out = np.zeros(n, dtype=np.uint8)
    nz = np.where(v != 0)[0]
    if nz.size == 0:
        return out
    lv = LOG[v[nz].astype(np.int32)]  # logs of nonzero entries
    for idx, t in enumerate(nz):
        row = M[t]
        nz2 = row != 0
        if not nz2.any():
            continue
        lr = LOG[row[nz2].astype(np.int32)]
        contrib = EXP[(lv[idx] + lr) % 255]
        out[nz2] ^= contrib
    return out


def gauss_elim_select_columns(A: np.ndarray, col_order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian elimination over GF(256) to bring `k` selected columns of `A` to identity.

    A: (k, n) generator matrix.
    col_order: length-n permutation; we attempt to make first len(col_order) ones systematic.

    For an MDS code (RS) any k columns are linearly independent; we still implement a
    pivot-fallback search that swaps in further columns if a chosen one is dependent.

    Returns:
        Gs: (k, n) matrix whose columns at positions selected_idx form I_k
        final_perm: (n,) actual permutation used (selected k MRB columns first)
    """
    A = np.array(A, dtype=np.uint8)
    k, n = A.shape
    perm = np.array(col_order, dtype=np.int64).copy()
    # Apply current permutation as a working view
    # We'll work on M = A[:, perm], modify rows of M in place.
    M = A[:, perm].copy()
    for r in range(k):
        # Find a pivot in column r at row >= r
        pivot_row = -1
        for rr in range(r, k):
            if M[rr, r] != 0:
                pivot_row = rr
                break
        if pivot_row < 0:
            # Need to swap in another column from positions r+1..n-1 into column r
            # (search for first column j>=k that has any nonzero in rows r..k-1)
            swap_col = -1
            for j in range(k, n):
                col = M[r:, j]
                if np.any(col != 0):
                    swap_col = j
                    break
            if swap_col < 0:
                raise RuntimeError("Matrix is rank-deficient even with column search; should not happen for RS.")
            # swap columns in M and in perm
            M[:, [r, swap_col]] = M[:, [swap_col, r]]
            perm[[r, swap_col]] = perm[[swap_col, r]]
            for rr in range(r, k):
                if M[rr, r] != 0:
                    pivot_row = rr
                    break
        # swap rows
        if pivot_row != r:
            M[[r, pivot_row]] = M[[pivot_row, r]]
        # normalize row r
        piv = int(M[r, r])
        inv = gf_inv(piv)
        M[r] = vec_mul_scalar(M[r], inv)
        # eliminate other rows
        for rr in range(k):
            if rr == r:
                continue
            c = int(M[rr, r])
            if c == 0:
                continue
            M[rr] ^= vec_mul_scalar(M[r], c)
    return M, perm


__all__ = [
    "PRIM_POLY", "LOG", "EXP",
    "gf_mul", "gf_div", "gf_inv",
    "vec_mul_scalar", "matmul_gf256", "vecmat_gf256",
    "gauss_elim_select_columns",
]
