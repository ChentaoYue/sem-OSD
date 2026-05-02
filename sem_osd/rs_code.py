"""
Shortened RS(n=80, k=64) over GF(2^8) for the BG-RSD paper.

We use the `reedsolo` library for systematic encoding and BM decoding.
We additionally extract the (k x n) generator matrix G (in systematic form
[I_k | P]) for use by OSD over GF(256).

API:
    code = RSCode(n=80, k=64)
    cw  = code.encode(msg_bytes)         # bytes/np.uint8 of length k -> length-n np.uint8
    ok, dec = code.bm_decode(rcv_bytes)  # length-n np.uint8 input; returns (success, length-k message)
    G   = code.G                          # (k, n) np.uint8 systematic generator matrix
    H   = code.H                          # (n-k, n) parity-check matrix
    ok, dec_full = code.bm_decode_full(rcv_bytes)  # returns (success, length-n decoded codeword)
"""
from __future__ import annotations

import numpy as np

import reedsolo as _rs

from .gf256 import matmul_gf256, vecmat_gf256, LOG, EXP


class RSCode:
    def __init__(self, n: int = 80, k: int = 64):
        if n < k:
            raise ValueError("n must be >= k")
        if n > 255:
            raise ValueError("n must be <= 255 for byte-level RS over GF(2^8)")
        self.n = n
        self.k = k
        self.nsym = n - k  # number of parity symbols, t = nsym // 2
        self.t = self.nsym // 2

        # Configure reedsolo. Default uses prim_poly=0x11D, generator=2, fcr=0.
        # We reset RS lookup tables to the default to be safe.
        _rs.init_tables(c_exp=8, prim=0x11D, generator=2)
        self._rsc = _rs.RSCodec(self.nsym, nsize=n)

        # Build generator and parity-check matrices.
        self.G = self._build_generator_matrix()
        self.H = self._build_parity_check_matrix()

    # ---- core encode / decode ----
    def encode(self, msg) -> np.ndarray:
        """Systematic encoding of a length-k message into a length-n codeword.

        msg: bytes / bytearray / 1-D np.uint8 of length k.
        """
        msg = np.asarray(bytearray(msg) if not isinstance(msg, (bytes, bytearray, np.ndarray)) else msg, dtype=np.uint8)
        if msg.size != self.k:
            raise ValueError(f"message length {msg.size} != k={self.k}")
        # Use generator matrix multiplication so we are consistent with G
        cw = vecmat_gf256(msg, self.G)
        return cw

    def encode_via_lib(self, msg) -> np.ndarray:
        """Reference encoding via reedsolo (kept for cross-checking)."""
        msg = bytes(bytearray(msg))
        if len(msg) != self.k:
            raise ValueError(f"message length {len(msg)} != k={self.k}")
        cw = self._rsc.encode(msg)  # bytearray of length n
        return np.frombuffer(bytes(cw), dtype=np.uint8)

    def bm_decode(self, rcv: np.ndarray) -> tuple[bool, np.ndarray]:
        """Berlekamp-Massey decoding. Returns (success, decoded message) of length k.

        On failure, returns (False, hard-decision msg estimate from rcv[:k]).
        """
        rcv = np.asarray(rcv, dtype=np.uint8)
        if rcv.size != self.n:
            raise ValueError(f"rcv length {rcv.size} != n={self.n}")
        try:
            decoded_msg, _decoded_full, _errata = self._rsc.decode(bytes(rcv.tolist()))
            decoded_msg = np.frombuffer(bytes(decoded_msg), dtype=np.uint8)
            if decoded_msg.size != self.k:
                # Library may sometimes include the codeword; trim
                decoded_msg = decoded_msg[: self.k]
            return True, decoded_msg
        except _rs.ReedSolomonError:
            return False, rcv[: self.k].copy()

    def bm_decode_full(self, rcv: np.ndarray) -> tuple[bool, np.ndarray]:
        """BM decode and re-encode to obtain the full length-n codeword."""
        ok, msg = self.bm_decode(rcv)
        if not ok:
            return False, rcv.copy()
        return True, self.encode(msg)

    # ---- syndrome / parity check ----
    def syndrome(self, rcv: np.ndarray) -> np.ndarray:
        """Compute the (n-k)-symbol syndrome of received vector rcv. Zero iff in code."""
        rcv = np.asarray(rcv, dtype=np.uint8)
        return vecmat_gf256(rcv, self.H.T)  # because H @ rcv^T -> use rcv @ H^T

    def is_codeword(self, rcv: np.ndarray) -> bool:
        return bool(np.all(self.syndrome(rcv) == 0))

    # ---- internal: build matrices ----
    def _build_generator_matrix(self) -> np.ndarray:
        """Build (k, n) systematic generator matrix by encoding standard basis."""
        G = np.zeros((self.k, self.n), dtype=np.uint8)
        for i in range(self.k):
            msg = bytearray(self.k)
            msg[i] = 1
            cw = self._rsc.encode(bytes(msg))
            G[i, :] = np.frombuffer(bytes(cw), dtype=np.uint8)
        # Verify systematic structure: first k columns should form identity
        if not np.array_equal(G[:, : self.k], np.eye(self.k, dtype=np.uint8)):
            # Some RS implementations use prepended-parity systematic form.
            # In reedsolo, output is [msg | parity], so the first k cols should be I.
            raise RuntimeError("Generator matrix is not in systematic [I|P] form; "
                                "check reedsolo version / nsize argument.")
        return G

    def _build_parity_check_matrix(self) -> np.ndarray:
        """Given G = [I_k | P], H = [P^T | I_{n-k}].

        Verifies G @ H^T == 0 (mod 256).
        """
        P = self.G[:, self.k :]            # (k, n-k)
        H = np.zeros((self.n - self.k, self.n), dtype=np.uint8)
        H[:, : self.k] = P.T               # (n-k, k)
        H[:, self.k :] = np.eye(self.n - self.k, dtype=np.uint8)
        # Sanity: G @ H^T = 0
        check = matmul_gf256(self.G, H.T)
        if not np.all(check == 0):
            raise RuntimeError("Generator/parity-check inconsistency over GF(256)")
        return H


__all__ = ["RSCode"]
