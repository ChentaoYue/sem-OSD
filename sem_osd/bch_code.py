"""Binary BCH(n, k) wrapper using `galois`."""
from __future__ import annotations
import numpy as np


def _to_bits_array(x) -> np.ndarray:
    if hasattr(x, '__array__'):
        x = np.asarray(x).astype(np.uint8) & 1
    else:
        x = np.array(list(x), dtype=np.uint8) & 1
    return x


class BCHCode:
    def __init__(self, n: int = 127, k: int = 64):
        try:
            import galois
        except ImportError as e:
            raise ImportError("Install galois: pip install galois") from e
        self._galois = galois
        self.n = int(n); self.k = int(k)
        self.bch = galois.BCH(n, k)
        self.d_min = int(self.bch.d)
        self.t = (self.d_min - 1) // 2
        self.G = np.array(self.bch.G, dtype=np.uint8) & 1
        self.H = np.array(self.bch.H, dtype=np.uint8) & 1
        if self.G.shape != (self.k, self.n):
            raise RuntimeError(f"Bad G shape {self.G.shape}")
        if self.H.shape != (self.n - self.k, self.n):
            raise RuntimeError(f"Bad H shape {self.H.shape}")
        check = (self.G @ self.H.T) % 2
        if np.any(check):
            raise RuntimeError("G H^T != 0")
        if np.array_equal(self.G[:, :self.k], np.eye(self.k, dtype=np.uint8)):
            self._sys_layout = 'left'
        elif np.array_equal(self.G[:, -self.k:], np.eye(self.k, dtype=np.uint8)):
            self._sys_layout = 'right'
        else:
            self._sys_layout = 'nonsystem'

    @property
    def info_positions(self) -> np.ndarray:
        if self._sys_layout == 'left':
            return np.arange(self.k)
        elif self._sys_layout == 'right':
            return np.arange(self.n - self.k, self.n)
        else:
            return np.arange(self.k)

    def encode(self, msg_bits) -> np.ndarray:
        msg_bits = _to_bits_array(msg_bits)
        if msg_bits.size != self.k:
            raise ValueError(f"msg has {msg_bits.size} bits, expected {self.k}")
        cw = (msg_bits @ self.G) & 1
        return cw.astype(np.uint8)

    def encode_bytes(self, msg_bytes) -> np.ndarray:
        msg_bytes = np.asarray(bytearray(msg_bytes) if not isinstance(msg_bytes, np.ndarray)
                                else msg_bytes, dtype=np.uint8)
        if msg_bytes.size * 8 != self.k:
            raise ValueError(f"need {self.k//8} bytes, got {msg_bytes.size}")
        bits = np.unpackbits(msg_bytes, bitorder='big').astype(np.uint8)
        return self.encode(bits)

    def bm_decode(self, rcv_bits) -> tuple[bool, np.ndarray]:
        """BM decode via galois with errors=True mode for proper failure detection."""
        rcv_bits = _to_bits_array(rcv_bits)
        if rcv_bits.size != self.n:
            raise ValueError(f"received has {rcv_bits.size} bits, expected {self.n}")
        gf2 = self._galois.GF2
        try:
            result = self.bch.decode(gf2(rcv_bits), errors=True)
            if isinstance(result, tuple) and len(result) == 2:
                decoded_full, n_err = result
                if int(n_err) < 0:
                    return False, rcv_bits[self.info_positions].copy()
            else:
                decoded_full = result
            decoded_arr = np.array(decoded_full, dtype=np.uint8) & 1
            if decoded_arr.size == self.k:
                return True, decoded_arr
            elif decoded_arr.size == self.n:
                return True, decoded_arr[self.info_positions]
            else:
                return True, decoded_arr[:self.k]
        except Exception:
            return False, rcv_bits[self.info_positions].copy()

    def syndrome(self, rcv_bits) -> np.ndarray:
        rcv_bits = _to_bits_array(rcv_bits)
        return (rcv_bits @ self.H.T) & 1

    def is_codeword(self, rcv_bits) -> bool:
        return bool(np.all(self.syndrome(rcv_bits) == 0))


__all__ = ["BCHCode"]
