"""Binary OSD on the bit expansion of a byte-aligned linear code, with optional ByT5 fusion.

Supports two code families through a thin uniform wrapper:
  * RS over GF(2^8)        : info length k = number of GF(2^8) symbols (= bytes)
  * Binary BCH(n, k)       : info length k must be a multiple of 8

Both yield the same Sem-OSD-Hybrid procedure, with a code-specific byte-delta
computation:
  RS  : delta_c[i] = (v XOR m_j) * G_RS[j, i]   (GF(256) scalar mult)
  BCH : delta_c[i] = sum_{r: bit_r(v XOR m_j)} G_b[8j+r, i]   (binary row-XOR)
"""
from __future__ import annotations
from itertools import combinations, product
from math import comb as math_comb
from typing import Tuple
import numpy as np
from .gf256 import EXP, gf_mul, vec_mul_scalar

try:
    from scipy.special import erfc as _scipy_erfc
    def _erfc(x):
        return _scipy_erfc(x)
except Exception:
    # Fallback: numerically stable erfc via numpy (loses precision for large |x|)
    from math import erfc as _math_erfc
    def _erfc(x):
        x_arr = np.asarray(x, dtype=np.float64)
        return np.vectorize(_math_erfc)(x_arr)


def _phi_norm(z):
    """Standard-normal CDF Phi(z) = 1 - 0.5*erfc(z/sqrt(2))."""
    return 1.0 - 0.5 * _erfc(z / np.sqrt(2.0))


def rs_to_binary_generator(G_rs: np.ndarray, m: int = 8) -> np.ndarray:
    """Big-endian bit expansion of a GF(2^m) generator matrix."""
    G_rs = np.asarray(G_rs, dtype=np.uint8)
    k_s, n_s = G_rs.shape
    G_b = np.zeros((k_s * m, n_s * m), dtype=np.uint8)
    for i in range(k_s):
        for j in range(n_s):
            alpha = int(G_rs[i, j])
            for r in range(m):
                if alpha == 0:
                    val = 0
                else:
                    val = gf_mul(alpha, int(EXP[m - 1 - r]))
                for c in range(m):
                    G_b[i * m + r, j * m + c] = (val >> (m - 1 - c)) & 1
    return G_b


def binary_ge(G: np.ndarray, col_order: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    G = np.array(G, dtype=np.uint8) & 1
    k, n = G.shape
    perm = np.array(col_order, dtype=np.int64).copy()
    M = G[:, perm].copy()
    for r in range(k):
        pivot = -1
        for rr in range(r, k):
            if M[rr, r] == 1:
                pivot = rr; break
        if pivot < 0:
            swap = -1
            for j in range(k, n):
                if np.any(M[r:, j]):
                    swap = j; break
            if swap < 0:
                raise RuntimeError("binary_ge: rank-deficient")
            M[:, [r, swap]] = M[:, [swap, r]]
            perm[[r, swap]] = perm[[swap, r]]
            for rr in range(r, k):
                if M[rr, r] == 1:
                    pivot = rr; break
        if pivot != r:
            M[[r, pivot]] = M[[pivot, r]]
        for rr in range(k):
            if rr == r: continue
            if M[rr, r] == 1:
                M[rr] ^= M[r]
    return M, perm


class _CodeAdapter:
    """Uniform interface over RS-over-GF(2^8) and binary BCH for Sem-OSD-Hybrid.

    Exposes:
      kind ('rs' | 'bch'), k_b, n_b, G_b (binary, k_b x n_b), k_byte (= k_b // 8),
      bm_decode_bits(hd_bits) -> (ok, msg_bytes [length k_byte]),
      encode_bytes(msg_bytes [length k_byte]) -> cw_bits [length n_b],
      byte_delta_bits(byte_pos, byte_xor) -> delta_bits [length n_b]
    """

    def __init__(self, code):
        # Detect code family by attribute presence (avoids cyclic imports)
        is_bch = hasattr(code, 'bch') and hasattr(code, 'H') and not hasattr(code, 't')\
                  or (hasattr(code, 'd_min') and hasattr(code, 'bch'))
        # Robust check: BCHCode uses bit-level G of shape (k, n), RSCode uses GF(256) G of shape (k, n) too.
        # The actual distinguishing feature: BCHCode has self.bch (galois.BCH instance),
        # RSCode has self._rsc (reedsolo.RSCodec).
        if hasattr(code, '_rsc'):
            self.kind = 'rs'
            self._code = code
            self.k_b = 8 * code.k
            self.n_b = 8 * code.n
            self.k_byte = code.k
            self.G_b = rs_to_binary_generator(code.G, m=8)
        elif hasattr(code, 'bch'):
            self.kind = 'bch'
            self._code = code
            if int(code.k) % 8 != 0:
                raise ValueError(f"BCH info length {code.k} bits is not byte-aligned")
            self.k_b = int(code.k)
            self.n_b = int(code.n)
            self.k_byte = self.k_b // 8
            self.G_b = np.asarray(code.G, dtype=np.uint8) & 1
        else:
            raise TypeError(f"Unknown code type {type(code).__name__!r}; "
                            f"need RSCode (with ._rsc) or BCHCode (with .bch).")
        if self.G_b.shape != (self.k_b, self.n_b):
            raise RuntimeError(f"G_b shape {self.G_b.shape} != ({self.k_b}, {self.n_b})")

    def bm_decode_bits(self, hd_bits):
        if self.kind == 'rs':
            hd_bytes = np.packbits(hd_bits, bitorder='big')
            if hd_bytes.size != self._code.n:
                hd_bytes = hd_bytes[: self._code.n]
            ok, msg = self._code.bm_decode(hd_bytes)
            return ok, (msg if ok else np.zeros(self.k_byte, dtype=np.uint8))
        else:
            ok, msg_bits = self._code.bm_decode(hd_bits[: self.n_b])
            if not ok:
                return False, np.zeros(self.k_byte, dtype=np.uint8)
            msg_bytes = np.packbits(np.asarray(msg_bits[: self.k_b], dtype=np.uint8),
                                       bitorder='big')[: self.k_byte]
            return True, msg_bytes

    def encode_bytes(self, msg_bytes):
        msg_bytes = np.asarray(msg_bytes, dtype=np.uint8)
        if self.kind == 'rs':
            cw_bytes = self._code.encode(msg_bytes)
            cw_bits = np.unpackbits(cw_bytes, bitorder='big')[: self.n_b]
            return cw_bits.astype(np.uint8)
        else:
            msg_bits = np.unpackbits(msg_bytes, bitorder='big')[: self.k_b]
            return self._code.encode(msg_bits).astype(np.uint8)

    def byte_delta_bits(self, byte_pos, byte_xor):
        """Codeword bit-delta when info byte at position `byte_pos` XORs by `byte_xor`."""
        byte_xor = int(byte_xor) & 0xff
        if self.kind == 'rs':
            delta_bytes = vec_mul_scalar(self._code.G[byte_pos], byte_xor)
            delta_bits = np.unpackbits(delta_bytes, bitorder='big')[: self.n_b]
            return delta_bits.astype(np.uint8)
        else:
            delta_bits = np.zeros(self.n_b, dtype=np.uint8)
            for r in range(8):
                if (byte_xor >> (7 - r)) & 1:
                    delta_bits = np.bitwise_xor(delta_bits, self.G_b[8 * byte_pos + r])
            return delta_bits


class BinaryOSDForRS:
    """Sem-OSD on the bit expansion of a byte-aligned linear code (RS or BCH).

    Modes selected via constructor flags:
      - osd_binary       : pure binary OSD, no LM (byt5_prior=None)
      - sem_osd_binary   : binary OSD + byte->bit-marginalised LM prior in fused
                           reliability ordering & scoring
      - sem_osd_hybrid   : sem_osd_binary + byte-level LM-augmented TEPs
                           (set byte_tep_T > 0 and byte_tep_W > 0)

    The class name is kept for backward compatibility; the implementation now
    works on any (RS, BCH) code with byte-aligned information.
    """

    def __init__(self, code, order_W: int = 4, byt5_prior=None,
                 alpha: float = 0.5,
                 byte_tep_T: int = 0, byte_tep_W: int = 0,
                 pb_accel: bool = False,
                 pb_sucset: float = 0.99,
                 pb_proset: float = 1e-3):
        self.adapter = _CodeAdapter(code)
        self.W = int(order_W)
        self.k_b = self.adapter.k_b
        self.n_b = self.adapter.n_b
        self.k_s = self.adapter.k_byte    # number of info bytes
        # n_s kept for backwards compatibility; meaningless for non-byte-aligned BCH.
        self.n_s = (self.n_b + 7) // 8
        self.G_b = self.adapter.G_b
        self.byt5 = byt5_prior
        self.alpha = float(alpha)
        self.byte_tep_T = int(byte_tep_T)
        self.byte_tep_W = int(byte_tep_W)
        # PB-OSD acceleration (only used on AWGN-style channels; do not enable for GE).
        # Defaults follow the PB-OSD paper: sucset=0.99 stop, proset=1e-3 prune scale.
        self.pb_accel = bool(pb_accel)
        self.pb_sucset = float(pb_sucset)
        self.pb_proset = float(pb_proset)
        bytes_arr = np.arange(256, dtype=np.uint8)
        self._byte_bit_pat = np.unpackbits(bytes_arr[:, None], axis=1,
                                              bitorder='big')   # (256, 8)

    # convenience aliases for older callers
    @property
    def rs(self):
        return self.adapter._code

    def decode(self, y, sigma, context_text: str = "", invoke_byt5: bool = True):
        y = np.asarray(y, dtype=np.float64)
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        if sigma_arr.ndim == 0:
            llr = 2.0 * y / (float(sigma_arr) ** 2)
        else:
            llr = 2.0 * y / (sigma_arr ** 2)
        # Truncate received signal to the codeword length in bits.
        if llr.size > self.n_b:
            llr = llr[: self.n_b]
        hd_bits = (llr < 0).astype(np.uint8)

        # BM fast path
        bm_ok, bm_msg = self.adapter.bm_decode_bits(hd_bits)
        if bm_ok:
            cw_bits = self.adapter.encode_bytes(bm_msg)
            cw_bytes_full = np.packbits(cw_bits, bitorder='big')[: self.n_s]
            return {'msg': bm_msg, 'codeword': cw_bytes_full,
                    'method': 'bm', 'invoked_byt5': False,
                    'tep_count': 1, 'success': True}

        # Per-bit channel log-posterior
        log_p0 = -np.logaddexp(0.0, -llr)
        log_p1 = -np.logaddexp(0.0,  llr)
        ch_logp = np.stack([log_p0, log_p1], axis=1)
        ch_norm = ch_logp - ch_logp.max(axis=1, keepdims=True)

        # Optional ByT5 prior on info bits
        byt5_used = False
        hd_info_bits = hd_bits[: self.k_b]
        hd_info_bytes = np.packbits(hd_info_bits, bitorder='big')[: self.k_s]
        if invoke_byt5 and self.byt5 is not None:
            byt5_bit_post = self.byt5.bit_posterior(hd_info_bytes,
                                                     context_text=context_text)
            byt5_logp = np.log(np.clip(byt5_bit_post, 1e-12, 1.0))
            byt5_norm = byt5_logp - byt5_logp.max(axis=1, keepdims=True)
            ch_norm = ch_norm.copy()
            ch_norm[: self.k_b] = self.alpha * ch_norm[: self.k_b] + \
                                   (1.0 - self.alpha) * byt5_norm
            byt5_used = True

        # Reliability ordering on (possibly fused) log-posterior
        margin = np.abs(ch_norm[:, 0] - ch_norm[:, 1])
        col_order = np.argsort(-margin)
        G_s, perm = binary_ge(self.G_b, col_order)
        log_post_perm = ch_norm[perm]
        u_0 = (log_post_perm[: self.k_b, 1] >
                log_post_perm[: self.k_b, 0]).astype(np.uint8)

        idx_n = np.arange(self.n_b)
        def cw_distance(cw_perm):
            return -float(np.sum(log_post_perm[idx_n, cw_perm]))

        base_cw = (u_0 @ G_s) & 1
        best_cw = base_cw.copy()
        best_d = cw_distance(best_cw)
        tep_count = 1

        # ---- (A) bit-level TEP enumeration over MRB ----
        # When pb_accel is on (intended for AWGN), use PB-OSD-style cheap cutoff +
        # promising-probability prune + Psuc early-termination. The math runs in
        # the *fused* domain (using log_post_perm margins), so the same code path
        # works for plain binary OSD (alpha=1, byt5=None) and Sem-OSD's bit-level
        # part.  On bursty channels (GE) the PB-OSD assumptions break down, so the
        # caller should pass pb_accel=False there.
        if self.pb_accel:
            best_cw, best_d, tep_count = self._tep_search_bit_pb(
                base_cw, G_s, log_post_perm, best_cw, best_d, tep_count)
        else:
            for w in range(1, self.W + 1):
                for S in combinations(range(self.k_b), w):
                    cw_perm = base_cw.copy()
                    for p in S:
                        cw_perm = cw_perm ^ G_s[p]
                    d = cw_distance(cw_perm)
                    tep_count += 1
                    if d < best_d:
                        best_d = d
                        best_cw = cw_perm

        # ---- (B) byte-level LM-augmented TEPs (Hybrid mode) ----
        hybrid_used = False
        if (byt5_used and self.byte_tep_T > 0 and self.byte_tep_W > 0):
            hybrid_used = True
            byt5_byte_post = self.byt5.posterior(hd_info_bytes,
                                                   context_text=context_text)
            byt5_byte_logp = np.log(np.clip(byt5_byte_post, 1e-12, 1.0))   # (k_s, 256)

            ch_byte_logp = np.zeros((self.k_s, 256), dtype=np.float64)
            for j in range(self.k_s):
                for r in range(8):
                    bi = 8 * j + r
                    bit_r_of_b = self._byte_bit_pat[:, r]
                    ch_byte_logp[j] += np.where(bit_r_of_b == 0,
                                                  log_p0[bi], log_p1[bi])


            ch_byte_norm = ch_byte_logp - ch_byte_logp.max(axis=1, keepdims=True)
            byt5_byte_norm = byt5_byte_logp - byt5_byte_logp.max(axis=1, keepdims=True)
            fused_byte = (self.alpha * ch_byte_norm
                           + (1.0 - self.alpha) * byt5_byte_norm)

            u_0_bytes = np.argmax(fused_byte, axis=1).astype(np.uint8)

            sort_idx_byte = np.argsort(-fused_byte, axis=1)
            cands_per_pos = []
            for j in range(self.k_s):
                cs = []
                for c in sort_idx_byte[j]:
                    ic = int(c)
                    if ic == int(u_0_bytes[j]):
                        continue
                    cs.append(ic)
                    if len(cs) >= self.byte_tep_T:
                        break
                cands_per_pos.append(np.array(cs, dtype=np.uint8))

            base_cw_bits = self.adapter.encode_bytes(u_0_bytes)
            base_cw_byte_perm = base_cw_bits[perm].astype(np.uint8)

            byte_delta_perm = {}
            for j in range(self.k_s):
                u_j = int(u_0_bytes[j])
                for v in cands_per_pos[j]:
                    iv = int(v)
                    delta_bits = self.adapter.byte_delta_bits(j, iv ^ u_j)
                    byte_delta_perm[(j, iv)] = delta_bits[perm].astype(np.uint8)

            d_base_byte = cw_distance(base_cw_byte_perm)
            tep_count += 1
            if d_base_byte < best_d:
                best_d = d_base_byte
                best_cw = base_cw_byte_perm

            for w in range(1, self.byte_tep_W + 1):
                for S in combinations(range(self.k_s), w):
                    if any(len(cands_per_pos[j]) == 0 for j in S):
                        continue
                    cand_lists = [cands_per_pos[j] for j in S]
                    for vals in product(*cand_lists):
                        cw_hyp_perm = base_cw_byte_perm.copy()
                        for j, v in zip(S, vals):
                            cw_hyp_perm = np.bitwise_xor(
                                cw_hyp_perm, byte_delta_perm[(int(j), int(v))])
                        d_hyp = cw_distance(cw_hyp_perm)
                        tep_count += 1
                        if d_hyp < best_d:
                            best_d = d_hyp
                            best_cw = cw_hyp_perm

        inv_perm = np.argsort(perm)
        best_cw_bits = best_cw[inv_perm]
        cw_bytes_full = np.packbits(best_cw_bits, bitorder='big')[: self.n_s]
        decoded_msg_bytes = np.packbits(best_cw_bits[: self.k_b],
                                            bitorder='big')[: self.k_s].copy()
        if hybrid_used:
            method = 'sem_osd_hybrid'
        elif byt5_used:
            method = 'sem_osd_binary'
        else:
            method = 'osd_binary'
        return {'msg': decoded_msg_bytes, 'codeword': cw_bytes_full,
                'method': method,
                'invoked_byt5': byt5_used,
                'tep_count': tep_count, 'success': True}

    # --------------------------------------------------------------------- #
    # PB-OSD-style accelerated bit-level TEP search.                        #
    # Cheap cutoff + promising-probability prune + Psuc early termination,  #
    # operating on the (possibly LM-fused) bit log-posterior margins.       #
    # --------------------------------------------------------------------- #
    def _tep_search_bit_pb(self, base_cw, G_s, log_post_perm,
                            best_cw, best_d, tep_count):
        # Per-bit reliability margin in MRB ordering. log_post_perm is the
        # row-normalised (and possibly LM-fused) log-posterior with peak 0.
        margin = np.abs(log_post_perm[:, 0] - log_post_perm[:, 1])
        margin = np.clip(margin, 0.0, 60.0)   # avoid overflow in exp(-margin)
        aR0m = margin[: self.k_b]
        aLRB = margin[self.k_b:]

        # Fused-domain bit-flip probability (Bernoulli sigmoid of the margin).
        EPini = 1.0 / (1.0 + np.exp(margin))
        EP_mrb = EPini[: self.k_b]
        EP_lrb = EPini[self.k_b:]

        # Probability that the MRB has at most W errors (PbSuc target).
        PPPord = float(np.mean(EP_mrb)) if self.k_b > 0 else 0.0
        PbSuc = 0.0
        for j in range(self.W + 1):
            PbSuc += math_comb(self.k_b, j) * (PPPord ** j) * \
                       ((1.0 - PPPord) ** (self.k_b - j))
        PbSuc = float(np.clip(PbSuc, 1e-300, 1.0))

        baseEt = float(np.exp(np.sum(np.log(np.clip(1.0 - EP_mrb,
                                                         1e-300, 1.0)))))
        NonEP = float(np.exp(np.sum(np.log(np.clip(1.0 - EPini,
                                                        1e-300, 1.0)))))
        baseEp = NonEP / max(baseEt, 1e-300)

        n_lrb = self.n_b - self.k_b
        EpLRB = float(np.mean(aLRB)) if n_lrb > 0 else 1.0
        EpLRB = max(EpLRB, 1e-6)
        ErrPLRB = float(np.sum(EP_lrb))
        M1 = ErrPLRB
        V1 = max(ErrPLRB * max(1.0 - ErrPLRB / max(n_lrb, 1), 0.0), 1e-6)
        M2 = n_lrb * 0.5
        V2 = max(n_lrb * 0.25, 1e-6)

        Tsz_total = sum(math_comb(self.k_b, j) for j in range(self.W + 1))
        PbPro = self.pb_proset * np.sqrt(max((1.0 - PbSuc) /
                                                  max(Tsz_total, 1), 1e-300))

        idx_n = np.arange(self.n_b)
        minTEPWeight = float('inf')
        minDis = float(best_d)

        early_exit = False
        for w in range(1, self.W + 1):
            if early_exit:
                break
            # Anchor enumeration at the least-reliable MRB positions first so
            # that low-TEPdis subsets are encountered early; this maximises the
            # chance of an early Psuc termination.
            for S in combinations(range(self.k_b - 1, -1, -1), w):
                TEPdis = 0.0
                for p in S:
                    TEPdis += float(aR0m[p])

                # Cheap cutoff: skip if TEPdis already exceeds best or any
                # previously pruned distance (whose Promising was below PbPro).
                if TEPdis >= minDis or TEPdis >= minTEPWeight:
                    continue

                # Promising probability (mixture of TEP-correct vs random).
                Ptep = float(np.exp(-TEPdis) * baseEt)
                Ptep = float(np.clip(Ptep, 0.0, 1.0))
                EstNum = (minDis - TEPdis) / EpLRB
                Promising = (Ptep * _phi_norm((EstNum - M1) / np.sqrt(V1)) +
                              (1.0 - Ptep) * _phi_norm((EstNum - M2) /
                                                          np.sqrt(V2)))
                if Promising < PbPro:
                    if TEPdis < minTEPWeight:
                        minTEPWeight = TEPdis
                    continue

                # Score this TEP.
                cw_perm = base_cw.copy()
                for p in S:
                    cw_perm = cw_perm ^ G_s[p]
                d_curr = -float(np.sum(log_post_perm[idx_n, cw_perm]))
                tep_count += 1

                if d_curr < best_d:
                    best_d = d_curr
                    best_cw = cw_perm
                    minDis = d_curr

                    # Psuc check (PB-OSD eq.): mixture posterior that the
                    # current best is the true codeword.
                    Ppar = float(np.exp(-(d_curr - TEPdis)) * baseEp)
                    num = Ptep * Ppar
                    den = num + (1.0 - Ptep) * (2.0 ** (self.k_b - self.n_b))
                    Psuc = num / max(den, 1e-300)
                    if Psuc > self.pb_sucset * PbSuc:
                        early_exit = True
                        break
        return best_cw, best_d, tep_count


__all__ = ["BinaryOSDForRS", "rs_to_binary_generator", "binary_ge"]
