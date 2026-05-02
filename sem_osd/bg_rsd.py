"""
BG-RSD decoder: ByT5-Guided RS Decoder.
"""
from __future__ import annotations

from itertools import combinations, product
from typing import Optional

import numpy as np

from .channel import symbol_logposterior, hard_decision_symbols
from .gf256 import gauss_elim_select_columns, vecmat_gf256, vec_mul_scalar
from .rs_code import RSCode


class BGRSD:
    def __init__(self, code, byt5_prior=None,
                 alpha=0.5, top_T=4, order_W=2,
                 mrb_mode="standard",
                 bit_flip_aug=0,
                 verbose=False):
        """ByT5-Guided RS Decoder.

        Args:
            code: RSCode instance
            byt5_prior: ByT5Prior instance (or None for channel-only OSD baseline)
            alpha: weight on channel log-posterior; ByT5 weight = 1 - alpha
            top_T: per-MRB-position number of candidate symbol values
            order_W: maximum OSD order
            mrb_mode: 'standard' (reliability-based MRB, may include parity)
                       or 'info_only' (force MRB = info positions only;
                       ByT5 always provides candidates at every MRB pos)
            bit_flip_aug: 0 = no bit-flip augmentation (default),
                           1 = add 1-bit Hamming-distance neighbours of u0
                                and hd_byte to the candidate set,
                           2 = also add 2-bit neighbours.
                           Helps recover from "1-bit-error-not-in-LM-top-T"
                           failures at high SNR.
        """
        self.code = code
        self.byt5 = byt5_prior
        self.alpha = float(alpha)
        self.T = int(top_T)
        self.W = int(order_W)
        if mrb_mode not in ("standard", "info_only"):
            raise ValueError(f"mrb_mode must be 'standard' or 'info_only', got {mrb_mode!r}")
        self.mrb_mode = mrb_mode
        self.bit_flip_aug = int(bit_flip_aug)
        self.verbose = bool(verbose)

    def decode(self, y, sigma, context_text="", invoke_byt5=True):
        # Dispatch on code family. RS over GF(2^8) is symbol-level throughout;
        # binary BCH stores its codeword at bit level but the information word
        # is byte-aligned, so we run a byte-level OSD on the k = k_b/8 info
        # bytes and score candidates against a bit-level fused log-posterior.
        if hasattr(self.code, 'bch'):
            return self._decode_bch(y, sigma, context_text, invoke_byt5)

        n = self.code.n
        k = self.code.k

        # ---- 1. channel posterior + hard decision ----
        ch_logp = symbol_logposterior(y, sigma, n_symbols=n)
        hd = hard_decision_symbols(ch_logp)

        # ---- 2. fast path: BM decode ----
        ok, decoded_msg = self.code.bm_decode(hd)
        if ok:
            cw = self.code.encode(decoded_msg)
            return {'msg': decoded_msg, 'codeword': cw,
                    'method': 'bm', 'invoked_byt5': False, 'tep_count': 1, 'success': True}

        # ---- 3. ByT5 prior ----
        byt5_used = False
        byt5_logp = None
        if invoke_byt5 and self.byt5 is not None:
            byt5_post = self.byt5.posterior(hd[:k], context_text=context_text)
            byt5_logp = np.log(np.clip(byt5_post, 1e-12, 1.0))
            byt5_used = True

        # ---- 4. fuse ----
        L = self._fuse(ch_logp, byt5_logp, k, alpha=self.alpha)

        # ---- 5. MRB selection ----
        sort_idx = np.argsort(-L, axis=1)
        L_top1 = np.take_along_axis(L, sort_idx[:, :1], axis=1).flatten()
        L_top2 = np.take_along_axis(L, sort_idx[:, 1:2], axis=1).flatten()
        margins = L_top1 - L_top2

        if self.mrb_mode == "info_only":
            # Force MRB = info positions; sorted by margin within info & parity blocks.
            info_order = np.argsort(-margins[:k])
            parity_order = np.argsort(-margins[k:]) + k
            col_order = np.concatenate([info_order, parity_order])
        else:
            col_order = np.argsort(-margins)

        # ---- 6. GE on permuted G ----
        Gs, perm = gauss_elim_select_columns(self.code.G, col_order)
        L_perm = L[perm]
        hd_perm = hd[perm]

        # u0 = fused MAP at MRB positions (with info_only mode: pure ByT5+ch on info)
        u0 = np.argmax(L_perm[:k], axis=1).astype(np.uint8)

        # ---- 7. candidate sets per MRB position ----
        # Each position contributes up to (T-1) fused-posterior top candidates,
        # plus optional bit-flip neighbours of u0 and hd_perm to guarantee that
        # small-Hamming-distance variants are always tried (covers high-SNR
        # "1-bit error not in LM top-T" failures).
        sort_idx_perm = np.argsort(-L_perm, axis=1)
        candidates_by_pos = []
        for i in range(k):
            cands_set = set()
            # (a) fused-posterior top-(T-1)
            for c in sort_idx_perm[i]:
                ic = int(c)
                if ic == int(u0[i]):
                    continue
                cands_set.add(ic)
                if len(cands_set) >= max(0, self.T - 1):
                    break
            # (b) bit-flip augmentation
            if self.bit_flip_aug >= 1:
                bases = {int(u0[i]), int(hd_perm[i])}
                for base in bases:
                    for b in range(8):
                        cands_set.add(base ^ (1 << b))
                if self.bit_flip_aug >= 2:
                    for base in bases:
                        for b1 in range(8):
                            for b2 in range(b1 + 1, 8):
                                cands_set.add(base ^ (1 << b1) ^ (1 << b2))
                cands_set.discard(int(u0[i]))   # u0 is base, not a candidate
            candidates_by_pos.append(np.array(sorted(cands_set), dtype=np.uint8))

        base_cw = vecmat_gf256(u0, Gs)

        # Pre-compute single-position deltas
        single_deltas = {}
        for i in range(k):
            for v in candidates_by_pos[i]:
                scalar = int(v) ^ int(u0[i])
                single_deltas[(i, int(v))] = vec_mul_scalar(Gs[i], scalar)

        idx_n = np.arange(n)
        def cw_distance(cw_perm):
            return -float(np.sum(L_perm[idx_n, cw_perm]))

        best_cw_perm = base_cw.copy()
        best_dist = cw_distance(best_cw_perm)
        tep_count = 1

        for w in range(1, self.W + 1):
            for S in combinations(range(k), w):
                cand_lists = [candidates_by_pos[i] for i in S]
                if any(len(c) == 0 for c in cand_lists):
                    continue
                for vals in product(*cand_lists):
                    cw_perm = base_cw.copy()
                    for pos, v in zip(S, vals):
                        cw_perm = np.bitwise_xor(cw_perm, single_deltas[(int(pos), int(v))])
                    d = cw_distance(cw_perm)
                    tep_count += 1
                    if d < best_dist:
                        best_dist = d
                        best_cw_perm = cw_perm

        # ---- 8. un-permute ----
        inv_perm = np.argsort(perm)
        best_cw = best_cw_perm[inv_perm]
        is_cw = self.code.is_codeword(best_cw)
        decoded_msg = best_cw[: self.code.k].copy()

        return {'msg': decoded_msg, 'codeword': best_cw,
                'method': 'osd_byt5' if byt5_used else 'osd',
                'invoked_byt5': byt5_used,
                'tep_count': tep_count,
                'success': bool(is_cw)}

    @staticmethod
    def _fuse(ch_logp, byt5_logp, k, alpha=None):
        L = ch_logp.copy()
        L = L - L.max(axis=1, keepdims=True)
        if byt5_logp is None:
            return L
        a = 0.5 if alpha is None else alpha
        Lb = byt5_logp - byt5_logp.max(axis=1, keepdims=True)
        L[:k] = a * L[:k] + (1.0 - a) * Lb
        return L

    # ------------------------------------------------------------------ #
    #  BCH branch: byte-level OSD on the k_b/8 information bytes, with   #
    #  bit-level scoring against the fused channel--LM log-posterior.    #
    # ------------------------------------------------------------------ #
    def _decode_bch(self, y, sigma, context_text="", invoke_byt5=True):
        n_b = int(self.code.n)            # codeword length in bits
        k_b = int(self.code.k)            # info length in bits
        if k_b % 8 != 0:
            raise ValueError(f"BGRSD on BCH requires k_b divisible by 8; got k_b={k_b}")
        k_byte = k_b // 8                  # number of info bytes
        G_b = np.asarray(self.code.G, dtype=np.uint8) & 1  # (k_b, n_b)

        # Codeword positions of the information bits, regardless of whether
        # the systematic G has 'left', 'right', or non-trivial layout.
        info_pos = np.asarray(self.code.info_positions, dtype=np.int64)
        if info_pos.size != k_b:
            raise RuntimeError(f"info_positions has {info_pos.size} entries, "
                                  f"expected k_b={k_b}")

        # ---- bit-level channel observations and posteriors ----
        y = np.asarray(y, dtype=np.float64)
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        if sigma_arr.ndim == 0:
            llr = 2.0 * y / (float(sigma_arr) ** 2)
        else:
            llr = 2.0 * y / (sigma_arr ** 2)
        if llr.size > n_b:
            llr = llr[: n_b]
        log_p0 = -np.logaddexp(0.0, -llr)        # (n_b,)
        log_p1 = -np.logaddexp(0.0,  llr)        # (n_b,)
        hd_bits = (llr < 0).astype(np.uint8)     # (n_b,)

        # ---- BM fast path on the bit-level hard decision ----
        ok, msg_bits = self.code.bm_decode(hd_bits)
        if ok:
            msg_bits = np.asarray(msg_bits, dtype=np.uint8)[: k_b]
            decoded_msg_bytes = np.packbits(msg_bits, bitorder='big')[: k_byte]
            cw_bits = self.code.encode(msg_bits)
            return {'msg': decoded_msg_bytes, 'codeword': cw_bits,
                    'method': 'bm', 'invoked_byt5': False,
                    'tep_count': 1, 'success': True}

        # ---- byte-level channel log-posterior on info bytes ----
        # Each info byte i covers info bits 8i..8i+7, sitting at codeword
        # positions info_pos[8i..8i+7].
        bytes_arr = np.arange(256, dtype=np.uint8)
        byte_bit_pat = np.unpackbits(bytes_arr[:, None], axis=1,
                                          bitorder='big')   # (256, 8)
        ch_byte_logp = np.zeros((k_byte, 256), dtype=np.float64)
        for i in range(k_byte):
            for j in range(8):
                ell = int(info_pos[8 * i + j])
                bit_j = byte_bit_pat[:, j]               # (256,)
                ch_byte_logp[i] += np.where(bit_j == 0, log_p0[ell], log_p1[ell])

        # ---- LM byte posterior on info bytes ----
        hd_info_bits = hd_bits[info_pos]                 # (k_b,) info bit HD
        hd_info_bytes = np.packbits(hd_info_bits, bitorder='big')[: k_byte]
        byt5_used = False
        byt5_logp_byte = None
        if invoke_byt5 and self.byt5 is not None:
            byt5_post = self.byt5.posterior(hd_info_bytes,
                                                context_text=context_text)
            byt5_logp_byte = np.log(np.clip(byt5_post, 1e-12, 1.0))
            byt5_used = True

        # ---- fused byte score on info bytes ----
        L_byte = self._fuse(ch_byte_logp, byt5_logp_byte, k_byte, alpha=self.alpha)

        # ---- top-T candidate set per byte position ----
        u0_bytes = np.argmax(L_byte, axis=1).astype(np.uint8)
        sort_idx = np.argsort(-L_byte, axis=1)
        candidates_by_pos = []
        for i in range(k_byte):
            cands_set = set()
            for c in sort_idx[i]:
                ic = int(c)
                if ic == int(u0_bytes[i]):
                    continue
                cands_set.add(ic)
                if len(cands_set) >= max(0, self.T - 1):
                    break
            if self.bit_flip_aug >= 1:
                bases = {int(u0_bytes[i]), int(hd_info_bytes[i])}
                for base in bases:
                    for b in range(8):
                        cands_set.add(base ^ (1 << b))
                if self.bit_flip_aug >= 2:
                    for base in bases:
                        for b1 in range(8):
                            for b2 in range(b1 + 1, 8):
                                cands_set.add(base ^ (1 << b1) ^ (1 << b2))
                cands_set.discard(int(u0_bytes[i]))
            candidates_by_pos.append(np.array(sorted(cands_set), dtype=np.uint8))

        # ---- base codeword (bit-level) and per-position byte-deltas ----
        base_cw_bits = np.asarray(self.code.encode_bytes(u0_bytes), dtype=np.uint8)
        single_deltas_bits = {}
        for i in range(k_byte):
            for v in candidates_by_pos[i]:
                xor = int(v) ^ int(u0_bytes[i])
                delta_bits = np.zeros(n_b, dtype=np.uint8)
                for r in range(8):
                    if (xor >> (7 - r)) & 1:
                        delta_bits = np.bitwise_xor(delta_bits, G_b[8 * i + r])
                single_deltas_bits[(i, int(v))] = delta_bits

        # ---- bit-level fused score for candidate scoring ----
        # Channel bit log-posterior, row-normalised so that max is 0.
        ch_bit = np.column_stack([log_p0, log_p1])               # (n_b, 2)
        ch_bit = ch_bit - ch_bit.max(axis=1, keepdims=True)
        fused_bit = ch_bit.copy()
        if byt5_used and byt5_logp_byte is not None:
            # Marginalise the LM byte posterior to a per-bit log-posterior on
            # information bits, indexed by info-bit-index 0..k_b-1.
            byt5_post = np.exp(byt5_logp_byte)                   # (k_byte, 256)
            lm_bit = np.zeros((k_b, 2), dtype=np.float64)
            for ell in range(k_b):
                i = ell // 8; j = ell % 8
                mask0 = byte_bit_pat[:, j] == 0
                p0 = float(np.sum(byt5_post[i, mask0]))
                p1 = 1.0 - p0
                lm_bit[ell, 0] = np.log(max(p0, 1e-12))
                lm_bit[ell, 1] = np.log(max(p1, 1e-12))
            lm_bit = lm_bit - lm_bit.max(axis=1, keepdims=True)
            # Apply LM evidence to the actual codeword positions of the info
            # bits, which may be non-contiguous depending on G's layout.
            fused_bit[info_pos] = (self.alpha * fused_bit[info_pos]
                                      + (1.0 - self.alpha) * lm_bit)

        idx_n = np.arange(n_b)
        def cw_distance(cw_bits):
            return -float(np.sum(fused_bit[idx_n, cw_bits]))

        best_cw_bits = base_cw_bits.copy()
        best_dist = cw_distance(best_cw_bits)
        tep_count = 1

        # ---- enumerate up to W byte substitutions ----
        for w in range(1, self.W + 1):
            for S in combinations(range(k_byte), w):
                cand_lists = [candidates_by_pos[i] for i in S]
                if any(len(c) == 0 for c in cand_lists):
                    continue
                for vals in product(*cand_lists):
                    cw_bits = base_cw_bits.copy()
                    for pos, v in zip(S, vals):
                        cw_bits = np.bitwise_xor(
                            cw_bits, single_deltas_bits[(int(pos), int(v))])
                    d = cw_distance(cw_bits)
                    tep_count += 1
                    if d < best_dist:
                        best_dist = d
                        best_cw_bits = cw_bits

        # Extract decoded info bytes via info_pos (works under any layout).
        decoded_info_bits = best_cw_bits[info_pos]
        decoded_msg_bytes = np.packbits(decoded_info_bits,
                                            bitorder='big')[: k_byte]
        return {'msg': decoded_msg_bytes, 'codeword': best_cw_bits,
                'method': 'osd_byt5' if byt5_used else 'osd',
                'invoked_byt5': byt5_used,
                'tep_count': tep_count, 'success': True}


__all__ = ["BGRSD"]
