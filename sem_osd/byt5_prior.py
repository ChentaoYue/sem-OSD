"""
ByT5 prior wrapper for the BG-RSD decoder.

Loads the trained `byt5_prefix_ckpt` (ByT5ByteTagger from train_byt5_prefix_group.py)
and exposes:

    P = prior.posterior(noisy_block_text, context_text="")
    # P is shape (k, 256), entries are P_BYT5(c_i = b | context, noisy_block)

Notes
-----
- The model architecture is `ByT5ByteTagger` = T5EncoderModel + Linear(d_model, vocab_size).
- It tokenizes WITHOUT special tokens and outputs a per-token logit over the tokenizer vocab
  (which for ByT5 is byte-level, so token id b in [3..258] corresponds to byte (b-3)).
- We map the (V,) vocab logits to a (256,) GF(256) posterior by selecting only the byte tokens.
- The model was trained on intra-sentence prefix-group noise; we generalise the inference
  pattern to: input = [optional clean cross-sentence context] + [current noisy block].

Public class:
    ByT5Prior(model_dir, base_model_name="google/byt5-small", device=None, fp16=True)
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np


class ByT5Prior:
    """Symbol-level prior over GF(256) for current noisy block, computed by trained ByT5."""

    def __init__(self,
                 model_dir: str,
                 base_model_name: str = "google/byt5-small",
                 device: Optional[str] = None,
                 fp16: bool = True,
                 max_length: int = 256):
        # Lazy torch import so the rest of the package is sandbox-importable
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, T5EncoderModel
        self.torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.eos_token else "<pad>"

        # Build encoder from base name (state dict will overwrite weights)
        enc = T5EncoderModel.from_pretrained(base_model_name)

        # Re-create the ByT5ByteTagger architecture inline (we don't import the training script)
        class _ByT5ByteTagger(nn.Module):
            def __init__(self, enc, vocab_size):
                super().__init__()
                self.enc = enc
                self.classifier = nn.Linear(self.enc.config.d_model, vocab_size)

            def forward(self, input_ids, attention_mask=None):
                outputs = self.enc(input_ids=input_ids, attention_mask=attention_mask)
                return {"logits": self.classifier(outputs.last_hidden_state)}

        model = _ByT5ByteTagger(enc, vocab_size=self.tokenizer.vocab_size)

        # Load trained weights
        sd_path = os.path.join(model_dir, "pytorch_model.bin")
        if not os.path.isfile(sd_path):
            raise FileNotFoundError(f"Could not find pytorch_model.bin under {model_dir}")
        sd = torch.load(sd_path, map_location="cpu")
        # Some training scripts save with key prefix; load non-strict to be safe.
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # Filter expected misses (encoder might load all keys but classifier is critical)
        critical_missing = [k for k in missing if k.startswith("classifier")]
        if critical_missing:
            raise RuntimeError(f"Critical keys missing when loading classifier head: {critical_missing}")
        model.eval()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if fp16 and self.device == "cuda":
            try:
                model = model.half()
                self.dtype = torch.float16
            except Exception:
                self.dtype = torch.float32
        else:
            self.dtype = torch.float32
        self.model = model.to(self.device)
        self.max_length = max_length

        # Pre-compute mapping from vocab token id -> byte value (0..255), or -1 if not a byte token.
        self.byte_token_ids = self._build_byte_token_ids()

    def _build_byte_token_ids(self) -> np.ndarray:
        """Return shape (256,) array `byte_to_id` such that byte_to_id[b] is the
        ByT5 vocab id that decodes to byte value b. For ByT5: id b+3 corresponds to byte b.
        We compute it explicitly from the tokenizer for safety.
        """
        byte_to_id = -np.ones(256, dtype=np.int64)
        for tid in range(self.tokenizer.vocab_size):
            try:
                s = self.tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            except Exception:
                continue
            if isinstance(s, str) and len(s) == 1:
                code = ord(s)
                if 0 <= code <= 255 and byte_to_id[code] < 0:
                    byte_to_id[code] = tid
        # Fill any remaining slots using ByT5 convention (offset of 3)
        for b in range(256):
            if byte_to_id[b] < 0:
                cand = b + 3  # ByT5 convention
                if cand < self.tokenizer.vocab_size:
                    byte_to_id[b] = cand
        return byte_to_id

    @property
    def torch_device(self):
        return self.device

    # ---- main inference ----
    def posterior(self,
                  noisy_block_bytes: np.ndarray,
                  context_text: str = "") -> np.ndarray:
        """Compute symbol-level posterior over GF(256) for each byte position in the noisy block.

        Args:
            noisy_block_bytes: (k,) np.uint8, the hard-decision (or otherwise-estimated)
                bytes of the current RS info block.
            context_text: optional str with the clean cross-sentence prefix
                (concatenation of previously-decoded sentences).

        Returns:
            P: (k, 256) np.float64 matrix of probabilities (rows sum to ~1).
        """
        torch = self.torch
        noisy_block_bytes = np.asarray(noisy_block_bytes, dtype=np.uint8)
        k = int(noisy_block_bytes.size)
        # Build text. Map each byte to a printable char with placeholder for non-printable
        # (consistent with the training pipeline that used '?' for non-printable).
        block_text = self._bytes_to_safe_text(noisy_block_bytes)
        if context_text:
            full_text = context_text + block_text
            ctx_offset_in_chars = len(context_text)
        else:
            full_text = block_text
            ctx_offset_in_chars = 0

        ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        L = len(ids)
        if L > self.max_length:
            # Truncate from the LEFT so the current block is preserved at the right side.
            cut = L - self.max_length
            ids = ids[cut:]
            ctx_offset_in_chars = max(0, ctx_offset_in_chars - cut)
            L = len(ids)

        pad_id = self.tokenizer.pad_token_id
        input_ids = ids + [pad_id] * (self.max_length - L)
        attn = [1] * L + [0] * (self.max_length - L)
        input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        attn_t = torch.tensor([attn], dtype=torch.long, device=self.device)

        with torch.no_grad():
            out = self.model(input_ids=input_ids_t, attention_mask=attn_t)
            logits = out["logits"][0, :L, :]  # (L, V)

        # Slice the block region: positions ctx_offset_in_chars..ctx_offset_in_chars+k
        block_start = ctx_offset_in_chars
        block_end = block_start + k
        # In ByT5 the tokenizer is byte-level, so token positions == char positions for ASCII text.
        # Defensive guard:
        if block_end > L:
            # Pad missing positions with uniform distribution
            pad_len = block_end - L
            block_logits = logits[block_start:L, :]
            uniform = torch.zeros((pad_len, logits.size(1)), dtype=logits.dtype, device=logits.device)
            block_logits = torch.cat([block_logits, uniform], dim=0)
        else:
            block_logits = logits[block_start:block_end, :]

        # Map vocab logits -> 256 byte logits via byte_token_ids
        byte_ids = torch.tensor(self.byte_token_ids, dtype=torch.long, device=self.device)
        # Some entries may still be -1 if vocab smaller; clamp to 0 and post-mask
        mask_valid = (byte_ids >= 0)
        clamped = torch.where(mask_valid, byte_ids, torch.zeros_like(byte_ids))
        # Index columns
        byte_logits = block_logits.index_select(dim=1, index=clamped)  # (k, 256)
        # For invalid byte ids, set logit to a very negative number.
        # Promote to fp32 first so that -1e9 is representable (fp16 max ~65504).
        byte_logits = byte_logits.float()
        if (~mask_valid).any():
            neg_inf = torch.full_like(byte_logits, -1e9)
            byte_logits = torch.where(mask_valid.unsqueeze(0), byte_logits, neg_inf)

        # Softmax to probabilities
        probs = torch.softmax(byte_logits, dim=-1).cpu().numpy().astype(np.float64)
        return probs  # (k, 256)

    def log_posterior(self, noisy_block_bytes: np.ndarray, context_text: str = "",
                       eps: float = 1e-12) -> np.ndarray:
        P = self.posterior(noisy_block_bytes, context_text=context_text)
        return np.log(np.clip(P, eps, 1.0))

    # ---- bit-level marginalization (for binary BCH / LDPC integration) ----
    def bit_posterior(self, noisy_block_bytes: np.ndarray, context_text: str = "",
                       bitorder: str = "big") -> np.ndarray:
        """Return per-bit posterior of shape (k_bytes * 8, 2).

        For each byte position i and each bit position j in that byte:
            P(bit_j(c_i) = 0 | y, ctx) = sum over byte values b with bit_j(b) = 0
                                                 of P(c_i = b | y, ctx)

        Args:
            noisy_block_bytes: (k_bytes,) np.uint8 — current segment bytes
            context_text: clean prefix
            bitorder: 'big' (MSB first, matches our convention)

        Returns:
            (k_bytes*8, 2) float64; row i has [P(bit=0), P(bit=1)] for bit i
            (in the bit-stream order produced by np.unpackbits with that bitorder).
        """
        byte_post = self.posterior(noisy_block_bytes, context_text=context_text)  # (k_bytes, 256)
        k_bytes = byte_post.shape[0]
        # Build bit-pattern table: bit_pat[v, j] = j-th bit of byte value v
        vals = np.arange(256, dtype=np.uint8)
        bit_pat = np.unpackbits(vals[:, None], axis=1, bitorder=bitorder)  # (256, 8)

        out = np.zeros((k_bytes * 8, 2), dtype=np.float64)
        for j in range(8):
            mask0 = (bit_pat[:, j] == 0)  # (256,)
            # For each byte position, p0 = sum_{v: bit_j(v)=0} byte_post[i, v]
            p0_col = byte_post[:, mask0].sum(axis=1)   # (k_bytes,)
            p1_col = 1.0 - p0_col
            # Map (byte_idx, bit_j) -> linear bit index = byte_idx*8 + j
            out[j::8, 0] = p0_col
            out[j::8, 1] = p1_col
        # Numerical: ensure nonnegative + normalize per row
        out = np.clip(out, 0.0, 1.0)
        out /= out.sum(axis=1, keepdims=True)
        return out

    def bit_log_posterior(self, noisy_block_bytes: np.ndarray, context_text: str = "",
                           eps: float = 1e-12) -> np.ndarray:
        P = self.bit_posterior(noisy_block_bytes, context_text=context_text)
        return np.log(np.clip(P, eps, 1.0))

    # ---- helpers ----
    @staticmethod
    def _bytes_to_safe_text(bs: np.ndarray) -> str:
        """Map bytes to a printable-ASCII string, replacing non-printable with '?'.

        Matches the convention used during ByT5 training (`make_bitflip_noisy_group`).
        """
        chars = []
        for b in bs.tolist():
            if 32 <= int(b) <= 126:
                chars.append(chr(int(b)))
            else:
                chars.append('?')
        return "".join(chars)


__all__ = ["ByT5Prior"]
