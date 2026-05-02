"""End-to-end Sem-OSD demo (channel-only path; no language-model prior).

Runs the channel-only baseline of the Sem-OSD pipeline on both code families
covered in the paper:

    BPSK + AWGN  ->  Berlekamp-Massey fast path  ->  binary OSD with PB-OSD

For each of (RS(16, 8) over GF(2^8), binary BCH(127, 64)), two SNR points are
shown: a high-SNR point where the BM fast path should resolve the block, and
a low-SNR point where BM fails and the binary OSD must recover the codeword.

The full Sem-OSD modes (sem_osd_binary, sem_osd_hybrid; bg_rsd's T_B-only
ablation) require a trained ByT5 semantic-prior checkpoint and are documented
in the top-level README. They are intentionally omitted from this
self-contained demo so that it can run without `torch`, `transformers`, or
the held-back checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Union

# Make the repository root importable when this script is run from anywhere
# (e.g. `python examples/decode_demo.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from sem_osd.rs_code import RSCode
from sem_osd.bch_code import BCHCode
from sem_osd.channel import bytes_to_bits, bpsk_modulate, awgn, ebn0_db_to_sigma
from sem_osd.binary_osd_for_rs import BinaryOSDForRS


def _encode_to_bits(code, message: bytes) -> np.ndarray:
    """Encode `message` (bytes) and return the codeword bit sequence."""
    if isinstance(code, RSCode):
        msg = np.frombuffer(message, dtype=np.uint8)
        cw_bytes = code.encode(msg)
        return bytes_to_bits(cw_bytes)
    elif isinstance(code, BCHCode):
        msg_bytes = np.frombuffer(message, dtype=np.uint8)
        return code.encode_bytes(msg_bytes).astype(np.uint8)
    raise TypeError(type(code).__name__)


def _check_match(result: dict, message: bytes) -> bool:
    msg = result['msg']
    if isinstance(msg, np.ndarray):
        msg = bytes(msg.astype(np.uint8))
    elif not isinstance(msg, (bytes, bytearray)):
        msg = bytes(msg)
    return msg[: len(message)] == message


def run_one(code, message: bytes, ebn0_db: float, rng: np.random.Generator):
    rate = code.k / code.n if isinstance(code, RSCode) else code.k / code.n
    sigma = ebn0_db_to_sigma(ebn0_db, rate=rate)
    cw_bits = _encode_to_bits(code, message)

    # BPSK + AWGN over the codeword bits.
    x = bpsk_modulate(cw_bits)
    y = awgn(x, sigma, rng=rng)

    # Binary OSD with PB-OSD acceleration; no LM prior in this demo.
    decoder = BinaryOSDForRS(code, order_W=3, pb_accel=True)
    result = decoder.decode(y, sigma)

    print(f"  Eb/N0 = {ebn0_db:>4.1f} dB,  sigma = {sigma:.3f}")
    print(f"  method      : {result['method']}")
    print(f"  candidates  : {result['tep_count']}")
    print(f"  success     : {result['success']}")
    print(f"  match input : {_check_match(result, message)}")
    print()


def demo_section(code, ebn0_high: float, ebn0_low: float, label: str,
                  message: bytes, rng: np.random.Generator):
    print(f"=== {label} ===")
    print(f"Code        : {type(code).__name__}(n={code.n}, k={code.k})")
    print(f"             rate = {code.k / code.n:.3f}")
    print()
    print(f"--- High SNR: BM is expected to succeed ---")
    run_one(code, message, ebn0_db=ebn0_high, rng=rng)
    print(f"--- Low SNR: BM expected to fail; binary OSD should recover ---")
    run_one(code, message, ebn0_db=ebn0_low, rng=rng)


def main():
    rng = np.random.default_rng(42)

    # RS(16, 8) over GF(2^8), 8-byte message.
    rs_code = RSCode(n=16, k=8)
    demo_section(rs_code,
                 ebn0_high=6.0, ebn0_low=2.0,
                 label="RS(16, 8) over GF(2^8)",
                 message=b"Hello!\x01\x02",
                 rng=rng)

    # Binary BCH(127, 64): k = 64 bits = 8 bytes.
    try:
        bch_code = BCHCode(n=127, k=64)
    except ImportError:
        print("BCH demo skipped: install `galois` to enable BCH support.")
        return
    demo_section(bch_code,
                 ebn0_high=4.0, ebn0_low=1.0,
                 label="Binary BCH(127, 64)",
                 message=b"Hello!\x01\x02",
                 rng=rng)


if __name__ == "__main__":
    main()
