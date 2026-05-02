"""sem_osd — Semantic Ordered Statistics Decoding (Sem-OSD).

A soft decoder for short byte-aligned linear block codes that injects a
byte-level language-model prior into ordered statistics decoding.

Modules
-------
gf256              GF(2^8) arithmetic and Gaussian elimination.
rs_code            Shortened Reed-Solomon code RS(n, k) over GF(2^8).
bch_code           Binary BCH(n, k) code construction.
channel            BPSK + AWGN / Gilbert-Elliott channel models with bit-LLRs.
binary_osd_for_rs  Binary ordered statistics decoder with PB-OSD acceleration.
bg_rsd             The Sem-OSD decoder (T_b + T_B candidate families).
byt5_prior         ByT5-encoder semantic-prior wrapper (per-position byte posterior).

The module name `bg_rsd` is retained for backward compatibility; the class it
exports (`BGRSD`) implements the Sem-OSD algorithm of the paper.
"""

__version__ = "0.1.0"
