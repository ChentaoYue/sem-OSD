# Sem-OSD

Reference implementation for the paper

> **Semantic Ordered Statistics Decoding.**

Sem-OSD is a soft decoder for short byte-aligned linear block codes
(Reed-Solomon over GF(2^8) and binary BCH whose information word is a multiple
of eight bits). It augments classical Fossorier-style ordered statistics
decoding with a byte-level language-model (LM) prior, fuses the LM evidence
with the channel reliability into a unified bit/byte score, and enumerates two
complementary candidate families on top of it:

- **T_b** — bit-flip family of Hamming radius m, accelerated by Probability-based
  OSD (PB-OSD) cutoff and pruning;
- **T_B** — LM-driven byte-substitution family of count w with top-T
  alternatives per position.

A Berlekamp-Massey fast path keeps the LM idle on blocks the channel decoder
already resolves; the LM is invoked only on hard blocks. The same algorithm
runs without modification on RS(16, 8) over GF(2^8) and on binary BCH(127, 64).

## Repository status

This repository currently contains the **decoder core** so that readers of the
paper can follow the algorithm at the level of source code:

| Released now                                   | Held back until acceptance               |
|------------------------------------------------|------------------------------------------|
| Sem-OSD decoder (`bg_rsd.py`)                  | Trained ByT5 semantic-prior checkpoint   |
| Binary OSD with PB-OSD acceleration            | Training script and SNLI preprocessing   |
| RS / BCH code construction, GF(2^8) arithmetic | Full simulation harness and SNR sweeper  |
| AWGN / Gilbert-Elliott channel models          | Semantic similarity (SBERT) evaluation   |
| ByT5 prior wrapper (inference-time only)       | Profiling and ablation scripts           |
| Minimal demo                                   | Per-figure result CSVs and logs          |

The held-back pieces will be released here on paper acceptance, together with
the trained `byt5_prefix_ckpt`.

## Supported modes (in the released code)

Both code families (RS over GF(2^8) and binary BCH with byte-aligned k) and all
three decoder modes used in the paper are exercised by the two top-level
classes `BinaryOSDForRS` and `BGRSD`. The `BinaryOSDForRS` class name is kept
for backward compatibility; it works on either code family through an internal
adapter.

| Mode (paper)              | Class / constructor                                                                              |
|---------------------------|--------------------------------------------------------------------------------------------------|
| BM fast path (baseline)   | both classes invoke it before any OSD work                                                       |
| Binary OSD (channel-only) | `BinaryOSDForRS(code, byt5_prior=None, order_W=m)`                                               |
| Sem-OSD, T_b-only         | `BinaryOSDForRS(code, byt5_prior=lm, order_W=m)`                                                 |
| Sem-OSD, T_B-only         | `BGRSD(code, byt5_prior=lm, order_W=w, top_T=T)`                                                 |
| **Sem-OSD (full)**        | `BinaryOSDForRS(code, byt5_prior=lm, order_W=m, byte_tep_T=T, byte_tep_W=w)`                     |
| PB-OSD acceleration       | add `pb_accel=True` to any `BinaryOSDForRS` call (AWGN only; do not enable on Gilbert-Elliott)   |

`code` may be either `RSCode(n, k)` (with `k` info bytes) or `BCHCode(n, k)`
(with `k` info bits, `k % 8 == 0`). The same constructor calls work on both;
no separate BCH decoder is required.

## Project layout

```
sem-OSD/
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- sem_osd/
|   |-- __init__.py
|   |-- gf256.py             GF(2^8) arithmetic + Gaussian elimination over GF(2^8)
|   |-- rs_code.py           shortened RS(n, k) over GF(2^8) (uses `reedsolo`)
|   |-- bch_code.py          binary BCH(n, k) construction
|   |-- channel.py           BPSK + AWGN / Gilbert-Elliott + per-bit LLR
|   |-- binary_osd_for_rs.py **Sem-OSD-Hybrid (T_b + T_B); works on RS *and* BCH**
|   |-- bg_rsd.py            T_B-only ablation (`BGRSD`); RS + BCH dispatched internally
|   `-- byt5_prior.py        ByT5 encoder + per-position byte head
`-- examples/
    `-- decode_demo.py       end-to-end RS(16,8) and BCH(127,64) AWGN demo without LM
```

The filename `binary_osd_for_rs.py` is legacy; the class `BinaryOSDForRS` it
exports actually supports both code families (RS over GF(2^8) and binary BCH
with byte-aligned k) through an internal `_CodeAdapter`. We have left the
filename unchanged for backward compatibility with prior internal references.

## Installation

Tested with Python 3.10. CUDA is optional but recommended if you plan to plug
in your own LM.

```bash
git clone https://github.com/<your-org>/sem-OSD.git
cd sem-OSD
pip install -r requirements.txt
```

`requirements.txt` lists only the strictly necessary packages; in particular
`torch` and `transformers` are required only when you instantiate
`ByT5Prior`. You can run the channel-only baseline (`BinaryOSD`) without
either.

## Quickstart

A minimal end-to-end example using RS(16, 8) over AWGN, *without* the LM:

```bash
python examples/decode_demo.py
```

Expected output (truncated):

```
RS(16, 8) over GF(2^8), t = 4
Eb/N0 = 4.0 dB, sigma = 0.398
sent     : b'Hello!\x01\x02'
received : b'Helmo!\x01\x02'        (1 byte error)
BM       : success, decoded = b'Hello!\x01\x02'

Eb/N0 = 1.0 dB, sigma = 0.629  (BM expected to fail)
BM       : failure
Sem-OSD (channel only, m = 3) : decoded = b'Hello!\x01\x02'  (success)
```

To enable the LM-driven family `T_B`, instantiate `ByT5Prior` with your own
checkpoint and pass it to `BGRSD`:

```python
from sem_osd.byt5_prior import ByT5Prior
from sem_osd.bg_rsd import BGRSD
from sem_osd.rs_code import RSCode

code  = RSCode(n=16, k=8)
prior = ByT5Prior(model_dir="path/to/your/byt5_checkpoint")
dec   = BGRSD(code, byt5_prior=prior, alpha=0.5, top_T=4, order_W=2,
              mrb_mode="info_only")

ctx = "The cat is sleeping on t"           # clean linguistic prefix
ok, decoded = dec.decode(y, sigma, context_text=ctx)
```

## Algorithm in one diagram

```
y --> byte hard decision  -->  semantic-prior model  -->  score fusion
                                                              (lambda_f, Lambda_f)
                                                                |
                                              +-----------------+-----------------+
                                              |                                   |
                                          T_b: bit-flip                      T_B: byte-sub
                                          radius m                           count w, top-T
                                          (PB-OSD)                                |
                                              +-----------------+-----------------+
                                                                |
                                              re-encoding via G + score + argmax
                                                                |
                                                              c_hat
```

The fast path (Berlekamp-Massey) sits *before* this pipeline and short-circuits
to its own output whenever bounded-distance decoding succeeds. See the paper
for the precise score definitions and the cutoff/pruning rules of PB-OSD.

## Citation

A preprint and the corresponding BibTeX entry will be added here once the
manuscript is posted on arXiv.

## License

The decoder source (this repository) is released under the MIT License - see
`LICENSE`. The ByT5-small base model is released by Google under the Apache
2.0 License; the fine-tuned semantic-prior checkpoint will be released
separately, also under Apache 2.0, after acceptance.
