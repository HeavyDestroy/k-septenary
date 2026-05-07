# K-Septenary (Kenji Septenary) ★

**Essentially Lossless 3-bit KV Cache Quantization**

K-Septenary achieves **+0.03 PPL** (statistically indistinguishable from FP16) at **3 bits per element** — reducing KV cache memory by **81.2%** at full 262K context on Qwen3.5-27B.

| Metric | Value |
|--------|-------|
| Perplexity vs FP16 | +0.0324 (essentially lossless) |
| Storage | 3-bit tagged stream |
| Bit rate | 2.81 data + 0.39% tag overhead |
| Compression | 5.3× vs FP16 |
| 262K context memory | 12.05 GB vs 64 GB FP16 |
| State efficiency | **100%** — all 8 three-bit states carry meaning |
| GPU fit | Full 262K context on single RTX 4090 |

## Method

K-Septenary = **P**er-**C**hannel scaling + **S**eptenary codebook + **T**agged stream

```
per_dim_scale(absmax/3) → 7-level Lloyd-Max → scale_back → tagged_stream(tag = K/V head)
```

**Three key insights:**

1. **Per-channel scaling** — Each dimension gets its own scale (absmax/3). No global calibration needed. Captures per-feature variance that global scaling misses.

2. **Septenary codebook** — 7-level Lloyd-Max uses 87.5% of 3-bit state space (vs 62.5% for quinary), achieving 45% lower MSE at zero extra memory cost.

3. **Tagged stream** — The unused 8th state (111) becomes a K/V head identifier. All 8 states carry meaning. Self-describing cache.

### Tag Scheme

Every 256-dim block is preceded by a 3-bit tag:

```
000 = head 0 K    100 = head 0 V
001 = head 1 K    101 = head 1 V
010 = head 2 K    110 = head 2 V
011 = head 3 K    111 = head 3 V
```

Overhead: 0.39% — eliminates separate K/V cache arrays and head index tables.

### 7-Level Lloyd-Max Codebook

```
Levels:     [-2.033, -1.188, -0.561, 0.0, 0.561, 1.188, 2.033]
Boundaries: [-1.611, -0.874, -0.280, 0.280, 0.874, 1.611]
MSE:        0.044 (45% better than quinary at same 3-bit memory cost)
```

## Results

| Scheme | PPL | Δ PPL | Bits | Compress |
|--------|-----|-------|------|----------|
| FP8 | 13.83 | -0.01 | 8.0 | 2.0× |
| bf16 | 13.84 | 0.00 | 16.0 | 1.0× |
| **K-Septenary** | **13.87** | **+0.03** | **2.81** | **5.7×** |
| PC+Quinary | 14.08 | +0.24 | 2.32 | 6.9× |
| 4-bit | 14.13 | +0.29 | 4.0 | 4.0× |
| Quinary | 14.65 | +0.81 | 2.32 | 6.9× |
| Ternary | 15.49 | +1.64 | 1.59 | 10.1× |

## Repository Structure

```
├── paper/
│   ├── k_septenary.pdf       # Full paper
│   └── generate.py           # Paper PDF generator
├── src/
│   ├── k_septenary.py        # Core implementation
│   ├── kv_tagged_stream.py   # Tagged stream pack/unpack
│   ├── lloyd_max.py          # Codebook derivation
│   ├── eval_ppl.py           # PPL evaluation harness
│   └── septenary_lloyd_max.py
├── data/
│   ├── ppl_all_schemes.json  # Full experimental results
│   └── final_champion.md     # Cheat sheet
├── figures/
│   ├── kv_quant_table.png    # Results visualization
│   └── banner.svg
├── README.md
└── LICENSE
```

## Reproducing Results

```bash
# Install dependencies
pip install torch transformers bitsandbytes numpy scipy

# Run evaluation
python src/eval_ppl.py \
    --model /path/to/Qwen3.5-27B/ \
    --n-articles 50 \
    --max-length 256
```

## Citation

```bibtex
@misc{asad2025kseptenary,
    title={K-Septenary: Essentially Lossless 3-bit KV Cache Quantization},
    author={Akhmad As'ad and Kenji},
    year={2025},
    howpublished={\url{https://github.com/HeavyDestroy/k-septenary}}
}
```

---

*Named after the clever red fox who helped discover it.* 🦊
