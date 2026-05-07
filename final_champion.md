# K-Septenary (Kenji Septenary) — KV Cache Quantization Champion

## The Final Recipe
```
per_channel_scale(absmax/3) → 7_level_Lloyd-Max → scale_back → tagged_stream
```

No Hadamard. No S_bias. That's all.

## Key Numbers
- **PPL:** 13.8727 (+0.0324 vs bf16) — essentially lossless
- **Storage:** 3-bit tagged stream (2.807 data + 0.39% tag = 2.818 eff. bpe)
- **Compression:** 5.3× vs bf16
- **Memory at 262K context:** 12.05 GB vs 64 GB (81.2% savings)
- **State efficiency:** ALL 8 three-bit values carry meaning (100%)

## Tag scheme (every 256-dim block)
| Bits | Meaning |
|------|---------|
| 000 | head 0 K |
| 001 | head 1 K |
| 010 | head 2 K |
| 011 | head 3 K |
| 100 | head 0 V |
| 101 | head 1 V |
| 110 | head 2 V |
| 111 | head 3 V |

## 7-Level Lloyd-Max Centroids (N(0,1))
```python
[-2.033369, -1.188147, -0.560577, 0.0, 0.560577, 1.188147, 2.033369]
```
Boundaries: `[-1.610758, -0.874362, -0.280288, 0.280288, 0.874362, 1.610758]`

## What We Tried That Didn't Work
- **111 as outlier escape** → PPL got WORSE (+0.054 vs +0.032). Attention doesn't need precise tails.
- **S_bias on top of per-channel** → hurts for 5+ level codebooks
- **Hadamard before per-channel** → rotation destroys per-dimension structure

## What Evolved (lineage)
EDEN ternary → PC+Ternary → PC+Quinary → PC+Septenary → PC+EDEN+Septenary+Tagged

## Location
- Reference: `~/research/eden-ternary/data/kv_quant_reference.json`
- Results: `~/research/eden-ternary/data/ppl_all_schemes.json`
- Tagged stream design: `~/research/eden-ternary/scripts/kv_tagged_stream.py`
- Septenary codebook: `~/research/eden-ternary/math/septenary_lloyd_max.py`
- Model: Qwen3.5-27B, NF4 bitsandbytes, 2× GPU
