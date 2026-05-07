"""
K-Septenary (Kenji Septenary) — KV Cache Quantization

Per-channel scaling → 7-level Lloyd-Max → tagged K/V stream.

Reference: https://github.com/HeavyDestroy/k-septenary
"""

import math
import torch
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# 7-Level Lloyd-Max Codebook (optimal for N(0,1))
# ═══════════════════════════════════════════════════════════════════════

SEPTENARY_LEVELS = torch.tensor([
    -2.033369, -1.188147, -0.560577, 0.0,
    0.560577,  1.188147,  2.033369
])

SEPTENARY_BOUNDARIES = torch.tensor([
    -1.610758, -0.874362, -0.280288,
    0.280288,  0.874362,  1.610758
])

DATA_STD = 1.515625  # Empirical std of Qwen3.5-27B K/V activations
HEAD_DIM = 256

# Tag scheme for the self-describing stream
# 000-011 = K heads 0-3, 100-111 = V heads 0-3
TAG_K_HEADS = {0: 0, 1: 1, 2: 2, 3: 3}      # head_id → tag
TAG_V_HEADS = {0: 4, 1: 5, 2: 6, 3: 7}      # head_id → tag
TAG_TO_HEAD = {0: (0, 'K'), 1: (1, 'K'), 2: (2, 'K'), 3: (3, 'K'),
               4: (0, 'V'), 5: (1, 'V'), 6: (2, 'V'), 7: (3, 'V')}


# ═══════════════════════════════════════════════════════════════════════
# Core Quantization
# ═══════════════════════════════════════════════════════════════════════

def batch_hadamard(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform along last dimension."""
    N, d = x.shape
    h = 1
    while h < d:
        xv = x.view(N, d // (2 * h), 2, h)
        a, b = xv[:, :, 0, :].clone(), xv[:, :, 1, :].clone()
        xv[:, :, 0, :] = a + b
        xv[:, :, 1, :] = a - b
        h <<= 1
    return x / math.sqrt(d)


def quantize_septenary_scaled(vecs: torch.Tensor) -> torch.Tensor:
    """
    Quantize K/V vectors using per-channel scaling + 7-level Lloyd-Max.
    
    Args:
        vecs: (batch * kv_heads, seq_len, head_dim) float tensor
    
    Returns:
        indices: (batch * kv_heads, seq_len, head_dim) uint8 values 0..6
        scales: (batch * kv_heads, head_dim) float32 per-dimension scales
    """
    bnh, s, d = vecs.shape
    
    # Per-dimension absmax across sequence
    per_dim_absmax = vecs.abs().amax(dim=1)  # (bnh, d)
    scales = per_dim_absmax / 3.0
    scales = torch.clamp(scales, min=1e-8)
    
    # Scale to approximately N(0,1)
    scaled = vecs / scales.unsqueeze(1)
    
    # 7-level Lloyd-Max quantization
    levels = SEPTENARY_LEVELS.to(scaled.device)
    boundaries = SEPTENARY_BOUNDARIES.to(scaled.device)
    indices = torch.bucketize(scaled, boundaries)  # returns 0..6
    indices = indices.clamp(0, 6)
    
    return indices.to(torch.uint8), scales


def dequantize_septenary_scaled(
    indices: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """
    Reconstruct float vectors from quantized indices.
    
    Args:
        indices: (bnh, s, d) uint8 values 0..6
        scales: (bnh, d) float32 per-dimension scales
    
    Returns:
        vecs: (bnh, s, d) float32 reconstructed vectors
    """
    levels = SEPTENARY_LEVELS.to(indices.device)
    q = levels[indices.long()]  # (bnh, s, d)
    return q * scales.unsqueeze(1)


# ═══════════════════════════════════════════════════════════════════════
# Tagged Stream
# ═══════════════════════════════════════════════════════════════════════

def pack_tagged_stream(
    K: torch.Tensor, V: torch.Tensor, kv_heads: int = 4
) -> tuple[torch.Tensor, dict]:
    """
    Pack K and V tensors into a self-describing tagged bitstream.
    
    Each 256-dim block is preceded by a 3-bit tag:
        000-011: K heads 0-3
        100-111: V heads 0-3
    
    Format per KV pair: [K_tag:3b] [K_data:256×3b] [V_tag:3b] [V_data:256×3b]
    Overhead: 6 tag bits / 1536 data bits = 0.39%
    
    Args:
        K: (batch, kv_heads, seq_len, head_dim) float tensor
        V: same shape
        kv_heads: number of KV heads
    
    Returns:
        stream: 1D uint8 tensor of packed 3-bit values
        metadata: dict describing the stream layout
    """
    b, nh, s, d = K.shape
    assert nh == kv_heads
    assert V.shape == K.shape
    assert d == HEAD_DIM
    
    N = b * nh * s  # total KV pairs
    block_len = 1 + d  # 257 elements per block (1 tag + 256 data)
    stream_len = N * 2 * block_len
    
    # Quantize
    k_idx, k_scales = quantize_septenary_scaled(K.reshape(b * nh, s, d))
    v_idx, v_scales = quantize_septenary_scaled(V.reshape(b * nh, s, d))
    
    # Build stream
    stream = torch.zeros(stream_len, dtype=torch.uint8)
    
    k_flat = k_idx.reshape(N, d).cpu()
    v_flat = v_idx.reshape(N, d).cpu()
    
    for i in range(N):
        base = i * 2 * block_len
        head = i % nh
        
        # K block: tag 0-3
        stream[base] = TAG_K_HEADS[head]
        stream[base + 1: base + 1 + d] = k_flat[i]
        
        # V block: tag 4-7
        stream[base + block_len] = TAG_V_HEADS[head]
        stream[base + block_len + 1: base + block_len + 1 + d] = v_flat[i]
    
    metadata = {
        'batch': b,
        'kv_heads': nh,
        'seq_len': s,
        'head_dim': d,
        'n_pairs': N,
        'block_format': '[tag:3b] [256_septenary_values:256×3b]',
        'tag_scheme': {str(k): f'head {k % 4} {"V" if k >= 4 else "K"}'
                       for k in range(8)},
        'overhead_pct': 3 / (d * 3) * 100,
        'stream_elements': stream_len,
        'stream_bits': stream_len * 3,
        'data_bits': 2 * N * d * 3,
        'tags_used': 8,
        'tags_unused': 0,
    }
    
    return stream, metadata, (k_scales, v_scales)


def unpack_tagged_stream(
    stream: torch.Tensor, metadata: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reconstruct K and V from a tagged stream.
    
    Args:
        stream: 1D uint8 tensor of packed 3-bit values
        metadata: dict from pack_tagged_stream
    
    Returns:
        K, V: (batch, kv_heads, seq_len, head_dim) float tensors
    """
    b = metadata['batch']
    nh = metadata['kv_heads']
    s = metadata['seq_len']
    d = metadata['head_dim']
    N = metadata['n_pairs']
    block_len = 1 + d
    
    K_flat = torch.zeros(N, d)
    V_flat = torch.zeros(N, d)
    
    for i in range(N):
        base = i * 2 * block_len
        
        # K block
        k_tag = stream[base].item()
        assert k_tag < 4, f"Expected K tag 0-3, got {k_tag}"
        K_flat[i] = stream[base + 1: base + 1 + d].float()
        
        # V block
        v_tag = stream[base + block_len].item()
        assert v_tag >= 4, f"Expected V tag 4-7, got {v_tag}"
        V_flat[i] = stream[base + block_len + 1: base + block_len + 1 + d].float()
    
    K = K_flat.view(b, nh, s, d)
    V = V_flat.view(b, nh, s, d)
    return K, V


# ═══════════════════════════════════════════════════════════════════════
# Compression Statistics
# ═══════════════════════════════════════════════════════════════════════

def compression_ratio(seq_len: int = 262144, fp16_bits: int = 16,
                      quant_bits: int = 3, overhead_pct: float = 0.39,
                      layers: int = 64, kv_heads: int = 4,
                      head_dim: int = 256) -> dict:
    """
    Compute memory savings vs FP16 for a given context length.
    """
    elem_per_token = layers * kv_heads * head_dim  # K only
    
    fp16_bytes = elem_per_token * seq_len * 2 * 2  # K+V, 2 bytes each
    quant_bytes = (elem_per_token * seq_len * 2 * quant_bits / 8) * \
                  (1 + overhead_pct / 100)
    
    return {
        'context_length': seq_len,
        'fp16_gb': fp16_bytes / 1e9,
        'k_septenary_gb': quant_bytes / 1e9,
        'saved_gb': (fp16_bytes - quant_bytes) / 1e9,
        'saved_pct': (1 - quant_bytes / fp16_bytes) * 100,
        'compression_ratio': fp16_bytes / quant_bytes,
    }


def stream_efficiency(seq_len: int = 262144, layers: int = 64,
                      kv_heads: int = 4, head_dim: int = 256) -> dict:
    """Compute detailed stream efficiency metrics."""
    total_pairs = layers * kv_heads * seq_len
    data_bits_per_pair = 2 * head_dim * 3
    tag_bits_per_pair = 2 * 3
    
    return {
        'total_pairs': total_pairs,
        'data_bits': total_pairs * data_bits_per_pair,
        'tag_bits': total_pairs * tag_bits_per_pair,
        'total_bits': total_pairs * (data_bits_per_pair + tag_bits_per_pair),
        'tag_overhead_pct': tag_bits_per_pair / data_bits_per_pair * 100,
        'state_efficiency': 100,  # All 8 states used
    }
