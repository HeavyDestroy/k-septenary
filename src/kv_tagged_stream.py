#!/usr/bin/env python3
"""
KV Tagged Stream — uses ALL 8 three-bit states as block tags.

For each 256-element KV block (head_dim), prepend a 3-bit tag:
  000 = K head 0      100 = V head 0
  001 = K head 1      101 = V head 1  
  010 = K head 2      110 = V head 2
  011 = K head 3      111 = V head 3

Overhead: 3 tag bits per 768 data bits = 0.39%
System win: eliminates separate K/V cache + head indexing
"""
import torch, json
from pathlib import Path

SEPT_LEVELS = torch.tensor([-2.033369, -1.188147, -0.560577, 0.0, 0.560577, 1.188147, 2.033369])
SEPT_BOUNDARIES = torch.tensor([-1.610758, -0.874362, -0.280288, 0.280288, 0.874362, 1.610758])
HEAD_DIM = 256

def _quantize(vec):
    """Float vector → septenary indices 0..6."""
    return torch.bucketize(vec, SEPT_BOUNDARIES.to(vec.device)).clamp(0, 6)

def _dequantize(idx):
    """Septenary indices → float centroids."""
    return SEPT_LEVELS.to(idx.device)[idx.long()]

def _make_tag(head_idx, is_value=True):
    """Create tag: 0-3 for K heads, 4-7 for V heads."""
    return head_idx + (4 if is_value else 0)

def pack_tagged_stream(K, V):
    """
    Pack K and V tensors into self-describing tagged stream.
    
    Input:  K, V each (batch, kv_heads, seq_len, 256) float
    Output: stream uint8 1D, metadata dict
    
    Stream format (per 256-dim block):
      [tag: 0-7] [val: 0-6] × 256
    
    Layout: ... K0 V0 K1 V1 K2 V2 K3 V3 ... (interleaved per head per position)
    """
    b, nh, s, d = K.shape
    assert d == HEAD_DIM, f"head_dim must be {HEAD_DIM}"
    assert V.shape == K.shape and nh == 4
    
    N = b * nh * s  # total blocks (one per head per position)
    block_len = 1 + d  # 257
    
    # Interleaved stream: K0 V0 K1 V1 ...
    stream_len = N * 2 * block_len
    stream = torch.zeros(stream_len, dtype=torch.uint8)
    
    k_flat = _quantize(K).reshape(N, d).cpu()
    v_flat = _quantize(V).reshape(N, d).cpu()
    
    for i in range(N):
        base = i * 2 * block_len
        head_idx = i % nh
        
        # K block: tag 0-3
        stream[base] = _make_tag(head_idx, is_value=False)
        stream[base+1:base+1+d] = k_flat[i].to(torch.uint8)
        
        # V block: tag 4-7
        stream[base+block_len] = _make_tag(head_idx, is_value=True)
        stream[base+block_len+1:base+block_len+1+d] = v_flat[i].to(torch.uint8)
    
    metadata = {
        'batch': b, 'kv_heads': nh, 'seq_len': s, 'head_dim': d,
        'n_blocks': N * 2,
        'block_format': '[tag:3bits] [256_data_values:256×3bits]',
        'tag_scheme': {str(k): f'head {k%4} {"V" if k >= 4 else "K"}' for k in range(8)},
        'tag_bits_per_block': 3,
        'data_bits_per_block': d * 3,
        'overhead_pct': 3 / (d * 3) * 100,
        'stream_bits': stream.numel() * 3,
        'separate_tensors_bits': 2 * N * d * 3,
        'overhead_vs_separate_bytes': (stream.numel() - 2 * N * d) * 3 / 8,
        'tags_used': 8,
        'tags_unused': 0,
        'system_benefits': 'Zero waste. Self-describing stream eliminates separate K/V cache arrays and head index tables.',
    }
    
    return stream, metadata

def unpack_tagged_stream(stream, metadata):
    """
    Unpack tagged stream back to K and V float tensors.
    """
    b = metadata['batch']
    nh = metadata['kv_heads']
    s = metadata['seq_len']
    d = metadata['head_dim']
    N = b * nh * s
    block_len = 1 + d  # 257
    
    K_flat = torch.zeros(N, d)
    V_flat = torch.zeros(N, d)
    
    for i in range(N):
        base = i * 2 * block_len
        
        # K block
        k_tag = stream[base].item()
        k_head = k_tag % 4
        assert k_tag < 4, f"Expected K tag 0-3, got {k_tag}"
        k_data = stream[base+1:base+1+d].long()
        K_flat[i] = k_data
        
        # V block
        v_tag = stream[base+block_len].item()
        v_head = v_tag % 4
        assert v_tag >= 4, f"Expected V tag 4-7, got {v_tag}"
        v_data = stream[base+block_len+1:base+block_len+1+d].long()
        V_flat[i] = v_data
        
        # Verify head consistency
        assert k_head == v_head, f"Head mismatch at pair {i}: K head {k_head}, V head {v_head}"
    
    K = K_flat.view(b, nh, s, d)
    V = V_flat.view(b, nh, s, d)
    return K, V

def round_trip_test():
    """Verify bit-exact reconstruction."""
    torch.manual_seed(42)
    b, nh, s, d = 2, 4, 16, 256
    
    K = torch.randn(b, nh, s, d)
    V = torch.randn(b, nh, s, d)
    
    K_idx = _quantize(K)
    V_idx = _quantize(V)
    
    stream, meta = pack_tagged_stream(K, V)
    K_rt, V_rt = unpack_tagged_stream(stream, meta)
    
    k_ok = torch.allclose(K_idx.float(), K_rt.float())
    v_ok = torch.allclose(V_idx.float(), V_rt.float())
    
    print(f"✓ Round-trip: {'PASS' if k_ok and v_ok else 'FAIL'} (bit-exact)")
    
    # Stats
    data_per_block = meta['data_bits_per_block']
    tag_per_block = meta['tag_bits_per_block']
    overhead = meta['overhead_pct']
    total_bits = meta['stream_bits']
    separate_bits = meta['separate_tensors_bits']
    extra_bytes = meta['overhead_vs_separate_bytes']
    
    print(f"\n  Tagged Stream Efficiency:")
    print(f"    Blocks (K+V total):  {meta['n_blocks']:,}")
    print(f"    Tag scheme:          {meta['tag_scheme']}")
    print(f"    Tags used:           8/8 ({meta['tags_unused']} unused)")
    print(f"    Overhead:            {tag_per_block} bits/{data_per_block} data bits = {overhead:.3f}%")
    print(f"    Stream:              {total_bits:,} bits ({total_bits/8:,} bytes)")
    print(f"    Separate:            {separate_bits:,} bits ({separate_bits/8:,} bytes)")
    print(f"    Extra bytes:         {int(extra_bytes):,}")
    print(f"    System win:          {meta['system_benefits']}")
    
    # Memory at scale
    layers = 64
    seq = 65536
    total_blocks = layers * nh * seq * 2
    stream_bytes = total_blocks * (1 + d) * 3 / 8
    separate_bytes = layers * nh * seq * d * 2 * 3 / 8
    print(f"\n  At 64K context (Qwen3.5-27B):")
    print(f"    Stream:              {stream_bytes/1e6:.2f} MB")
    print(f"    Separate:            {separate_bytes/1e6:.2f} MB")
    print(f"    Overhead:            +{(stream_bytes/separate_bytes - 1)*100:.3f}%")
    print(f"    But eliminates:      2 cache arrays + 2 index pointers + head dim tables")
    
    return K_rt, V_rt

if __name__ == '__main__':
    print("=" * 60)
    print("  Tagged KV Stream — ALL 8 states used!")
    print("=" * 60)
    K_rt, V_rt = round_trip_test()
    
    design = {
        'format': 'tagged_KV_stream',
        'tags': {str(k): f'head {k%4} {"V" if k >= 4 else "K"}' for k in range(8)},
        'block_format': '[tag:3bits] [256_septenary_values:256×3bits]',
        'overhead_pct': 0.39,
        'tags_used': 8,
        'tags_unused': 0,
        'benefits': [
            'ALL 8 three-bit states carry meaning',
            'No separate K/V cache arrays',
            'No head index tables needed',
            'Self-describing stream',
            'Contiguous K+V per position',
            'Single memory allocation vs 2+',
        ],
    }
    out = Path(__file__).parent.parent / 'data' / 'kv_tagged_design.json'
    with open(out, 'w') as f:
        json.dump(design, f, indent=2)
    print(f"\n  Design saved to {out}")
