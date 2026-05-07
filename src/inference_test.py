#!/usr/bin/env python3
"""
K-Septenary Inference Test — Generate text with quantized KV cache.

Compares output from:
  1. BF16 KV cache (baseline)
  2. K-Septenary compressed KV cache

Usage:
    python src/inference_test.py \
        --model /path/to/Qwen3.6-27B/ \
        --prompt "Write a story about a fox who discovers mathematics."
"""
import os, sys, gc, json, time, math, warnings
from pathlib import Path
import torch
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.k_septenary import quantize_septenary_scaled, dequantize_septenary_scaled

warnings.filterwarnings("ignore")

FULL_ATTN = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63}

# ═══ DynamicCache monkey-patch with mode switching ═══════════════════

from transformers import DynamicCache

_orig_update = DynamicCache.update
_patch_active = False
_compress_fn = None

def _patched_update(self, key_states, value_states, layer_idx, *args, **kwargs):
    global _patch_active, _compress_fn
    if _patch_active and _compress_fn is not None and layer_idx in FULL_ATTN:
        key_states = _compress_fn(key_states)
        value_states = _compress_fn(value_states)
    return _orig_update(self, key_states, value_states, layer_idx, *args, **kwargs)

def enable_compression(fn):
    global _patch_active, _compress_fn
    _patch_active = True
    _compress_fn = fn

def disable_compression():
    global _patch_active
    _patch_active = False

# ═══ K-Septenary compressor ═════════════════════════════════════════

def k_septenary_compress(tensor):
    """Apply K-Septenary per-channel scaling + 7-level quantization."""
    b, nh, s, d = tensor.shape
    orig_dtype = tensor.dtype
    vecs = tensor.float().reshape(b * nh, s, d)
    
    indices, scales = quantize_septenary_scaled(vecs)
    recon = dequantize_septenary_scaled(indices.float(), scales)
    return recon.view(b, nh, s, d).to(orig_dtype)


# Apply patch once
DynamicCache.update = _patched_update

# ═══ Inference function ═════════════════════════════════════════════

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=128, 
             temperature=0.7, top_p=0.9, use_compression=False):
    """
    Generate text, optionally with K-Septenary compressed KV cache.
    Returns generated text and timing info.
    """
    if use_compression:
        enable_compression(k_septenary_compress)
    else:
        disable_compression()
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda:0')
    input_len = inputs.input_ids.shape[1]
    
    # Prefill
    t0 = time.time()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    elapsed = time.time() - t0
    
    # Decode
    generated = out[0][input_len:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    
    tokens_per_sec = (out.shape[1] - input_len) / elapsed
    
    return text, {
        'input_tokens': input_len,
        'output_tokens': out.shape[1] - input_len,
        'time_seconds': elapsed,
        'tokens_per_sec': tokens_per_sec,
    }


# ═══ Main ═══════════════════════════════════════════════════════════

def main():
    import argparse
    from transformers import (
        Qwen3_5ForConditionalGeneration, AutoTokenizer, BitsAndBytesConfig
    )
    
    parser = argparse.ArgumentParser(description='K-Septenary Inference Test')
    parser.add_argument('--model', default='/home/as-ad/models/Qwen3.6-27B/')
    parser.add_argument('--prompt', default=None)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--seeds', type=int, default=3,
                       help='Number of seeds to try for each mode')
    args = parser.parse_args()
    
    # Default prompt if none given
    if args.prompt is None:
        args.prompt = (
            "Explain the concept of entropy in information theory. "
            "Include Claude Shannon's contribution and give a concrete example."
        )
    
    print("=" * 65)
    print("  K-Septenary Inference Test — Qwen3.5-27B")
    print("=" * 65)
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params\n")
    
    # Test prompt
    print(f"Prompt ({len(args.prompt)} chars):")
    print(f"  \"{args.prompt}\"\n")
    
    # ── BF16 baseline ──
    print("─" * 65)
    print("  BASELINE: BF16 KV Cache")
    print("─" * 65)
    
    bf16_outputs = []
    bf16_timing = []
    for seed in range(args.seeds):
        torch.manual_seed(42 + seed)
        text, info = generate(model, tokenizer, args.prompt,
                               args.max_tokens, args.temperature,
                               use_compression=False)
        bf16_outputs.append(text)
        bf16_timing.append(info)
        print(f"\n  Seed {seed}: {info['output_tokens']} tokens "
              f"in {info['time_seconds']:.1f}s "
              f"({info['tokens_per_sec']:.1f} tok/s)")
        print(f"  Output: {text[:120]}...")
    
    avg_bf16_tps = np.mean([t['tokens_per_sec'] for t in bf16_timing])
    print(f"\n  Avg: {avg_bf16_tps:.1f} tok/s")
    
    # ── K-Septenary ──
    print("\n" + "─" * 65)
    print("  K-SEPTENARY: 3-bit Quantized KV Cache")
    print("─" * 65)
    
    ks_outputs = []
    ks_timing = []
    for seed in range(args.seeds):
        torch.manual_seed(42 + seed)
        text, info = generate(model, tokenizer, args.prompt,
                               args.max_tokens, args.temperature,
                               use_compression=True)
        ks_outputs.append(text)
        ks_timing.append(info)
        print(f"\n  Seed {seed}: {info['output_tokens']} tokens "
              f"in {info['time_seconds']:.1f}s "
              f"({info['tokens_per_sec']:.1f} tok/s)")
        print(f"  Output: {text[:120]}...")
    
    avg_ks_tps = np.mean([t['tokens_per_sec'] for t in ks_timing])
    print(f"\n  Avg: {avg_ks_tps:.1f} tok/s")
    
    # ── Comparison ──
    print("\n" + "=" * 65)
    print("  COMPARISON")
    print("=" * 65)
    print(f"  BF16:        {avg_bf16_tps:.1f} tok/s")
    print(f"  K-Septenary: {avg_ks_tps:.1f} tok/s")
    print(f"  Speed diff:  {(avg_ks_tps / avg_bf16_tps - 1) * 100:+.1f}%")
    
    # Quality comparison (visual)
    print("\n  Quality comparison (seed 0):")
    print(f"  BF16:  {bf16_outputs[0][:200]}...")
    print(f"  K-Sep: {ks_outputs[0][:200]}...")
    
    # Check if outputs match (they should be very close)
    match = bf16_outputs[0] == ks_outputs[0]
    if match:
        print("\n  ✓ Exact match on first 128 tokens!")
    else:
        # Check first 50 chars
        common = 0
        for i in range(min(len(bf16_outputs[0]), len(ks_outputs[0]))):
            if bf16_outputs[0][i] == ks_outputs[0][i]:
                common = i + 1
            else:
                break
        print(f"\n  △ First divergence at character {common}")
    
    # Save results
    result = {
        'model': args.model.split('/')[-1],
        'prompt': args.prompt,
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'seeds': args.seeds,
        'bf16': {
            'avg_tokens_per_sec': avg_bf16_tps,
            'outputs': bf16_outputs,
        },
        'k_septenary': {
            'avg_tokens_per_sec': avg_ks_tps,
            'outputs': ks_outputs,
        },
        'speed_delta_pct': (avg_ks_tps / avg_bf16_tps - 1) * 100,
    }
    
    out_path = ROOT / 'data' / 'inference_test.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()
