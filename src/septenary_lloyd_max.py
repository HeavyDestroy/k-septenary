#!/usr/bin/env python3
"""Compute Lloyd-Max 7-level (septenary) centroids for N(0,1)."""
import numpy as np
from scipy.stats import norm

n_levels = 7
# Initial guess: equal-probability intervals
p = np.linspace(0, 1, n_levels + 1)
boundaries = norm.ppf(p)
boundaries[0] = -np.inf
boundaries[-1] = np.inf

for it in range(200):
    # Compute centroids: E[x | a < x < b] for N(0,1)
    levels = np.zeros(n_levels)
    for i in range(n_levels):
        a, b = boundaries[i], boundaries[i+1]
        if np.isneginf(a):
            levels[i] = -norm.pdf(b) / norm.cdf(b)
        elif np.isposinf(b):
            levels[i] = norm.pdf(a) / (1 - norm.cdf(a))
        else:
            levels[i] = (norm.pdf(a) - norm.pdf(b)) / (norm.cdf(b) - norm.cdf(a))
    
    # Update boundaries: midpoint between adjacent centroids
    for i in range(1, n_levels):
        boundaries[i] = (levels[i-1] + levels[i]) / 2
    
    # Compute MSE
    mse = 0
    for i in range(n_levels):
        a, b = boundaries[i], boundaries[i+1]
        pa = 0 if np.isneginf(a) else norm.cdf(a)
        pb = 1 if np.isposinf(b) else norm.cdf(b)
        phi_a = 0 if np.isneginf(a) else norm.pdf(a)
        phi_b = 0 if np.isposinf(b) else norm.pdf(b)
        prob = pb - pa
        if prob > 1e-15:
            ex = (phi_a - phi_b) / prob
            ex2 = 1 - (b*phi_b - a*phi_a) / prob
            mse += prob * (ex2 - 2*levels[i]*ex + levels[i]**2)

levels_sorted = np.sort(levels)
mid_boundaries = (levels_sorted[:-1] + levels_sorted[1:]) / 2

# Empirical MSE on 2M samples
samples = np.random.randn(2_000_000)

# Ternary
t = 1.224006
d = 0.612003
q3 = np.zeros_like(samples)
q3[samples > d] = t
q3[samples < -d] = -t
q3_mse = np.mean((samples - q3)**2)

# Quinary
q5 = np.zeros_like(samples)
q5[np.abs(samples) < 0.382284] = 0
m1 = (np.abs(samples) >= 0.382284) & (np.abs(samples) < 1.244357)
q5[m1] = np.sign(samples[m1]) * 0.764568
q5[np.abs(samples) >= 1.244357] = np.sign(samples[np.abs(samples) >= 1.244357]) * 1.724147
q5_mse = np.mean((samples - q5)**2)

# Septenary
q7 = levels_sorted[np.digitize(samples, mid_boundaries)]
q7_mse = np.mean((samples - q7)**2)

print("=" * 60)
print("  Lloyd-Max Codebooks for N(0,1)")
print("=" * 60)
print()
print(f"Ternary ({np.log2(3):.4f} bits, 2-bit storage):")
print(f"  Levels: [-1.224006, 0, 1.224006]")
print(f"  MSE:    {q3_mse:.6f}")
print(f"  States used: 3/4 (75%)")
print()
print(f"Quinary ({np.log2(5):.4f} bits, 3-bit storage):")
print(f"  Levels: [-1.724147, -0.764568, 0, 0.764568, 1.724147]")
print(f"  MSE:    {q5_mse:.6f}")
print(f"  States used: 5/8 (62.5%)")
print()
print(f"Septenary ({np.log2(7):.4f} bits, 3-bit storage):")
print(f"  Levels: {np.round(levels_sorted, 6).tolist()}")
print(f"  Boundaries: {np.round(mid_boundaries, 6).tolist()}")
print(f"  MSE:    {q7_mse:.6f}")
print(f"  States used: 7/8 (87.5%)")
print()
print("─" * 60)
print(f"Septenary vs Quinary:")
print(f"  MSE improvement: {(q5_mse - q7_mse) / q5_mse * 100:.1f}%")
print(f"  Same memory cost: 3 bits/element")
print(f"  Compression vs bf16: {16 / np.log2(7):.2f}x (same as quinary)")
print(f"  Extra precision: 7 levels vs 5 levels (40% more)")
