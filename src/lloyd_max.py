"""
Lloyd-Max optimal quantizer for N(0,1) distribution.

Computes the optimal codebook levels and decision boundaries
that minimize mean squared error for a standard normal distribution.
"""

import math
import numpy as np
from scipy.stats import norm


def lloyd_max_7(n_levels: int = 7, trunc_limit: float = 4.0,
                max_iter: int = 500, tol: float = 1e-10) -> dict:
    """
    Compute optimal Lloyd-Max codebook for N(0,1).
    
    Args:
        n_levels: Number of quantization levels
        trunc_limit: Truncation limit (±L) for tail-bound computation
        max_iter: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        dict with 'levels', 'boundaries', 'mse'
    """
    # Initial: equal probability intervals
    p = np.linspace(0, 1, n_levels + 1)
    boundaries = norm.ppf(p)
    boundaries[0] = -trunc_limit
    boundaries[-1] = trunc_limit
    
    for iteration in range(max_iter):
        # Compute centroids: E[x | a < x < b] for N(0,1)
        levels = np.zeros(n_levels)
        for i in range(n_levels):
            a, b = boundaries[i], boundaries[i + 1]
            phi_a = norm.pdf(a) if abs(a) < trunc_limit else 0.0
            phi_b = norm.pdf(b) if abs(b) < trunc_limit else 0.0
            Phi_a = norm.cdf(a)
            Phi_b = norm.cdf(b)
            if Phi_b - Phi_a > 1e-15:
                levels[i] = (phi_a - phi_b) / (Phi_b - Phi_a)
            else:
                levels[i] = (a + b) / 2.0
        
        # Update boundaries: midpoint between adjacent centroids
        new_boundaries = np.zeros_like(boundaries)
        new_boundaries[0] = -trunc_limit
        new_boundaries[-1] = trunc_limit
        for i in range(1, n_levels):
            new_boundaries[i] = (levels[i - 1] + levels[i]) / 2.0
        
        if np.allclose(boundaries, new_boundaries, atol=tol):
            break
        boundaries = new_boundaries
    
    # Compute MSE
    mse = 0.0
    for i in range(n_levels):
        a, b = boundaries[i], boundaries[i + 1]
        pa = 0.0 if a <= -trunc_limit else norm.cdf(a)
        pb = 1.0 if b >= trunc_limit else norm.cdf(b)
        phi_a = 0.0 if abs(a) >= trunc_limit else norm.pdf(a)
        phi_b = 0.0 if abs(b) >= trunc_limit else norm.pdf(b)
        prob = pb - pa
        if prob > 1e-15:
            ex = (phi_a - phi_b) / prob
            ex2 = 1.0 - (b * phi_b - a * phi_a) / prob
            mse += prob * (ex2 - 2 * levels[i] * ex + levels[i] ** 2)
    
    levels_sorted = np.sort(levels)
    mid_boundaries = (levels_sorted[:-1] + levels_sorted[1:]) / 2.0
    
    return {
        'n_levels': n_levels,
        'bits_per_element': math.log2(n_levels),
        'levels': levels_sorted.tolist(),
        'boundaries': mid_boundaries.tolist(),
        'mse': mse,
    }


_KNOWN_CODEBOOKS = {
    3: {
        'levels': [-1.224006, 0.0, 1.224006],
        'boundaries': [0.612003],
        'mse': 0.190,
        'name': 'Ternary',
    },
    5: {
        'levels': [-1.724147, -0.764568, 0.0, 0.764568, 1.724147],
        'boundaries': [0.382284, 1.244357],
        'mse': 0.080,
        'name': 'Quinary',
    },
    7: {
        'levels': [-2.033369, -1.188147, -0.560577, 0.0,
                   0.560577, 1.188147, 2.033369],
        'boundaries': [-1.610758, -0.874362, -0.280288,
                       0.280288, 0.874362, 1.610758],
        'mse': 0.044,
        'name': 'Septenary',
    },
}


def get_codebook(n_levels: int) -> dict:
    """Get pre-computed Lloyd-Max codebook for common sizes."""
    if n_levels in _KNOWN_CODEBOOKS:
        return _KNOWN_CODEBOOKS[n_levels]
    return lloyd_max_7(n_levels=n_levels)


def codebook_comparison() -> dict:
    """Compare MSE and bitrate across all known codebooks."""
    return {
        name: {
            'levels': cb['levels'],
            'bits': math.log2(cb['n_levels']) if 'n_levels' in cb
                    else math.log2(len(cb['levels'])),
            'mse': cb['mse'],
        }
        for name, cb in [
            ('Ternary', get_codebook(3)),
            ('Quinary', get_codebook(5)),
            ('Septenary', get_codebook(7)),
        ]
    }


if __name__ == '__main__':
    import json
    print("=" * 55)
    print("  Lloyd-Max Codebooks for N(0,1)")
    print("=" * 55)
    
    for n in [3, 5, 7]:
        cb = get_codebook(n)
        bpe = math.log2(n)
        print(f"\n  {cb['name']} ({n} levels, {bpe:.4f} bits):")
        print(f"    Levels:     {[round(v, 6) for v in cb['levels']]}")
        print(f"    Boundaries: {[round(v, 6) for v in cb['boundaries']]}")
        print(f"    MSE:        {cb['mse']:.6f}")
    
    print(f"\n  Saved to math/lloyd_max_codebooks.json")
    Path(__file__).parent.parent / 'math' / 'lloyd_max_codebooks.json'
    with open(Path(__file__).parent.parent / 'math' / 'lloyd_max_codebooks.json', 'w') as f:
        json.dump(codebook_comparison(), f, indent=2)
