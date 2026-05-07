# Mathematical Foundations of K-Septenary

This document provides the complete mathematical derivation for each component
of K-Septenary KV cache quantization.

---

## 1. Lloyd-Max Quantization for N(0,1)

### 1.1 Problem Formulation

Given a random variable $X \sim \mathcal{N}(0, 1)$ with PDF $\phi(x)$ and CDF
$\Phi(x)$, we seek $L$ quantization levels $c_1, \ldots, c_L$ and $L-1$
decision boundaries $b_1, \ldots, b_{L-1}$ that minimize the mean squared error:

$$
\text{MSE} = \mathbb{E}\left[(X - Q(X))^2\right] =
\sum_{i=1}^{L} \int_{b_{i-1}}^{b_i} (x - c_i)^2 \phi(x) \, dx
$$

where $b_0 = -\infty$ and $b_L = \infty$.

### 1.2 Optimality Conditions

The necessary conditions for optimality are:

**Centroid condition** (level equals conditional mean within its interval):

$$
\frac{\partial \text{MSE}}{\partial c_i} = 0
\;\Longrightarrow\;
c_i = \frac{\int_{b_{i-1}}^{b_i} x \phi(x) \, dx}
          {\int_{b_{i-1}}^{b_i} \phi(x) \, dx}
     = \mathbb{E}[X \mid b_{i-1} < X \leq b_i]
$$

**Boundary condition** (boundary is midpoint of adjacent levels):

$$
\frac{\partial \text{MSE}}{\partial b_i} = 0
\;\Longrightarrow\;
b_i = \frac{c_i + c_{i+1}}{2}
$$

### 1.3 Closed Form for N(0,1)

For the standard normal distribution, the centroid has a closed form.
Using $\phi'(x) = -x\phi(x)$:

$$
\int x \phi(x) \, dx = -\phi(x) + C
$$

Therefore:

$$
c_i = \frac{\phi(b_{i-1}) - \phi(b_i)}{\Phi(b_i) - \Phi(b_{i-1})}
$$

This is iterated with the boundary update until convergence.

### 1.4 Convergence

The Lloyd-Max algorithm is a special case of the Expectation-Maximization (EM)
algorithm and is guaranteed to converge monotonically to a local optimum.
The MSE decreases at each iteration:

$$
\text{MSE}^{(t+1)} \leq \text{MSE}^{(t)}
$$

For N(0,1), the algorithm converges to the global optimum within ~50 iterations.

### 1.5 7-Level Codebook

The optimal 7-level codebook for N(0,1) is:

$$
\begin{aligned}
\text{Levels:} \quad & \{-2.033369,\; -1.188147,\; -0.560577,\; 0.0,\\
                     & \quad\; 0.560577,\; 1.188147,\; 2.033369\} \\[4pt]
\text{Boundaries:} \quad & \{-1.610758,\; -0.874362,\; -0.280288,\\
                         & \quad\; 0.280288,\; 0.874362,\; 1.610758\} \\[4pt]
\text{MSE:} \quad & 0.044
\end{aligned}
$$

#### Comparison with Fewer Levels

| Levels | $L=3$ (Ternary) | $L=5$ (Quinary) | $L=7$ (Septenary) |
|--------|:-:|:-:|:-:|
| MSE | 0.190 | 0.080 | **0.044** |
| Improvement | — | 2.38× | **4.32×** vs ternary |
| Bit rate | 1.585 | 2.322 | **2.807** |

### 1.6 Information-Theoretic Interpretation

The rate-distortion function for a unit-variance Gaussian source under MSE
distortion is [Shannon, 1948]:

$$
R(D) = \frac{1}{2} \log_2 \frac{1}{D} \quad \text{bits/sample}
$$

For K-Septenary's 7-level quantizer with $D = 0.044$:

$$
R(0.044) = \frac{1}{2} \log_2 \frac{1}{0.044} = 2.25 \text{ bits}
$$

Our actual bitrate is $\log_2 7 = 2.807$ bits. The gap of $2.807 - 2.25 = 0.56$
bits is the price of finite-level scalar quantization vs. the theoretical
optimal (infinite-dimensional vector quantizer).

---

## 2. Per-Channel Scaling

### 2.1 Motivation

KV cache activations exhibit significant variance across dimensions.
This is a consequence of the attention mechanism: different dimensions encode
different positional and semantic features.

Let $\mathbf{x}_i \in \mathbb{R}^{S \times d}$ be the activations for
dimension $i$ across a sequence of length $S$. Each dimension has its own
empirical standard deviation $\sigma_i$.

### 2.2 Scaling Procedure

For each dimension $i$, we compute:

$$
\sigma_{\text{max}, i} = \max_{1 \leq j \leq S} |x_{ij}|
$$

We then scale each dimension independently:

$$
z_{ij} = \frac{x_{ij}}{\sigma_{\text{max}, i} / 3}
$$

The factor 3 maps the max to approximately 3 standard deviations, ensuring
that the scaled data $z_{ij} \sim \mathcal{N}(0, 1)$ approximates the optimal
codebook distribution.

### 2.3 Storage Overhead

The per-dimension scale factors are stored as FP32 (4 bytes each):

$$
\text{Overhead per head} = d \times 4 \text{ bytes} = 256 \times 4 = 1024 \text{ bytes}
$$

Amortized over sequence length $S$:

$$
\text{Effective overhead} = \frac{1024 \times 8}{S \times 256} = \frac{32}{S} \text{ bits/element}
$$

At $S=256$: $+0.125$ bpe. At $S=4096$: $+0.008$ bpe. At $S=65536$: $+0.0005$ bpe.

For long-context inference ($S \geq 4096$), the overhead is negligible.

---

## 3. Tagged Stream Format

### 3.1 Encoding Efficiency

A 3-bit integer can represent $2^3 = 8$ distinct values. Standard septenary
quantization uses 7 of these (000-110), leaving 111 unused.

**State utilization:**

$$
\eta = \frac{L}{2^B} = \frac{7}{8} = 87.5\%
$$

### 3.2 Tag Scheme

We repurpose the unused state 111 as part of a block-level tag. Each 256-element
block (one head-dimension vector) is preceded by a 3-bit tag:

```
tag ∈ {000, 001, 010, 011, 100, 101, 110, 111}
```

Where:

$$
\text{tensor}(t) =
\begin{cases}
K_{\text{head }t}, & 0 \leq t \leq 3 \\
V_{\text{head }(t-4)}, & 4 \leq t \leq 7
\end{cases}
$$

### 3.3 Overhead

Per 256-element block, one 3-bit tag is added:

$$
\text{Overhead} = \frac{3}{256 \times 3} = \frac{1}{256} \approx 0.39\%
$$

### 3.4 Complete Bit Budget

The total bits per element in the tagged stream:

$$
b_{\text{total}} = \underbrace{\log_2 7}_{\text{data}} +
                   \underbrace{\frac{3}{256}}_{\text{tag}} +
                   \underbrace{\frac{32}{S}}_{\text{per-channel scales}}
$$

At $S = 65536$:

$$
b_{\text{total}} = 2.807 + 0.012 + 0.0005 = 2.819 \text{ bits/element}
$$

---

## 4. Memory Analysis

### 4.1 Model Parameters

For Qwen3.5-27B:

| Parameter | Value |
|-----------|-------|
| Layers ($L$) | 64 |
| KV heads ($H$) | 4 |
| Head dimension ($d$) | 256 |
| Context length ($S$) | 262,144 |

### 4.2 FP16 Baseline

Total KV cache size:

$$
\begin{aligned}
M_{\text{FP16}} &= 2 \times L \times H \times d \times S \times 2 \text{ bytes} \\
                &= 2 \times 64 \times 4 \times 256 \times 262144 \times 2 \\
                &= 64.0 \text{ GB}
\end{aligned}
$$

### 4.3 K-Septenary

$$
\begin{aligned}
M_{\text{K-Sep}} &= \left( 2 \times L \times H \times d \times S \times \frac{3}{8} \right) \times (1 + 0.0039) \\
                 &= \left( 2 \times 64 \times 4 \times 256 \times 262144 \times 0.375 \right) \times 1.0039 \\
                 &= 12.05 \text{ GB}
\end{aligned}
$$

### 4.4 Savings

$$
\text{Savings} = 1 - \frac{M_{\text{K-Sep}}}{M_{\text{FP16}}}
               = 1 - \frac{12.05}{64.0}
               = 81.2\%
$$

---

## 5. Why Simpler Is Better

### 5.1 The EDEN Pipeline

EDEN [2024] applies three transformations:

$$
\mathbf{y} = \mathbf{Hx}, \quad
q_i = Q\left(\frac{y_i}{\sigma}\right), \quad
\hat{x} = s \cdot \mathbf{H}^{-1}\mathbf{q}
$$

Where $\mathbf{H}$ is a random Hadamard matrix, $Q$ is scalar quantization,
and $s$ is a per-vector learned scale.

### 5.2 Why We Drop Each Component

**Hadamard rotation** spreads information across dimensions. After rotation,
each dimension has similar variance, making per-channel scaling ineffective.
This can be shown:

$$
\text{Var}[(\mathbf{Hx})_i] = \frac{1}{d} \sum_{j=1}^d \text{Var}[x_j]
$$

After rotation, all dimensions have the same variance, negating the benefit
of per-channel normalization.

**EDEN S_bias** provides a per-vector refinement. But with per-channel scaling
already in place, the residual error after quantization is uncorrelated with
the signal, so S_bias cannot improve it:

$$
\mathbb{E}[S \mid \mathbf{q}, \mathbf{y}] = \frac{\mathbf{y} \cdot \mathbf{q}}
                                                 {\|\mathbf{q}\|^2}
$$

For coarser codebooks (ternary), this correction is meaningful because the
quantization error is large and correlated. For septenary (MSE = 0.044),
the error is small and the correction adds noise.

### 5.3 K-Septenary Minimal Pipeline

The complete compression pipeline:

$$
\hat{x}_{ij} = c_k \cdot \sigma_i,
\quad\text{where}\quad
k = \underset{1 \leq j \leq 7}{\arg\min}\,
\left| \frac{x_{ij}}{\sigma_i/3} - c_j \right|
$$

where $c_j$ are the 7 Lloyd-Max centroids and $\sigma_i$ is the per-dimension
max absolute value. The tagged stream then adds a 3-bit header to each
256-element block without any further transformation.

---

## References

1. Lloyd, S. (1982). "Least Squares Quantization in PCM." *IEEE Trans. Info. Theory*.
2. Max, J. (1960). "Quantizing for Minimum Distortion." *IRE Trans. Info. Theory*.
3. Shannon, C. E. (1948). "A Mathematical Theory of Communication." *Bell System Tech. J.*
4. Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory.* Wiley.
5. Gersho, A. & Gray, R. M. (1992). *Vector Quantization and Signal Compression.* Springer.
