# SA Optimization: Mathematical Formulation

## Grid Graph

Let $G = (V, E)$ be a grid graph on $W \times H$ vertices where

$$V = \{(x,y) : 0 \le x < W,\; 0 \le y < H\}$$

The edge set is represented by two boolean matrices:

- $\mathbf{H}[y][x] \in \{0,1\}$ — horizontal edge between $(x,y)$ and $(x+1,y)$, for $0 \le x < W-1$
- $\mathbf{V}[y][x] \in \{0,1\}$ — vertical edge between $(x,y)$ and $(x,y+1)$, for $0 \le y < H-1$

## Hamiltonian Cycle Constraint

The edge matrices $(\mathbf{H}, \mathbf{V})$ must encode a **Hamiltonian cycle**: a single closed path visiting every vertex exactly once. Every vertex has degree exactly 2.

## Zone Function

A zone assignment $z: V \to \{0, 1, \ldots, K-1\}$ partitions the grid into $K$ material regions. Six patterns are implemented:

| Pattern | Definition |
|---------|-----------|
| `left_right` | $z(x,y) = \lfloor x \,/\, (W/K) \rfloor$ |
| `top_bottom` | $z(x,y) = \lfloor y \,/\, (H/K) \rfloor$ |
| `stripes` | Periodic bands (vertical or horizontal) |
| `checkerboard` | $z(x,y) = (x+y) \bmod 2$ |
| `voronoi` | $z(x,y) = \arg\min_k \lVert (x,y) - s_k \rVert$ for random seeds $s_k$ |
| `islands` | Background $= 0$, square regions $= 1$ |

## Cost Function (Zone Crossings)

$$C(\mathbf{H}, \mathbf{V}, z) = \sum_{y=0}^{H-1}\sum_{x=0}^{W-2} \mathbf{H}[y][x] \cdot \mathbf{1}\!\big[z(x,y) \neq z(x\!+\!1,y)\big] \;+\; \sum_{y=0}^{H-2}\sum_{x=0}^{W-1} \mathbf{V}[y][x] \cdot \mathbf{1}\!\big[z(x,y) \neq z(x,y\!+\!1)\big]$$

Each term counts an edge that is **both present in the path and crosses a zone boundary**. Every such crossing requires a material change during 3D printing.

## Decision Variables (Operations)

The optimizer modifies $(\mathbf{H}, \mathbf{V})$ via two families of **path-preserving local operations**:

### Transpose $\tau_v(x,y)$

Acts on a $3\times 3$ subgrid anchored at $(x,y)$. The 9 vertices are indexed $0$–$8$ in row-major order:

$$\begin{pmatrix} 0 & 1 & 2 \\ 3 & 4 & 5 \\ 6 & 7 & 8 \end{pmatrix}$$

Each variant $v \in \{\text{sr, sl, nl, nr, ea, wa, eb, wb}\}$ is a deterministic edge rewiring: replace edge set $E_{\text{old}}$ with edge set $E_{\text{new}}$. Both sets contain exactly 7 edges (the Hamiltonian path uses 7 of the 12 possible grid edges in a $3\times3$ block).

### Flip $\phi_v(x,y)$

Acts on a $2\times 3$ (variants $n, s$) or $3\times 2$ (variants $e, w$) subgrid. The 6 vertices are indexed $0$–$5$:

$$2\times3:\quad\begin{pmatrix} 0 & 1 \\ 2 & 3 \\ 4 & 5 \end{pmatrix} \qquad 3\times2:\quad\begin{pmatrix} 0 & 1 & 2 \\ 3 & 4 & 5 \end{pmatrix}$$

Each variant replaces 4 edges with 4 new edges, reversing the path traversal direction within the subgrid.

### Feasibility

Both operations are **conditional**: they only apply if the current edge configuration exactly matches $E_{\text{old}}$ **and** the result still forms a valid Hamiltonian cycle. Otherwise the state is unchanged.

## Optimization Problem

$$\min_{(\mathbf{H}', \mathbf{V}')} \; C(\mathbf{H}', \mathbf{V}', z)$$

subject to:

1. $(\mathbf{H}', \mathbf{V}')$ encodes a Hamiltonian cycle on the $W \times H$ grid
2. $(\mathbf{H}', \mathbf{V}')$ is reachable from initial state $(\mathbf{H}_0, \mathbf{V}_0)$ via a finite sequence of transpose/flip operations

## SA Algorithm

Standard Metropolis–Hastings with a sigmoid temperature schedule:

$$T(i) = T_{\min} + (T_{\max} - T_{\min}) \cdot \sigma\!\left(\frac{10}{\texttt{iters}}\cdot\left(\frac{\texttt{iters}}{2} - i\right)\right)$$

where $\sigma(x) = \frac{1}{1+e^{-x}}$.

At iteration $i$, a candidate move $m$ is sampled from a precomputed pool (biased toward zone boundaries). If the move applies successfully:

$$\Delta = C_{\text{new}} - C_{\text{current}}$$

$$P(\text{accept}) = \begin{cases} 1 & \text{if } \Delta < 0 \\[4pt] \exp\!\left(\dfrac{-\Delta}{T(i)}\right) & \text{otherwise} \end{cases}$$

### Two-Phase Strategy

| Phase | Iterations | Operation mix |
|-------|-----------|---------------|
| **Phase 1** | $0$ to $\lfloor 0.6 \cdot \texttt{iters}\rfloor$ | 98% transpose, 2% flip |
| **Phase 2** | remainder | 90% flip, 10% transpose |

### Reheating

After $p$ consecutive iterations with no improvement to the best-known cost:

$$T_{\max} \leftarrow \min\!\big(T_{\text{cap}},\; T_{\max} \cdot 1.5\big)$$

The move pool is also refreshed upon reheating.

### Move Pool

Every 250 iterations, a pool of feasible moves is rebuilt by:

1. Enumerating candidate $(x,y)$ positions biased toward zone boundaries
2. Testing each (position, variant) pair for pattern match feasibility
3. Retaining up to 5000 feasible moves

### Key Empirical Property

> Every effective operation (one that improves the global best cost) reduces crossings by exactly 2.

This means effective operations cluster at zone boundaries and the theoretical minimum number of effective operations for a run is $\frac{C_0 - C^*}{2}$.
