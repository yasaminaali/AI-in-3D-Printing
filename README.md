# Zone Crossing Minimization for Neural Toolpath Optimization

This project generates training data for deep learning models that predict optimal sequences of Flip/Transpose operations to minimize zone crossings in Hamiltonian toolpaths for multi-material 3D printing.

## Problem Overview

In multi-material 3D printing, the print head must traverse all points in a layer while minimizing transitions between different material zones. Each zone transition (crossing) requires a material change, which:
- Slows down the printing process
- Can affect print quality
- Increases wear on the print head

This project uses **Simulated Annealing (SA)** to find good operation sequences that reduce zone crossings, and records these sequences as training data for neural networks.

## Project Structure

```
AI-in-3D-Printing/
├── README.md                 # This file
├── operations.py             # HamiltonianSTL class - grid operations
├── Zones.py                  # Zone pattern generators
├── SA_generation.py          # Simulated Annealing + dataset generation
├── Dataset/                  # Generated dataset output (JSONL files)
└── DATA_GENERATION_SPEC.md   # Detailed specification for dataset generation
```

## File Descriptions

### operations.py
Core module providing the `HamiltonianSTL` class for manipulating Hamiltonian paths on grid graphs.

**Features:**
- **Initial path patterns:** zigzag, snake_bends, hilbert, fermat_spiral
- **Transpose operations:** 8 variants (sr, wa, sl, ea, nl, eb, nr, wb) on 3×3 subgrids
- **Flip operations:** 4 variants (w, e, n, s) on 3×2 or 2×3 subgrids
- **Path validation:** Ensures Hamiltonian property is preserved
- **ASCII visualization:** Debug output of grid state

### Zones.py
Zone pattern generators for simulating multi-material print regions.

**Available Patterns:**
| Pattern | Description | Parameters |
|---------|-------------|------------|
| `zones_left_right` | Vertical bands | k (number of bands) |
| `zones_top_bottom` | Horizontal bands | k (number of bands) |
| `zones_diagonal` | Diagonal split | Random offset |
| `zones_stripes` | Parallel stripes | direction (v/h), k |
| `zones_checkerboard` | Alternating grid | kx, ky |
| `zones_voronoi` | Irregular regions | k (seed points) |

### SA_generation.py
Simulated Annealing optimizer and dataset generator.

**Key Functions:**
- `run_sa()` - Single SA optimization run
- `run_sa_multiple_seeds()` - Batch runs with different seeds
- `compute_crossings()` - Count zone boundary crossings
- `save_sa_dataset_record()` - Write results to JSONL

**SA Configurations:**
| Config | Iterations | Tmax | Tmin | Use Case |
|--------|------------|------|------|----------|
| short | 1,000 | 60 | 0.5 | High-crossing trajectories |
| medium | 3,000 | 80 | 0.5 | Balanced optimization |
| long | 8,000 | 100 | 0.3 | Near-optimal solutions |
| extra_long | 15,000 | 120 | 0.2 | Large grids (64×64+) |

## Installation

### Prerequisites
- Python 3.8+
- Required packages: `matplotlib`, `numpy`

```bash
pip install matplotlib numpy
```

## Usage

### Basic Example

```python
from SA_generation import run_sa, run_sa_multiple_seeds

# Single optimization run
best_crossings, operations = run_sa(
    width=32,
    height=32,
    iterations=3000,
    zone_mode="left_right",
    write_dataset=True
)
print(f"Reduced crossings to {best_crossings} with {len(operations)} operations")
```

### Generate Training Dataset

```python
from SA_generation import run_sa_multiple_seeds

# Generate data for 100 different seeds
results = run_sa_multiple_seeds(
    seeds=list(range(100)),
    width=32,
    height=32,
    iterations=3000,
    Tmax=80.0,
    Tmin=0.5,
    zone_mode="left_right",
    dataset_dir="Dataset",
    write_dataset=True
)
```

### Using Different Zone Patterns

```python
# Stripes pattern (3 vertical bands)
run_sa(zone_mode="stripes", stripe_k=3, stripe_direction="v")

# Islands pattern (3 square islands on background)
run_sa(zone_mode="islands", num_islands=3, island_size=8)

# Voronoi pattern (3 irregular regions)
run_sa(zone_mode="voronoi", voronoi_k=3)
```

## Dataset Format

Output is written to `Dataset/Dataset.jsonl` as JSON Lines format:

```json
{
  "run_id": "sa_left_right_W32H32_seed0_1706745600",
  "seed": 0,
  "grid_W": 32,
  "grid_H": 32,
  "zone_pattern": "left_right",
  "zone_grid": [0, 0, 0, ..., 1, 1, 1],
  "initial_crossings": 32,
  "final_crossings": 4,
  "sequence_len": 156,
  "sequence_ops": [
    {"kind": "T", "x": 5, "y": 3, "variant": "sr"},
    {"kind": "F", "x": 10, "y": 7, "variant": "w"},
    ...
  ],
  "runtime_sec": 12.5
}
```

**Field Descriptions:**
| Field | Description |
|-------|-------------|
| `run_id` | Unique identifier for the run |
| `seed` | Random seed used |
| `grid_W`, `grid_H` | Grid dimensions |
| `zone_pattern` | Type of zone pattern |
| `zone_grid` | Flattened zone assignments (row-major, normalized to 0..K-1) |
| `initial_crossings` | Crossings before optimization |
| `final_crossings` | Crossings after optimization |
| `sequence_len` | Number of operations applied |
| `sequence_ops` | List of operations: T=Transpose, F=Flip |
| `runtime_sec` | Execution time in seconds |

## Algorithm Details

### Simulated Annealing Process
1. **Initialization:** Create zigzag Hamiltonian path
2. **Build move pool:** Find all feasible transpose/flip operations
3. **Main loop:**
   - Select random move from pool
   - Compute new crossing count
   - Accept if better, or probabilistically if worse (Metropolis criterion)
   - Periodically refresh move pool and apply reheating if stuck
4. **Output:** Best operation sequence and final crossing count

### Temperature Schedule
Uses sigmoid-based cooling:
```
T(i) = Tmin + (Tmax - Tmin) * sigmoid(k * (iters/2 - i))
```

### Reheating Strategy
When stuck (no improvement for `reheat_patience` iterations):
- Multiply Tmax by `reheat_factor` (up to `reheat_cap`)
- Rebuild move pool

## Contributing

This project is part of ongoing research in AI-assisted 3D printing optimization.

## License

[Add appropriate license information]
