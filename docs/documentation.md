# AI-in-3D-Printing - Technical Documentation

This document provides detailed technical documentation for each module in the project,
explaining their purpose, functionality, and interconnections.

---

## Table of Contents

1. [Project Architecture](#project-architecture)
2. [Core Module (src/core/)](#core-module)
3. [Optimization Module (src/optimization/)](#optimization-module)
4. [Data Collection Module (src/data/)](#data-collection-module)
5. [Machine Learning Module (src/ml/)](#machine-learning-module)
6. [Scripts (scripts/)](#scripts)
7. [Data Flow](#data-flow)
8. [Configuration](#configuration)

---

## Project Architecture

```
AI-in-3D-Printing/
|
|-- src/                          # Source code packages
|   |-- __init__.py
|   |-- core/                     # Core data structures
|   |   |-- __init__.py
|   |   |-- hamiltonian.py        # HamiltonianSTL class
|   |   |-- zones.py              # Zone generation functions
|   |
|   |-- optimization/             # Optimization algorithms
|   |   |-- __init__.py
|   |   |-- simulated_annealing.py    # SA with left-right zones
|   |   |-- sa_patterns.py            # SA with multiple zone patterns
|   |   |-- genetic_algorithm.py      # GA over operation sequences
|   |
|   |-- data/                     # Data collection
|   |   |-- __init__.py
|   |   |-- collector.py          # Core collection infrastructure
|   |   |-- collector_helper.py   # Helper functions
|   |   |-- collect_full.py       # Full-scale collection config
|   |   |-- collect_medium.py     # Medium-scale collection config
|   |   |-- collect_quick.py      # Quick test collection config
|   |
|   |-- ml/                       # Machine learning
|       |-- __init__.py
|       |-- cnn_rnn.py            # CNN+RNN model definition
|
|-- scripts/                      # Entry point scripts
|   |-- main.py                   # Pipeline orchestrator
|   |-- evaluate_model.py         # Model evaluation
|   |-- optimize_paths.py         # Path optimization demos
|
|-- docs/                         # Documentation
|   |-- documentation.md          # This file
|
|-- reports/                      # Generated reports
|   |-- EVALUATION_REPORT.txt
|   |-- evaluation_results.json
|
|-- Dataset/                      # Training data (gitignored)
|   |-- states.csv
|   |-- actions.csv
|   |-- features/*.npz
|
|-- models/                       # Trained models
    |-- global_seq_policy.pt
```

---

## Core Module

### src/core/hamiltonian.py

**Purpose**: Defines the fundamental `HamiltonianSTL` class for managing grid-based Hamiltonian paths.

**Key Class**: `HamiltonianSTL`

**Edge Representation**:
- `H[y][x]`: Boolean matrix for horizontal edges. True if edge connects (x,y) to (x+1,y)
- `V[y][x]`: Boolean matrix for vertical edges. True if edge connects (x,y) to (x,y+1)

**Initial Path Patterns**:
| Method | Description | Constraints |
|--------|-------------|-------------|
| `zigzag()` | Default serpentine pattern | Any grid size |
| `snake_bends()` | Alternating vertical traversal | Best for odd dimensions |
| `hilbert()` | Space-filling Hilbert curve | Square grids of size 2^n |
| `fermat_spiral()` | Inward spiral pattern | Any grid size |

**Transformation Operations**:

1. **Transpose (3x3 subgrid)**: 8 variants
   - Compass-based naming: `sr`, `wa`, `sl`, `ea`, `nl`, `eb`, `nr`, `wb`
   - Each variant defines old/new edge patterns in `transpose_patterns` dict

2. **Flip (3x2 or 2x3 subgrid)**: 4 variants
   - `n`, `s`: 3x2 vertical subgrid flips
   - `e`, `w`: 2x3 horizontal subgrid flips

**Key Methods**:
```python
h = HamiltonianSTL(width, height)     # Initialize with zigzag
sub = h.get_subgrid((x1,y1), (x2,y2)) # Extract subgrid
h.transpose_subgrid(sub, 'sr')        # Apply transpose
h.flip_subgrid(sub, 'n')              # Apply flip
h.validate_full_path()                 # Check Hamiltonicity
```

### src/core/zones.py

**Purpose**: Generate zone partitions for the grid to define crossing penalties.

**Zone Functions**:

| Function | Description | Parameters |
|----------|-------------|------------|
| `zones_left_right(W, H)` | Vertical split into 2 zones | Width, Height |
| `zones_top_bottom(W, H)` | Horizontal split into 2 zones | Width, Height |
| `zones_diagonal(W, H)` | Diagonal split into 2 zones | Width, Height |
| `zones_stripes(W, H, direction, k)` | k parallel stripes | 'v' or 'h', count |
| `zones_checkerboard(W, H, size)` | Checkerboard pattern | Square size |
| `zones_voronoi(W, H, k, seed)` | Voronoi diagram with k seeds | Seed count, random seed |

**Return Format**: `Dict[Tuple[int,int], int]` mapping (x,y) to zone ID (1-indexed)

---

## Optimization Module

### src/optimization/simulated_annealing.py

**Purpose**: Simulated Annealing optimization for minimizing zone crossings.

**Key Classes**:

1. **HamiltonianZoningSA**
   - Wraps HamiltonianSTL with zone assignments
   - Computes zone crossings as optimization objective
   - Applies moves and visualizes path

2. **ZoningAdapterForSA**
   - Adapter for data collection compatibility
   - Tracks step counter for sequence building

**Temperature Schedule**:
```python
T(l) = Tmin + (Tmax - Tmin) * sigmoid(k * (-l + L/2))
```
Where l = current iteration, L = total iterations

**Move Pool System**:
- Pre-computes feasible moves for efficiency
- Biases toward zone boundaries
- Periodically refreshes as path changes

**Reheating Mechanism**:
- Triggers after `reheat_patience` iterations without improvement
- Increases Tmax by `reheat_factor` up to `reheat_cap`

**Main Function**:
```python
best_cost, best_ops = run_sa(
    width=32, height=32,
    iterations=2000,
    Tmax=80.0, Tmin=0.5,
    seed=42,
    plot_live=True,
    dataset_dir="Dataset"
)
```

### src/optimization/sa_patterns.py

**Purpose**: Extended SA supporting multiple zone patterns.

**Zone Modes**:
- `islands`: Non-overlapping rectangular islands
- `diagonal`: Diagonal split
- `stripes`: Parallel stripes (vertical/horizontal)
- `voronoi`: Voronoi diagram partitioning

**Additional Parameters**:
```python
run_sa(
    zone_mode="voronoi",
    voronoi_k=5,           # Number of Voronoi regions
    stripe_direction="v",   # 'v' or 'h'
    stripe_k=3,            # Number of stripes
    num_islands=3,         # Island count
    island_size=8,         # Island dimension
)
```

### src/optimization/genetic_algorithm.py

**Purpose**: Genetic Algorithm operating on operation sequences.

**Population**: Built from SA trajectory operations

**Crossover**: One-point crossover on operation sequences

**Validity Constraints**:
- Child must apply >= MIN_APPLIED_VALID operations
- Child crossings must not exceed parent crossings

---

## Data Collection Module

### src/data/collector.py

**Purpose**: Core infrastructure for logging states and actions during optimization.

**Data Classes**:

1. **RunMeta**: Run-level metadata
   ```python
   @dataclass
   class RunMeta:
       run_id: str
       timestamp: str
       algorithm: str
       init_path: str
       grid_w: int
       grid_h: int
       random_seed: int
       zones_max: int
   ```

2. **StateRow**: Grid state snapshot
   ```python
   @dataclass
   class StateRow:
       sample_id: str
       run_id: str
       instance_id: str
       step_t: int
       layer_id: int
       grid_w: int
       grid_h: int
       num_zones: int
       zone_pattern: str
       zone_params: str        # JSON
       crossings_before: int
       features_file: str      # Path to .npz
   ```

3. **ActionRow**: Operation attempt record
   ```python
   @dataclass
   class ActionRow:
       sample_id: str
       x: int, y: int
       subgrid_kind: str       # "3x3", "3x2", "2x3"
       orientation: str        # Variant name
       op: str                 # "transpose" or "flip"
       valid: int              # 1 if Hamiltonicity preserved
       crossings_before: int
       crossings_after: int
       delta_cross: int        # Improvement (positive = better)
       reward: float
       best_in_state: int      # 1 if best action for this state
   ```

**Feature Builder**:
```python
build_features_multizone(h, zones, layer_id, Z_MAX=6)
```
Returns tensor `[C, H, W]` with channels:
- Zone one-hot encoding (Z_MAX channels)
- Edge maps (2 channels: right, down)
- Boundary map (1 channel)
- Layer indicator (1 channel)

### src/data/collector_helper.py

**Purpose**: Helper functions for logged data collection during optimization.

**Main Function**:
```python
mutate_layer_logged(
    zoning,                # SA object with zones
    collector,             # ZoningCollector instance
    run_id, instance_id,
    layer_id, attempts,
    zone_pattern, zone_params,
    num_zones,
    allow_flip=True,
    Z_MAX=6
)
```

Workflow:
1. Build feature tensor from current state
2. Save tensor to .npz file
3. Log StateRow to states.csv
4. Try all operations at layer positions
5. Log ActionRows to actions.csv
6. Apply best valid operation to grid

### src/data/collect_*.py

**Purpose**: Configuration scripts for data collection at different scales.

| Script | Grid Sizes | Patterns | Instances | Iterations |
|--------|------------|----------|-----------|------------|
| collect_quick.py | 10x10, 15x15 | 2 | 2 | 500 |
| collect_medium.py | 20-50 | 4 | 3 | 2000 |
| collect_full.py | 30-200 | 6 | 10 | 5000 |

---

## Machine Learning Module

### src/ml/cnn_rnn.py

**Purpose**: CNN+RNN model for learning optimal operation sequences.

**Architecture**:
```
Input: x_states [B, T, C, H, W], cfg_vec [B, cfg_dim]

CNN Encoder (per timestep):
  Conv2d(C_in, 32, 3x3) -> ReLU
  Conv2d(32, 64, 3x3) -> ReLU
  AdaptiveAvgPool2d(4, 4)
  FC(64*16, 128) -> ReLU
  Output: state_emb [B, 128]

Config Encoder:
  FC(cfg_dim, 32) -> ReLU
  Output: cfg_emb [B, 32]

RNN:
  GRU(128+32, 128)
  Input: concat(state_emb, cfg_emb) per timestep
  Output: hidden [B, T, 128]

Output Heads (per timestep):
  head_op: Linear(128, 2)      # flip vs transpose
  head_sg: Linear(128, 3)      # 3x3, 3x2, 2x3
  head_ori: Linear(128, 12)    # 8 transpose + 4 flip variants
```

**Configuration Vector** (cfg_dim = 3 + num_patterns):
- `w_norm`: Normalized grid width
- `h_norm`: Normalized grid height
- `k_norm`: Normalized zone count
- One-hot encoded zone pattern

**Dataset Split**:
- Train: instance seeds 0-6
- Validation: instance seeds 7-8
- Test: instance seed 9

**Key Functions**:
```python
# Metadata scanning
meta = scan_action_metadata("Dataset/actions.csv")

# Pattern list
patterns = get_pattern_list("Dataset/states.csv")

# Instance splitting
train, val, test = split_instances_by_seed("Dataset/states.csv")

# Training
train_seq_model(model, train_loader, val_loader, epochs=10)

# Evaluation
evaluate_seq_model(model, test_loader, device)
```

---

## Scripts

### scripts/main.py

**Purpose**: Pipeline orchestrator for complete workflow.

**Phases**:
1. Dependency check (torch, numpy, matplotlib)
2. GPU detection
3. Data collection (imports collect_quick)
4. Model training (imports cnn_rnn)

**Usage**:
```bash
# Full pipeline
python scripts/main.py

# Skip collection, use existing data
python scripts/main.py --skip-collection

# Collection only
python scripts/main.py --collect-only

# Training only
python scripts/main.py --train-only
```

### scripts/evaluate_model.py

**Purpose**: Comprehensive model evaluation and benchmarking.

**Evaluations**:
1. Model architecture analysis
2. Dataset statistics
3. SA optimization benchmark (multiple grid sizes)
4. Zone pattern comparison

**Output**:
- `reports/EVALUATION_REPORT.txt`: Human-readable report
- `reports/evaluation_results.json`: Detailed JSON data

### scripts/optimize_paths.py

**Purpose**: Demonstrate path optimization using trained model and SA.

**Demos**:
- SA optimization on various grid sizes
- Zone pattern comparison
- Model inference (when implemented)

---

## Data Flow

```
                    +------------------+
                    |   HamiltonianSTL |
                    |   (Initial Path) |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |   Zone Generator |
                    |   (zones_*.py)   |
                    +--------+---------+
                             |
                             v
+----------------+  +------------------+  +----------------+
|  SA Optimizer  |->|  ZoningCollector |->|  Dataset/      |
| (run_sa)       |  |  (log states,    |  |  states.csv    |
|                |  |   actions, .npz) |  |  actions.csv   |
+----------------+  +------------------+  |  features/*.npz|
                                          +-------+--------+
                                                  |
                                                  v
                                          +----------------+
                                          |  CNN+RNN Model |
                                          |  (training)    |
                                          +-------+--------+
                                                  |
                                                  v
                                          +----------------+
                                          |  Trained Model |
                                          |  (.pt file)    |
                                          +----------------+
```

---

## Configuration

### Environment Variables

- `KMP_DUPLICATE_LIB_OK=TRUE`: Required for PyTorch on some systems

### Key Parameters

**SA Optimization**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `iterations` | 2000 | Total SA iterations |
| `Tmax` | 80.0 | Maximum temperature |
| `Tmin` | 0.5 | Minimum temperature |
| `pool_refresh_period` | 250 | Iterations between pool refreshes |
| `pool_max_moves` | 5000 | Maximum moves in pool |
| `reheat_patience` | 3000 | Iterations before reheat |
| `reheat_factor` | 1.5 | Temperature multiplier on reheat |
| `reheat_cap` | 600.0 | Maximum temperature after reheat |

**CNN+RNN Model**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `T_seq` | 10 | Sequence length |
| `hidden_dim` | 128 | GRU hidden dimension |
| `state_dim` | 128 | CNN output dimension |
| `cfg_dim` | 32 | Config embedding dimension |

### File Paths

- Dataset: `Dataset/`
- Models: `models/`
- Reports: `reports/`

---

## Dependencies

- **PyTorch** >= 2.0.0: Deep learning framework
- **NumPy** >= 1.24.0: Numerical operations
- **Matplotlib** >= 3.7.0: Visualization

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Running the Project

1. **Quick Test**:
   ```bash
   python scripts/main.py
   ```

2. **Medium Data Collection**:
   ```bash
   python src/data/collect_medium.py
   python src/ml/cnn_rnn.py
   ```

3. **Evaluate Model**:
   ```bash
   python scripts/evaluate_model.py
   ```

4. **Optimize Paths**:
   ```bash
   python scripts/optimize_paths.py
   ```
