# AI-in-3D-Printing

Optimization of Hamiltonian paths and cycles on grid graphs for 3D printing toolpath generation using metaheuristic algorithms and deep learning.

## Overview

This project addresses toolpath optimization for 3D printing by minimizing zone crossings in Hamiltonian cycles on grid graphs. The system uses Simulated Annealing (SA) and Genetic Algorithms (GA) to optimize paths, then trains a CNN+RNN model to learn optimal operation sequences from the collected data.

## Project Structure

```
AI-in-3D-Printing/
|-- src/                              # Source code packages
|   |-- core/                         # Core data structures
|   |   |-- hamiltonian.py            # HamiltonianSTL class
|   |   |-- zones.py                  # Zone generation functions
|   |
|   |-- optimization/                 # Optimization algorithms
|   |   |-- simulated_annealing.py    # SA with left-right zones
|   |   |-- sa_patterns.py            # SA with multiple zone patterns
|   |   |-- genetic_algorithm.py      # GA over operation sequences
|   |
|   |-- data/                         # Data collection
|   |   |-- collector.py              # Core collection infrastructure
|   |   |-- collector_helper.py       # Helper functions
|   |   |-- collect_full.py           # Full-scale collection config
|   |   |-- collect_medium.py         # Medium-scale collection
|   |   |-- collect_quick.py          # Quick test collection
|   |
|   |-- ml/                           # Machine learning
|       |-- cnn_rnn.py                # CNN+RNN model definition
|
|-- scripts/                          # Entry point scripts
|   |-- main.py                       # Pipeline orchestrator
|   |-- evaluate_model.py             # Model evaluation
|   |-- optimize_paths.py             # Path optimization demos
|
|-- docs/                             # Documentation
|   |-- documentation.md              # Technical documentation
|
|-- reports/                          # Generated reports
|-- Dataset/                          # Training data (gitignored)
|-- models/                           # Trained models
```

## Core Components

### Hamiltonian Path Operations (`src/core/hamiltonian.py`)

Foundation class `HamiltonianSTL` for managing grid-based Hamiltonian paths:
- Edge representation using horizontal `H` and vertical `V` boolean matrices
- Initial path generators: `zigzag()`, `hilbert()`, `snake_bends()`, `fermat_spiral()`
- Local transformation operations:
  - `transpose_subgrid()`: 3x3 subgrid transformations with 8 variants
  - `flip_subgrid()`: 3x2 or 2x3 subgrid flips with 4 variants
- Path validation ensuring Hamiltonicity is preserved

### Optimization Algorithms (`src/optimization/`)

**simulated_annealing.py** - Simulated Annealing implementation with zone-crossing minimization:
- `HamiltonianZoningSA`: Manages zone assignments and computes crossings
- Dynamic temperature scheduling with sigmoid-based cooling
- Move pool system with periodic refresh for efficient exploration
- Reheating mechanism to escape local minima

**sa_patterns.py** - Extended SA with multiple zone patterns:
- Zone patterns: `islands`, `diagonal`, `stripes_v`, `stripes_h`, `voronoi`, `left_right`
- Configurable zone generation parameters

**genetic_algorithm.py** - Genetic algorithm operating on operation sequences:
- Population-based search over sequences of local operations
- Strict validity constraints ensuring Hamiltonicity
- Fitness based on zone-crossing reduction

### Data Collection (`src/data/`)

**collector.py / collector_helper.py** - Dataset generation infrastructure:
- `StateRow`: Grid state snapshots with zone patterns and crossing counts
- `ActionRow`: Operation attempts with validity, delta crossings, and rewards
- Feature tensor generation saved as compressed `.npz` files
- CSV logging for states and actions

**collect_*.py** - Data collection configurations:
- `collect_full.py`: Full-scale collection across all parameters
- `collect_medium.py`: Medium-scale (96 runs, ~3 hours)
- `collect_quick.py`: Quick test runs for validation

### Machine Learning (`src/ml/cnn_rnn.py`)

Deep learning model for sequence prediction:
- CNN extracts spatial features from grid states
- GRU processes temporal sequences of states
- Multi-head prediction: operation type, subgrid kind, and orientation
- Train/validation/test split based on random seeds (0-6 train, 7-8 val, 9 test)

## Key Concepts

### Edge Representation
```python
# Horizontal edges: H[y][x] connects (x,y) to (x+1,y)
# Vertical edges: V[y][x] connects (x,y) to (x,y+1)
```

### Operation Variants
- **Transpose**: 8 variants (`sr`, `wa`, `sl`, `ea`, `nl`, `eb`, `nr`, `wb`)
- **Flip**: 4 variants (`n`, `s` for 3x2, `e`, `w` for 2x3)

### Zone Patterns
Zones partition the grid to define crossing penalties. Each cell `(x,y)` maps to a zone ID. Available patterns include islands, diagonal splits, stripes, and Voronoi diagrams.

## Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Run Complete Pipeline
```bash
python scripts/main.py
```

### Data Collection
Configure parameters in collection scripts, then run:
```bash
python src/data/collect_medium.py  # Medium-scale (~3 hours)
python src/data/collect_quick.py   # Quick test
python src/data/collect_full.py    # Full-scale
```
This generates `Dataset/` containing:
- `states.csv`: Grid snapshots with metadata
- `actions.csv`: All operation attempts with outcomes
- `features/*.npz`: Feature tensors for CNN input

### Training
```bash
python src/ml/cnn_rnn.py
```
Requires `Dataset/` with collected data. Trains the model to predict optimal operation sequences.

### Evaluation
```bash
python scripts/evaluate_model.py
```
Runs comprehensive model evaluation including:
- Model architecture analysis
- Dataset statistics
- Optimization benchmarks across grid sizes
- Zone pattern comparison tests
- Results saved to `reports/`

## Model Performance

### Trained Model Summary
| Metric | Value |
|--------|-------|
| Total Parameters | 266,417 |
| Model Size | 1.02 MB |
| Training Instances | 96 |
| Labeled Actions | 10,241 |

### Architecture
```
CNN: 2 conv layers (32 → 64 filters) → AdaptiveAvgPool2d(4,4) → FC(128)
RNN: GRU with 128 hidden units
Heads: 3 output heads (op, subgrid_kind, orientation)
```

### Optimization Results (SA Benchmark)

| Grid Size | Initial Crossings | Final (Avg) | Reduction |
|-----------|-------------------|-------------|-----------|
| 20×20 | 20 | 10.0 | **50.0%** |
| 25×25 | 25 | 17.0 | 32.0% |
| 30×30 | 30 | 26.7 | 11.1% |

### Zone Pattern Comparison (25×25 Grid)

| Pattern | Final Crossings | Time |
|---------|-----------------|------|
| left_right | 15 | 11.9s |
| diagonal | 25 | 11.5s |
| stripes_v | 30 | 10.9s |
| voronoi | 29 | 12.3s |

### Training Dataset
- **States**: 10,241 grid snapshots
- **Actions**: 9,626,592 total attempts (33,342 valid)
- **Grid Sizes**: 20x20, 30x30, 40x40, 50x50
- **Zone Patterns**: left_right, diagonal, stripes, voronoi

## Documentation

See [docs/documentation.md](docs/documentation.md) for comprehensive technical documentation including:
- Detailed module descriptions
- Data flow diagrams
- Configuration parameters
- API reference

## Dependencies
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0

## License
See LICENSE file for details. 
