# AI-in-3D-Printing

Optimization of Hamiltonian paths and cycles on grid graphs for 3D printing toolpath generation using metaheuristic algorithms and deep learning.

## Overview

This project addresses toolpath optimization for 3D printing by minimizing zone crossings in Hamiltonian cycles on grid graphs. The system uses Simulated Annealing (SA) and Genetic Algorithms (GA) to optimize paths, then trains a CNN+RNN model to learn optimal operation sequences from the collected data.

## Core Components

### Hamiltonian Path Operations

**Flip_Transpose2.py** - Foundation class `HamiltonianSTL` for managing grid-based Hamiltonian paths:
- Edge representation using horizontal `H` and vertical `V` boolean matrices
- Initial path generators: `zigzag()`, `hilbert()`, `snake_bends()`, `fermat_spiral()`
- Local transformation operations:
  - `transpose_subgrid()`: 3x3 subgrid transformations with 8 variants
  - `flip_subgrid()`: 3x2 or 2x3 subgrid flips with 4 variants
- Path validation ensuring Hamiltonicity is preserved

### Optimization Algorithms

**SA.py** - Simulated Annealing implementation with zone-crossing minimization:
- `HamiltonianZoningSA`: Manages zone assignments and computes crossings
- Dynamic temperature scheduling with sigmoid-based cooling
- Move pool system with periodic refresh for efficient exploration
- Reheating mechanism to escape local minima

**SA_patterns.py** - Extended SA with multiple zone patterns:
- Zone patterns: `islands`, `diagonal`, `stripes_v`, `stripes_h`, `voronoi`, `left_right`
- Configurable zone generation parameters

**ga_sequence.py** - Genetic algorithm operating on operation sequences:
- Population-based search over sequences of local operations
- Strict validity constraints ensuring Hamiltonicity
- Fitness based on zone-crossing reduction

### Data Collection and Training

**Collector.py / Collector_helper.py** - Dataset generation infrastructure:
- `StateRow`: Grid state snapshots with zone patterns and crossing counts
- `ActionRow`: Operation attempts with validity, delta crossings, and rewards
- Feature tensor generation saved as compressed `.npz` files
- CSV logging for states and actions

**Collect_SA.py** - Orchestrates data collection:
- Runs SA across multiple grid sizes, zone patterns, and random seeds
- Configurable hyperparameters for iterations, temperature, and move strategies
- Generates structured dataset in `Dataset/` directory

**CNN+RNN.py** - Deep learning model for sequence prediction:
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

### Data Collection
Configure parameters in `Collect_SA.py`, then run:
```bash
python Collect_SA.py
```
This generates `Dataset/` containing:
- `states.csv`: Grid snapshots with metadata
- `actions.csv`: All operation attempts with outcomes
- `features/*.npz`: Feature tensors for CNN input

### Training
```bash
python CNN+RNN.py
```
Requires `Dataset/` with collected data. Trains the model to predict optimal operation sequences.

## Project Structure
```
Flip_Transpose2.py      # Hamiltonian path operations and edge management
SA.py                   # Simulated annealing with left-right zones
SA_patterns.py          # SA with multiple zone patterns
ga_sequence.py          # Genetic algorithm for operation sequences
Collector.py            # Dataset generation infrastructure
Collector_helper.py     # Helper functions for data collection
Collect_SA.py           # Data collection orchestration
CNN+RNN.py              # Deep learning model for sequence prediction
README.md               # This file
```

## Dependencies
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0

## License
See LICENSE file for details. 
