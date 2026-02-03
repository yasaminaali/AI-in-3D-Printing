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
├── run_pipeline.py           # Entry point for parallel data generation
├── pipeline/                 # Pipeline modules
│   ├── runner.py             # Multiprocessing orchestrator with Rich UI
│   ├── worker.py             # Task execution worker
│   ├── task_generator.py     # Task generation from configs
│   ├── config.py             # Configuration data classes
│   ├── checkpoint.py         # Progress tracking & resume
│   └── merge.py              # Dataset merging utility
├── config/                   # Configuration files
│   └── global_config.yaml    # SA algorithm presets
└── output/                   # Generated datasets
```

## Installation

```bash
# Install dependencies
pip install matplotlib numpy pyyaml rich

# Verify installation
python -c "import matplotlib, numpy, yaml, rich; print('OK')"
```

## Quick Start

### Basic SA Optimization

```python
from SA_generation import run_sa

# Single optimization run
initial, final, operations = run_sa(
    width=32,
    height=32,
    iterations=3000,
    zone_mode="left_right",
    write_dataset=True
)
print(f"Reduced crossings from {initial} to {final} with {len(operations)} operations")
```

### Using the Pipeline

The pipeline provides parallel data generation with a Rich UI, automatic checkpointing, and resume capability.

```bash
# Run the pipeline
python run_pipeline.py <machine_id> --workers 8

# Check status without running
python run_pipeline.py <machine_id> --status

# Resume from where you left off (automatic)
python run_pipeline.py <machine_id> --workers 8

# Retry failed tasks
python run_pipeline.py <machine_id> --retry-failed
```

**Command-line Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `machine_id` | Machine identifier (required) | - |
| `--workers` | Number of parallel processes | From config or CPU count |
| `--config-dir` | Config directory path | `config` |
| `--retry-failed` | Retry previously failed tasks | False |
| `--status` | Show status and exit without running | False |

### Resume Functionality

The pipeline automatically tracks progress via checkpoint files. When you stop the pipeline (Ctrl+C) and restart it:
1. The pipeline loads the checkpoint file
2. Skips all previously completed tasks
3. Resumes from where it left off
4. The progress bar shows both resumed and new progress

Checkpoint files are stored at: `output/<machine_id>/checkpoint_<machine_id>.json`

To start fresh, delete the checkpoint file.

## Zone Patterns

| Pattern | Description | Parameters |
|---------|-------------|------------|
| `left_right` | Vertical bands | k (number of bands) |
| `stripes` | Parallel stripes | direction (v/h), k |
| `islands` | Square islands on background | num_islands, island_size |
| `voronoi` | Irregular regions | k (seed points) |

```python
# Stripes pattern (3 vertical bands)
run_sa(zone_mode="stripes", stripe_k=3, stripe_direction="v")

# Islands pattern (3 square islands)
run_sa(zone_mode="islands", num_islands=3, island_size=8)

# Voronoi pattern (3 irregular regions)
run_sa(zone_mode="voronoi", voronoi_k=3)
```

## SA Configuration Presets

| Config | Iterations | Tmax | Tmin | Use Case |
|--------|------------|------|------|----------|
| short | 3,000 | 60 | 0.5 | Quick runs, small grids |
| medium | 5,000 | 80 | 0.5 | Balanced optimization |
| long | 10,000 | 100 | 0.3 | Thorough optimization |
| extra_long | 20,000 | 120 | 0.2 | Large grids (64x64+) |

## Dataset Format

Output is written to `output/<machine_id>/Dataset.jsonl` as JSON Lines:

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
    {"kind": "F", "x": 10, "y": 7, "variant": "w"}
  ],
  "runtime_sec": 12.5
}
```

## Merging Datasets

After running on multiple machines, merge the datasets:

```python
from pipeline.merge import merge_datasets

report = merge_datasets(
    input_dirs=["output/machine1", "output/machine2"],
    output_path="Dataset/Complete_Dataset.jsonl",
    deduplicate=True,
    validate=True
)
```

## Troubleshooting

**Workers crashing with memory errors:**
```bash
python run_pipeline.py <machine_id> --workers 4  # Reduce workers
```

**Tasks failing consistently:**
```bash
python run_pipeline.py <machine_id> --retry-failed
```

**Want to restart from scratch:**
```bash
rm output/<machine_id>/checkpoint_<machine_id>.json
python run_pipeline.py <machine_id>
```

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
Uses sigmoid-based cooling with reheating when stuck.

## License

[Add appropriate license information]
