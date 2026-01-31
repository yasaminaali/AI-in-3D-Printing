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

---

## Pipeline Setup and Data Generation

The project includes a **multiprocessing pipeline** for large-scale dataset generation across multiple machines. This system automatically distributes work, tracks progress via checkpoints, and merges results.

### Project Structure

```
AI-in-3D-Printing/
├── run_pipeline.py              # Entry point script
├── pipeline/                    # Pipeline modules
│   ├── runner.py               # Multiprocessing orchestrator
│   ├── worker.py               # Task execution worker
│   ├── task_generator.py       # Task generation from configs
│   ├── config.py               # Configuration data classes
│   ├── checkpoint.py           # Progress tracking & resume
│   └── merge.py                # Dataset merging utility
├── config/                      # Configuration files
│   ├── global_config.yaml      # SA algorithm presets
│   ├── yasamin.yaml            # Yasamin's work assignment
│   ├── istiaq.yaml             # Istiaq's work assignment
│   └── kazi.yaml               # Kazi's work assignment
└── output/                      # Generated datasets
    ├── yasamin/
    │   ├── Dataset.jsonl       # Training data
    │   └── checkpoint_yasamin.json
    ├── istiaq/
    └── kazi/
```

### Configuration System

#### 1. Global Configuration (`config/global_config.yaml`)

Defines SA algorithm presets used across all machines:

```yaml
sa_configs:
  short:        # 3,000 iterations - quick runs
    iterations: 3000
    Tmax: 60.0
    Tmin: 0.5
    transpose_phase_ratio: 0.6
    
  medium:       # 5,000 iterations - balanced
    iterations: 5000
    Tmax: 80.0
    Tmin: 0.5
    
  long:         # 10,000 iterations - thorough
    iterations: 10000
    Tmax: 100.0
    Tmin: 0.3
    
  extra_long:   # 20,000 iterations - large grids
    iterations: 20000
    Tmax: 120.0
    Tmin: 0.2
```

#### 2. Machine Configuration (`config/{machine_id}.yaml`)

Each machine has a config file defining its work assignment:

```yaml
machine_id: yasamin
num_workers: 8                    # Parallel processes
output_dir: output/yasamin        # Output directory

assignments:
  - grid: [32, 32]                # Grid size
    patterns:                     # Zone patterns to run
      - left_right
      - stripes
      - islands
      - voronoi
    sa_configs:                   # Which SA presets
      - short
      - medium
      - long
    seeds:                        # Number of seeds per pattern
      left_right: 2300
      stripes: 600
      islands: 450
      voronoi: 700
```

### Quick Start

#### Step 1: Setup Environment

```bash
# Install dependencies
pip install matplotlib numpy pyyaml

# Verify installation
python -c "import matplotlib, numpy, yaml; print('OK')"
```

#### Step 2: Run the Pipeline

```bash
# Basic usage - run with default workers
python run_pipeline.py yasamin

# Specify number of workers
python run_pipeline.py yasamin --workers 8

# Check progress without running
python run_pipeline.py yasamin --status

# Retry failed tasks
python run_pipeline.py yasamin --retry-failed

# Custom progress logging interval (seconds)
python run_pipeline.py yasamin --progress-interval 30
```

**Command-line Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `machine_id` | Machine identifier (required) | - |
| `--workers` | Number of parallel processes | From config or CPU count |
| `--config-dir` | Config directory path | `config` |
| `--retry-failed` | Retry previously failed tasks | False |
| `--progress-interval` | Seconds between progress logs | 60 |
| `--status` | Show status and exit | False |

### Multiprocessing Architecture

The pipeline uses Python's `multiprocessing.Pool` for parallel execution:

```
┌─────────────────────────────────────────┐
│           Main Process                  │
│  ┌─────────────────────────────────┐    │
│  │  ParallelRunner                 │    │
│  │  - Load configs                 │    │
│  │  - Generate all tasks           │    │
│  │  - Filter pending tasks         │    │
│  │  - Submit to Pool               │    │
│  └─────────────────────────────────┘    │
│                   │                     │
│         ┌─────────┴─────────┐           │
│         ▼                 ▼             │
│  ┌─────────────┐   ┌─────────────┐      │
│  │  Worker 1   │   │  Worker 2   │      │
│  │  ...        │   │  ...        │      │
│  └─────────────┘   └─────────────┘      │
│                                         │
└─────────────────────────────────────────┘
```

**Key Features:**

1. **Automatic Task Distribution**: Tasks are distributed across workers as they become available
2. **Checkpoint-Based Resume**: Progress saved to JSON file; restart resumes where left off
3. **Graceful Shutdown**: SIGINT/SIGTERM handlers ensure clean exit
4. **Progress Tracking**: Real-time stats with ETA calculation

### Monitoring Progress

During execution, you'll see output like:

```
=== SA Dataset Generation Pipeline ===
Machine: yasamin
Config dir: config

[Runner] Machine: yasamin
[Runner] Workers: 8
[Runner] Total tasks: 12150
[Runner] Already completed: 0
[Runner] Pending: 12150
[Runner] Task breakdown:
  By grid: {'32x32': 12150}
  By pattern: {'left_right': 6900, 'stripes': 1800, 'islands': 1350, 'voronoi': 2100}
  By config: {'short': 4050, 'medium': 4050, 'long': 4050}

[Runner] Starting 8 workers...
[OK] yasamin_32x32_left_right_short_seed0 - crossings=4 time=12.5s
[OK] yasamin_32x32_left_right_short_seed1 - crossings=6 time=11.8s
...
[Progress] 1000/12150 completed (8.2%) | Failed: 0 | Elapsed: 2h 15m | ETA: 25h 10m
```

### Checkpoint and Resume

The pipeline automatically creates a checkpoint file at `output/{machine_id}/checkpoint_{machine_id}.json`:

```json
{
  "completed_tasks": {
    "yasamin_32x32_left_right_short_seed0": {
      "final_crossings": 4,
      "runtime_sec": 12.5,
      "timestamp": 1706745600.0
    }
  },
  "failed_tasks": {},
  "stats": {
    "total_completed": 1000,
    "total_failed": 0,
    "start_time": 1706745600.0
  }
}
```

**Resume behavior:**
- On restart, completed tasks are skipped automatically
- Failed tasks can be retried with `--retry-failed`
- To restart from scratch, delete the checkpoint file

### Merging Datasets

After all machines complete their work, merge the datasets:

```python
from pipeline.merge import merge_datasets

# Merge all machine outputs
report = merge_datasets(
    input_dirs=["output/yasamin", "output/istiaq", "output/kazi"],
    output_path="Dataset/Complete_Dataset.jsonl",
    deduplicate=True,      # Remove duplicate run_ids
    validate=True          # Validate merged dataset
)

print(f"Total records: {report['final_record_count']}")
print(f"Duplicates removed: {report['duplicates_removed']}")
```

Or use the standalone script:

```bash
python merge_datasets.py \
    --inputs output/yasamin output/istiaq output/kazi \
    --output Dataset/Complete_Dataset.jsonl \
    --deduplicate \
    --validate
```

### Work Distribution

The dataset generation is split across 3 machines:

| Machine | Grid Sizes | Total Tasks | Est. Time (8 workers) |
|---------|------------|-------------|----------------------|
| **Yasamin** | 32×32, 64×64, 32×128, 32×256 | 36,050 | ~4-5 days |
| **Istiaq** | 30×30, 80×80 | 19,750 | ~2-3 days |
| **Kazi** | 50×50, 100×100 | 14,350 | ~2-3 days |
| **TOTAL** | 8 grids | **70,150** | **~500K trajectories** |

*Time estimates assume medium SA config (~30-60 sec per task on modern CPU)*

### Performance Tuning

#### Choosing Worker Count

```bash
# Rule of thumb: workers = CPU cores - 1 (leave one for OS)
# For a 16-core machine:
python run_pipeline.py yasamin --workers 15

# For shared machines, use fewer workers:
python run_pipeline.py yasamin --workers 4
```

**Guidelines:**
- **CPU-bound tasks**: Use all cores (or cores-1)
- **Memory-constrained**: Reduce workers if RAM usage is high
- **Shared machines**: Use 25-50% of available cores

#### SA Configuration Selection

| Grid Size | Recommended SA Configs | Why |
|-----------|----------------------|-----|
| ≤32×32 | short, medium, long | Fast convergence |
| 50×50 to 64×64 | medium, long, extra_long | Need more iterations |
| ≥80×80 or rectangular | medium, long, extra_long | Large search space |

### Troubleshooting

#### Issue: Workers crashing with memory errors

**Solution**: Reduce worker count or use smaller grids first:
```bash
python run_pipeline.py yasamin --workers 4
```

#### Issue: Tasks failing consistently

**Solution**: Check the checkpoint file for error details, then retry:
```bash
python run_pipeline.py yasamin --retry-failed
```

#### Issue: Want to restart from scratch

**Solution**: Delete the checkpoint file:
```bash
rm output/yasamin/checkpoint_yasamin.json
python run_pipeline.py yasamin
```

#### Issue: Progress seems stuck

**Solution**: Check status to see if tasks are still running:
```bash
python run_pipeline.py yasamin --status
```

---

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
