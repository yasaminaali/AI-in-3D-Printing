# Neural Toolpath Optimization for Multi-Material 3D Printing

Zone crossing minimization in Hamiltonian toolpaths using **Simulated Annealing (SA)**, **Genetic Algorithms (GA)**, and a **FusionNet (CNN+RNN+FiLM)** neural network.

## Problem

In multi-material 3D printing, the print head traverses all cells in a layer via a Hamiltonian cycle. Each transition between different material zones (a "crossing") requires a material change that slows printing and degrades quality. This project minimizes zone crossings through local path-rerouting operations (transposes on 3x3 subgrids, flips on 2x3/3x2 subgrids) that preserve the Hamiltonian property by construction.

## Project Structure

```
AI-in-3D-Printing/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore
│
├── src/                               # All Python source code
│   ├── operations.py                  # HamiltonianSTL class (grid operations)
│   ├── zones.py                       # Zone pattern generators (left_right, voronoi, islands, stripes)
│   ├── sa_generation.py               # Simulated Annealing data generation
│   ├── sa_generation_gpu.py           # GPU-accelerated SA
│   ├── ga_sequence.py                 # Genetic Algorithm with state-aware crossover
│   ├── ga_sequence_gpu.py             # GPU-accelerated GA
│   ├── numba_ops.py                   # Numba-accelerated grid operations
│   ├── run_pipeline.py                # Parallel data generation entry point
│   ├── run_pipeline_gpu.py            # GPU pipeline entry point
│   ├── run_ga_pipeline_gpu.py         # GA GPU pipeline
│   ├── merge_datasets.py              # Dataset merging utility
│   │
│   ├── pipeline/                      # Data generation pipeline package
│   │   ├── __init__.py
│   │   ├── config.py                  # Configuration data classes
│   │   ├── runner.py                  # Multiprocessing orchestrator with Rich UI
│   │   ├── worker.py                  # Task execution worker
│   │   ├── task_generator.py          # Task generation from configs
│   │   ├── checkpoint.py              # Progress tracking & resume
│   │   ├── merge.py                   # Dataset merging utility
│   │   ├── ga_config.py               # GA-specific config
│   │   └── ga_task_generator.py       # GA task generation
│   │
│   └── model/                         # FusionNet neural network
│       ├── __init__.py
│       ├── fusion_model.py            # Architecture (4-level ResU-Net + GRU + FiLM)
│       ├── fusion_dataset.py          # PyTorch Dataset with augmentation
│       ├── constructive.py            # Constructive crossing optimization (no model)
│       ├── build_fusion_data.py       # Precompute training data from trajectories
│       ├── train_fusion.py            # DDP training with Rich UI dashboard
│       ├── inference_fusion.py        # Constructive + model-guided inference
│       └── inference_comparison.py    # Comparison inference (model+SA on all patterns)
│
├── config/                            # Configuration files
│   ├── global_config.yaml             # SA algorithm presets
│   ├── ga_global_config.yaml          # GA algorithm presets
│   ├── istiaq.yaml                    # Per-machine configs
│   ├── kazi.yaml
│   ├── kazi_ga.yaml
│   └── kazi_stripes.yaml
│
├── scripts/                           # Shell/batch/sbatch scripts
│   ├── setup_env.sh                   # SA environment setup
│   ├── setup_gpu.sh                   # GPU environment setup
│   ├── setup_runpod.sh                # RunPod cloud setup
│   ├── setup_nn.bat                   # Windows NN environment setup
│   ├── run_training.sh / .bat         # Training launchers
│   ├── run_inference.sh / .bat        # Inference launchers
│   ├── resume_training.bat            # Resume from checkpoint
│   ├── evaluate_model.sh              # Model evaluation suite
│   ├── compare_checkpoints.sh         # Checkpoint comparison
│   ├── quick_test.sh                  # Quick smoke test
│   ├── help.sh                        # Script usage help
│   ├── make_executable.sh             # chmod helper
│   └── sbatch_*.sh                    # SLURM job scripts (8 files)
│
├── datasets/                          # Training data
│   ├── Readme.txt
│   └── analyze_dataset.py             # Dataset analysis script
│
├── checkpoints/                       # Model checkpoints & results (gitignored)
│   ├── best.pt                        # Best model (by val_loss)
│   ├── fusion_data.pt                 # Preprocessed training tensors
│   ├── training_log.csv               # Epoch-by-epoch metrics
│   ├── vis/                           # Visualization PNGs
│   └── comparison/                    # Comparison inference results
│
├── docs/                              # Documentation
│   ├── SYSTEM_DOCUMENTATION.md        # Detailed system documentation
│   ├── CHANGELOG.md                   # Change history
│   └── dataset_analysis.md            # Dataset analysis report
│
├── paper/                             # CCAI 2026 paper (gitignored)
│
└── output/                            # Pipeline output (gitignored)
```

## FusionNet v2 Architecture

FusionNet is a hybrid CNN+RNN model that predicts per-pixel Hamiltonian path operations by fusing spatial grid state with temporal operation history via **FiLM (Feature-wise Linear Modulation)**.

### Input Encoding (9 channels, 128x128)

| Channel | Content |
|---------|---------|
| 0 | Normalized zone map |
| 1 | Horizontal edges (H) |
| 2 | Vertical edges (V) |
| 3 | Grid validity mask |
| 4 | Boundary mask (dilated) |
| 5 | Crossing density |
| 6 | Progress (fraction of trajectory completed) |
| 7 | Y coordinate |
| 8 | X coordinate |

### CNN Branch: 4-Level ResU-Net

| Level | Channels | Resolution |
|-------|----------|------------|
| Encoder 1 | 48 | 128 -> 64 |
| Encoder 2 | 96 | 64 -> 32 |
| Encoder 3 | 192 | 32 -> 16 |
| Encoder 4 | 192 | 16 -> 8 |
| Bottleneck | 192 + Self-Attention (4 heads) | 8x8 |
| Decoder 4-1 | 192 -> 192 -> 96 -> 48 | 8 -> 128 |

- **GroupNorm** replaces BatchNorm (handles variable grid sizes)
- **Bottleneck self-attention** at 8x8 resolution (64 tokens, 4 heads) for global reasoning

### RNN Branch: GRU History Encoder

| Parameter | Value |
|-----------|-------|
| Input | K=32 recent effective operations |
| Embeddings | action (16d) + pos_y (16d) + pos_x (16d) + crossings_before (16d) + crossings_after (16d) = 80d |
| GRU | 2 layers, 192 hidden, dropout=0.15 |
| Output | 192-dim context vector |

### FiLM Fusion

The RNN context vector is projected to per-channel scale (gamma) and shift (beta) parameters that modulate CNN decoder features at all 4 resolution levels:

```
FiLM(features, gamma, beta) = gamma * features + beta
```

**Identity initialization** (gamma=1, beta=0) guarantees the model starts as a pure U-Net and gradually learns temporal modulation.

### Output: Dual Heads

| Head | Output Shape | Description |
|------|-------------|-------------|
| **Position** | 1 x 128 x 128 | Single hypothesis position scores |
| **Action** | 12 x 128 x 128 | Per-pixel action logits (8 transpose + 4 flip) |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Total parameters | ~2.3M |
| Optimizer | Adam (lr=4e-4) |
| Schedule | Cosine annealing with warmup |
| Loss | Margin-based ranking hinge loss + hard negative mining + neighborhood action supervision |
| Epochs | 200 |
| Early stopping patience | 40 |
| DDP | 4x H100 GPUs with bf16 autocast |

### 12 Operation Classes

| Type | Variants | Subgrid |
|------|----------|---------|
| Transpose (8) | nl, nr, sl, sr, eb, ea, wa, wb | 3x3 |
| Flip (4) | n, s, e, w | 2x3 or 3x2 |

## Inference Strategies

### 1. Constructive (left_right, stripes)

Deterministic, no model required. Two-phase approach:
- **Phase 1 (propagate)**: Start from optimal zigzag (k-1 crossings), greedily propagate crossings to near-max coverage.
- **Phase 2 (trim)**: Selectively remove crossings using spread ordering (binary subdivision) to bring count into target range [60%-80% of max] while keeping crossings evenly distributed.

### 2. Model-Guided (voronoi, islands)

Alternating model-SA loop:
1. Start from zigzag, model predicts positions + actions to reduce crossings
2. Boundary-biased random sampling for exploration
3. When model stagnates (150-step stagnation), light SA perturbation (3000 steps, `restore_best=False`) to escape local minima
4. 5 model-SA cycles total
5. Phase 2: greedy redistribution for CV uniformity

## Simulated Annealing (SA)

`src/sa_generation.py` implements the primary data generation algorithm. SA optimizes a Hamiltonian cycle on a zone-partitioned grid by iteratively applying local path-rerouting operations and accepting/rejecting based on the Metropolis criterion.

### SA Algorithm Overview

```
1. INITIALIZE
   ├── Create Hamiltonian cycle on WxH grid (auto-select initial pattern)
   ├── Build zone map for the chosen pattern
   └── Compute initial crossing count

2. BUILD MOVE POOL
   ├── Enumerate all feasible transpose anchors (3x3 subgrids)
   ├── Enumerate all feasible flip anchors (2x3 / 3x2 subgrids)
   ├── Rank moves by zone-boundary proximity (bias_to_boundary)
   └── Optionally order border-to-inner (border_to_inner)

3. MAIN LOOP (for each iteration i = 0 .. N)
   │
   ├── PHASE 1 (i < 60% of iterations): Transpose-only
   │   └── Only transpose operations in the move pool
   │
   ├── PHASE 2 (i >= 60%): Mixed transpose + flip
   │   └── Both operation types in pool, flips preferred (90% probability)
   │
   ├── TEMPERATURE: T = Tmin + (Tmax - Tmin) * sigmoid(k * (N/2 - i))
   │
   ├── SELECT MOVE from pre-built move pool
   │
   ├── APPLY MOVE
   │   ├── Snapshot edges before application
   │   ├── Apply transpose/flip operation
   │   ├── Verify Hamiltonian cycle preserved
   │   └── Compute new crossing count
   │
   ├── METROPOLIS ACCEPTANCE
   │   ├── If delta < 0 (improvement): ALWAYS accept
   │   └── If delta >= 0 (worse): accept with P = exp(-delta / T)
   │
   ├── PERIODIC REFRESH (every 250 iterations)
   │   └── Rebuild move pool from current edge configuration
   │
   └── REHEATING (if no improvement for reheat_patience iterations)
       ├── Multiply Tmax by 1.5 (capped at 600)
       └── Refresh move pool to escape local minimum

4. OUTPUT
   ├── Restore best-seen edge state
   └── Write trajectory to JSONL (operation sequence + metadata)
```

### Initial Path Selection

| Zone Pattern | Initial Path | Rationale |
|-------------|-------------|-----------|
| `left_right` | `vertical_zigzag` | Vertical traversal aligns with vertical zone boundary |
| `stripes` (vertical) | `vertical_zigzag` | Same as left_right |
| `stripes` (horizontal) | `zigzag` (horizontal) | Horizontal traversal aligns with horizontal boundaries |
| `voronoi`, `islands` | `zigzag` (horizontal) | Default; irregular boundaries have no preferred direction |

### SA Configuration Presets (from `config/global_config.yaml`)

| Config | Iterations | Tmax | Tmin | Reheat Patience |
|--------|-----------|------|------|-----------------|
| `short` | 3,000 | 60 | 0.5 | 500 |
| `medium` | 5,000 | 80 | 0.5 | 1,500 |
| `long` | 10,000 | 100 | 0.3 | 2,500 |
| `extra_long` | 20,000 | 120 | 0.2 | 4,000 |

### SA Usage

```python
from sa_generation import run_sa, run_sa_multiple_seeds

# Single run
initial, final, ops = run_sa(
    width=30, height=30,
    iterations=3000,
    zone_mode="left_right",
    write_dataset=True
)

# Multiple seeds for dataset generation
results = run_sa_multiple_seeds(
    seeds=list(range(100)),
    width=30, height=30,
    iterations=3000,
    zone_mode="voronoi",
    voronoi_k=3
)
```

## Genetic Algorithm (GA)

`src/ga_sequence.py` implements a Genetic Algorithm that further optimizes crossing counts beyond what SA achieves. The GA operates on **operation sequences** (genomes) and uses the SA dataset as its initial population.

### GA Algorithm Overview

```
1. LOAD INITIAL POPULATION
   ├── Read SA trajectories from JSONL dataset
   ├── Convert operation sequences to Individual genomes
   └── Select top-N by final_crossings

2. EVOLUTION LOOP (for each generation g = 1 .. G)
   │
   ├── ELITISM: carry top ELITE_K=6 individuals unchanged
   │
   ├── PARENT CARRYOVER: keep top 70% of population (KEEP_RATE)
   │
   ├── FILL REMAINING SLOTS via crossover:
   │   ├── SELECT two parents via tournament selection (k=3)
   │   ├── ONE-POINT CROSSOVER at 50% of genome length
   │   ├── EVALUATE child (replay on fresh grid)
   │   └── ACCEPTANCE GATE: child must be within EPS_CROSSINGS=2 of best parent
   │
   └── NO MUTATION (crossover-only GA)

3. OUTPUT: Best individual across all generations
```

### Key Design Decisions

- **No Mutation**: Operations are context-dependent; random perturbation produces mostly invalid operations.
- **Strong Elitism (70% carryover)**: Prevents catastrophic loss of SA-quality solutions.
- **Acceptance Gate**: Children must achieve crossing counts within `EPS_CROSSINGS=2` of their best parent.

### GA Usage

```python
from ga_sequence import run_ga_sequences_dataset_init

best = run_ga_sequences_dataset_init(
    dataset_jsonl="datasets/combined_dataset.jsonl",
    W=30, H=30,
    zone_pattern="left_right",
    pop_size=30,
    generations=20,
    genome_len=200,
)
```

## Data Generation Pipeline

The pipeline (`src/run_pipeline.py` + `src/pipeline/`) orchestrates parallel SA data generation across multiple CPU/GPU cores with automatic checkpointing and resume.

```bash
# Run parallel data generation
python src/run_pipeline.py <machine_id> --workers 8

# GPU-accelerated pipeline
python src/run_pipeline_gpu.py <machine_id>

# Check status
python src/run_pipeline.py <machine_id> --status

# Merge datasets from multiple machines
python src/merge_datasets.py
```

Pipeline features:
- **Rich UI dashboard** with real-time progress tracking
- **Automatic checkpointing**: stop and resume without losing progress
- **Per-machine configs** in `config/` (grid sizes, zone patterns, SA presets, seeds)
- **Fault tolerance**: failed tasks tracked and retryable

## Training & Inference

### 1. Build Training Data

Precompute tensor dataset from SA trajectories:

```bash
PYTHONPATH=$(pwd)/src:$PYTHONPATH python src/model/build_fusion_data.py \
    --input datasets/combined_dataset.jsonl \
    --output checkpoints/fusion_data.pt
```

### 2. Train

```bash
# Single GPU
PYTHONPATH=$(pwd)/src:$PYTHONPATH python src/model/train_fusion.py \
    --data_path checkpoints/fusion_data.pt \
    --checkpoint_dir checkpoints \
    --epochs 200 --batch_size 128

# 4x GPU DDP
PYTHONPATH=$(pwd)/src:$PYTHONPATH torchrun --nproc_per_node=4 src/model/train_fusion.py
```

### 3. Inference

```bash
PYTHONPATH=$(pwd)/src:$PYTHONPATH python src/model/inference_fusion.py \
    --checkpoint checkpoints/best.pt \
    --jsonl datasets/combined_dataset.jsonl \
    --n_per_pattern 50 \
    --visualize
```

### 4. Comparison Inference

Forces model+SA on all patterns (including ones that normally use constructive):

```bash
PYTHONPATH=$(pwd)/src:$PYTHONPATH python src/model/inference_comparison.py \
    --checkpoint checkpoints/best.pt
```

## Zone Patterns

| Pattern | Description |
|---------|-------------|
| `left_right` | Vertical bands |
| `stripes` | Parallel stripes (horizontal or vertical) |
| `islands` | Square islands on background |
| `voronoi` | Irregular Voronoi regions |

## Dataset Format

Each trajectory in the JSONL dataset:

```json
{
  "run_id": "sa_left_right_W30H30_seed0_...",
  "grid_W": 30, "grid_H": 30,
  "zone_pattern": "left_right",
  "zone_grid": [0, 0, ..., 1, 1],
  "initial_crossings": 32,
  "final_crossings": 4,
  "sequence_ops": [
    {"kind": "T", "x": 5, "y": 3, "variant": "sr"},
    {"kind": "F", "x": 10, "y": 7, "variant": "w"}
  ]
}
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- matplotlib, numpy, rich, pyyaml, scipy

```bash
pip install -r requirements.txt
```

## SLURM (HPC)

All sbatch scripts are in `scripts/`. Submit from the repo root:

```bash
sbatch scripts/sbatch_kazi_gpu.sh        # GPU data generation
sbatch scripts/sbatch_train_fusion.sh     # DDP training (4x H100)
sbatch scripts/sbatch_inference_fusion.sh # Inference
```
