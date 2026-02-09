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
├── operations.py                      # HamiltonianSTL class (grid operations)
├── Zones.py                           # Zone pattern generators (left_right, voronoi, islands, stripes)
├── SA_generation.py                   # Simulated Annealing data generation
├── GA_squence.py                      # Genetic Algorithm with state-aware crossover
├── run_pipeline.py                    # Parallel data generation entry point
├── merge_datasets_safe.py             # Dataset merging utility
│
├── FusionModel/                       # FusionNet neural network
│   ├── fusion/                        # Model source code
│   │   ├── fusion_model.py            #   Architecture (ResU-Net + GRU + FiLM)
│   │   ├── fusion_dataset.py          #   PyTorch Dataset with augmentation
│   │   ├── build_fusion_data.py       #   Precompute training data from trajectories
│   │   ├── train_fusion.py            #   Training script with Rich UI dashboard
│   │   ├── inference_fusion.py        #   Iterative inference with history context
│   │   └── fusion_data.pt             #   Precomputed training tensors
│   └── nn_checkpoints/fusion/         # Trained model checkpoints & results
│       ├── best.pt                    #   Best model (by val_loss)
│       ├── epoch_10.pt / 20.pt / 30.pt
│       ├── training_log.csv           #   Epoch-by-epoch metrics
│       ├── inference_results.json     #   Evaluation results vs SA baseline
│       └── vis/                       #   200 visualization PNGs (before/after)
│
├── FusionNet_Paper_Description/       # Architecture description for research paper
│   ├── FusionModel_Architecture_Description.txt
│   ├── FusionNet_Architecture.png
│   ├── FusionNet_Detailed_Blocks.png
│   ├── FusionNet_FiLM_Detail.png
│   └── FusionNet_Inference_Pipeline.png
│
├── CCAI/                              # CCAI 2026 paper (LaTeX source + figures)
│
├── pipeline/                          # Parallel data generation pipeline
│   ├── runner.py                      #   Multiprocessing orchestrator with Rich UI
│   ├── worker.py                      #   Task execution worker
│   ├── task_generator.py              #   Task generation from configs
│   ├── config.py                      #   Configuration data classes
│   ├── checkpoint.py                  #   Progress tracking & resume
│   └── merge.py                       #   Dataset merging utility
│
├── config/                            # Pipeline configuration files
│   ├── global_config.yaml             #   SA algorithm presets
│   ├── istiaq.yaml                    #   Per-machine configs
│   └── kazi.yaml
│
├── datasets/                          # Training datasets
│   └── combined_dataset.jsonl         #   Merged SA trajectories (all zone patterns)
│
├── scripts/                           # Utility shell/batch scripts
│   ├── run_training.sh / .bat         #   Training launchers
│   ├── run_inference.sh / .bat        #   Inference launchers
│   ├── evaluate_model.sh              #   Model evaluation
│   ├── compare_checkpoints.sh         #   Checkpoint comparison
│   ├── setup_gpu.sh                   #   GPU environment setup
│   ├── setup_runpod.sh                #   RunPod cloud setup
│   ├── setup_nn.bat                   #   Windows NN environment setup
│   ├── quick_test.sh                  #   Quick smoke test
│   ├── help.sh                        #   Script usage help
│   └── make_executable.sh             #   chmod helper
│
└── nn_venv/                           # Python virtual environment (PyTorch + CUDA)
```

## FusionNet Architecture

FusionNet is a hybrid CNN+RNN model that predicts per-pixel Hamiltonian path operations by fusing spatial grid state with temporal operation history via **FiLM (Feature-wise Linear Modulation)**.

### Two-Branch Design

| Branch | Input | Architecture | Output |
|--------|-------|-------------|--------|
| **CNN** (spatial) | 5-channel 32x32 tensor | 3-level ResU-Net (48/96/192 channels) | Multi-scale feature maps |
| **RNN** (temporal) | K=8 recent effective ops | 2-layer GRU (80 -> 192 hidden) | 192-dim context vector |

### 5-Channel Input Encoding

| Channel | Content |
|---------|---------|
| 0 | Normalized zone map |
| 1 | Horizontal edges (H) |
| 2 | Vertical edges (V) |
| 3 | Boundary mask (dilated) |
| 4 | Crossing indicator (derived) |

### FiLM Fusion

The RNN context vector is projected to per-channel scale (gamma) and shift (beta) parameters that modulate CNN decoder features at each resolution level:

```
FiLM(features, gamma, beta) = gamma * features + beta
```

**Identity initialization** (gamma=1, beta=0) guarantees the model starts as a pure U-Net and gradually learns temporal modulation -- it can never perform worse than the spatial-only baseline.

### Output: Dual Heads

- **Position Head**: K=4 Winner-Takes-All hypotheses over 32x32 boundary positions
- **Action Head**: 12-class per-pixel prediction (8 transpose + 4 flip variants)

### Key Specs

| Parameter | Value |
|-----------|-------|
| Total parameters | ~2.3M |
| Base features (f) | 48 |
| RNN hidden dim | 192 |
| GRU layers | 2 |
| Position hypotheses (K) | 4 |
| History window | 8 operations |
| Optimizer | AdamW (lr=3e-4, wd=1e-4) |
| Schedule | Cosine anneal with 500-step warmup |
| Loss | 10 * WTA_pos + CE_action + 0.5 * diversity |

### 12 Operation Classes

| Type | Variants | Subgrid |
|------|----------|---------|
| Transpose (8) | nl, nr, sl, sr, eb, ea, wa, wb | 3x3 |
| Flip (4) | n, s, e, w | 2x3 or 3x2 |

## Simulated Annealing (SA)

`SA_generation.py` implements the primary data generation algorithm. SA optimizes a Hamiltonian cycle on a zone-partitioned grid by iteratively applying local path-rerouting operations and accepting/rejecting based on the Metropolis criterion.

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
   │       (flips rarely feasible on early zigzag-like paths)
   │
   ├── PHASE 2 (i >= 60%): Mixed transpose + flip
   │   └── Both operation types in pool, flips preferred (90% probability)
   │
   ├── TEMPERATURE: T = Tmin + (Tmax - Tmin) * sigmoid(k * (N/2 - i))
   │   └── Sigmoid-based cooling (smooth, not step-wise)
   │
   ├── SELECT MOVE
   │   ├── Sample from pre-built move pool (fast path)
   │   └── Fallback: random position + random variant (if pool empty)
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
   ├── TRACK BEST
   │   ├── Record full operation sequence leading to best state
   │   └── Save edge state snapshot at best crossing count
   │
   ├── PERIODIC REFRESH (every 250 iterations)
   │   └── Rebuild move pool from current edge configuration
   │
   └── REHEATING (if no improvement for 3000 iterations)
       ├── Multiply Tmax by 1.5 (capped at 600)
       └── Refresh move pool to escape local minimum

4. OUTPUT
   ├── Restore best-seen edge state
   ├── Write trajectory to JSONL (operation sequence + metadata)
   └── Return (initial_crossings, final_crossings, best_ops)
```

### Two-Phase Operation Strategy

The SA uses a two-phase approach because the feasibility of operations depends on the current path structure:

| Phase | Iterations | Operations | Rationale |
|-------|-----------|------------|-----------|
| **Phase 1** | 0 - 60% | Transpose only (98%) | Starting from a zigzag, most flip patterns don't match. Transposes restructure the path to create flip-feasible configurations. |
| **Phase 2** | 60% - 100% | Flip preferred (90%) | After transposes have restructured the path, flips become feasible and are more effective at eliminating crossings near zone boundaries. |

### Move Pool Construction

Rather than randomly sampling operations each iteration, SA pre-builds a pool of verified-feasible moves:

1. **Enumerate** all possible anchor positions for each operation type
2. **Feasibility check**: trial-apply each move, verify it succeeds and preserves the Hamiltonian cycle, then undo
3. **Boundary bias**: sort moves by zone-boundary overlap count (more boundary edges in the subgrid = higher priority)
4. **Border-to-inner ordering**: optionally process moves from grid edges inward, optimizing boundary crossings layer by layer
5. **Refresh periodically**: rebuild the pool every 250 iterations since feasibility changes as the path evolves

### Temperature Schedule

SA uses a **sigmoid-based** cooling schedule rather than the traditional linear or geometric decay:

```
T(i) = Tmin + (Tmax - Tmin) * sigmoid(k * (N/2 - i))
```

where `k = 10/N`. This produces a smooth S-curve that spends more time at moderate temperatures (where the most productive exploration happens) compared to linear cooling.

**Reheating**: if no improvement occurs for `reheat_patience` iterations (default 3000), the maximum temperature is multiplied by `reheat_factor` (1.5x, capped at 600). This allows the search to escape deep local minima.

### Initial Path Selection

The initial Hamiltonian path is automatically selected based on the zone pattern to minimize starting crossings:

| Zone Pattern | Initial Path | Rationale |
|-------------|-------------|-----------|
| `left_right` | `vertical_zigzag` | Vertical traversal naturally aligns with vertical zone boundary |
| `stripes` (vertical) | `vertical_zigzag` | Same as left_right |
| `stripes` (horizontal) | `zigzag` (horizontal) | Horizontal traversal aligns with horizontal boundaries |
| `voronoi`, `islands` | `zigzag` (horizontal) | Default; irregular boundaries have no preferred direction |

### SA Configuration Presets

| Config | Iterations | Tmax | Tmin | Use Case |
|--------|-----------|------|------|----------|
| `short` | 1,000 | 60 | 0.5 | Quick runs, high-crossing trajectories for diverse training data |
| `medium` | 3,000 | 80 | 0.5 | Balanced optimization |
| `long` | 8,000 | 100 | 0.3 | Near-optimal solutions |
| `extra_long` | 15,000 | 120 | 0.2 | Large grids, thorough search |

### SA Usage

```python
from SA_generation import run_sa, run_sa_multiple_seeds

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

### SA Hyperparameter Summary

| Parameter | Default | Description |
|-----------|---------|-------------|
| `iterations` | 5000 | Total SA iterations |
| `Tmax` | 80.0 | Starting maximum temperature |
| `Tmin` | 0.5 | Minimum temperature |
| `pool_refresh_period` | 250 | Rebuild move pool every N iterations |
| `pool_max_moves` | 5000 | Max moves in pool |
| `reheat_patience` | 3000 | Iterations without improvement before reheating |
| `reheat_factor` | 1.5 | Temperature multiplier on reheat |
| `reheat_cap` | 600.0 | Maximum Tmax after reheating |
| `transpose_phase_ratio` | 0.6 | Fraction of iterations for transpose-only phase |
| `border_to_inner` | True | Process moves from grid border inward |
| `max_move_tries` | 25 | Fallback random attempts if pool empty |

---

## Genetic Algorithm (GA)

`GA_squence.py` implements a Genetic Algorithm that further optimizes crossing counts beyond what SA achieves. The GA operates on **operation sequences** (genomes) and uses the SA dataset as its initial population.

### GA Algorithm Overview

```
1. LOAD INITIAL POPULATION
   ├── Read SA trajectories from JSONL dataset
   ├── Filter by grid size and zone pattern
   ├── Convert operation sequences to Individual genomes
   ├── Select top-N by final_crossings (or random sample)
   └── Pad/truncate all genomes to fixed length (GENOME_LEN)

2. EVALUATE GENERATION 0
   ├── For each Individual, replay ops on fresh zigzag grid
   ├── Apply each operation sequentially (skip failures)
   ├── Track best-seen crossing count during replay
   └── fitness = -best_seen_crossings (maximizing fitness = minimizing crossings)

3. EVOLUTION LOOP (for each generation g = 1 .. G)
   │
   ├── SORT population by fitness (descending)
   │
   ├── ELITISM: carry top ELITE_K=6 individuals unchanged
   │
   ├── PARENT CARRYOVER: keep top 70% of population (KEEP_RATE)
   │   └── Strong carryover preserves SA-quality solutions
   │
   ├── FILL REMAINING SLOTS via crossover:
   │   │
   │   ├── SELECT two parents via tournament selection (k=3)
   │   │
   │   ├── ONE-POINT CROSSOVER at cut = 50% of genome length:
   │   │   ├── child_ops = parent_A[:cut] + parent_B[cut:]
   │   │   └── Applied with CX_RATE = 90% (10% chance: clone parent A)
   │   │
   │   ├── EVALUATE child (replay on fresh grid)
   │   │
   │   ├── ACCEPTANCE GATE:
   │   │   ├── child.applied >= 1 (at least one op succeeded)
   │   │   └── child.best_seen <= min(parent_best) + EPS_CROSSINGS(2)
   │   │       (child must not be much worse than best parent)
   │   │
   │   ├── If accepted: add child to next generation
   │   └── If rejected after MAX_TRIES(80): clone a tournament-selected parent
   │
   └── NO MUTATION (crossover-only GA)

4. OUTPUT
   ├── Best individual across all generations
   └── Write final best to Dataset.jsonl + all accepted children to Children.jsonl
```

### Key Design Decisions

**No Mutation**: The GA deliberately omits mutation. Since operations are context-dependent (an operation that succeeds on one path configuration may fail on another), random perturbation of genomes mostly produces invalid operations. The SA-generated initial population already provides high-quality, diverse genomes.

**Strong Elitism (70% carryover)**: Rather than replacing most of the population each generation, 70% of parents survive unchanged. This prevents catastrophic loss of SA-quality solutions while still allowing crossover to explore new combinations.

**Acceptance Gate**: Children must achieve crossing counts within `EPS_CROSSINGS=2` of their best parent. This prevents population quality from degrading due to destructive crossover.

**One-Point Crossover at 50%**: The genome is split at the midpoint -- the child inherits the first half of parent A's operations and the second half of parent B's. Since early operations restructure the path (transpose phase) and late operations fine-tune (flip phase), crossover combines different restructuring strategies with different fine-tuning strategies.

### GA Genome Representation

Each individual is a fixed-length sequence of operations:

```
Individual.ops = [Op(kind, x, y, variant), ...]   # length = GENOME_LEN
```

| Field | Values | Description |
|-------|--------|-------------|
| `kind` | `T`, `F`, `N` | Transpose, Flip, or No-op |
| `x`, `y` | 0..W-1, 0..H-1 | Anchor position of the operation |
| `variant` | `sr`, `nl`, `n`, `w`, ... | Operation variant identifier |

Genomes shorter than `GENOME_LEN` are padded with no-ops. Operations that fail during evaluation (invalid position, pattern mismatch, cycle violation) are silently skipped.

### Fitness Evaluation

Each individual is evaluated by replaying its operation sequence on a fresh zigzag-initialized grid:

1. Create `HamiltonianSTL(W, H)` with zigzag initial pattern
2. For each operation in the genome:
   - Attempt to apply it (snapshot edges, apply, verify cycle, rollback on failure)
   - Track `best_seen` crossing count (minimum seen during replay)
3. `fitness = -best_seen` (negative because GA maximizes fitness)

The `applied` count tracks how many operations actually succeeded (vs. were skipped due to infeasibility).

### GA Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GENOME_LEN` | 100-200 | Fixed genome length (must match SA sequence lengths) |
| `pop_size` | 30 | Population size |
| `generations` | 20-30 | Number of GA generations |
| `ELITE_K` | 6 | Number of elite individuals always preserved |
| `KEEP_RATE` | 0.70 | Fraction of population carried unchanged |
| `CX_RATE` | 0.90 | Crossover probability (vs. clone) |
| `CX_RATIO` | 0.50 | Crossover cut point (fraction of genome) |
| `tourn_k` | 3 | Tournament selection size |
| `EPS_CROSSINGS` | 2 | Max allowed degradation for child acceptance |
| `MIN_APPLIED_VALID` | 1 | Minimum successful ops for a valid child |
| `MAX_TRIES_PER_SLOT` | 80 | Max crossover attempts per population slot |

### GA Usage

```python
# Command-line
python GA_squence.py

# Programmatic
from GA_squence import run_ga_sequences_dataset_init

best = run_ga_sequences_dataset_init(
    dataset_jsonl="datasets/combined_dataset.jsonl",
    W=30, H=30,
    zone_pattern="left_right",
    pop_size=30,
    generations=20,
    genome_len=200,
)
```

---

## Data Generation Pipeline

The pipeline (`run_pipeline.py` + `pipeline/`) orchestrates parallel SA data generation across multiple CPU cores with automatic checkpointing and resume.

```bash
# Run parallel data generation
python run_pipeline.py <machine_id> --workers 8

# Check status
python run_pipeline.py <machine_id> --status

# Retry failed tasks
python run_pipeline.py <machine_id> --retry-failed

# Merge datasets from multiple machines
python merge_datasets_safe.py
```

Pipeline features:
- **Rich UI dashboard** with real-time progress tracking
- **Automatic checkpointing**: stop and resume without losing progress
- **Per-machine configs** in `config/` (grid sizes, zone patterns, SA presets, seeds)
- **Task generation**: each (zone_pattern, seed, SA_config) combination becomes one task
- **Fault tolerance**: failed tasks tracked and retryable

## Training the FusionNet

### 1. Build Training Data

Precompute tensor dataset from SA trajectories (replays trajectories, extracts effective operations with history context):

```bash
PYTHONPATH=$(pwd):$PYTHONPATH python FusionModel/fusion/build_fusion_data.py \
    --input datasets/combined_dataset.jsonl \
    --output FusionModel/fusion/fusion_data.pt
```

### 2. Train

```bash
PYTHONPATH=$(pwd):$PYTHONPATH python FusionModel/fusion/train_fusion.py \
    --data_path FusionModel/fusion/fusion_data.pt \
    --checkpoint_dir FusionModel/nn_checkpoints/fusion \
    --epochs 30 --batch_size 128
```

Training features Rich UI dashboard, cosine LR schedule with warmup, early stopping (patience=15), and periodic checkpoints.

### 3. Inference & Evaluation

Iterative inference with autoregressive history accumulation across all 4 zone patterns:

```bash
PYTHONPATH=$(pwd):$PYTHONPATH python FusionModel/fusion/inference_fusion.py \
    --checkpoint FusionModel/nn_checkpoints/fusion/best.pt \
    --jsonl datasets/combined_dataset.jsonl \
    --n_per_pattern 50 \
    --visualize
```

The inference loop: encode state -> build history -> forward pass -> mean-softmax pool K hypotheses -> top-k position/action search -> apply if crossings decrease -> update history -> repeat.

## Zone Patterns

| Pattern | Description |
|---------|-------------|
| `left_right` | Vertical bands |
| `stripes` | Parallel stripes (horizontal or vertical) |
| `islands` | Square islands on background |
| `voronoi` | Irregular Voronoi regions |

## Dataset Format

Each trajectory in `combined_dataset.jsonl`:

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
- matplotlib, numpy, rich, tqdm, pyyaml

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib numpy pyyaml rich tqdm
```

## License

[Add appropriate license information]
