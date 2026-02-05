# Zone Crossing Minimization for Neural Toolpath Optimization

This project uses **deep learning** to predict optimal sequences of Flip/Transpose operations to minimize zone crossings in Hamiltonian toolpaths for multi-material 3D printing.

The project includes:
- **Data Generation**: Simulated Annealing (SA) and Genetic Algorithm (GA) to generate training data
- **CNN+RNN Model**: Paper-compliant neural network that learns to predict optimal operation sequences
- **Complete Pipeline**: Automated training with Rich UI dashboard and configurable hyperparameters

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
├── EXPERIMENT_RESULTS.md     # GA experiment findings
├── run_training.bat          # One-click training launcher
├── setup_nn.bat              # Environment setup
├── model/                    # CNN+RNN Neural Network
│   ├── config.yaml           # Model & training configuration
│   ├── train.py              # Main training script with Rich UI
│   ├── inference.py          # Model inference for new problems
│   ├── models/
│   │   └── cnn_rnn.py        # CNN+RNN architecture
│   ├── data/
│   │   ├── preprocess.py     # Dataset preprocessing
│   │   └── dataset.py        # PyTorch Dataset implementation
│   └── training/
│       └── loss.py           # Multi-task loss function
├── operations.py             # HamiltonianSTL class - grid operations
├── Zones.py                  # Zone pattern generators
├── SA_generation.py          # Simulated Annealing + dataset generation
├── ga_final.py               # Genetic Algorithm with state-aware crossover
├── run_pipeline.py           # Entry point for parallel data generation
├── merge_datasets.py         # Dataset merging utilities
├── nn_data/                  # Preprocessed training data
│   ├── train_all.jsonl       # Training split
│   ├── val_all.jsonl         # Validation split
│   ├── test_all.jsonl        # Test split
│   └── 30x30/                # Per-grid-size splits
├── nn_checkpoints/           # Model checkpoints
│   └── best_model.pt         # Best model checkpoint
├── nn_logs/                  # Training logs
│   ├── training_log.csv      # Epoch-by-epoch metrics
│   └── training_summary.json # Final training summary
├── requirements.txt          # Dependencies
├── pipeline/                 # Pipeline modules
│   ├── runner.py             # Multiprocessing orchestrator with Rich UI
│   ├── worker.py             # Task execution worker
│   ├── task_generator.py     # Task generation from configs
│   ├── config.py             # Configuration data classes
│   ├── checkpoint.py         # Progress tracking & resume
│   └── merge.py              # Dataset merging utility
└── config/                   # Configuration files
    └── global_config.yaml    # SA algorithm presets
```

## CNN+RNN Neural Network

The project includes a **paper-compliant CNN+RNN model** that learns from SA-generated data to predict optimal operation sequences.

### Architecture (CCAI 2026)

**CNN Backbone:**
- 4 convolutional layers (16→32→64→128 channels)
- 3×3 kernels with batch normalization
- Global average pooling to 128-dim embedding

**RNN Solver:**
- GRU with 256 hidden units, 2 layers
- Processes sequence of CNN embeddings
- Maintains temporal dependencies

**Multi-Head Predictor:**
- Operation type: T (transpose), F (flip), N (no-op)
- Position: (x, y) coordinates
- Variant: Pattern-specific variants

### Configuration

All training parameters are in `model/config.yaml`:

```yaml
model:
  predictor:
    sequence_length: 100        # Operations to predict (reduce for speed)
    max_positions: 30           # Grid size support

training:
  epochs: 100
  batch_sizes:
    "30x30": 64                 # Batch size per grid
    "50x50": 32
  learning_rate: 0.001
  
validation:
  compute_performance_stats: false  # Toggle for speed
  validate_every_n_epochs: 1
checkpointing:
  save_every_n_epochs: 10     # Periodic checkpoints
```

### Performance Tuning

**Speed vs Quality Trade-offs:**

| Config | Epoch Time | Quality |
|--------|-----------|---------|
| `sequence_length: 200` | ~24 min | Best |
| `sequence_length: 100` | ~12 min | Good |
| `sequence_length: 50` | ~6 min | OK |

**Recommended:** `sequence_length: 100` for 2× speedup with minimal quality loss.

### Model Output Location

After training completes, your models are saved in:

```
nn_checkpoints/
├── best_model.pt              ← Your best model (use this!)
├── checkpoint_epoch_1.pt      ← Periodic checkpoints
├── checkpoint_epoch_10.pt
└── checkpoint_epoch_20.pt
```

**Use `nn_checkpoints/best_model.pt`** for inference - it has the lowest validation loss.

### Inference

Use the trained model to optimize new Hamiltonian paths:

```bash
python model/inference.py \
    --checkpoint nn_checkpoints/best_model.pt \
    --grid-W 30 \
    --grid-H 30 \
    --zone-pattern left_right
```

## Quick Start

### Neural Network Training (Recommended)

The fastest way to train the CNN+RNN model:

```bash
# 1. Setup environment (one-time)
.\setup_nn.bat

# 2. Run training (auto-preprocesses data and trains)
.\run_training.bat
```

That's it! The script will:
- Check for preprocessed data or create it automatically
- Train the CNN+RNN model with Rich UI dashboard
- Save checkpoints to `nn_checkpoints/`
- Log metrics to `nn_logs/`

**After training, your model is at:** `nn_checkpoints/best_model.pt`

### Installation (Manual)

```bash
# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib numpy pyyaml rich tqdm

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
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

SA and GA output is written to `output/<machine_id>/Dataset.jsonl` as JSON Lines:

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

This data is preprocessed into `nn_data/` for neural network training:
- Stratified split (80/10/10) by grid size
- Filtered to records with improvement > 0
- Multi-channel grid state representation (H_edges, V_edges, zones, mask)
- Encoded operation sequences for supervised learning

### Merging Datasets

After generating data on multiple machines, merge them:

```bash
python merge_datasets_safe.py
```

This creates `output/datasets/combined_dataset.jsonl` for training.

## Troubleshooting

### Neural Network Training

**Training is very slow (>20 min/epoch):**
```yaml
# In model/config.yaml, reduce sequence_length:
model:
  predictor:
    sequence_length: 100  # Instead of 200 (2× speedup)
```

**GPU out of memory:**
```yaml
# Reduce batch size in model/config.yaml:
training:
  batch_sizes:
    "30x30": 32  # Instead of 64
```

**Want faster training (sacrifice some monitoring):**
```yaml
validation:
  compute_performance_stats: false  # Skip detailed stats
```

**Training not using GPU:**
```bash
# Check GPU is detected
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Data Generation

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

## Genetic Algorithm (ga_final.py)

The GA further optimizes paths beyond what SA achieves using **state-aware crossover**.

### Key Innovation

Traditional sequence-based crossover fails because operations are context-dependent. Our state-aware crossover splices actual path states (H/V edge matrices) instead of operation sequences.

### Results

| Zone Pattern | SA Dataset Best | GA Best | Improvement |
|--------------|-----------------|---------|-------------|
| left_right   | 12              | **5**   | 58%         |
| voronoi      | 12              | **9**   | 25%         |
| stripes      | 26              | **19**  | 27%         |
| islands      | 25              | **19**  | 24%         |

### Usage

```bash
# Run improved GA (default)
python ga_final.py --generations 300 --zone left_right

# Run paper's methodology for comparison
python ga_final.py --paper-compliant --generations 100

# Test different zone patterns
python ga_final.py --zone voronoi
python ga_final.py --zone stripes
```

See `EXPERIMENT_RESULTS.md` for detailed findings.

## Workflow Summary

### Complete Pipeline

```
1. Generate Training Data (SA/GA)
   └─→ python run_pipeline.py <machine_id> --workers 8

2. Merge Datasets (if multiple machines)
   └─→ python merge_datasets_safe.py

3. Train Neural Network
   └─→ .\run_training.bat

4. Use Trained Model
   └─→ python model/inference.py --checkpoint nn_checkpoints/best_model.pt ...
```

### Key Files

| File | Purpose |
|------|---------|
| `run_training.bat` | One-click training launcher |
| `model/config.yaml` | All hyperparameters |
| `nn_checkpoints/best_model.pt` | Best model checkpoint |
| `nn_logs/training_log.csv` | Training metrics |
| `setup_nn.bat` | Environment setup |

## License

[Add appropriate license information]
