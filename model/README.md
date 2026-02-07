# Neural Network Models for Hamiltonian Path Optimization

This directory contains three model approaches for predicting crossing-reducing operations on Hamiltonian paths in multi-material 3D printing grids. **OperationNet (U-Net)** is the current primary approach.

## Directory Structure

```
model/
├── README.md                          # This file
├── unet/                              # OperationNet (U-Net) — primary approach
│   ├── unet_model.py                 # ResU-Net architecture + WTA loss
│   ├── unet_dataset.py               # Dataset with augmentation + boundary masking
│   ├── train_unet.py                 # Training script with Rich dashboard
│   ├── inference_unet.py             # Iterative inference + evaluation
│   ├── precompute_unet_data.py       # Build unet_data.pt (flat tensors, fast)
│   └── build_unet_data.py            # Build unet_data.pkl (pickle, legacy)
├── decision_transformer/             # Decision Transformer v2 — alternative
│   ├── dt_model.py
│   ├── dt_dataset_v2.py
│   ├── train_dt_v2.py
│   ├── inference_dt_v2.py
│   ├── build_dt_training_data.py
│   └── precompute_rtg.py
├── models/
│   └── cnn_rnn.py                    # CNN+RNN baseline (reference only)
├── data/
│   ├── preprocess.py                 # CNN+RNN dataset preprocessing
│   └── dataset.py                    # CNN+RNN PyTorch Dataset
├── training/
│   └── loss.py                       # CNN+RNN multi-task loss
├── utils/
│   └── visualization.py
├── config.yaml                       # CNN+RNN hyperparameters
├── train.py                          # CNN+RNN training entry point
└── inference.py                      # CNN+RNN inference script
```

---

## OperationNet (U-Net) — Primary Approach

OperationNet is a ResU-Net that maintains full spatial resolution to predict **where** to apply operations (position) and **what** to apply (action) on a per-pixel basis. Unlike the Decision Transformer, it avoids spatial compression bottlenecks, preserving fine-grained position information across the entire grid.

### Architecture Overview

The model is defined in `unet/unet_model.py` as the `OperationNet` class.

**Input:** 5-channel 32x32 tensor `[B, 5, 32, 32]`

| Channel | Content | Encoding |
|---------|---------|----------|
| 0 | Zone IDs | Normalized to [0, 1] by dividing by max zone ID |
| 1 | Horizontal edges (H) | Binary — 1 if edge present between (y, x) and (y, x+1) |
| 2 | Vertical edges (V) | Binary — 1 if edge present between (y, x) and (y+1, x) |
| 3 | Boundary mask + grid validity | 1.0 at zone boundaries, 0.5 for valid grid cells, 0.0 outside grid |
| 4 | Crossing indicator | 1.0 at cells adjacent to a zone-crossing edge (computed on-the-fly) |

Grids smaller than 32x32 are zero-padded. The maximum supported grid size is 32x32.

**Output:** Two heads producing dense per-pixel predictions:
- **Position logits** `[B, K, 32, 32]` — K independent hypotheses for where to apply an operation
- **Action logits** `[B, 12, 32, 32]` — 12-class per-pixel action prediction

#### Encoder-Decoder Structure

```
Input [B, 5, 32, 32]
  │
  ├─ DownBlock 1: 5 → f channels      → skip e1 [B, f, 32, 32]     → pool [B, f, 16, 16]
  ├─ DownBlock 2: f → 2f channels     → skip e2 [B, 2f, 16, 16]    → pool [B, 2f, 8, 8]
  ├─ DownBlock 3: 2f → 4f channels    → skip e3 [B, 4f, 8, 8]      → pool [B, 4f, 4, 4]
  │
  ├─ Bottleneck: ResBlock(4f) + Dropout2d(0.1)                       → [B, 4f, 4, 4]
  │
  ├─ UpBlock 3: upsample + concat e3  → 4f+4f → 2f                  → [B, 2f, 8, 8]
  ├─ UpBlock 2: upsample + concat e2  → 2f+2f → f                   → [B, f, 16, 16]
  ├─ UpBlock 1: upsample + concat e1  → f+f → f                     → [B, f, 32, 32]
  │
  ├─ Position Head: Conv2d(f → K, 1x1)                               → [B, K, 32, 32]
  └─ Action Head:   Conv2d(f → 12, 1x1)                              → [B, 12, 32, 32]
```

Where `f = base_features` (default 48).

#### Building Blocks

- **ResBlock**: Two 3x3 convolutions with BatchNorm and LeakyReLU (slope=0.01), with an additive skip connection. `x + Conv(BN(LeakyReLU(Conv(BN(x)))))`.
- **DownBlock**: 1x1 projection (channel expansion) → ResBlock → MaxPool2d(2). Returns both the pre-pool features (for skip connections) and the downsampled output.
- **UpBlock**: Bilinear upsample(2x) → concatenate with skip connection → 1x1 projection (channel reduction) → ResBlock.

#### Action Classes

12 canonical action classes shared across all models:

| Index | Variant | Operation Type |
|-------|---------|---------------|
| 0-7 | nl, nr, sl, sr, eb, ea, wa, wb | Transpose (T) — reroutes a 3x3 subgrid |
| 8-11 | n, s, e, w | Flip (F) — reverses edges in a 2x3 or 3x2 subgrid |

### Winner-Takes-All (WTA) Loss

The loss function (`compute_loss` in `unet_model.py`) addresses multimodality in position prediction — multiple valid positions may exist for crossing reduction at any given state.

**Position loss (WTA):**
1. Each of K hypotheses produces a softmax distribution over boundary-masked positions
2. Cross-entropy loss is computed for each hypothesis against the ground truth position
3. Only the hypothesis closest to GT (lowest loss) receives gradients — the "winner"
4. This encourages each hypothesis to specialize on a different mode of the position distribution

**Diversity regularizer:**
- Computes soft winner assignment probabilities via softmin over per-hypothesis losses
- Penalizes deviation from uniform usage: `entropy_loss = sum(avg_usage * log(avg_usage)) + log(K)`
- When all hypotheses win equally often, diversity_loss = 0; when one hypothesis dominates, the penalty increases
- Prevents mode collapse where a single hypothesis wins everything

**Action loss:** Standard cross-entropy with label smoothing (default 0.1) at the ground truth position.

**Total loss:** `pos_weight * pos_loss + act_loss + diversity_weight * diversity_loss`

Default weights: `pos_weight=10.0`, `diversity_weight=0.5`.

### Dataset and Augmentation

Defined in `unet/unet_dataset.py`.

**Data format:** Loads from `unet_data.pt` containing:
- `states` `[N, 4, 32, 32]` — 4-channel grid states (channel 5 computed on-the-fly)
- `targets` `[N, 3]` — (y, x, action_class) per sample
- `traj_ids` `[N]` — trajectory membership for split integrity
- `n_trajectories` — total trajectory count

**On-the-fly computation:**
- **Crossing indicator** (channel 5): Marks cells adjacent to edges that cross zone boundaries. Computed from zones and edge matrices using vectorized operations.
- **Boundary mask**: Zone boundary positions dilated by a configurable radius (default 1-2 pixels) to cover operation neighborhoods. Masked to valid grid area.

**Data augmentation** (training only):
- **Horizontal flip** (50% probability): Flips grid channels within the valid grid region and remaps action variants (e.g., nl↔nr, e↔w)
- **Vertical flip** (50% probability): Flips grid channels and remaps action variants (e.g., nl↔sl, n↔s)
- Remapping tables (`HFLIP_ACTION`, `VFLIP_ACTION`) ensure geometric consistency

**Train/val split:** Trajectory-level split (default 90/10) prevents data leakage — all operations from a single SA trajectory go to the same split.

### Data Preprocessing Pipeline

Two preprocessing scripts convert raw JSONL trajectories into training data:

#### Option 1: `precompute_unet_data.py` → `unet_data.pt` (recommended)

Produces flat PyTorch tensors. Faster to load, smaller on disk.

```bash
PYTHONPATH=$(pwd):$PYTHONPATH python model/unet/precompute_unet_data.py \
    --input combined_dataset.jsonl \
    --output model/unet/unet_data.pt \
    --workers 0
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `combined_dataset.jsonl` | Source JSONL dataset |
| `--output` | `model/unet/unet_data.pt` | Output tensor file |
| `--workers` | `0` | Worker processes (0 = 30% of CPU cores) |
| `--limit` | `0` | Process only first N trajectories (0 = all) |

#### Option 2: `build_unet_data.py` → `unet_data.pkl` (legacy)

Produces a pickle with richer per-trajectory metadata. Useful for debugging.

```bash
PYTHONPATH=$(pwd):$PYTHONPATH python model/unet/build_unet_data.py \
    --input combined_dataset.jsonl \
    --output model/unet/unet_data.pkl \
    --min-reduction 0.0
```

**Both scripts perform the same core logic:**
1. Load each SA/GA trajectory from JSONL
2. Initialize a zigzag Hamiltonian path via `operations.py`
3. Replay every operation in the trajectory sequence
4. After each operation, count crossings — if crossings decreased, this is an "effective" operation
5. Capture the 4-channel grid state from **before** the effective operation as a training sample
6. Target = `(y, x, action_class)` of the effective operation

### Training

```bash
PYTHONPATH=$(pwd):$PYTHONPATH python model/unet/train_unet.py \
    --data_path model/unet/unet_data.pt \
    --epochs 20 \
    --batch_size 128 \
    --base_features 48
```

#### Training Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--data_path` | `model/unet/unet_data.pt` | Path to preprocessed training data |
| `--checkpoint_dir` | `nn_checkpoints/unet` | Directory for model checkpoints |
| `--epochs` | `20` | Number of training epochs |
| `--batch_size` | `128` | Batch size |
| `--learning_rate` | `3e-4` | Initial learning rate |
| `--weight_decay` | `1e-4` | AdamW weight decay |
| `--warmup_steps` | `500` | Linear warmup steps before cosine annealing |
| `--grad_clip` | `1.0` | Gradient clipping max norm |
| `--val_split` | `0.1` | Validation split ratio (trajectory-level) |
| `--base_features` | `48` | Base channel count `f` (encoder: f, 2f, 4f) |
| `--n_hypotheses` | `4` | Number of WTA position hypotheses (K) |
| `--pos_weight` | `10.0` | Position loss multiplier |
| `--diversity_weight` | `0.5` | Winner-assignment entropy regularizer weight |
| `--boundary_dilation` | `1` | Boundary mask dilation radius |
| `--num_workers` | `4` | DataLoader worker processes |
| `--no_augment` | `false` | Disable horizontal/vertical flip augmentation |

#### Training Features

- **Optimizer:** AdamW with betas=(0.9, 0.999)
- **Scheduler:** Linear warmup → cosine annealing (decays to 1% of initial LR)
- **Gradient clipping:** Max norm = 1.0
- **Rich dashboard:** Live training/validation metrics updated each epoch
- **Checkpointing:** Best model saved to `best.pt`, periodic checkpoints every 10 epochs
- **CSV logging:** Full metrics logged to `training_log.csv`
- **CUDA required:** Training requires a GPU

#### Validation Metrics

| Metric | Description |
|--------|-------------|
| `pos_acc_top1` | Position accuracy — mean-softmax pooled argmax matches GT |
| `pos_acc_top5` | Position in top-5 pooled predictions |
| `oracle_acc_top1` | Any of K hypotheses' argmax matches GT |
| `act_acc_at_gt` | Action accuracy at ground truth position |
| `act_acc_e2e` | End-to-end: correct action at correctly predicted position |

### Inference

```bash
PYTHONPATH=$(pwd):$PYTHONPATH python model/unet/inference_unet.py \
    --checkpoint nn_checkpoints/unet/best.pt \
    --data model/unet/unet_data.pt \
    --n_samples 50 \
    --max_steps 50 \
    --visualize
```

#### Inference Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | `nn_checkpoints/unet/best.pt` | Path to trained model checkpoint |
| `--data` | `model/unet/unet_data.pt` | Data source for evaluation samples |
| `--n_samples` | `50` | Number of test samples to evaluate |
| `--max_steps` | `50` | Maximum inference steps per sample |
| `--top_k_pos` | `15` | Top-k boundary positions to try per step |
| `--top_k_act` | `3` | Top-k actions to try per candidate position |
| `--device` | `cuda` | Device (cuda or cpu) |
| `--verbose` | `false` | Print per-step operation details |
| `--visualize` | `false` | Save before/after comparison PNGs |
| `--vis_dir` | `nn_checkpoints/unet/vis` | Directory for visualization output |

#### Iterative Inference Algorithm

At each step:
1. **Encode** the current grid state as a 5-channel 32x32 tensor
2. **Forward pass** through OperationNet → position logits + action logits
3. **Position selection:** Mean-softmax pooling across K hypotheses, masked to dilated boundary positions. Select top-k candidates.
4. **Action selection:** For each candidate position, take top-k action logits
5. **Validation:** Check bounds and subgrid validity for each (position, action) pair
6. **Application:** Apply the first valid, crossing-reducing operation via `operations.py` (guarantees Hamiltonicity)
7. **Rollback:** If an operation doesn't reduce crossings, restore the saved edge state
8. **Termination:** Stop after `max_steps` or 10 consecutive steps without improvement

Outputs include per-sample crossing reduction, operation sequences, and optional before/after visualizations.

#### Output Files

- `nn_checkpoints/unet/inference_results.json` — per-sample results with crossing histories and operation sequences
- `nn_checkpoints/unet/vis/sample_N.png` — initial vs. U-Net comparison plots (if `--visualize`)

### Design Rationale

**Why U-Net over Decision Transformer?**
The Decision Transformer compresses a 30x30 grid into a 256-dimensional vector via adaptive average pooling, losing fine-grained spatial information. Position accuracy plateaus at ~22% (x) and ~11% (y). The U-Net maintains full 32x32 spatial resolution through its encoder-decoder with skip connections, enabling precise per-pixel position prediction.

**Why WTA over standard cross-entropy for position?**
At any given grid state, multiple positions may validly reduce crossings. Standard CE forces the model to predict a single "correct" answer, leading to blurry/averaged predictions. WTA allows K hypotheses to specialize on different valid modes, with the diversity regularizer ensuring all hypotheses remain active.

**Why boundary masking?**
Operations that reduce crossings must occur near zone boundaries by definition. Masking the position softmax to boundary-adjacent cells (with dilation) prevents the model from wasting capacity predicting positions in zone interiors, focusing learning on the relevant spatial regions.

**Why trajectory-level splits?**
Operations within a single SA trajectory are correlated (each depends on the previous state). Splitting at the operation level would leak information about trajectory dynamics into the validation set. Splitting at the trajectory level ensures the model generalizes to unseen optimization runs.

---

## CNN+RNN Neural Network (Baseline — Reference Only)

Paper-compliant hybrid architecture implementing the CCAI 2026 approach.

### Architecture Overview

**CNN Backbone (Section 3.3.1):**
- 4 convolutional layers with 3x3 kernels
- Channel progression: 16 → 32 → 64 → 128
- Batch normalization and ReLU activation
- Global average pooling to 128-dim embedding
- Combines with global features (crossings, grid size)

**RNN Solver (Section 3.3.2):**
- GRU with 256 hidden units, 2 layers
- Processes sequence of CNN embeddings
- Maintains hidden state for temporal dependencies
- Captures operation sequence patterns

**Operation Predictor:**
- Multi-head prediction:
  - Operation type: T (transpose), F (flip), N (no-op)
  - Position: (x, y) coordinates on 30x30 grid
  - Variant: Pattern-specific (nl/nr/sl/sr/eb for T, n/s/e/w for F)
- Auxiliary crossing reduction prediction

### Usage

#### 1. Preprocess Dataset

```bash
python model/data/preprocess.py
```

This will:
- Load `output/ansh/Dataset.jsonl`
- Sort records by grid sizes (10x10, 15x15, 20x20, etc.)
- Filter records with no improvement
- Create stratified 80/10/10 splits per grid size
- Save to `nn_data/` directory

#### 2. Train Model

```bash
python model/train.py \
    --train-file nn_data/train_all.jsonl \
    --val-file nn_data/val_all.jsonl \
    --config model/config.yaml
```

Features:
- Rich UI with real-time metrics dashboard
- Live loss component tracking (type, x, y, variant)
- Training history visualization
- Automatic checkpointing (saves best model)
- Early stopping with patience
- Learning rate scheduling

**Config Adjustments:**
Edit `model/config.yaml` to change:
- `training.epochs`: Number of training epochs (default: 100)
- `training.batch_size`: Batch size (default: 32)
- `training.learning_rate`: Initial learning rate (default: 0.001)
- `training.early_stopping_patience`: Early stopping patience (default: 20)

#### 3. Run Inference

```bash
python model/inference.py \
    --checkpoint nn_checkpoints/best_model.pt \
    --grid-W 30 \
    --grid-H 30 \
    --zone-pattern left_right \
    --initial-path zigzag \
    --output result.json
```

### Configuration (config.yaml)

#### Model Architecture
```yaml
model:
  cnn:
    input_channels: 4        # H_edges, V_edges, zones, mask
    embedding_dim: 128       # CNN output dimension (paper)
    num_layers: 4            # Number of conv layers (paper)
    kernel_size: 3           # 3x3 convolutions (paper)

  rnn:
    type: "GRU"              # Gated Recurrent Unit (paper)
    hidden_size: 256         # GRU hidden dimension
    num_layers: 2            # GRU layers
    dropout: 0.2             # Dropout rate

  predictor:
    sequence_length: 100     # Max operations to predict
    max_positions: 30        # Grid size (30x30)
```

#### Training Parameters
```yaml
training:
  epochs: 100                # Modify as needed
  batch_size: 32            # Modify as needed
  learning_rate: 0.001      # Modify as needed
  early_stopping_patience: 20

  loss_weights:
    operation_type: 1.0
    position_x: 1.0
    position_y: 1.0
    variant: 0.5
    crossing_reduction: 2.0
```

#### Data Processing
```yaml
data:
  filter:
    min_improvement: 1              # Discard no-improvement records
    min_crossing_reduction_ratio: 0.1  # At least 10% improvement

  split:
    train_ratio: 0.8
    val_ratio: 0.1
    test_ratio: 0.1
```

### Output Files

**After Preprocessing:**
- `nn_data/preprocessing_stats.json` - Dataset statistics
- `nn_data/train_all.jsonl` - Combined training data
- `nn_data/val_all.jsonl` - Combined validation data
- `nn_data/test_all.jsonl` - Combined test data
- `nn_data/{W}x{H}/train.jsonl` - Per-grid-size splits

**After Training:**
- `nn_checkpoints/best_model.pt` - Best model checkpoint
- `nn_logs/` - Training logs (if TensorBoard enabled)

---

## Quick Reference — Commands

```bash
# Always set PYTHONPATH first
export PYTHONPATH=$(pwd):$PYTHONPATH

# ---- OperationNet (U-Net) Full Pipeline ----
# 1. Preprocess (choose one)
python model/unet/precompute_unet_data.py --input combined_dataset.jsonl     # → unet_data.pt (recommended)
python model/unet/build_unet_data.py --input combined_dataset.jsonl          # → unet_data.pkl (legacy)

# 2. Train
python model/unet/train_unet.py --epochs 20 --batch_size 128

# 3. Evaluate
python model/unet/inference_unet.py --checkpoint nn_checkpoints/unet/best.pt --visualize

# ---- CNN+RNN Baseline ----
python model/data/preprocess.py
python model/train.py
python model/inference.py --checkpoint nn_checkpoints/best_model.pt --grid-W 30 --grid-H 30 --zone-pattern left_right
```
