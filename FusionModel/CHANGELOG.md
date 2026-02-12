# FusionNet v2 Changelog

Complete rewrite to fix 86% failure rate. 18 changes across 5 files addressing data corruption, architecture limitations, loss design, inference strategy, and generalization.

---

## Data-Level Fixes

### Change 1: MAX_GRID_SIZE 32 → 128
**Files**: `build_fusion_data.py`, `fusion_dataset.py`, `inference_fusion.py`
**Why**: New dataset has grids up to 100x100. Old 32x32 truncated them.

### Change 15: Position Balancing
**File**: `build_fusion_data.py`
**Why**: SA's `border_to_inner=True` creates training data dominated by outer-edge operations (the "only works on edges" symptom).
**What**: Compute layer index `min(y, x, grid_h-1-y, grid_w-1-x)` per sample. Oversample inner layers to match outer layer count.

### Change 16: Fix Init Pattern Inference + Safe Edge Replay (CRITICAL)
**File**: `build_fusion_data.py`
**Why**: Builder hardcoded `init_pattern='zigzag'` but SA uses `init_pattern='auto'` which resolves to `vertical_zigzag` for `left_right` and vertical `stripes`. **61.7% of trajectories (11,130/18,043) replayed with wrong initial path** — edge diffs produced non-Hamiltonian corrupted states.
**What**: Added `_infer_init_pattern(zone_pattern, zone_grid, grid_w, grid_h)` to resolve the correct pattern. Switched to safe `transpose_subgrid`/`flip_subgrid` methods with status checking (auto-rollback on failure).

### Change 17: Compact Grouped Storage
**File**: `build_fusion_data.py`
**Why**: 350K samples × 128×128 × float32 = 92 GB. Impractical.
**What**: Group samples by grid size, store at natural resolution in float16. Pad to 128x128 on-the-fly in dataset. Result: ~1.7 GB (54x smaller).

---

## Architecture-Level Fixes

### Change 2: 4th Encoder/Decoder Level
**File**: `fusion_model.py`
**Why**: 128x128 with 3 levels gives 16x16 bottleneck (too large). 4 levels gives 8x8.
**What**: Added `enc4`/`dec4` with corresponding FiLM layer. ~700K new parameters.

### Change 3: GroupNorm Replacing BatchNorm
**File**: `fusion_model.py`
**Why**: BN statistics contaminated by variable zero-padding ratios (30x30 = 94.5% zeros vs 100x100 = 39% zeros in 128x128 tensor). GroupNorm normalizes per-sample.
**What**: All `nn.BatchNorm2d(ch)` → `nn.GroupNorm(min(8, ch), ch)`.

### Change 4: Refinement Heads
**File**: `fusion_model.py`
**Why**: Position head was single 1×1 conv (196 params, zero spatial context). Action head supervised at one pixel only.
**What**: Position head: 3-layer CNN (3×3 → 3×3 → 1×1). Action head: 2-layer CNN (3×3 → 1×1).

### Change 5: 9-Channel Input (Split Ch3 + Crossing Count + Progress + CoordConv)
**Files**: `build_fusion_data.py`, `fusion_dataset.py`, `inference_fusion.py`, `fusion_model.py`
**Why**: Channel 3 overloaded (grid extent + zone boundary in one number). Binary crossing channel loses density info. CNNs are translation-equivariant but crossings are position-dependent.
**What**: 9 channels:
| Ch | Content | Source |
|----|---------|--------|
| 0 | zones / max_zone | Saved in .pt |
| 1 | H edges | Saved in .pt |
| 2 | V edges | Saved in .pt |
| 3 | Grid validity (1.0 inside grid) | On-the-fly |
| 4 | Zone boundary | Saved in .pt (extracted from old ch3) |
| 5 | Crossing count (normalized) | On-the-fly |
| 6 | Progress (current/initial crossings) | On-the-fly |
| 7 | Y coordinate (0→1) | On-the-fly |
| 8 | X coordinate (0→1) | On-the-fly |

### Change 13: Bottleneck Self-Attention
**File**: `fusion_model.py`
**Why**: CNN receptive field ~25px can't capture how one move at (5,5) affects crossings at (80,80). Self-attention at 8×8 bottleneck (64 tokens) gives global reasoning.
**What**: `BottleneckAttention` module with `nn.MultiheadAttention(channels, 4)` + residual + LayerNorm. ~150K parameters.

---

## Loss-Level Fixes

### Change 6: Gaussian Heatmap Position Targets
**File**: `fusion_model.py` (`compute_loss`)
**Why**: One-hot targets give zero gradient signal for close-but-wrong predictions.
**What**: Generate 2D Gaussian (σ=2.0) at target position, mask to boundary, renormalize to sum=1. KL divergence loss between predicted softmax and Gaussian target. WTA: only best hypothesis gets gradient.

### Change 7: Fix Crossing Normalization
**Files**: `build_fusion_data.py`, `fusion_dataset.py`, `inference_fusion.py`
**Why**: Hardcoded `/60.0` produces values >1.0 for grids larger than 30x30. GRU gets out-of-distribution inputs.
**What**: Save `initial_crossings` per sample. Normalize by `max(initial_crossings, 1)`.

### Change 14: Neighborhood Action Supervision
**File**: `fusion_model.py` (`compute_loss`)
**Why**: Action head only supervised at exact GT pixel. 1px away from target = untrained noise.
**What**: Add action cross-entropy loss at 4-connected neighbors with 0.3 weight + label smoothing 0.2.

---

## Inference-Level Fixes

### Change 8: Per-Hypothesis Candidate Selection
**File**: `inference_fusion.py`
**Why**: WTA training specializes hypotheses. Mean-softmax averaging destroys specialization (averaging two peaks = a valley).
**What**: Each of K hypotheses produces top candidates independently. All K×top_k candidates are de-duplicated by position and sorted by score.

### Change 9: Adaptive Give-Up Threshold
**File**: `inference_fusion.py`
**Why**: Hardcoded `max_failures=10` is too small for large grids.
**What**: `max_failures = max(30, initial_crossings * 2)`, `max_steps = max(150, initial_crossings * 5)`.

### Change 10: History Seeding
**File**: `inference_fusion.py`
**Why**: Empty history → zero GRU context → FiLM identity → model never benefits from temporal context.
**What**: Before neural inference, run random boundary operations. Keep first 3-5 that reduce crossings. Seed the history buffer.

### Change 11: Stochastic Exploration
**File**: `inference_fusion.py`
**Why**: Pure greedy cascades into repeated failures at similar positions.
**What**: With probability 0.15, sample from position distribution instead of argmax.

---

## Generalization-Level Fixes

### Change 12: State Perturbation (DAgger-lite)
**File**: `fusion_dataset.py`
**Why**: Model trained on perfect teacher states, infers on its own imperfect states.
**What**: `perturbation_prob` parameter in dataset (default 0.0, set to 0.3 for training). Stub for applying random valid operations before presenting state to model.

---

## Training Infrastructure

### Change 18: DDP + bf16 + Staged Training
**File**: `train_fusion.py`
**What**:
- DistributedDataParallel via `torchrun --nproc_per_node=4`
- bf16 autocast (H100 native)
- lr=4e-4, warmup=2000 steps, patience=40 epochs
- 200 epochs max, batch_size=64/GPU (256 effective)
- Checkpoint saving: every 10 epochs + best, with optimizer/scheduler state
- Resume from checkpoint support (`--resume`)
- Text-based progress dashboard (DDP-compatible)

| Setting | Old | New |
|---------|-----|-----|
| MAX_GRID_SIZE | 32 | 128 |
| in_channels | 5 | 9 |
| Encoder levels | 3 | 4 |
| Normalization | BatchNorm | GroupNorm |
| Position head | 1×1 conv | 3-layer refinement |
| Action head | 1×1 conv | 2-layer refinement |
| Attention | None | MHSA at bottleneck |
| batch_size/GPU | 128 | 64 |
| Effective batch | 128 | 256 (4 GPUs) |
| Mixed precision | None | bf16 |
| Epochs | 30 | 200 |
| Learning rate | 3e-4 | 4e-4 |
| Warmup steps | 500 | 2000 |
| Patience | N/A | 40 |
| History depth | 8 | 32 |
| Position loss | CE one-hot | Gaussian heatmap σ=2 + KL |
| Action loss | GT pixel only | GT + 4 neighbors |
| Crossing norm | /60.0 | /initial_crossings |

---

## Deployment

### SLURM Scripts (CCRI TamIA, 4x H100 SXM 80GB)

Pipeline order:
1. `sbatch sbatch_build_fusion_data.sh` — CPU job, builds compact .pt from JSONL (~15-30 min)
2. `sbatch sbatch_train_fusion.sh` — 4x H100 DDP training (~5-7 hours for 200 epochs)
3. `sbatch sbatch_inference_fusion.sh` — 1x H100, evaluates on test set (~30 min)

Environment variables for customization:
```bash
# Data building
DATASET_JSONL=path/to/file.jsonl sbatch sbatch_build_fusion_data.sh

# Training
EPOCHS=30 BATCH_SIZE=64 LR=4e-4 sbatch sbatch_train_fusion.sh
RESUME_CKPT=path/to/checkpoint.pt sbatch sbatch_train_fusion.sh

# Inference
CHECKPOINT=path/to/best.pt TEST_JSONL=path/to/test.jsonl N_SAMPLES=100 sbatch sbatch_inference_fusion.sh
```

### Step-by-Step Usage

**1. Push code & pull on TamIA**
```bash
# Local machine
git add -A && git commit -m "FusionNet v2" && git push

# SSH into TamIA
cd ~/links/projects/aip-rnishat/shared/AI-in-3D-Printing
git pull
```

**2. Build training data** (CPU job, ~15-30 min)
```bash
sbatch sbatch_build_fusion_data.sh
# Monitor: tail -f fusion_build_data_<jobid>.out
```

**3. Staged training** — start small, verify, scale up

For **1-GPU stages** (Stage 1 & 2), use an interactive session:
```bash
srun --account=aip-rnishat --qos=normal --gres=gpu:h100:1 \
     --mem=64G --cpus-per-task=8 --time=01:00:00 --pty bash

# Once on the node:
module load python/3.11.5 cuda/12.6
cd ~/links/projects/aip-rnishat/shared/AI-in-3D-Printing
source sa_gpu_env/bin/activate

# Stage 1: Smoke test (10 epochs, ~5 min)
PYTHONPATH=$(pwd):$PYTHONPATH python3 FusionModel/fusion/train_fusion.py \
    --epochs 10 --batch_size 64 --learning_rate 4e-4 --patience 10

# Stage 2: Architecture validation (15 epochs, ~15 min)
PYTHONPATH=$(pwd):$PYTHONPATH python3 FusionModel/fusion/train_fusion.py \
    --epochs 15 --batch_size 64 --learning_rate 4e-4 --patience 15
```

For **4-GPU stages** (Stage 3 & 4), use sbatch:
```bash
# Stage 3: Short DDP run (30 epochs, ~1-2 hr)
EPOCHS=30 PATIENCE=30 sbatch --time=03:00:00 sbatch_train_fusion.sh

# Stage 4: Full training (200 epochs, ~5-7 hr) — only if Stage 3 passes
sbatch sbatch_train_fusion.sh
```

**4. Resume from checkpoint** (if job times out)
```bash
RESUME_CKPT=nn_checkpoints/fusion/checkpoint_epoch_30.pt sbatch sbatch_train_fusion.sh
```

**5. Run inference evaluation**
```bash
sbatch sbatch_inference_fusion.sh
# Results: FusionModel/nn_checkpoints/fusion/inference_results.json
```

### Staged Validation Criteria

| Stage | Duration | GPUs | Epochs | Pass If |
|-------|----------|------|--------|---------|
| 1 (Smoke) | ~5 min | 1 | 10 | val_pos_acc_top1 trending > 20% |
| 2 (Arch) | ~15 min | 1 | 15 | No NaN, loss decreasing |
| 3 (Short) | ~1-2 hr | 4 | 30 | pos_acc > 25%, any inference reductions |
| 4 (Full) | ~5-7 hr | 4 | 200 | pos_acc > 35%, avg reduction > 5.0, >50% samples improved |

### Dataset Note
`final_dataset.jsonl` contains 18,386 trajectories (18,263 SA + 123 GA). The 123 GA records lack `zone_grid` and are automatically skipped during data building (0.7% of data).
