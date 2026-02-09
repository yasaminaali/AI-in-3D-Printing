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

## Data Generation

### Simulated Annealing (SA)

Iterative optimization using Metropolis-Hastings acceptance on random transpose/flip operations with sigmoid cooling and reheating.

```bash
python SA_generation.py
```

### Genetic Algorithm (GA)

State-aware crossover that splices actual edge matrices (not operation sequences), avoiding context-dependency issues of traditional sequence crossover.

```bash
python GA_squence.py --generations 300 --zone left_right
```

### Pipeline (Parallel Generation)

```bash
python run_pipeline.py <machine_id> --workers 8
python merge_datasets_safe.py
```

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
