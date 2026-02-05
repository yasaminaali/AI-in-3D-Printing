# CNN+RNN Neural Network for Hamiltonian Path Optimization

Paper-compliant hybrid architecture implementing the CCAI 2026 approach.

## Architecture Overview

**CNN Backbone (Section 3.3.1):**
- 4 convolutional layers with 3×3 kernels
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
  - Position: (x, y) coordinates on 30×30 grid
  - Variant: Pattern-specific (nl/nr/sl/sr/eb for T, n/s/e/w for F)
- Auxiliary crossing reduction prediction

## Directory Structure

```
model/
├── config.yaml              # Hyperparameters & architecture config
├── data/
│   ├── preprocess.py       # Dataset preprocessing & stratified split
│   └── dataset.py          # PyTorch Dataset implementation
├── models/
│   └── cnn_rnn.py         # CNN+RNN architecture
├── training/
│   ├── loss.py            # Multi-task loss function
│   └── trainer.py         # Training loop with Rich UI
├── train.py               # Main training entry point
├── inference.py           # Inference & optimization script
└── README.md             # This file
```

## Usage

### 1. Preprocess Dataset

```bash
python model/data/preprocess.py
```

This will:
- Load `output/ansh/Dataset.jsonl`
- Sort records by grid sizes (10x10, 15x15, 20x20, etc.)
- Filter records with no improvement
- Create stratified 80/10/10 splits per grid size
- Save to `nn_data/` directory

### 2. Train Model

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

### 3. Run Inference

```bash
python model/inference.py \
    --checkpoint nn_checkpoints/best_model.pt \
    --grid-W 30 \
    --grid-H 30 \
    --zone-pattern left_right \
    --initial-path zigzag \
    --output result.json
```

## Configuration (config.yaml)

### Model Architecture
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

### Training Parameters
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

### Data Processing
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

## Output Files

**After Preprocessing:**
- `nn_data/preprocessing_stats.json` - Dataset statistics
- `nn_data/train_all.jsonl` - Combined training data
- `nn_data/val_all.jsonl` - Combined validation data
- `nn_data/test_all.jsonl` - Combined test data
- `nn_data/{W}x{H}/train.jsonl` - Per-grid-size splits

**After Training:**
- `nn_checkpoints/best_model.pt` - Best model checkpoint
- `nn_logs/` - Training logs (if TensorBoard enabled)

## Paper Compliance

This implementation follows the CCAI 2026 paper specifications:

1. **Sequence-based representation** (Section 3.2): Model learns to predict sequences of operations rather than direct grid manipulation

2. **CNN architecture** (Section 3.3.1): Exact 4-layer architecture with specified channels and 128-dim output

3. **RNN architecture** (Section 3.3.2): GRU-based temporal modeling with hidden state maintenance

4. **Multi-task prediction**: Joint prediction of operation type, position, and variant

5. **Hamiltonian validity**: All predicted operations preserve Hamiltonian path properties

## Key Differences from GA

Unlike the GA (`inital_ga_squence.py`), this NN approach:
- Learns from SA-generated training data
- Generalizes to unseen zone patterns
- Predicts operation sequences in one forward pass
- Can be deployed for real-time optimization
- Learns implicit heuristics from data rather than explicit rules

## Commands Summary

```bash
# 1. Preprocess
python model/data/preprocess.py

# 2. Train (adjust epochs in config.yaml)
python model/train.py

# 3. Inference
python model/inference.py --checkpoint nn_checkpoints/best_model.pt --grid-W 30 --grid-H 30 --zone-pattern left_right
```
