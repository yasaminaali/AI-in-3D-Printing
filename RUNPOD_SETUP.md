# RunPod RTX 5090 Setup Guide

## Quick Start (3 steps)

### Step 1: Connect to RunPod

Use either connection method:

**Option A - Direct TCP (supports SCP/SFTP for file transfer):**
```bash
ssh root@149.36.0.215 -p 14758 -i ~/.ssh/id_ed25519
```

**Option B - RunPod SSH:**
```bash
ssh kh35y2zmjyd62s-64411e69@ssh.runpod.io -i ~/.ssh/id_ed25519
```

### Step 2: Upload Your Data

From your local machine (in a new terminal), upload the necessary files:

```bash
# Upload preprocessed data
scp -P 14758 -i ~/.ssh/id_ed25519 -r nn_data/ root@149.36.0.215:/workspace/

# Upload your checkpoint (to resume training)
scp -P 14758 -i ~/.ssh/id_ed25519 -r nn_checkpoints/ root@149.36.0.215:/workspace/

# Upload the code
scp -P 14758 -i ~/.ssh/id_ed25519 -r AI-in-3D-Printing/ root@149.36.0.215:/workspace/
```

Or use the upload helper script (see below).

### Step 3: Run Setup and Training

Once connected to RunPod via SSH:

```bash
cd /workspace/AI-in-3D-Printing

# Make scripts executable
chmod +x setup_runpod.sh run_training.sh

# Run setup (installs PyTorch, dependencies)
./setup_runpod.sh

# Start training
./run_training.sh
```

## Manual Setup (if automated script fails)

```bash
# 1. Update system
apt-get update

# 2. Install git
apt-get install -y git

# 3. Clone repo (or use uploaded files)
cd /workspace
git clone https://github.com/anshjaiswal/AI-in-3D-Printing.git
cd AI-in-3D-Printing

# 4. Create virtual environment
python3 -m venv nn_venv
source nn_venv/bin/activate

# 5. Install PyTorch for RTX 5090 (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 6. Install other deps
pip install rich pyyaml matplotlib numpy tqdm

# 7. Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# 8. Run training
python model/train.py
```

## File Upload Helper Script

Create this on your local machine as `upload_to_runpod.sh`:

```bash
#!/bin/bash
# Upload files to RunPod

RUNPOD_IP="149.36.0.215"
RUNPOD_PORT="14758"
SSH_KEY="~/.ssh/id_ed25519"

echo "Uploading to RunPod RTX 5090..."

# Upload data
scp -P $RUNPOD_PORT -i $SSH_KEY -r nn_data/ root@$RUNPOD_IP:/workspace/
scp -P $RUNPOD_PORT -i $SSH_KEY -r nn_checkpoints/ root@$RUNPOD_IP:/workspace/

# Upload code
scp -P $RUNPOD_PORT -i $SSH_KEY -r model/ root@$RUNPOD_IP:/workspace/AI-in-3D-Printing/
scp -P $RUNPOD_PORT -i $SSH_KEY *.sh *.py root@$RUNPOD_IP:/workspace/AI-in-3D-Printing/

echo "Upload complete!"
```

Then run: `chmod +x upload_to_runpod.sh && ./upload_to_runpod.sh`

## Monitoring Training

Once training starts, you'll see:
- Rich UI dashboard with progress bars
- Loss metrics updating in real-time
- ETA for completion

To monitor GPU usage in another terminal:
```bash
ssh root@149.36.0.215 -p 14758 -i ~/.ssh/id_ed25519
watch -n 1 nvidia-smi
```

## Download Results

After training completes, download your trained model:

```bash
# From local machine
scp -P 14758 -i ~/.ssh/id_ed25519 root@149.36.0.215:/workspace/AI-in-3D-Printing/nn_checkpoints/best_model.pt .
scp -P 14758 -i ~/.ssh/id_ed25519 root@149.36.0.215:/workspace/AI-in-3D-Printing/nn_logs/training_log.csv .
```

## Expected Performance on RTX 5090

With the optimized config:
- **Batch size:** 256 (4× larger than RTX 4080)
- **Sequence length:** 100
- **Mixed precision:** Enabled (2× speedup)
- **Epoch time:** ~2-3 minutes
- **Total training:** Up to 300 epochs (~10-15 hours)
- **Best val loss target:** < 3.0 (good predictions), < 2.0 (excellent)

## Troubleshooting

**Connection refused:**
- Make sure RunPod pod is "Running" (not "Pending")
- Check the port number in RunPod dashboard

**CUDA out of memory:**
- Reduce batch_size in model/config.yaml
- RTX 5090 has 32GB, so batch_size=256 should work fine

**Training not using GPU:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

**Slow data upload:**
- Compress data first: `tar -czf nn_data.tar.gz nn_data/`
- Upload the tar file, then extract on RunPod
