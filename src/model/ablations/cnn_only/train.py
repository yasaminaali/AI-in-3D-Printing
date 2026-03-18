"""Train CNN-Only ablation model."""
import sys
from pathlib import Path

# Add the cnn_only/ directory itself first so 'model' resolves to ./model.py
_this_dir = str(Path(__file__).resolve().parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

_ablations_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ablations_dir))
sys.path.insert(0, str(_ablations_dir.parent))
sys.path.insert(0, str(_ablations_dir.parent.parent))

from model import CNNOnly
from train_common import train_ablation

if __name__ == '__main__':
    train_ablation(
        model_class=CNNOnly,
        model_name="CNN-Only",
        default_checkpoint_dir="checkpoints/ablations/cnn_only",
    )
