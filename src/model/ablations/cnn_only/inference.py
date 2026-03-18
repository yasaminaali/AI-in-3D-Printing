"""Run inference for CNN-Only ablation model."""
import sys
from pathlib import Path

_ablations_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ablations_dir))
sys.path.insert(0, str(_ablations_dir.parent))
sys.path.insert(0, str(_ablations_dir.parent.parent))

from model import CNNOnly
from inference_common import run_ablation_inference

if __name__ == '__main__':
    run_ablation_inference(
        model_class=CNNOnly,
        model_name="CNN-Only",
        default_output_dir="checkpoints/ablations/cnn_only",
    )
