"""Run inference for U-Net+RNN (Concat) ablation model."""
import sys
from pathlib import Path

_ablations_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ablations_dir))
sys.path.insert(0, str(_ablations_dir.parent))
sys.path.insert(0, str(_ablations_dir.parent.parent))

from model import UNetRNNConcat
from inference_common import run_ablation_inference

if __name__ == '__main__':
    run_ablation_inference(
        model_class=UNetRNNConcat,
        model_name="U-Net+RNN (Concat)",
        default_output_dir="checkpoints/ablations/unet_rnn_concat",
    )
