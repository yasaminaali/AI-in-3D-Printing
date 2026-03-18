"""Run inference for U-Net Spatial-Only ablation model."""
import sys
from pathlib import Path

_ablations_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ablations_dir))
sys.path.insert(0, str(_ablations_dir.parent))
sys.path.insert(0, str(_ablations_dir.parent.parent))

from model import UNetSpatial
from inference_common import run_ablation_inference

if __name__ == '__main__':
    run_ablation_inference(
        model_class=UNetSpatial,
        model_name="U-Net Spatial",
        default_output_dir="checkpoints/ablations/unet_spatial",
    )
