"""Run inference for CNN-Only ablation model."""
import sys
from pathlib import Path

_this_dir = Path(__file__).resolve().parent
_ablations_dir = _this_dir.parent
_model_dir = _ablations_dir.parent
_src_dir = _model_dir.parent
for p in [str(_this_dir), str(_ablations_dir), str(_model_dir), str(_src_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from cnn_only_model import CNNOnly
from inference_common import run_ablation_inference

if __name__ == '__main__':
    run_ablation_inference(
        model_class=CNNOnly,
        model_name="CNN-Only",
        default_output_dir="checkpoints/ablations/cnn_only",
    )
