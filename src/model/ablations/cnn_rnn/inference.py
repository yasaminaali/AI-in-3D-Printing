"""Run inference for CNN+RNN ablation model."""
import sys
from pathlib import Path

_this_dir = str(Path(__file__).resolve().parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

_ablations_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ablations_dir))
sys.path.insert(0, str(_ablations_dir.parent))
sys.path.insert(0, str(_ablations_dir.parent.parent))

from model import CNNRNN
from inference_common import run_ablation_inference

if __name__ == '__main__':
    run_ablation_inference(
        model_class=CNNRNN,
        model_name="CNN+RNN",
        default_output_dir="checkpoints/ablations/cnn_rnn",
    )
