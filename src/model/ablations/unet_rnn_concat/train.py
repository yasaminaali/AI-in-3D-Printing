"""Train U-Net+RNN (Concat) ablation model."""
import sys
from pathlib import Path

_this_dir = Path(__file__).resolve().parent
_ablations_dir = _this_dir.parent
_model_dir = _ablations_dir.parent
_src_dir = _model_dir.parent
for p in [str(_this_dir), str(_ablations_dir), str(_model_dir), str(_src_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from unet_rnn_concat_model import UNetRNNConcat
from train_common import train_ablation

if __name__ == '__main__':
    train_ablation(
        model_class=UNetRNNConcat,
        model_name="U-Net+RNN (Concat)",
        default_checkpoint_dir="checkpoints/ablations/unet_rnn_concat",
    )
