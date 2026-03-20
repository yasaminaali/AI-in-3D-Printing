"""
Compare ablation study results across all model variants.

Reads training logs and inference results from each variant's checkpoint
directory and produces a summary comparison table + JSON.
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path

_model_dir = Path(__file__).resolve().parent.parent
_src_dir = _model_dir.parent
for p in [str(_model_dir), str(_src_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from fusion_model import count_parameters
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

VARIANTS = [
    ('cnn_only', 'CNN-Only'),
    ('unet_spatial', 'U-Net Spatial'),
    ('cnn_rnn', 'CNN+RNN'),
    ('unet_rnn_concat', 'U-Net+RNN (Concat)'),
]


def load_training_log(csv_path):
    """Load training_log.csv and return best epoch metrics."""
    if not os.path.exists(csv_path):
        return None
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return None

    # Find best epoch by val_loss
    best = min(rows, key=lambda r: float(r.get('val_loss', 1e9)))
    return {
        'best_epoch': int(best.get('epoch', 0)),
        'total_epochs': int(rows[-1].get('epoch', 0)),
        'best_val_loss': float(best.get('val_loss', 0)),
        'best_recall_at_5': float(best.get('val_recall_at_5', 0)),
        'best_recall_at_10': float(best.get('val_recall_at_10', 0)),
        'best_pos_rank_median': float(best.get('val_pos_rank_median', 0)),
        'best_act_acc_at_gt': float(best.get('val_act_acc_at_gt', 0)),
        'total_train_time': sum(float(r.get('epoch_time', 0)) for r in rows),
    }


def load_inference_results(json_path):
    """Load inference_results.json and compute summary stats."""
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r') as f:
        results = json.load(f)
    if not results:
        return None

    reductions = [r.get('reduction_pct', 0) for r in results if 'reduction_pct' in r]
    ops = [r.get('num_operations', 0) for r in results if 'num_operations' in r]
    in_target = [r.get('in_target_range', False) for r in results]
    times = [r.get('sample_time', 0) for r in results if 'sample_time' in r]

    return {
        'n_samples': len(results),
        'avg_reduction_pct': sum(reductions) / max(len(reductions), 1),
        'avg_operations': sum(ops) / max(len(ops), 1),
        'pct_in_target': sum(in_target) / max(len(in_target), 1) * 100,
        'avg_time': sum(times) / max(len(times), 1),
    }


def get_param_count(checkpoint_path):
    """Extract parameter count from checkpoint."""
    if not os.path.exists(checkpoint_path):
        return None
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        n_params = sum(p.numel() for p in ckpt['model_state_dict'].values())
        return n_params
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description='Compare ablation study results')
    parser.add_argument('--ablation_dir', type=str, default='checkpoints/ablations')
    parser.add_argument('--fusionnet_dir', type=str, default='checkpoints')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.ablation_dir, 'ablation_summary.json')

    # Collect all variants + FusionNet
    all_models = VARIANTS + [('__fusionnet__', 'FusionNet (FiLM)')]

    summary = {}

    table = Table(title="Ablation Study Results", box=box.ROUNDED, show_lines=True)
    table.add_column("Model", style="bold cyan", min_width=20)
    table.add_column("Params", justify="right")
    table.add_column("Best Epoch", justify="center")
    table.add_column("Val Loss", justify="right")
    table.add_column("R@5", justify="right")
    table.add_column("R@10", justify="right")
    table.add_column("Rank Med", justify="right")
    table.add_column("Act Acc", justify="right")
    table.add_column("Avg Red%", justify="right")
    table.add_column("Avg Ops", justify="right")
    table.add_column("In Target%", justify="right")
    table.add_column("Train Time", justify="right")

    for variant_id, variant_name in all_models:
        if variant_id == '__fusionnet__':
            base_dir = args.fusionnet_dir
        else:
            base_dir = os.path.join(args.ablation_dir, variant_id)

        train_log = load_training_log(os.path.join(base_dir, 'training_log.csv'))
        # Try comparison_results.json first, fall back to inference_results.json
        comp_path = os.path.join(base_dir, 'comparison_results.json')
        inf_path = os.path.join(base_dir, 'inference_results.json')
        if variant_id == '__fusionnet__':
            # FusionNet comparison results are in checkpoints/comparison/
            comp_path = os.path.join(base_dir, 'comparison', 'comparison_results.json')
        inf_results = load_inference_results(comp_path) or load_inference_results(inf_path)
        n_params = get_param_count(os.path.join(base_dir, 'best.pt'))

        entry = {
            'name': variant_name,
            'params': n_params,
            'training': train_log,
            'inference': inf_results,
        }
        summary[variant_id] = entry

        # Format table row
        params_str = f"{n_params:,}" if n_params else "N/A"
        if train_log:
            epoch_str = f"{train_log['best_epoch']}/{train_log['total_epochs']}"
            vloss_str = f"{train_log['best_val_loss']:.4f}"
            r5_str = f"{train_log['best_recall_at_5']:.3f}"
            r10_str = f"{train_log['best_recall_at_10']:.3f}"
            rank_str = f"{train_log['best_pos_rank_median']:.1f}"
            act_str = f"{train_log['best_act_acc_at_gt']:.3f}"
            time_str = f"{train_log['total_train_time'] / 3600:.1f}h"
        else:
            epoch_str = vloss_str = r5_str = r10_str = rank_str = act_str = time_str = "N/A"

        if inf_results:
            red_str = f"{inf_results['avg_reduction_pct']:.1f}%"
            ops_str = f"{inf_results['avg_operations']:.0f}"
            target_str = f"{inf_results['pct_in_target']:.0f}%"
        else:
            red_str = ops_str = target_str = "N/A"

        style = "bold green" if variant_id == '__fusionnet__' else ""
        table.add_row(
            variant_name, params_str, epoch_str, vloss_str,
            r5_str, r10_str, rank_str, act_str,
            red_str, ops_str, target_str, time_str,
            style=style,
        )

    console.print()
    console.print(table)

    # Save JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    console.print(f"\n[bold]Summary saved to {args.output}[/bold]")


if __name__ == '__main__':
    main()
