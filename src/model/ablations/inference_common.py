"""
Shared inference wrapper for ablation study models.

Loads a model checkpoint, runs the same inference pipeline as inference_fusion.py,
and saves results to the model's output directory.

Supports two modes:
  - Default: evaluate_all_patterns (constructive + model-guided)
  - --comparison: evaluate_comparison (model+SA on voronoi/islands only)
"""

import sys
import os
import json
import argparse
import torch
from pathlib import Path

_this_dir = Path(__file__).resolve().parent
_model_dir = _this_dir.parent
_src_dir = _model_dir.parent
for p in [str(_this_dir), str(_model_dir), str(_src_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fusion_model import count_parameters
from rich.console import Console

console = Console()


def run_ablation_inference(model_class, model_name, default_output_dir,
                           model_kwargs_fn=None):
    """
    Run inference evaluation for an ablation model.

    Uses evaluate_all_patterns by default, or evaluate_comparison with --comparison.
    """
    parser = argparse.ArgumentParser(description=f'{model_name} Inference')
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(default_output_dir, 'best.pt'))
    parser.add_argument('--jsonl', type=str, default='datasets/final_dataset.jsonl')
    parser.add_argument('--n_per_pattern', type=int, default=25)
    parser.add_argument('--output_dir', type=str, default=default_output_dir)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--comparison', action='store_true',
                        help='Run model+SA comparison inference (voronoi/islands only)')
    parser.add_argument('--in_channels', type=int, default=9)
    parser.add_argument('--base_features', type=int, default=48)
    parser.add_argument('--max_grid_size', type=int, default=128)
    parser.add_argument('--n_hypotheses', type=int, default=1)
    parser.add_argument('--rnn_hidden', type=int, default=192)
    parser.add_argument('--rnn_layers', type=int, default=2)
    parser.add_argument('--rnn_dropout', type=float, default=0.15)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    console.print(f"\n[bold]{model_name} — Loading checkpoint: {args.checkpoint}[/bold]")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Build model kwargs
    kwargs = {
        'in_channels': args.in_channels,
        'base_features': args.base_features,
        'n_hypotheses': args.n_hypotheses,
        'max_grid_size': args.max_grid_size,
    }
    if model_kwargs_fn:
        kwargs.update(model_kwargs_fn(args))
    else:
        kwargs['max_history'] = 32
        kwargs['rnn_hidden'] = args.rnn_hidden
        kwargs['rnn_layers'] = args.rnn_layers
        kwargs['rnn_dropout'] = args.rnn_dropout

    model = model_class(**kwargs).to(device)

    # Load state dict (handle DDP prefix)
    state_dict = ckpt['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    n_params = count_parameters(model)
    console.print(f"  Parameters: [green]{n_params:,}[/green]")
    console.print(f"  Best epoch: {ckpt.get('epoch', '?')}")
    if 'val_metrics' in ckpt:
        vm = ckpt['val_metrics']
        console.print(f"  Val loss: {vm.get('loss', '?'):.4f}")
        console.print(f"  Recall@5: {vm.get('recall_at_5', '?'):.4f}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.comparison:
        # Model+SA comparison inference (voronoi/islands)
        from inference_comparison import evaluate_comparison

        mode_str = "Comparison (Model+SA)"
        console.print(f"  Mode: [bold yellow]{mode_str}[/bold yellow]")

        results = evaluate_comparison(
            model=model,
            jsonl_path=args.jsonl,
            n_per_pattern=args.n_per_pattern,
            device=device,
            visualize=args.visualize,
            output_dir=args.output_dir,
        )

        results_path = os.path.join(args.output_dir, 'comparison_results.json')
    else:
        # Standard inference (constructive + model-guided)
        from inference_fusion import evaluate_all_patterns

        mode_str = "Standard (Constructive + Model)"
        console.print(f"  Mode: [bold cyan]{mode_str}[/bold cyan]")

        vis_dir = os.path.join(args.output_dir, 'vis') if args.visualize else None
        if vis_dir:
            os.makedirs(vis_dir, exist_ok=True)

        results = evaluate_all_patterns(
            model=model,
            jsonl_path=args.jsonl,
            n_per_pattern=args.n_per_pattern,
            device=device,
            visualize=args.visualize,
            vis_dir=vis_dir if vis_dir else os.path.join(args.output_dir, 'vis'),
        )

        results_path = os.path.join(args.output_dir, 'inference_results.json')

    # Save results
    non_serializable = {'final_h', 'boundary_details'}
    serializable = []
    for r in results:
        sr = {k: v for k, v in r.items()
              if not isinstance(v, (torch.Tensor,)) and k not in non_serializable}
        if 'crossings_history' in r:
            sr['crossings_history'] = r['crossings_history']
        if 'sequence' in r:
            sr['sequence'] = [
                {'kind': op['kind'], 'x': op['x'], 'y': op['y'],
                 'variant': op['variant']}
                for op in r.get('sequence', [])
            ]
        serializable.append(sr)

    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)

    console.print(f"\n[bold green]{model_name} {mode_str} complete.[/bold green]")
    console.print(f"  Results: {results_path}")
    console.print(f"  Samples: {len(serializable)}")
