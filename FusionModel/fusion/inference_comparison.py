"""
Comparison Inference: Model+SA on ALL patterns (no constructive).

Forces the model+SA alternating cycle on left_right and stripes (which
normally use the fast constructive approach) to demonstrate WHY the
constructive approach is needed for regular patterns.

All patterns use horizontal zigzag init → model+SA cycle.
Output goes to a separate folder so existing results are untouched.

Usage:
    PYTHONPATH=$(pwd):$PYTHONPATH python FusionModel/fusion/inference_comparison.py \\
        --checkpoint FusionModel/nn_checkpoints/fusion/best.pt
"""

import torch
import json
import argparse
import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inference_fusion import (
    run_inference, load_model, compute_boundary_mask, plot_cycle_on_ax,
    MAX_GRID_SIZE, MAX_HISTORY,
)
from constructive import (
    select_init_and_strategy, compute_crossings,
    TARGET_RANGES, CV_THRESHOLDS,
)
from operations import HamiltonianSTL
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import time as _time

console = Console()

ALL_PATTERNS = ['left_right', 'stripes', 'voronoi', 'islands']


# ---------------------------------------------------------------------------
# Evaluation: model+SA on ALL patterns
# ---------------------------------------------------------------------------

def evaluate_comparison(
    model,
    jsonl_path,
    n_per_pattern=25,
    max_steps=200,
    max_history=32,
    n_candidates=10,
    n_random=10,
    device=torch.device('cuda'),
    visualize=False,
    output_dir='FusionModel/nn_checkpoints/fusion/comparison',
):
    console.print(Panel.fit(
        "[bold cyan]Comparison: Model+SA on ALL Patterns (No Constructive)[/bold cyan]\n"
        f"Data: {jsonl_path}\n"
        f"Patterns: {', '.join(ALL_PATTERNS)}\n"
        f"Samples per pattern: {n_per_pattern}\n"
        f"ALL patterns use model+SA cycle (horizontal zigzag init)\n"
        f"Model candidates: {n_candidates} + random: {n_random} per step\n"
        f"Target ranges: {dict(TARGET_RANGES)}",
        border_style="cyan"
    ))

    with open(jsonl_path) as f:
        lines = f.readlines()

    pattern_lines = defaultdict(list)
    for line in lines:
        traj = json.loads(line.strip())
        pattern = traj.get('zone_pattern', 'unknown')
        pattern_lines[pattern].append(line)

    # Show dataset composition
    comp_table = Table(title="[bold]Dataset Composition[/bold]", box=box.SIMPLE)
    comp_table.add_column("Zone Pattern", style="cyan")
    comp_table.add_column("Total Trajectories", justify="right")
    comp_table.add_column("Test Samples", justify="right")
    for p in ALL_PATTERNS:
        available = len(pattern_lines[p])
        n_test = min(n_per_pattern, available)
        comp_table.add_row(p, str(available), str(n_test))
    console.print(comp_table)

    # Stratified sampling (same as original)
    test_samples = []
    for pattern in ALL_PATTERNS:
        available = pattern_lines[pattern]
        n_test = min(n_per_pattern, len(available))
        if n_test == 0:
            console.print(f"  [yellow]Warning: no samples for '{pattern}'[/yellow]")
            continue
        by_size = defaultdict(list)
        for line in available:
            traj = json.loads(line.strip())
            if 'zone_grid' not in traj:
                continue
            sa_init = traj.get('initial_crossings', 0)
            sa_final = traj.get('final_crossings', sa_init)
            if sa_init <= 0 or sa_final >= sa_init:
                continue
            size_key = (traj.get('grid_W', 30), traj.get('grid_H', 30))
            by_size[size_key].append(line)
        sizes = sorted(by_size.keys())
        selected = []
        size_indices = {s: 0 for s in sizes}
        while len(selected) < n_test:
            added_any = False
            for s in sizes:
                if len(selected) >= n_test:
                    break
                lines_for_size = by_size[s]
                idx = size_indices[s]
                if idx < len(lines_for_size):
                    selected.append(lines_for_size[idx])
                    size_indices[s] = idx + 1
                    added_any = True
            if not added_any:
                break
        for line in selected:
            test_samples.append((pattern, line))

    total_samples = len(test_samples)
    console.print(f"\n  Total test samples: [green]{total_samples}[/green]")

    all_results = []
    t0 = _time.time()
    vis_dir = os.path.join(output_dir, 'vis')

    for sample_idx, (pattern, line) in enumerate(test_samples):
        traj = json.loads(line.strip())
        grid_w = traj.get('grid_W', 30)
        grid_h = traj.get('grid_H', 30)
        zones_np = np.array(traj['zone_grid']).reshape(grid_h, grid_w)
        n_zones = len(set(zones_np.flatten().tolist()))
        boundary_mask = compute_boundary_mask(zones_np, grid_h, grid_w)

        # Detect stripe params (for target computation) but FORCE model strategy
        init_pat, orig_strategy, s_dir, s_k = select_init_and_strategy(
            pattern, zones_np, grid_w, grid_h,
        )

        sample_t0 = _time.time()
        result = run_inference(
            model=model,
            zones_np=zones_np,
            boundary_mask=boundary_mask,
            grid_w=grid_w,
            grid_h=grid_h,
            zone_pattern=pattern,
            strategy='model',  # FORCE model+SA for all patterns
            stripe_direction=s_dir,
            stripe_k=s_k,
            max_history=max_history,
            max_steps=max_steps,
            n_candidates=n_candidates,
            n_random=n_random,
            device=device,
        )
        sample_time = _time.time() - sample_t0

        # SA baseline info
        sa_initial = traj.get('initial_crossings', result['initial_crossings'])
        sa_final = traj.get('final_crossings', result['final_crossings'])
        sa_reduction = sa_initial - sa_final if sa_initial else 0
        n_sa_ops = len(traj.get('sequence_ops', []))
        n_sa_effective = sum(
            1 for op in traj.get('sequence_ops', []) if op.get('kind') != 'N'
        )

        result['sa_initial'] = sa_initial
        result['sa_final'] = sa_final
        result['sa_reduction'] = sa_reduction
        result['sa_reduction_pct'] = (
            (sa_reduction / sa_initial * 100)
            if sa_initial and sa_initial > 0 else 0
        )
        result['sa_ops'] = n_sa_ops
        result['sa_effective_ops'] = n_sa_effective
        result['zone_pattern'] = pattern
        result['grid_size'] = f"{grid_w}x{grid_h}"
        result['n_zones'] = n_zones
        result['sample_time'] = sample_time
        result['original_strategy'] = orig_strategy

        all_results.append(result)

        # Visualization
        if visualize:
            os.makedirs(vis_dir, exist_ok=True)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            h_init = HamiltonianSTL(grid_w, grid_h, init_pattern='zigzag')
            plot_cycle_on_ax(ax1, h_init, zones_np,
                             f"Zigzag Init (crossings={result['initial_crossings']})")
            plot_cycle_on_ax(ax2, result['final_h'], zones_np,
                             f"Model+SA (crossings={result['final_crossings']}, "
                             f"ops={result['num_operations']})")
            fig.suptitle(
                f"[COMPARISON] Sample {len(all_results)} | {pattern} | "
                f"{result['grid_size']} | {n_zones} zones | "
                f"Normally: {orig_strategy}",
                fontsize=14, fontweight='bold')
            fig.tight_layout()
            fig.savefig(os.path.join(vis_dir, f"{pattern}_{len(all_results)}.png"),
                        dpi=200, bbox_inches='tight')
            plt.close(fig)

        # Per-sample print
        in_target = "Y" if result.get('in_target_range') else "N"
        elapsed = _time.time() - t0
        eta = elapsed / (sample_idx + 1) * (total_samples - sample_idx - 1)
        n_ops = result['num_operations']
        init_c = result['initial_crossings']
        final_c = result['final_crossings']
        red = result.get('reduction', 0)
        tgt_lo = result.get('target_lower', '?')
        tgt_hi = result.get('target_upper', '?')
        esc = "+esc" if result.get('sa_escape_used') else ""

        normal_tag = f" [normally: {orig_strategy}]" if orig_strategy != 'model' else ""
        print(
            f"  [{sample_idx+1}/{total_samples}] {pattern} {grid_w}x{grid_h}{normal_tag} | "
            f"{init_c}->{final_c} (SA:{sa_final}) target:[{tgt_lo},{tgt_hi}] | "
            f"model{esc}: -{red} in {n_ops}ops | "
            f"CV={result.get('final_cv', 0):.2f} | {in_target} | {sample_time:.1f}s",
            flush=True,
        )

    _display_comparison_results(all_results)
    return all_results


# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

def _display_comparison_results(results):
    if not results:
        console.print("[bold red]No results to display[/bold red]")
        return

    pattern_results = defaultdict(list)
    for r in results:
        pattern_results[r['zone_pattern']].append(r)

    # ---- Table 1: Per-Pattern Summary ----
    per_pattern = Table(
        title="[bold]Per-Pattern Results: Model+SA on ALL Patterns (No Constructive)[/bold]",
        box=box.ROUNDED
    )
    per_pattern.add_column("Pattern", style="cyan")
    per_pattern.add_column("Normal", style="dim")
    per_pattern.add_column("N", justify="right", style="dim")
    per_pattern.add_column("Grid", justify="center")
    per_pattern.add_column("Model Red", justify="right", style="green")
    per_pattern.add_column("SA Red", justify="right", style="yellow")
    per_pattern.add_column("Avg Ops", justify="right")
    per_pattern.add_column("Avg SA Ops", justify="right")
    per_pattern.add_column("Avg CV", justify="right")
    per_pattern.add_column("In Target", justify="right")
    per_pattern.add_column("Avg Time", justify="right")

    for pattern in ALL_PATTERNS:
        pr = pattern_results.get(pattern, [])
        if not pr:
            per_pattern.add_row(
                pattern, "-", "0", "-", "-", "-", "-", "-", "-", "-", "-"
            )
            continue

        n = len(pr)
        grid_sizes = set(r['grid_size'] for r in pr)
        grid_str = ', '.join(sorted(grid_sizes))
        orig_strat = pr[0].get('original_strategy', '?')

        fusion_reds = [r['reduction'] for r in pr]
        fusion_pcts = [r['reduction_pct'] for r in pr]
        sa_reds = [r['sa_reduction'] for r in pr]
        fusion_ops = [r['num_operations'] for r in pr]
        sa_eff_ops = [r.get('sa_effective_ops', 0) for r in pr]
        final_cvs = [r.get('final_cv', 0) for r in pr]
        in_target = sum(1 for r in pr if r.get('in_target_range', False))
        times = [r.get('sample_time', 0) for r in pr]

        cv_style = "green" if np.mean(final_cvs) < CV_THRESHOLDS.get(pattern, 0.5) else "yellow"

        per_pattern.add_row(
            pattern,
            orig_strat,
            str(n),
            grid_str,
            f"{np.mean(fusion_reds):.1f} ({np.mean(fusion_pcts):.1f}%)",
            f"{np.mean(sa_reds):.1f}",
            f"{np.mean(fusion_ops):.1f}",
            f"{np.mean(sa_eff_ops):.0f}",
            f"[{cv_style}]{np.mean(final_cvs):.2f}[/{cv_style}]",
            f"{in_target}/{n} ({in_target/n*100:.0f}%)",
            f"{np.mean(times):.1f}s",
        )

    console.print(per_pattern)

    # ---- Table 2: Per-Sample Detail ----
    for pattern in ALL_PATTERNS:
        pr = pattern_results.get(pattern, [])
        if not pr:
            continue

        orig_strat = pr[0].get('original_strategy', '?')
        n_show = min(15, len(pr))
        detail = Table(
            title=(
                f"[bold]{pattern}[/bold] — Per-Sample (first {n_show}) | "
                f"Normally: {orig_strat} | Now: model+SA"
            ),
            box=box.SIMPLE
        )
        detail.add_column("#", style="dim")
        detail.add_column("Grid")
        detail.add_column("Init", justify="right")
        detail.add_column("Final", justify="right")
        detail.add_column("Red%", justify="right")
        detail.add_column("SA Ops", justify="right")
        detail.add_column("CV", justify="right")
        detail.add_column("Target", justify="center")
        detail.add_column("Phase", justify="center")
        detail.add_column("Time", justify="right")

        for i, r in enumerate(pr[:n_show]):
            in_tgt = r.get('in_target_range', False)
            cv_val = r.get('final_cv', 0)
            cv_thresh = r.get('cv_threshold', 0.5)
            t = r.get('sample_time', 0)
            sa_eff = r.get('sa_effective_ops', 0)

            if in_tgt and cv_val < cv_thresh:
                style = "bold green"
            elif r['reduction'] > 0:
                style = "yellow"
            else:
                style = "red"

            detail.add_row(
                str(i + 1), r.get('grid_size', '?'),
                str(r['initial_crossings']), str(r['final_crossings']),
                f"{r['reduction_pct']:.1f}%",
                str(sa_eff),
                f"{cv_val:.2f}",
                "[green]Y[/green]" if in_tgt else "[red]N[/red]",
                r.get('phase_at_stop', '?'),
                f"{t:.1f}s",
                style=style,
            )

        console.print(detail)

    # ---- Table 3: Average Time per Grid Size per Pattern ----
    # Collect all grid sizes across all patterns
    all_grid_sizes = set()
    time_by_pattern_size = defaultdict(lambda: defaultdict(list))
    for r in results:
        gs = r['grid_size']
        all_grid_sizes.add(gs)
        time_by_pattern_size[r['zone_pattern']][gs].append(r.get('sample_time', 0))

    sorted_sizes = sorted(all_grid_sizes, key=lambda s: (
        int(s.split('x')[0]) * int(s.split('x')[1])
    ))

    time_table = Table(
        title="[bold]Average Time per Grid Size per Pattern[/bold]",
        box=box.ROUNDED
    )
    time_table.add_column("Pattern", style="cyan")
    for gs in sorted_sizes:
        time_table.add_column(gs, justify="right")

    for pattern in ALL_PATTERNS:
        row = [pattern]
        for gs in sorted_sizes:
            times = time_by_pattern_size[pattern].get(gs, [])
            if times:
                row.append(f"{np.mean(times):.1f}s")
            else:
                row.append("-")
        time_table.add_row(*row)

    console.print(time_table)

    # ---- Overall Summary ----
    reductions = [r['reduction'] for r in results]
    reduction_pcts = [r['reduction_pct'] for r in results]
    ops = [r['num_operations'] for r in results]
    sa_reductions = [r['sa_reduction'] for r in results]
    all_cvs = [r.get('final_cv', 0) for r in results]
    in_target_count = sum(1 for r in results if r.get('in_target_range', False))
    times = [r.get('sample_time', 0) for r in results]

    # Split by original strategy for comparison
    constructive_results = [r for r in results if r.get('original_strategy') == 'constructive']
    model_results = [r for r in results if r.get('original_strategy') == 'model']

    console.print(Panel.fit(
        f"[bold]Comparison Summary — Model+SA on ALL Patterns[/bold]\n"
        f"Total samples: {len(results)}\n"
        f"In target range: {in_target_count}/{len(results)} "
        f"({in_target_count/len(results)*100:.0f}%)\n"
        f"Avg reduction: {np.mean(reduction_pcts):.1f}%\n"
        f"Avg operations: {np.mean(ops):.1f}\n"
        f"Avg final CV: {np.mean(all_cvs):.3f}\n"
        f"Avg time: {np.mean(times):.1f}s | Total: {sum(times):.0f}s\n"
        f"\n"
        f"[bold yellow]Patterns that normally use constructive "
        f"(left_right, stripes):[/bold yellow]\n"
        f"  Samples: {len(constructive_results)}\n"
        f"  In target: {sum(1 for r in constructive_results if r.get('in_target_range', False))}"
        f"/{len(constructive_results)} "
        f"({sum(1 for r in constructive_results if r.get('in_target_range', False))/max(len(constructive_results),1)*100:.0f}%)\n"
        f"  Avg reduction: {np.mean([r['reduction_pct'] for r in constructive_results]):.1f}%\n"
        f"  Avg time: {np.mean([r.get('sample_time',0) for r in constructive_results]):.1f}s\n"
        f"  [dim](Constructive achieves 100% target hit in <3s)[/dim]\n"
        f"\n"
        f"[bold green]Patterns that normally use model "
        f"(voronoi, islands):[/bold green]\n"
        f"  Samples: {len(model_results)}\n"
        f"  In target: {sum(1 for r in model_results if r.get('in_target_range', False))}"
        f"/{len(model_results)} "
        f"({sum(1 for r in model_results if r.get('in_target_range', False))/max(len(model_results),1)*100:.0f}%)\n"
        f"  Avg reduction: {np.mean([r['reduction_pct'] for r in model_results]):.1f}%\n"
        f"  Avg time: {np.mean([r.get('sample_time',0) for r in model_results]):.1f}s"
        if model_results else "",
        border_style="cyan"
    ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Comparison: Model+SA on ALL patterns (no constructive)'
    )
    parser.add_argument('--checkpoint',
                        default='FusionModel/nn_checkpoints/fusion/best.pt')
    parser.add_argument('--jsonl', default='datasets/final_dataset.jsonl')
    parser.add_argument('--n_per_pattern', type=int, default=25)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--n_candidates', type=int, default=10)
    parser.add_argument('--n_random', type=int, default=10)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--output_dir',
                        default='FusionModel/nn_checkpoints/fusion/comparison')

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    console.print(f"\n[bold]Loading model from {args.checkpoint}...[/bold]")
    model, model_args = load_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    max_history = model_args.get('max_history', MAX_HISTORY)
    console.print(f"  Parameters: {n_params:,}")
    console.print(f"  Device: {device}")
    console.print(
        f"  [bold yellow]ALL patterns forced to model+SA "
        f"(no constructive)[/bold yellow]"
    )

    results = evaluate_comparison(
        model=model,
        jsonl_path=args.jsonl,
        n_per_pattern=args.n_per_pattern,
        max_steps=args.max_steps,
        max_history=max_history,
        n_candidates=args.n_candidates,
        n_random=args.n_random,
        device=device,
        visualize=args.visualize,
        output_dir=args.output_dir,
    )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'comparison_results.json')
    non_serializable = {'sequence', 'crossings_history', 'final_h',
                        'boundary_details'}
    serializable = []
    for r in results:
        sr = {k: v for k, v in r.items() if k not in non_serializable}
        sr['crossings_history'] = r.get('crossings_history', [])
        sr['sequence'] = [
            {'kind': op['kind'], 'x': op['x'], 'y': op['y'],
             'variant': op['variant']}
            for op in r.get('sequence', [])
        ]
        serializable.append(sr)

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    console.print(f"\n[bold]Results saved to {output_path}[/bold]")
    if args.visualize:
        console.print(
            f"[bold]Visualizations saved to "
            f"{os.path.join(args.output_dir, 'vis')}/[/bold]"
        )


if __name__ == '__main__':
    main()
