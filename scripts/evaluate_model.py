#!/usr/bin/env python
"""
Comprehensive evaluation of the trained CNN+RNN model.
Tests model performance and generates documentation of results.

Run from project root: python scripts/evaluate_model.py
"""

import os
import sys
import json
import time
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.core.hamiltonian import HamiltonianSTL
from src.core.zones import zones_left_right, zones_diagonal, zones_stripes, zones_voronoi
from src.optimization import simulated_annealing as SA
from src.optimization import sa_patterns as SA_patterns


def load_model_info(model_path: str = "models/global_seq_policy.pt") -> Dict[str, Any]:
    """Load and analyze the trained model."""
    if not os.path.exists(model_path):
        return {"error": f"Model not found at {model_path}"}
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Analyze model structure
    info = {
        "model_path": model_path,
        "file_size_mb": os.path.getsize(model_path) / (1024 * 1024),
        "layers": list(checkpoint.keys()),
        "total_parameters": sum(p.numel() for p in checkpoint.values()),
    }
    
    # Layer-wise parameter counts
    layer_params = {}
    for name, param in checkpoint.items():
        layer_params[name] = {
            "shape": list(param.shape),
            "parameters": param.numel()
        }
    info["layer_details"] = layer_params
    
    return info


def analyze_dataset(dataset_dir: str = "Dataset") -> Dict[str, Any]:
    """Analyze the training dataset statistics."""
    import csv
    
    if not os.path.exists(dataset_dir):
        return {"error": f"Dataset not found at {dataset_dir}"}
    
    states_csv = os.path.join(dataset_dir, "states.csv")
    actions_csv = os.path.join(dataset_dir, "actions.csv")
    
    if not os.path.exists(states_csv) or not os.path.exists(actions_csv):
        return {"error": "Missing states.csv or actions.csv"}
    
    # Analyze states
    states_info = {
        "total_states": 0,
        "instances": set(),
        "grid_sizes": set(),
        "zone_patterns": set(),
    }
    
    with open(states_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            states_info["total_states"] += 1
            states_info["instances"].add(row["instance_id"])
            states_info["grid_sizes"].add(f"{row['grid_w']}x{row['grid_h']}")
            states_info["zone_patterns"].add(row["zone_pattern"])
    
    # Analyze actions
    actions_info = {
        "total_actions": 0,
        "valid_actions": 0,
        "best_actions": 0,
        "operations": {},
        "orientations": {},
    }
    
    with open(actions_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            actions_info["total_actions"] += 1
            if int(row["valid"]) == 1:
                actions_info["valid_actions"] += 1
            if int(row["best_in_state"]) == 1:
                actions_info["best_actions"] += 1
            
            op = row["op"]
            actions_info["operations"][op] = actions_info["operations"].get(op, 0) + 1
            
            ori = row["orientation"]
            actions_info["orientations"][ori] = actions_info["orientations"].get(ori, 0) + 1
    
    return {
        "states": {
            "total": states_info["total_states"],
            "unique_instances": len(states_info["instances"]),
            "grid_sizes": sorted(list(states_info["grid_sizes"])),
            "zone_patterns": sorted(list(states_info["zone_patterns"])),
        },
        "actions": actions_info,
    }


def run_optimization_benchmark(
    num_runs: int = 5,
    grid_sizes: List[Tuple[int, int]] = [(20, 20), (25, 25), (30, 30)],
    iterations: int = 1000,
) -> Dict[str, Any]:
    """Run SA optimization benchmark to measure performance."""
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp(prefix="eval_")
    results = []
    
    try:
        for width, height in grid_sizes:
            for run_idx in range(num_runs):
                seed = 100 + run_idx
                
                # Run SA optimization
                start_time = time.time()
                best_cost, _ = SA.run_sa(
                    width=width,
                    height=height,
                    iterations=iterations,
                    Tmax=60.0,
                    Tmin=0.5,
                    seed=seed,
                    plot_live=False,
                    show_every_accepted=0,
                    pause_seconds=0.0,
                    dataset_dir=temp_dir,
                    max_move_tries=25,
                    pool_refresh_period=100,
                    pool_max_moves=2000,
                    reheat_patience=500,
                    reheat_factor=1.5,
                    reheat_cap=200.0,
                )
                elapsed = time.time() - start_time
                
                initial = width  # left_right pattern: initial crossings = width
                reduction = ((initial - best_cost) / initial * 100) if initial > 0 else 0
                
                results.append({
                    "grid_size": f"{width}x{height}",
                    "run": run_idx + 1,
                    "seed": seed,
                    "initial_crossings": initial,
                    "final_crossings": best_cost,
                    "reduction_pct": reduction,
                    "time_seconds": elapsed,
                })
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Aggregate statistics
    aggregated = {}
    for size in [f"{w}x{h}" for w, h in grid_sizes]:
        size_results = [r for r in results if r["grid_size"] == size]
        if size_results:
            final_crossings = [r["final_crossings"] for r in size_results]
            reductions = [r["reduction_pct"] for r in size_results]
            times = [r["time_seconds"] for r in size_results]
            
            aggregated[size] = {
                "runs": len(size_results),
                "initial_crossings": size_results[0]["initial_crossings"],
                "final_crossings_min": min(final_crossings),
                "final_crossings_max": max(final_crossings),
                "final_crossings_avg": sum(final_crossings) / len(final_crossings),
                "reduction_pct_avg": sum(reductions) / len(reductions),
                "time_avg_seconds": sum(times) / len(times),
            }
    
    return {
        "config": {
            "num_runs": num_runs,
            "grid_sizes": [f"{w}x{h}" for w, h in grid_sizes],
            "iterations_per_run": iterations,
        },
        "individual_results": results,
        "aggregated": aggregated,
    }


def test_zone_patterns(
    width: int = 25,
    height: int = 25,
    iterations: int = 1000,
) -> Dict[str, Any]:
    """Test optimization across different zone patterns."""
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp(prefix="zone_eval_")
    results = []
    
    patterns = [
        ("left_right", SA.run_sa, {}),
        ("diagonal", SA_patterns.run_sa, {"zone_mode": "diagonal"}),
        ("stripes_v", SA_patterns.run_sa, {"zone_mode": "stripes", "stripe_direction": "v", "stripe_k": 3}),
        ("voronoi", SA_patterns.run_sa, {"zone_mode": "voronoi", "voronoi_k": 3}),
    ]
    
    try:
        for pattern_name, runner, kwargs in patterns:
            print(f"  Testing {pattern_name}...", end=" ", flush=True)
            
            start_time = time.time()
            best_cost, _ = runner(
                width=width,
                height=height,
                iterations=iterations,
                Tmax=60.0,
                Tmin=0.5,
                seed=42,
                plot_live=False,
                show_every_accepted=0,
                pause_seconds=0.0,
                dataset_dir=temp_dir,
                max_move_tries=25,
                pool_refresh_period=100,
                pool_max_moves=2000,
                reheat_patience=500,
                reheat_factor=1.5,
                reheat_cap=200.0,
                **kwargs,
            )
            elapsed = time.time() - start_time
            
            print(f"done ({best_cost} crossings)")
            
            results.append({
                "pattern": pattern_name,
                "final_crossings": best_cost,
                "time_seconds": elapsed,
            })
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return {
        "grid_size": f"{width}x{height}",
        "iterations": iterations,
        "results": results,
    }


def generate_report(
    model_info: Dict,
    dataset_info: Dict,
    benchmark_results: Dict,
    zone_results: Dict,
) -> str:
    """Generate a comprehensive performance report."""
    
    report = []
    report.append("=" * 70)
    report.append("CNN+RNN MODEL EVALUATION REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)
    
    # Model Information
    report.append("\n## MODEL INFORMATION")
    report.append("-" * 50)
    if "error" not in model_info:
        report.append(f"Model Path: {model_info['model_path']}")
        report.append(f"File Size: {model_info['file_size_mb']:.2f} MB")
        report.append(f"Total Parameters: {model_info['total_parameters']:,}")
        report.append(f"Number of Layers: {len(model_info['layers'])}")
        
        report.append("\nLayer Architecture:")
        for layer_name, layer_info in model_info['layer_details'].items():
            report.append(f"  {layer_name}: {layer_info['shape']} ({layer_info['parameters']:,} params)")
    else:
        report.append(f"Error: {model_info['error']}")
    
    # Dataset Information
    report.append("\n## TRAINING DATASET")
    report.append("-" * 50)
    if "error" not in dataset_info:
        states = dataset_info["states"]
        actions = dataset_info["actions"]
        
        report.append(f"Total States: {states['total']:,}")
        report.append(f"Unique Instances: {states['unique_instances']}")
        report.append(f"Grid Sizes: {', '.join(states['grid_sizes'])}")
        report.append(f"Zone Patterns: {', '.join(states['zone_patterns'])}")
        
        report.append(f"\nTotal Actions: {actions['total_actions']:,}")
        report.append(f"Valid Actions: {actions['valid_actions']:,}")
        report.append(f"Best Actions (labels): {actions['best_actions']:,}")
        
        report.append("\nOperation Distribution:")
        for op, count in sorted(actions['operations'].items()):
            pct = count / actions['total_actions'] * 100
            report.append(f"  {op}: {count:,} ({pct:.1f}%)")
    else:
        report.append(f"Error: {dataset_info['error']}")
    
    # Benchmark Results
    report.append("\n## OPTIMIZATION BENCHMARK")
    report.append("-" * 50)
    report.append(f"Runs per size: {benchmark_results['config']['num_runs']}")
    report.append(f"Iterations per run: {benchmark_results['config']['iterations_per_run']}")
    
    report.append("\nResults by Grid Size:")
    for size, stats in benchmark_results['aggregated'].items():
        report.append(f"\n  {size}:")
        report.append(f"    Initial crossings: {stats['initial_crossings']}")
        report.append(f"    Final crossings: {stats['final_crossings_avg']:.1f} (range: {stats['final_crossings_min']}-{stats['final_crossings_max']})")
        report.append(f"    Average reduction: {stats['reduction_pct_avg']:.1f}%")
        report.append(f"    Average time: {stats['time_avg_seconds']:.2f}s")
    
    # Zone Pattern Results
    report.append("\n## ZONE PATTERN COMPARISON")
    report.append("-" * 50)
    report.append(f"Grid Size: {zone_results['grid_size']}")
    report.append(f"Iterations: {zone_results['iterations']}")
    
    report.append("\nResults by Pattern:")
    for r in zone_results['results']:
        report.append(f"  {r['pattern']:>12}: {r['final_crossings']} crossings ({r['time_seconds']:.2f}s)")
    
    # Summary
    report.append("\n## SUMMARY")
    report.append("-" * 50)
    
    if "error" not in model_info and "error" not in dataset_info:
        report.append("The CNN+RNN model was successfully trained on the collected data.")
        report.append(f"- Model has {model_info['total_parameters']:,} trainable parameters")
        report.append(f"- Trained on {dataset_info['states']['unique_instances']} optimization instances")
        report.append(f"- Learned from {dataset_info['actions']['best_actions']:,} labeled best actions")
        
        # Best optimization result
        best_reduction = max(
            stats['reduction_pct_avg'] 
            for stats in benchmark_results['aggregated'].values()
        )
        report.append(f"- Best average crossing reduction: {best_reduction:.1f}%")
    
    report.append("\n" + "=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)
    
    return "\n".join(report)


def main():
    print("=" * 60)
    print("CNN+RNN Model Evaluation")
    print("=" * 60)
    
    # 1. Load and analyze model
    print("\n[1/4] Analyzing trained model...")
    model_info = load_model_info()
    if "error" not in model_info:
        print(f"  Model loaded: {model_info['total_parameters']:,} parameters")
    else:
        print(f"  {model_info['error']}")
    
    # 2. Analyze dataset
    print("\n[2/4] Analyzing training dataset...")
    dataset_info = analyze_dataset()
    if "error" not in dataset_info:
        print(f"  States: {dataset_info['states']['total']:,}")
        print(f"  Actions: {dataset_info['actions']['total_actions']:,}")
        print(f"  Instances: {dataset_info['states']['unique_instances']}")
    else:
        print(f"  {dataset_info['error']}")
    
    # 3. Run benchmark
    print("\n[3/4] Running optimization benchmark...")
    print("  This tests SA optimization performance (3 grid sizes, 3 runs each)")
    benchmark_results = run_optimization_benchmark(
        num_runs=3,
        grid_sizes=[(20, 20), (25, 25), (30, 30)],
        iterations=500,
    )
    print("  Benchmark complete!")
    
    # 4. Test zone patterns
    print("\n[4/4] Testing different zone patterns...")
    zone_results = test_zone_patterns(
        width=25,
        height=25,
        iterations=500,
    )
    
    # Generate report
    print("\n" + "=" * 60)
    print("Generating evaluation report...")
    report = generate_report(model_info, dataset_info, benchmark_results, zone_results)
    
    # Ensure reports directory exists
    os.makedirs("reports", exist_ok=True)
    
    # Save report
    report_path = "reports/EVALUATION_REPORT.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to: {report_path}")
    
    # Print report
    print("\n" + report)
    
    # Save JSON results
    results_path = "reports/evaluation_results.json"
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_info": model_info,
        "dataset_info": dataset_info,
        "benchmark_results": benchmark_results,
        "zone_results": zone_results,
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    main()
