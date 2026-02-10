#!/usr/bin/env python3
"""
run_pipeline_gpu.py - Multi-GPU SA Dataset Generation Pipeline

Distributes SA tasks from a machine YAML config across multiple NVIDIA GPUs.
Each GPU runs a worker process that executes SA tasks sequentially using
GPU-accelerated crossings computation and batched pattern matching.

Usage:
    python run_pipeline_gpu.py kazi --gpus 4
    python run_pipeline_gpu.py kazi --gpus 4 --workers-per-gpu 8
    python run_pipeline_gpu.py kazi --gpus 4 --retry-failed

Designed for SLURM environments (CCRI TamIA with 4x H100 SXM 80GB).
"""

import argparse
import os
import sys
import time
import json
import traceback
import threading
from datetime import timedelta
from multiprocessing import Process, Queue, cpu_count
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

# Add script directory to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import torch
import torch.multiprocessing as mp


# ============================================================
# Config loading (reuse pipeline config module)
# ============================================================
from pipeline.config import (
    GlobalConfig, MachineConfig, Task, SAConfig, ZoneParams,
    load_config, parse_zone_params,
)
from pipeline.task_generator import generate_tasks, filter_pending_tasks
from pipeline.checkpoint import Checkpoint, TaskResult


# ============================================================
# GPU Worker
# ============================================================
def gpu_worker(
    gpu_id: int,
    device_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    workers_per_gpu: int,
):
    """
    Worker process for a single GPU. Pulls tasks from queue, runs SA with GPU acceleration.

    Args:
        gpu_id: Logical GPU index (0-3)
        device_id: CUDA device index
        task_queue: Queue of task dicts to process
        result_queue: Queue for completed results
        workers_per_gpu: Number of CPU sub-workers per GPU for parallel SA runs
    """
    import torch
    device = torch.device(f"cuda:{device_id}")

    # Verify GPU is accessible
    try:
        torch.cuda.set_device(device)
        mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        name = torch.cuda.get_device_name(device)
        print(f"[GPU {gpu_id}] Using {name} ({mem:.0f} GB) - cuda:{device_id}", flush=True)
    except Exception as e:
        print(f"[GPU {gpu_id}] ERROR: Cannot access cuda:{device_id}: {e}", flush=True)
        return

    from SA_generation_gpu import run_sa

    completed = 0
    while True:
        try:
            task_dict = task_queue.get(timeout=2)
        except Exception:
            # Queue empty or timeout - check if there's more work
            if task_queue.empty():
                break
            continue

        if task_dict is None:  # Poison pill
            break

        task_id = task_dict["task_id"]
        sa_cfg = task_dict["sa_config"]
        zp = task_dict["zone_params"]
        start = time.time()

        try:
            initial_crossings, final_crossings, best_ops = run_sa(
                width=task_dict["width"],
                height=task_dict["height"],
                iterations=sa_cfg["iterations"],
                Tmax=sa_cfg["Tmax"],
                Tmin=sa_cfg["Tmin"],
                seed=task_dict["seed"],
                plot_live=False,
                show_every_accepted=0,
                pause_seconds=0.0,
                dataset_dir=task_dict["output_dir"],
                write_dataset=True,
                max_move_tries=sa_cfg.get("max_move_tries", 200),
                pool_refresh_period=sa_cfg.get("pool_refresh_period", 250),
                pool_max_moves=sa_cfg.get("pool_max_moves", 5000),
                reheat_patience=sa_cfg.get("reheat_patience", 1500),
                reheat_factor=sa_cfg.get("reheat_factor", 1.5),
                reheat_cap=sa_cfg.get("reheat_cap", 600.0),
                transpose_phase_ratio=sa_cfg.get("transpose_phase_ratio", 0.6),
                border_to_inner=sa_cfg.get("border_to_inner", True),
                zone_mode=task_dict["zone_mode"],
                num_islands=zp.get("num_islands", 3),
                island_size=zp.get("island_size", 8),
                allow_touch=zp.get("allow_touch", False),
                stripe_direction=zp.get("stripe_direction", "v"),
                stripe_k=zp.get("stripe_k", 3),
                voronoi_k=zp.get("voronoi_k", 3),
                debug=False,
                device=device,
            )

            runtime = time.time() - start
            completed += 1

            result_queue.put({
                "task_id": task_id,
                "success": True,
                "initial_crossings": initial_crossings,
                "final_crossings": final_crossings,
                "runtime_sec": runtime,
                "error": None,
                "timestamp": time.time(),
                "gpu_id": gpu_id,
            })

            if completed % 10 == 0:
                print(f"[GPU {gpu_id}] Completed {completed} tasks "
                      f"(last: {initial_crossings}->{final_crossings} in {runtime:.1f}s)",
                      flush=True)

        except Exception as e:
            runtime = time.time() - start
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            result_queue.put({
                "task_id": task_id,
                "success": False,
                "initial_crossings": None,
                "final_crossings": None,
                "runtime_sec": runtime,
                "error": error_msg,
                "timestamp": time.time(),
                "gpu_id": gpu_id,
            })

    print(f"[GPU {gpu_id}] Worker finished. Completed {completed} tasks total.", flush=True)


# ============================================================
# Result collector thread
# ============================================================
def result_collector(
    result_queue: mp.Queue,
    checkpoint: Checkpoint,
    total_tasks: int,
    stop_event: threading.Event,
):
    """Background thread that collects results and updates checkpoint."""
    completed = 0
    failed = 0
    start_time = time.time()

    while not stop_event.is_set():
        try:
            result = result_queue.get(timeout=1)
        except Exception:
            continue

        task_result = TaskResult(
            task_id=result["task_id"],
            success=result["success"],
            initial_crossings=result.get("initial_crossings"),
            final_crossings=result.get("final_crossings"),
            runtime_sec=result.get("runtime_sec"),
            error=result.get("error"),
            timestamp=result.get("timestamp", time.time()),
        )

        if task_result.success:
            checkpoint.mark_completed(task_result)
            completed += 1
        else:
            checkpoint.mark_failed(task_result)
            failed += 1

        processed = completed + failed
        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0
        remaining = total_tasks - processed
        eta = remaining / rate if rate > 0 else 0

        if processed % 25 == 0 or processed == total_tasks:
            print(
                f"\r[Progress] {processed}/{total_tasks} "
                f"({completed} OK, {failed} fail) "
                f"| {rate:.2f} tasks/s "
                f"| ETA: {timedelta(seconds=int(eta))} "
                f"| Elapsed: {timedelta(seconds=int(elapsed))}",
                flush=True,
            )

        if processed >= total_tasks:
            break

    # Drain any remaining results
    while not result_queue.empty():
        try:
            result = result_queue.get(timeout=0.5)
            task_result = TaskResult(
                task_id=result["task_id"],
                success=result["success"],
                initial_crossings=result.get("initial_crossings"),
                final_crossings=result.get("final_crossings"),
                runtime_sec=result.get("runtime_sec"),
                error=result.get("error"),
                timestamp=result.get("timestamp", time.time()),
            )
            if task_result.success:
                checkpoint.mark_completed(task_result)
            else:
                checkpoint.mark_failed(task_result)
        except Exception:
            break


# ============================================================
# Main pipeline
# ============================================================
def run_gpu_pipeline(
    config_dir: str,
    machine_id: str,
    num_gpus: int = 4,
    workers_per_gpu: int = 1,
    retry_failed: bool = False,
):
    """
    Run the SA dataset generation pipeline across multiple GPUs.

    Args:
        config_dir: Path to config directory with YAML files
        machine_id: Machine identifier (e.g., 'kazi')
        num_gpus: Number of GPUs to use
        workers_per_gpu: CPU workers per GPU (for future use)
        retry_failed: Whether to retry previously failed tasks
    """
    print("=" * 70)
    print(f"  SA Dataset Generation Pipeline - GPU Accelerated")
    print(f"  Machine: {machine_id}")
    print(f"  GPUs: {num_gpus}")
    print("=" * 70)

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Please check your PyTorch installation.")
        sys.exit(1)

    available_gpus = torch.cuda.device_count()
    print(f"Available CUDA devices: {available_gpus}")
    for i in range(available_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i} - {props.name} ({props.total_memory / 1024**3:.0f} GB)")

    if num_gpus > available_gpus:
        print(f"WARNING: Requested {num_gpus} GPUs but only {available_gpus} available. "
              f"Using {available_gpus}.")
        num_gpus = available_gpus

    # Load config
    global_cfg, machine_cfg = load_config(config_dir, machine_id)
    all_tasks = generate_tasks(global_cfg, machine_cfg)

    # Setup checkpoint
    os.makedirs(machine_cfg.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(machine_cfg.output_dir, f"checkpoint_{machine_id}.json")
    checkpoint = Checkpoint(checkpoint_path)

    if retry_failed:
        cleared = checkpoint.clear_failed()
        print(f"Cleared {cleared} failed tasks for retry.")

    # Filter pending tasks
    completed_ids = checkpoint.get_completed_ids()
    pending_tasks = filter_pending_tasks(all_tasks, completed_ids)

    total_all = len(all_tasks)
    total_pending = len(pending_tasks)
    already_done = total_all - total_pending

    print(f"\nTasks: {total_all} total, {already_done} already done, {total_pending} pending")

    if total_pending == 0:
        print("All tasks already completed!")
        return

    # Convert tasks to dicts
    task_dicts = [t.to_dict() for t in pending_tasks]

    # Create queues
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # Fill task queue, distributing evenly
    for td in task_dicts:
        task_queue.put(td)

    # Add poison pills (one per GPU worker)
    for _ in range(num_gpus):
        task_queue.put(None)

    # Start result collector thread
    stop_event = threading.Event()
    collector = threading.Thread(
        target=result_collector,
        args=(result_queue, checkpoint, total_pending, stop_event),
        daemon=True,
    )
    collector.start()

    # Start GPU worker processes
    print(f"\nLaunching {num_gpus} GPU workers...")
    start_time = time.time()

    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(gpu_id, gpu_id, task_queue, result_queue, workers_per_gpu),
        )
        p.start()
        processes.append(p)

    # Wait for all workers to finish
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nInterrupted! Waiting for workers to finish current tasks...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join(timeout=10)

    # Stop collector
    stop_event.set()
    collector.join(timeout=5)

    # Final stats
    elapsed = time.time() - start_time
    stats = checkpoint.get_stats()

    print("\n" + "=" * 70)
    print(f"  Pipeline Complete!")
    print(f"  Elapsed: {timedelta(seconds=int(elapsed))}")
    print(f"  Completed: {stats['completed_count']}/{total_all}")
    print(f"  Failed: {stats['failed_count']}")
    if stats['completed_count'] > 0:
        rate = (stats['completed_count'] - already_done) / elapsed if elapsed > 0 else 0
        print(f"  Rate: {rate:.2f} tasks/sec")
    print("=" * 70)


# ============================================================
# CLI
# ============================================================
def main():
    # Use 'spawn' for CUDA multiprocessing
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(
        description="GPU-accelerated SA dataset generation pipeline",
    )
    parser.add_argument("machine_id", type=str, help="Machine ID (e.g., kazi)")
    parser.add_argument("--gpus", type=int, default=4, help="Number of GPUs (default: 4)")
    parser.add_argument("--workers-per-gpu", type=int, default=1,
                        help="CPU workers per GPU (default: 1)")
    parser.add_argument("--config-dir", type=str, default="config",
                        help="Config directory (default: config)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry failed tasks")
    args = parser.parse_args()

    config_dir = os.path.join(_script_dir, args.config_dir)
    if not os.path.isdir(config_dir):
        print(f"Error: Config directory not found: {config_dir}")
        sys.exit(1)

    run_gpu_pipeline(
        config_dir=config_dir,
        machine_id=args.machine_id,
        num_gpus=args.gpus,
        workers_per_gpu=args.workers_per_gpu,
        retry_failed=args.retry_failed,
    )


if __name__ == "__main__":
    main()
