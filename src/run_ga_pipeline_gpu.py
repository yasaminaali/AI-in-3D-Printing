#!/usr/bin/env python3
"""
run_ga_pipeline_gpu.py - Multi-GPU GA Dataset Generation Pipeline

Distributes GA tasks from a machine YAML config across multiple NVIDIA GPUs.
Each GPU runs a worker process that executes GA tasks sequentially using
GPU-accelerated crossings computation and Numba-compiled path validation.

Usage:
    python run_ga_pipeline_gpu.py kazi_ga --gpus 4
    python run_ga_pipeline_gpu.py kazi_ga --gpus 4 --retry-failed
    python run_ga_pipeline_gpu.py kazi_ga --gpus 4 --config-dir config

Designed for SLURM environments (CCRI TamIA with 4x H100 SXM 80GB).
"""

import argparse
import json
import os
import sys
import time
import traceback
import threading
from datetime import timedelta

# Add script directory to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import torch
import torch.multiprocessing as mp

# ============================================================
# Config loading (GA pipeline config module)
# ============================================================
from pipeline.ga_config import (
    GAGlobalConfig, GAMachineConfig, GATask, GAConfig,
    load_ga_config,
)
from pipeline.ga_task_generator import generate_ga_tasks, filter_pending_ga_tasks
from pipeline.checkpoint import Checkpoint, TaskResult


# ============================================================
# GPU Worker
# ============================================================
def ga_gpu_worker(
    gpu_id: int,
    device_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
):
    """
    Worker process for a single GPU. Pulls GA tasks from queue,
    runs GA with GPU-accelerated crossings.

    Args:
        gpu_id: Logical GPU index (0-3)
        device_id: CUDA device index
        task_queue: Queue of task dicts to process
        result_queue: Queue for completed results
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

    from ga_sequence_gpu import run_ga_sequences_dataset_init

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
        ga_cfg = task_dict["ga_config"]
        zp = task_dict["zone_params"]
        start = time.time()

        print(f"[GPU {gpu_id}] Starting task: {task_id}", flush=True)

        try:
            best_ind = run_ga_sequences_dataset_init(
                dataset_jsonl=task_dict["dataset_jsonl"],
                W=task_dict["width"],
                H=task_dict["height"],
                zone_pattern=task_dict["zone_mode"],
                pop_size=ga_cfg["pop_size"],
                generations=ga_cfg["generations"],
                tourn_k=ga_cfg["tourn_k"],
                genome_len=ga_cfg["genome_len"],
                # zone params
                num_islands=zp.get("num_islands", 3),
                island_size=zp.get("island_size", 8),
                allow_touch=zp.get("allow_touch", False),
                stripe_direction=zp.get("stripe_direction", "v"),
                stripe_k=zp.get("stripe_k", 3),
                voronoi_k=zp.get("voronoi_k", 3),
                # dataset selection
                dataset_choose="best",
                dataset_sample_seed=0,
                # output
                ga_out_dir=task_dict["output_dir"],
                # GPU
                device=device,
                # GA hyperparameters
                elite_k=ga_cfg.get("elite_k", 6),
                keep_rate=ga_cfg.get("keep_rate", 0.60),
                cx_rate=ga_cfg.get("cx_rate", 0.90),
                cx_ratio=ga_cfg.get("cx_ratio", 0.60),
                eps_crossings=ga_cfg.get("eps_crossings", 2),
                min_applied_valid=ga_cfg.get("min_applied_valid", 1),
                max_tries_per_slot=ga_cfg.get("max_tries_per_slot", 80),
            )

            runtime = time.time() - start
            completed += 1

            final_crossings = int(best_ind.best_seen) if best_ind else None

            result_queue.put({
                "task_id": task_id,
                "success": True,
                "initial_crossings": None,
                "final_crossings": final_crossings,
                "runtime_sec": runtime,
                "error": None,
                "timestamp": time.time(),
                "gpu_id": gpu_id,
            })

            print(
                f"[GPU {gpu_id}] Completed task {task_id}: "
                f"best_crossings={final_crossings} in {runtime:.1f}s "
                f"({completed} tasks done)",
                flush=True,
            )

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
            print(f"[GPU {gpu_id}] FAILED task {task_id}: {e}", flush=True)

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
# Dataset pre-filtering (load once, write small per-task files)
# ============================================================
def prefilter_dataset(task_dicts, output_dir):
    """
    Load the full SA dataset JSONL once in the main process, then write
    small filtered files per (grid, pattern) combination. Updates each
    task_dict's dataset_jsonl to point to the filtered file.

    This avoids 4 workers each independently parsing ~934MB of JSON.
    """
    if not task_dicts:
        return task_dicts

    # All tasks share the same source dataset
    source_jsonl = task_dicts[0]["dataset_jsonl"]
    if not os.path.exists(source_jsonl):
        print(f"WARNING: Dataset file not found: {source_jsonl}")
        return task_dicts

    print(f"\nPre-filtering dataset: {source_jsonl}")
    t0 = time.time()

    # Load all records once
    records = []
    with open(source_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                records.append(json.loads(s))

    print(f"  Loaded {len(records)} records in {time.time() - t0:.1f}s")

    # Group by (grid_W, grid_H, zone_pattern)
    filtered_dir = os.path.join(output_dir, "filtered_datasets")
    os.makedirs(filtered_dir, exist_ok=True)

    written_paths = {}  # (W, H, pattern) -> filtered file path

    for td in task_dicts:
        key = (td["width"], td["height"], td["zone_mode"].lower())
        if key in written_paths:
            td["dataset_jsonl"] = written_paths[key]
            continue

        # Filter matching records
        matched = [
            r for r in records
            if (int(r.get("grid_W", 0)) == key[0]
                and int(r.get("grid_H", 0)) == key[1]
                and str(r.get("zone_pattern", "")).lower() == key[2])
        ]

        path = os.path.join(filtered_dir, f"{key[0]}x{key[1]}_{key[2]}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in matched:
                f.write(json.dumps(r) + "\n")

        written_paths[key] = path
        td["dataset_jsonl"] = path
        print(f"  {key[0]}x{key[1]} {key[2]}: {len(matched)} records -> {path}")

    print(f"  Pre-filtering done in {time.time() - t0:.1f}s\n")
    return task_dicts


# ============================================================
# Main pipeline
# ============================================================
def run_ga_gpu_pipeline(
    config_dir: str,
    machine_id: str,
    num_gpus: int = 4,
    retry_failed: bool = False,
):
    """
    Run the GA dataset generation pipeline across multiple GPUs.

    Args:
        config_dir: Path to config directory with YAML files
        machine_id: Machine identifier (e.g., 'kazi_ga')
        num_gpus: Number of GPUs to use
        retry_failed: Whether to retry previously failed tasks
    """
    print("=" * 70)
    print(f"  GA Dataset Generation Pipeline - GPU Accelerated")
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
    global_cfg, machine_cfg = load_ga_config(config_dir, machine_id)
    all_tasks = generate_ga_tasks(global_cfg, machine_cfg)

    print(f"\nGA Tasks generated: {len(all_tasks)}")
    for t in all_tasks:
        print(f"  {t.task_id}: {t.width}x{t.height} {t.zone_mode} genome_len={t.ga_config.genome_len}")

    # Setup checkpoint
    os.makedirs(machine_cfg.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(machine_cfg.output_dir, f"checkpoint_{machine_id}.json")
    checkpoint = Checkpoint(checkpoint_path)

    if retry_failed:
        cleared = checkpoint.clear_failed()
        print(f"Cleared {cleared} failed tasks for retry.")

    # Filter pending tasks
    completed_ids = checkpoint.get_completed_ids()
    pending_tasks = filter_pending_ga_tasks(all_tasks, completed_ids)

    total_all = len(all_tasks)
    total_pending = len(pending_tasks)
    already_done = total_all - total_pending

    print(f"\nTasks: {total_all} total, {already_done} already done, {total_pending} pending")

    if total_pending == 0:
        print("All tasks already completed!")
        return

    # Convert tasks to dicts
    task_dicts = [t.to_dict() for t in pending_tasks]

    # Pre-filter dataset: load the large JSONL once in the main process,
    # write small per-(grid,pattern) files so workers don't each parse 934MB.
    task_dicts = prefilter_dataset(task_dicts, machine_cfg.output_dir)

    # Create queues
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # Fill task queue
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
            target=ga_gpu_worker,
            args=(gpu_id, gpu_id, task_queue, result_queue),
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
    print(f"  GA Pipeline Complete!")
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
        description="GPU-accelerated GA dataset generation pipeline",
    )
    parser.add_argument("machine_id", type=str, help="Machine ID (e.g., kazi_ga)")
    parser.add_argument("--gpus", type=int, default=4, help="Number of GPUs (default: 4)")
    parser.add_argument("--config-dir", type=str, default="config",
                        help="Config directory (default: config)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry failed tasks")
    args = parser.parse_args()

    config_dir = os.path.join(_script_dir, args.config_dir)
    if not os.path.isdir(config_dir):
        print(f"Error: Config directory not found: {config_dir}")
        sys.exit(1)

    run_ga_gpu_pipeline(
        config_dir=config_dir,
        machine_id=args.machine_id,
        num_gpus=args.gpus,
        retry_failed=args.retry_failed,
    )


if __name__ == "__main__":
    main()
