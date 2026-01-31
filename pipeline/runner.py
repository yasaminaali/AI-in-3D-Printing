"""
Main multiprocessing orchestrator for SA pipeline.
"""

import os
import signal
import sys
import time
from multiprocessing import Pool, cpu_count
from typing import Any, Callable, Dict, List, Optional

from .config import GlobalConfig, MachineConfig, Task, load_config
from .task_generator import generate_tasks, filter_pending_tasks, get_task_summary
from .checkpoint import Checkpoint, TaskResult
from .worker import execute_task_dict


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds == float("inf"):
        return "unknown"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


class ParallelRunner:
    """
    Multiprocessing orchestrator for running SA tasks in parallel.

    Features:
    - Configurable number of worker processes
    - Checkpoint-based resume capability
    - Graceful shutdown on SIGINT
    - Progress logging with ETA
    """

    def __init__(
        self,
        config_dir: str,
        machine_id: str,
        num_workers: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialize the parallel runner.

        Args:
            config_dir: Directory containing YAML config files
            machine_id: Machine identifier (e.g., "yasamin", "istiaq", "kazi")
            num_workers: Number of worker processes (default: from config or CPU count)
            checkpoint_dir: Directory for checkpoint files (default: output_dir)
        """
        self.config_dir = config_dir
        self.machine_id = machine_id

        # Load configurations
        self.global_cfg, self.machine_cfg = load_config(config_dir, machine_id)

        # Set number of workers
        if num_workers is not None:
            self.num_workers = num_workers
        else:
            self.num_workers = self.machine_cfg.num_workers

        # Ensure we don't use more workers than CPUs
        max_workers = cpu_count() or 4
        self.num_workers = min(self.num_workers, max_workers)

        # Set up checkpoint
        if checkpoint_dir is None:
            checkpoint_dir = self.machine_cfg.output_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{machine_id}.json")
        self.checkpoint = Checkpoint(checkpoint_path)

        # Generate all tasks
        self.all_tasks = generate_tasks(self.global_cfg, self.machine_cfg)

        # State
        self._pool: Optional[Pool] = None
        self._shutdown_requested = False
        self._results_callback: Optional[Callable] = None

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def handler(signum, frame):
            print("\n[Runner] Shutdown requested, finishing current tasks...")
            self._shutdown_requested = True
            if self._pool is not None:
                self._pool.terminate()

        # Only set up handlers on main process
        if sys.platform != "win32":
            signal.signal(signal.SIGINT, handler)
            signal.signal(signal.SIGTERM, handler)

    def _on_task_complete(self, result_dict: Dict[str, Any]) -> None:
        """Callback when a task completes."""
        result = TaskResult(
            task_id=result_dict["task_id"],
            success=result_dict["success"],
            final_crossings=result_dict.get("final_crossings"),
            runtime_sec=result_dict.get("runtime_sec"),
            error=result_dict.get("error"),
            timestamp=result_dict.get("timestamp", time.time()),
        )

        if result.success:
            self.checkpoint.mark_completed(result)
            print(
                f"[OK] {result.task_id} - crossings={result.final_crossings} "
                f"time={result.runtime_sec:.1f}s"
            )
        else:
            self.checkpoint.mark_failed(result)
            error_short = result.error.split("\n")[0] if result.error else "Unknown error"
            print(f"[FAIL] {result.task_id} - {error_short}")

        if self._results_callback:
            self._results_callback(result)

    def _on_task_error(self, error: Exception) -> None:
        """Callback when a task raises an exception in the pool."""
        print(f"[ERROR] Worker exception: {error}")

    def _log_progress(self, total_tasks: int) -> None:
        """Log current progress."""
        progress = self.checkpoint.get_progress(total_tasks)
        print(
            f"[Progress] {progress['completed']}/{total_tasks} completed "
            f"({progress['percent']:.1f}%) | "
            f"Failed: {progress['failed']} | "
            f"Elapsed: {format_time(progress['elapsed_sec'])} | "
            f"ETA: {format_time(progress['eta_seconds'])}"
        )

    def run(
        self,
        results_callback: Optional[Callable[[TaskResult], None]] = None,
        progress_interval: int = 60,
        retry_failed: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the pipeline.

        Args:
            results_callback: Optional callback for each completed task
            progress_interval: Seconds between progress logs
            retry_failed: If True, retry previously failed tasks

        Returns:
            Dictionary with run statistics
        """
        self._results_callback = results_callback
        self._shutdown_requested = False

        # Get pending tasks
        if retry_failed:
            cleared = self.checkpoint.clear_failed()
            if cleared > 0:
                print(f"[Runner] Cleared {cleared} failed tasks for retry")

        completed_ids = self.checkpoint.get_completed_ids()
        pending_tasks = filter_pending_tasks(self.all_tasks, completed_ids)

        total_all = len(self.all_tasks)
        total_pending = len(pending_tasks)

        print(f"[Runner] Machine: {self.machine_id}")
        print(f"[Runner] Workers: {self.num_workers}")
        print(f"[Runner] Total tasks: {total_all}")
        print(f"[Runner] Already completed: {len(completed_ids)}")
        print(f"[Runner] Pending: {total_pending}")

        if total_pending == 0:
            print("[Runner] All tasks already completed!")
            return self.checkpoint.get_stats()

        # Print task summary
        summary = get_task_summary(pending_tasks)
        print(f"[Runner] Task breakdown:")
        print(f"  By grid: {summary['by_grid']}")
        print(f"  By pattern: {summary['by_pattern']}")
        print(f"  By config: {summary['by_config']}")

        # Convert tasks to dicts for pickling
        task_dicts = [t.to_dict() for t in pending_tasks]

        # Set up signal handlers
        self._setup_signal_handlers()

        # Create output directory
        os.makedirs(self.machine_cfg.output_dir, exist_ok=True)

        # Run with multiprocessing pool
        print(f"\n[Runner] Starting {self.num_workers} workers...")
        start_time = time.time()
        last_progress_log = start_time

        try:
            with Pool(processes=self.num_workers) as pool:
                self._pool = pool

                # Submit all tasks asynchronously
                async_results = []
                for task_dict in task_dicts:
                    if self._shutdown_requested:
                        break
                    result = pool.apply_async(
                        execute_task_dict,
                        (task_dict,),
                        callback=self._on_task_complete,
                        error_callback=self._on_task_error,
                    )
                    async_results.append(result)

                # Wait for all tasks with progress logging
                completed = 0
                while completed < len(async_results) and not self._shutdown_requested:
                    # Count completed
                    new_completed = sum(1 for r in async_results if r.ready())

                    if new_completed > completed:
                        completed = new_completed

                    # Log progress periodically
                    now = time.time()
                    if now - last_progress_log >= progress_interval:
                        self._log_progress(total_all)
                        last_progress_log = now

                    # Small sleep to avoid busy waiting
                    time.sleep(0.5)

                self._pool = None

        except KeyboardInterrupt:
            print("\n[Runner] Interrupted by user")
            self._shutdown_requested = True

        # Final stats
        elapsed = time.time() - start_time
        stats = self.checkpoint.get_stats()
        stats["run_elapsed_sec"] = elapsed
        stats["shutdown_requested"] = self._shutdown_requested

        print(f"\n[Runner] Run complete!")
        print(f"  Elapsed: {format_time(elapsed)}")
        print(f"  Completed this run: {stats['completed_count'] - len(completed_ids)}")
        print(f"  Total completed: {stats['completed_count']}/{total_all}")
        print(f"  Failed: {stats['failed_count']}")

        return stats

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "machine_id": self.machine_id,
            "num_workers": self.num_workers,
            "total_tasks": len(self.all_tasks),
            "progress": self.checkpoint.get_progress(len(self.all_tasks)),
            "stats": self.checkpoint.get_stats(),
        }


def run_pipeline(
    config_dir: str,
    machine_id: str,
    num_workers: Optional[int] = None,
    retry_failed: bool = False,
    progress_interval: int = 60,
) -> Dict[str, Any]:
    """
    Convenience function to run the pipeline.

    Args:
        config_dir: Directory containing YAML config files
        machine_id: Machine identifier
        num_workers: Number of worker processes
        retry_failed: If True, retry previously failed tasks
        progress_interval: Seconds between progress logs

    Returns:
        Run statistics dictionary
    """
    runner = ParallelRunner(
        config_dir=config_dir,
        machine_id=machine_id,
        num_workers=num_workers,
    )

    return runner.run(
        retry_failed=retry_failed,
        progress_interval=progress_interval,
    )
