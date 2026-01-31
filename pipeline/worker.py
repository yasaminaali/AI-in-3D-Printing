"""
Worker function for SA pipeline.

Executes a single SA task by calling the existing run_sa() function.
"""

import sys
import os
import time
import traceback
from typing import Any, Dict

# Add parent directory to path for imports
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from .config import Task
from .checkpoint import TaskResult


def execute_task(task: Task) -> TaskResult:
    """
    Execute a single SA task.

    Imports and calls the existing run_sa() function from SA_generation.py.
    Catches all exceptions and returns a TaskResult with status info.

    Args:
        task: Task object containing all SA parameters

    Returns:
        TaskResult with execution status
    """
    start_time = time.time()

    try:
        # Import here to avoid issues with multiprocessing
        from SA_generation import run_sa

        # Call run_sa with all parameters from task
        final_crossings, best_ops = run_sa(
            width=task.width,
            height=task.height,
            iterations=task.sa_config.iterations,
            Tmax=task.sa_config.Tmax,
            Tmin=task.sa_config.Tmin,
            seed=task.seed,
            plot_live=False,
            show_every_accepted=0,
            pause_seconds=0.0,
            dataset_dir=task.output_dir,
            write_dataset=True,
            # Pool / attempts
            max_move_tries=task.sa_config.max_move_tries,
            pool_refresh_period=task.sa_config.pool_refresh_period,
            pool_max_moves=task.sa_config.pool_max_moves,
            # Reheating
            reheat_patience=task.sa_config.reheat_patience,
            reheat_factor=task.sa_config.reheat_factor,
            reheat_cap=task.sa_config.reheat_cap,
            # Phases
            transpose_phase_ratio=task.sa_config.transpose_phase_ratio,
            border_to_inner=task.sa_config.border_to_inner,
            # Zone mode
            zone_mode=task.zone_mode,
            # Islands
            num_islands=task.zone_params.num_islands,
            island_size=task.zone_params.island_size,
            allow_touch=task.zone_params.allow_touch,
            # Stripes
            stripe_direction=task.zone_params.stripe_direction,
            stripe_k=task.zone_params.stripe_k,
            # Voronoi
            voronoi_k=task.zone_params.voronoi_k,
            # Debug
            debug=False,
        )

        runtime = time.time() - start_time

        return TaskResult(
            task_id=task.task_id,
            success=True,
            final_crossings=final_crossings,
            runtime_sec=runtime,
            error=None,
            timestamp=time.time(),
        )

    except Exception as e:
        runtime = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        return TaskResult(
            task_id=task.task_id,
            success=False,
            final_crossings=None,
            runtime_sec=runtime,
            error=error_msg,
            timestamp=time.time(),
        )


def execute_task_dict(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a task from dictionary representation.

    Used by multiprocessing Pool which requires picklable arguments.

    Args:
        task_dict: Task as dictionary

    Returns:
        TaskResult as dictionary
    """
    from .config import Task, SAConfig, ZoneParams

    # Reconstruct Task object from dict
    sa_config = SAConfig(**task_dict["sa_config"])
    zone_params = ZoneParams(**task_dict["zone_params"])

    task = Task(
        task_id=task_dict["task_id"],
        width=task_dict["width"],
        height=task_dict["height"],
        zone_mode=task_dict["zone_mode"],
        zone_params=zone_params,
        sa_config=sa_config,
        seed=task_dict["seed"],
        output_dir=task_dict["output_dir"],
    )

    result = execute_task(task)

    return {
        "task_id": result.task_id,
        "success": result.success,
        "final_crossings": result.final_crossings,
        "runtime_sec": result.runtime_sec,
        "error": result.error,
        "timestamp": result.timestamp,
    }
