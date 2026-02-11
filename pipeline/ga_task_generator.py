"""
Task generator for GA pipeline.

Generates a list of GATask objects from machine configuration.
"""

from typing import List

from .config import parse_zone_params
from .ga_config import (
    GAGlobalConfig,
    GAMachineConfig,
    GATask,
    parse_ga_config,
)


def generate_ga_tasks(
    global_cfg: GAGlobalConfig,
    machine_cfg: GAMachineConfig,
) -> List[GATask]:
    """
    Generate all GA tasks from configuration.

    Each task is a unique combination of:
    - Grid size (width, height)
    - Zone pattern (left_right, stripes, islands, voronoi)

    Total: 3 grids x 4 patterns = 12 tasks.

    Args:
        global_cfg: GA global configuration with presets
        machine_cfg: Machine-specific assignments

    Returns:
        List of GATask objects
    """
    tasks: List[GATask] = []

    for assignment in machine_cfg.assignments:
        width, height = assignment.grid

        # Look up the GA config preset
        config_name = assignment.ga_config
        if config_name not in global_cfg.ga_configs:
            raise ValueError(
                f"Unknown GA config preset: {config_name}. "
                f"Available: {list(global_cfg.ga_configs.keys())}"
            )
        base_ga_config = global_cfg.ga_configs[config_name]

        # Use default zone params
        zone_params = global_cfg.default_zone_params

        for pattern in assignment.patterns:
            # Get genome length for this pattern
            genome_len = assignment.genome_len.get(pattern)
            if genome_len is None:
                raise ValueError(
                    f"No genome_len specified for pattern '{pattern}' "
                    f"in grid {width}x{height}"
                )

            # Create a GAConfig with the correct genome_len
            ga_config = parse_ga_config(
                {
                    "generations": base_ga_config.generations,
                    "pop_size": base_ga_config.pop_size,
                    "tourn_k": base_ga_config.tourn_k,
                    "elite_k": base_ga_config.elite_k,
                    "keep_rate": base_ga_config.keep_rate,
                    "cx_rate": base_ga_config.cx_rate,
                    "cx_ratio": base_ga_config.cx_ratio,
                    "eps_crossings": base_ga_config.eps_crossings,
                    "min_applied_valid": base_ga_config.min_applied_valid,
                    "max_tries_per_slot": base_ga_config.max_tries_per_slot,
                },
                genome_len=genome_len,
            )

            task_id = f"{machine_cfg.machine_id}_{width}x{height}_{pattern}"

            task = GATask(
                task_id=task_id,
                width=width,
                height=height,
                zone_mode=pattern,
                zone_params=zone_params,
                ga_config=ga_config,
                dataset_jsonl=machine_cfg.dataset_jsonl,
                output_dir=machine_cfg.output_dir,
            )
            tasks.append(task)

    return tasks


def filter_pending_ga_tasks(tasks: List[GATask], completed_ids: set) -> List[GATask]:
    """
    Filter out completed GA tasks.

    Args:
        tasks: All GA tasks
        completed_ids: Set of completed task IDs

    Returns:
        List of tasks that haven't been completed
    """
    return [t for t in tasks if t.task_id not in completed_ids]
