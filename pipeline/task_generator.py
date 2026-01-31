"""
Task generator for SA pipeline.

Generates a list of Task objects from machine configuration.
"""

from typing import List

from .config import (
    GlobalConfig,
    MachineConfig,
    Task,
    ZoneParams,
    parse_zone_params,
)


def generate_tasks(
    global_cfg: GlobalConfig,
    machine_cfg: MachineConfig,
) -> List[Task]:
    """
    Generate all tasks from configuration.

    Each task is a unique combination of:
    - Grid size (width, height)
    - Zone pattern (left_right, stripes, islands, voronoi)
    - SA config preset (short, medium, long, extra_long)
    - Seed number

    Args:
        global_cfg: Global configuration with SA presets
        machine_cfg: Machine-specific assignments

    Returns:
        List of Task objects
    """
    tasks: List[Task] = []

    for assignment in machine_cfg.assignments:
        width, height = assignment.grid

        # Get zone params for this assignment (or use defaults)
        if assignment.zone_params:
            zone_params = parse_zone_params(assignment.zone_params)
        else:
            zone_params = global_cfg.default_zone_params

        for pattern in assignment.patterns:
            # Get number of seeds for this pattern
            num_seeds = assignment.seeds.get(pattern, 0)
            if num_seeds <= 0:
                continue

            for sa_config_name in assignment.sa_configs:
                # Get SA config preset
                if sa_config_name not in global_cfg.sa_configs:
                    raise ValueError(
                        f"Unknown SA config preset: {sa_config_name}. "
                        f"Available: {list(global_cfg.sa_configs.keys())}"
                    )
                sa_config = global_cfg.sa_configs[sa_config_name]

                for seed in range(num_seeds):
                    # Generate unique task ID
                    task_id = (
                        f"{machine_cfg.machine_id}_"
                        f"{width}x{height}_"
                        f"{pattern}_"
                        f"{sa_config_name}_"
                        f"seed{seed}"
                    )

                    task = Task(
                        task_id=task_id,
                        width=width,
                        height=height,
                        zone_mode=pattern,
                        zone_params=zone_params,
                        sa_config=sa_config,
                        seed=seed,
                        output_dir=machine_cfg.output_dir,
                    )
                    tasks.append(task)

    return tasks


def filter_pending_tasks(tasks: List[Task], completed_ids: set) -> List[Task]:
    """
    Filter out completed tasks.

    Args:
        tasks: All tasks
        completed_ids: Set of completed task IDs

    Returns:
        List of tasks that haven't been completed
    """
    return [t for t in tasks if t.task_id not in completed_ids]


def get_task_summary(tasks: List[Task]) -> dict:
    """
    Get summary statistics for task list.

    Args:
        tasks: List of tasks

    Returns:
        Dictionary with summary stats
    """
    if not tasks:
        return {
            "total": 0,
            "by_grid": {},
            "by_pattern": {},
            "by_config": {},
        }

    by_grid = {}
    by_pattern = {}
    by_config = {}

    for task in tasks:
        grid_key = f"{task.width}x{task.height}"
        by_grid[grid_key] = by_grid.get(grid_key, 0) + 1
        by_pattern[task.zone_mode] = by_pattern.get(task.zone_mode, 0) + 1

        # Extract config name from task_id
        parts = task.task_id.split("_")
        if len(parts) >= 4:
            config_name = parts[-2]  # e.g., "short", "medium", etc.
            by_config[config_name] = by_config.get(config_name, 0) + 1

    return {
        "total": len(tasks),
        "by_grid": by_grid,
        "by_pattern": by_pattern,
        "by_config": by_config,
    }
