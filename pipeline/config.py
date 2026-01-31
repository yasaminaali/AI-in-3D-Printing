"""
Configuration loading and data classes for the SA pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class SAConfig:
    """SA algorithm configuration preset."""
    iterations: int
    Tmax: float
    Tmin: float
    max_move_tries: int = 200
    pool_refresh_period: int = 250
    pool_max_moves: int = 5000
    reheat_patience: int = 1500
    reheat_factor: float = 1.5
    reheat_cap: float = 600.0
    transpose_phase_ratio: float = 0.6
    border_to_inner: bool = True


@dataclass
class ZoneParams:
    """Zone pattern parameters."""
    # islands
    num_islands: int = 3
    island_size: int = 8
    allow_touch: bool = False
    # stripes
    stripe_direction: str = "v"
    stripe_k: int = 3
    # voronoi
    voronoi_k: int = 3


@dataclass
class Task:
    """Represents a single SA run task."""
    task_id: str
    width: int
    height: int
    zone_mode: str
    zone_params: ZoneParams
    sa_config: SAConfig
    seed: int
    output_dir: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "width": self.width,
            "height": self.height,
            "zone_mode": self.zone_mode,
            "zone_params": {
                "num_islands": self.zone_params.num_islands,
                "island_size": self.zone_params.island_size,
                "allow_touch": self.zone_params.allow_touch,
                "stripe_direction": self.zone_params.stripe_direction,
                "stripe_k": self.zone_params.stripe_k,
                "voronoi_k": self.zone_params.voronoi_k,
            },
            "sa_config": {
                "iterations": self.sa_config.iterations,
                "Tmax": self.sa_config.Tmax,
                "Tmin": self.sa_config.Tmin,
                "max_move_tries": self.sa_config.max_move_tries,
                "pool_refresh_period": self.sa_config.pool_refresh_period,
                "pool_max_moves": self.sa_config.pool_max_moves,
                "reheat_patience": self.sa_config.reheat_patience,
                "reheat_factor": self.sa_config.reheat_factor,
                "reheat_cap": self.sa_config.reheat_cap,
                "transpose_phase_ratio": self.sa_config.transpose_phase_ratio,
                "border_to_inner": self.sa_config.border_to_inner,
            },
            "seed": self.seed,
            "output_dir": self.output_dir,
        }


@dataclass
class Assignment:
    """Work assignment for a specific grid size."""
    grid: List[int]  # [width, height]
    patterns: List[str]
    sa_configs: List[str]
    seeds: Dict[str, int]  # pattern -> number of seeds
    zone_params: Optional[Dict[str, Any]] = None


@dataclass
class MachineConfig:
    """Configuration for a specific machine."""
    machine_id: str
    num_workers: int
    output_dir: str
    assignments: List[Assignment]


@dataclass
class GlobalConfig:
    """Global configuration with SA presets."""
    sa_configs: Dict[str, SAConfig]
    default_zone_params: ZoneParams


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_sa_config(data: Dict[str, Any]) -> SAConfig:
    """Parse SA config from dictionary."""
    return SAConfig(
        iterations=data["iterations"],
        Tmax=data["Tmax"],
        Tmin=data["Tmin"],
        max_move_tries=data.get("max_move_tries", 200),
        pool_refresh_period=data.get("pool_refresh_period", 250),
        pool_max_moves=data.get("pool_max_moves", 5000),
        reheat_patience=data.get("reheat_patience", 1500),
        reheat_factor=data.get("reheat_factor", 1.5),
        reheat_cap=data.get("reheat_cap", 600.0),
        transpose_phase_ratio=data.get("transpose_phase_ratio", 0.6),
        border_to_inner=data.get("border_to_inner", True),
    )


def parse_zone_params(data: Optional[Dict[str, Any]]) -> ZoneParams:
    """Parse zone parameters from dictionary."""
    if data is None:
        return ZoneParams()
    return ZoneParams(
        num_islands=data.get("num_islands", 3),
        island_size=data.get("island_size", 8),
        allow_touch=data.get("allow_touch", False),
        stripe_direction=data.get("stripe_direction", "v"),
        stripe_k=data.get("stripe_k", 3),
        voronoi_k=data.get("voronoi_k", 3),
    )


def load_global_config(config_dir: str) -> GlobalConfig:
    """Load global configuration from YAML."""
    path = os.path.join(config_dir, "global_config.yaml")
    data = load_yaml(path)

    sa_configs = {}
    for name, cfg_data in data.get("sa_configs", {}).items():
        sa_configs[name] = parse_sa_config(cfg_data)

    default_zone_params = parse_zone_params(data.get("default_zone_params"))

    return GlobalConfig(
        sa_configs=sa_configs,
        default_zone_params=default_zone_params,
    )


def load_machine_config(config_dir: str, machine_id: str) -> MachineConfig:
    """Load machine-specific configuration from YAML."""
    path = os.path.join(config_dir, f"{machine_id}.yaml")
    data = load_yaml(path)

    assignments = []
    for asgn_data in data.get("assignments", []):
        assignments.append(Assignment(
            grid=asgn_data["grid"],
            patterns=asgn_data["patterns"],
            sa_configs=asgn_data["sa_configs"],
            seeds=asgn_data["seeds"],
            zone_params=asgn_data.get("zone_params"),
        ))

    return MachineConfig(
        machine_id=data["machine_id"],
        num_workers=data.get("num_workers", 8),
        output_dir=data.get("output_dir", f"output/{machine_id}"),
        assignments=assignments,
    )


def load_config(config_dir: str, machine_id: str) -> tuple:
    """
    Load both global and machine configurations.

    Returns:
        (GlobalConfig, MachineConfig)
    """
    global_cfg = load_global_config(config_dir)
    machine_cfg = load_machine_config(config_dir, machine_id)
    return global_cfg, machine_cfg
