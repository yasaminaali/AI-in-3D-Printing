"""
Configuration loading and data classes for the GA pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from .config import ZoneParams, parse_zone_params, load_yaml


@dataclass
class GAConfig:
    """GA algorithm configuration preset."""
    generations: int
    pop_size: int
    tourn_k: int
    genome_len: int = 100
    elite_k: int = 6
    keep_rate: float = 0.60
    cx_rate: float = 0.90
    cx_ratio: float = 0.60
    eps_crossings: int = 2
    min_applied_valid: int = 1
    max_tries_per_slot: int = 80


@dataclass
class GATask:
    """Represents a single GA run task."""
    task_id: str
    width: int
    height: int
    zone_mode: str
    zone_params: ZoneParams
    ga_config: GAConfig
    dataset_jsonl: str
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
            "ga_config": {
                "generations": self.ga_config.generations,
                "pop_size": self.ga_config.pop_size,
                "tourn_k": self.ga_config.tourn_k,
                "genome_len": self.ga_config.genome_len,
                "elite_k": self.ga_config.elite_k,
                "keep_rate": self.ga_config.keep_rate,
                "cx_rate": self.ga_config.cx_rate,
                "cx_ratio": self.ga_config.cx_ratio,
                "eps_crossings": self.ga_config.eps_crossings,
                "min_applied_valid": self.ga_config.min_applied_valid,
                "max_tries_per_slot": self.ga_config.max_tries_per_slot,
            },
            "dataset_jsonl": self.dataset_jsonl,
            "output_dir": self.output_dir,
        }


@dataclass
class GAAssignment:
    """Work assignment for a specific grid size in GA pipeline."""
    grid: List[int]  # [width, height]
    patterns: List[str]
    ga_config: str  # name of GA config preset
    genome_len: Dict[str, int]  # pattern -> genome length


@dataclass
class GAMachineConfig:
    """Machine-specific configuration for GA pipeline."""
    machine_id: str
    num_workers: int
    output_dir: str
    dataset_jsonl: str
    assignments: List[GAAssignment]


@dataclass
class GAGlobalConfig:
    """Global configuration with GA presets."""
    ga_configs: Dict[str, GAConfig]
    default_zone_params: ZoneParams


def parse_ga_config(data: Dict[str, Any], genome_len: int = 100) -> GAConfig:
    """Parse GA config from dictionary."""
    return GAConfig(
        generations=data["generations"],
        pop_size=data["pop_size"],
        tourn_k=data.get("tourn_k", 3),
        genome_len=genome_len,
        elite_k=data.get("elite_k", 6),
        keep_rate=data.get("keep_rate", 0.60),
        cx_rate=data.get("cx_rate", 0.90),
        cx_ratio=data.get("cx_ratio", 0.60),
        eps_crossings=data.get("eps_crossings", 2),
        min_applied_valid=data.get("min_applied_valid", 1),
        max_tries_per_slot=data.get("max_tries_per_slot", 80),
    )


def load_ga_global_config(config_dir: str) -> GAGlobalConfig:
    """Load GA global configuration from YAML."""
    path = os.path.join(config_dir, "ga_global_config.yaml")
    data = load_yaml(path)

    ga_configs = {}
    for name, cfg_data in data.get("ga_configs", {}).items():
        ga_configs[name] = parse_ga_config(cfg_data)

    default_zone_params = parse_zone_params(data.get("default_zone_params"))

    return GAGlobalConfig(
        ga_configs=ga_configs,
        default_zone_params=default_zone_params,
    )


def load_ga_machine_config(config_dir: str, machine_id: str) -> GAMachineConfig:
    """Load GA machine-specific configuration from YAML."""
    path = os.path.join(config_dir, f"{machine_id}.yaml")
    data = load_yaml(path)

    assignments = []
    for asgn_data in data.get("assignments", []):
        assignments.append(GAAssignment(
            grid=asgn_data["grid"],
            patterns=asgn_data["patterns"],
            ga_config=asgn_data["ga_config"],
            genome_len=asgn_data["genome_len"],
        ))

    return GAMachineConfig(
        machine_id=data["machine_id"],
        num_workers=data.get("num_workers", 4),
        output_dir=data.get("output_dir", f"output/{machine_id}"),
        dataset_jsonl=data.get("dataset_jsonl", "datasets/combined_dataset.jsonl"),
        assignments=assignments,
    )


def load_ga_config(config_dir: str, machine_id: str) -> tuple:
    """
    Load both GA global and machine configurations.

    Returns:
        (GAGlobalConfig, GAMachineConfig)
    """
    global_cfg = load_ga_global_config(config_dir)
    machine_cfg = load_ga_machine_config(config_dir, machine_id)
    return global_cfg, machine_cfg
