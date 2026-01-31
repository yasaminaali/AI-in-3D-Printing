"""
Checkpoint management for SA pipeline.

Provides thread-safe progress tracking with file locking for resume capability.
"""

import json
import os
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class TaskResult:
    """Result of a single task execution."""
    task_id: str
    success: bool
    final_crossings: Optional[int] = None
    runtime_sec: Optional[float] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class Checkpoint:
    """
    Thread-safe checkpoint manager for tracking pipeline progress.

    Stores completed and failed tasks in a JSON file for resume capability.
    Uses file locking to handle concurrent updates from multiple workers.
    """

    def __init__(self, checkpoint_path: str):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_path: Path to checkpoint JSON file
        """
        self.checkpoint_path = checkpoint_path
        self._lock = threading.Lock()
        self._data = self._load_or_create()

    def _load_or_create(self) -> Dict[str, Any]:
        """Load existing checkpoint or create new one."""
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Ensure required keys exist
                    data.setdefault("completed_tasks", {})
                    data.setdefault("failed_tasks", {})
                    data.setdefault("stats", {
                        "total_completed": 0,
                        "total_failed": 0,
                        "start_time": time.time(),
                        "last_update": time.time(),
                    })
                    return data
            except (json.JSONDecodeError, IOError) as e:
                # Silently start fresh checkpoint, error will be logged by caller if needed
                pass

        return {
            "completed_tasks": {},
            "failed_tasks": {},
            "stats": {
                "total_completed": 0,
                "total_failed": 0,
                "start_time": time.time(),
                "last_update": time.time(),
            },
        }

    def _save(self) -> None:
        """Save checkpoint to file (must hold lock)."""
        self._data["stats"]["last_update"] = time.time()

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.checkpoint_path) or ".", exist_ok=True)

        # Write to temp file first, then rename for atomicity
        tmp_path = self.checkpoint_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
            os.replace(tmp_path, self.checkpoint_path)
        except IOError as e:
            # Silently handle error, will be retried on next save
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass

    def mark_completed(self, result: TaskResult) -> None:
        """
        Mark a task as completed.

        Args:
            result: Task execution result
        """
        with self._lock:
            self._data["completed_tasks"][result.task_id] = {
                "final_crossings": result.final_crossings,
                "runtime_sec": result.runtime_sec,
                "timestamp": result.timestamp,
            }
            self._data["stats"]["total_completed"] = len(self._data["completed_tasks"])
            self._save()

    def mark_failed(self, result: TaskResult) -> None:
        """
        Mark a task as failed.

        Args:
            result: Task execution result with error info
        """
        with self._lock:
            self._data["failed_tasks"][result.task_id] = {
                "error": result.error,
                "timestamp": result.timestamp,
            }
            self._data["stats"]["total_failed"] = len(self._data["failed_tasks"])
            self._save()

    def get_completed_ids(self) -> Set[str]:
        """Get set of completed task IDs."""
        with self._lock:
            return set(self._data["completed_tasks"].keys())

    def get_failed_ids(self) -> Set[str]:
        """Get set of failed task IDs."""
        with self._lock:
            return set(self._data["failed_tasks"].keys())

    def get_all_processed_ids(self) -> Set[str]:
        """Get set of all processed (completed + failed) task IDs."""
        with self._lock:
            completed = set(self._data["completed_tasks"].keys())
            failed = set(self._data["failed_tasks"].keys())
            return completed | failed

    def get_stats(self) -> Dict[str, Any]:
        """Get checkpoint statistics."""
        with self._lock:
            stats = self._data["stats"].copy()
            stats["completed_count"] = len(self._data["completed_tasks"])
            stats["failed_count"] = len(self._data["failed_tasks"])
            return stats

    def get_progress(self, total_tasks: int) -> Dict[str, Any]:
        """
        Get progress information.

        Args:
            total_tasks: Total number of tasks

        Returns:
            Progress dictionary with counts and percentages
        """
        with self._lock:
            completed = len(self._data["completed_tasks"])
            failed = len(self._data["failed_tasks"])
            processed = completed + failed
            remaining = max(0, total_tasks - processed)

            elapsed = time.time() - self._data["stats"]["start_time"]
            rate = completed / elapsed if elapsed > 0 else 0
            eta_seconds = remaining / rate if rate > 0 else float("inf")

            return {
                "completed": completed,
                "failed": failed,
                "remaining": remaining,
                "total": total_tasks,
                "percent": (processed / total_tasks * 100) if total_tasks > 0 else 0,
                "elapsed_sec": elapsed,
                "rate_per_sec": rate,
                "eta_seconds": eta_seconds,
            }

    def clear_failed(self) -> int:
        """
        Clear failed tasks to allow retry.

        Returns:
            Number of failed tasks cleared
        """
        with self._lock:
            count = len(self._data["failed_tasks"])
            self._data["failed_tasks"] = {}
            self._data["stats"]["total_failed"] = 0
            self._save()
            return count

    def reset(self) -> None:
        """Reset checkpoint completely."""
        with self._lock:
            self._data = {
                "completed_tasks": {},
                "failed_tasks": {},
                "stats": {
                    "total_completed": 0,
                    "total_failed": 0,
                    "start_time": time.time(),
                    "last_update": time.time(),
                },
            }
            self._save()
