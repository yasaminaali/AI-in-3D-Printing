"""
Main multiprocessing orchestrator for SA pipeline with Rich UI.
Uses multiprocessing.Manager for shared state with workers.
All logs go to files, only Rich UI is displayed on console.
"""

import os
import sys
import signal
import time
import threading
import logging
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count, Manager
from typing import Any, Callable, Dict, List, Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
)
from rich.table import Table
from rich.layout import Layout
from rich.text import Text
from rich import box

from .config import GlobalConfig, MachineConfig, Task, load_config
from .task_generator import generate_tasks, filter_pending_tasks, get_task_summary
from .checkpoint import Checkpoint, TaskResult
from .worker import execute_task_dict


# Global console for Rich output
console = Console()


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds == float("inf") or seconds < 0:
        return "unknown"
    
    td = timedelta(seconds=int(seconds))
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if days > 0:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    elif hours > 0:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    elif minutes > 0:
        return f"{minutes}m {seconds:02d}s"
    else:
        return f"{seconds}s"


def format_number(n: int) -> str:
    """Format large numbers with commas."""
    return f"{n:,}"


def setup_logging(output_dir: str, machine_id: str) -> logging.Logger:
    """Setup logging to file only (no console output)."""
    # Create logs directory
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"{machine_id}_{timestamp}.log")
    
    # Configure logging
    logger = logging.getLogger("sa_pipeline")
    logger.setLevel(logging.INFO)
    
    # File handler only - no console handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Clear any existing handlers and add file handler
    logger.handlers.clear()
    logger.addHandler(file_handler)
    
    return logger


class RichUIManager:
    """
    Manages Rich UI with multiprocessing using Manager for shared state.
    All logs go to file, only Rich UI is displayed on console.
    """
    
    def __init__(self, total_tasks: int, num_workers: int, machine_id: str, output_dir: str,
                 already_completed: int = 0):
        self.total_tasks = total_tasks  # Total including already completed
        self.already_completed = already_completed  # Tasks completed in previous runs
        self.num_workers = num_workers
        self.machine_id = machine_id
        self.output_dir = output_dir
        self.start_time = time.time()

        # Setup file logging
        self.logger = setup_logging(output_dir, machine_id)
        self.log_file = os.path.join(output_dir, "logs", f"{machine_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        self.logger.info(f"Starting pipeline for {machine_id} with {num_workers} workers")
        self.logger.info(f"Total tasks: {total_tasks}, Already completed: {already_completed}")

        # Create manager for shared state
        self.manager = Manager()
        self.shared_state = self.manager.dict()
        self.shared_state['completed'] = already_completed  # Start from already completed
        self.shared_state['failed'] = 0
        self.shared_state['current_tasks'] = self.manager.dict()
        self.shared_state['logs'] = self.manager.list()
        self.shared_state['initial_crossings'] = self.manager.list()
        self.shared_state['final_crossings'] = self.manager.list()
        self.shared_state['runtimes'] = self.manager.list()
        
        # Local cache for stats
        self.task_durations: List[float] = []
        self.max_durations_history = 100
        
        # Create progress
        self.progress = Progress(
            SpinnerColumn(spinner_name="dots", style="cyan"),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None, complete_style="green", finished_style="bright_green"),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%", style="yellow"),
            TimeElapsedColumn(),
            TextColumn("<", style="dim"),
            TimeRemainingColumn(),
            TextColumn("[@{task.fields[rate]:.2f} tasks/s]", style="cyan"),
            console=console,
            transient=False,
            expand=True
        )
        
        # Add main progress task - start with already completed tasks
        self.main_task = self.progress.add_task(
            f"[bold cyan]{machine_id}",
            total=total_tasks,
            completed=already_completed,
            rate=0.0
        )
        
        # Create detailed layout
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="header", size=1),
            Layout(name="progress", size=3),
            Layout(name="main", ratio=1)
        )
        
        # Split main into left and right
        self.layout["main"].split_row(
            Layout(name="logs", ratio=2),
            Layout(name="stats", ratio=1)
        )
        
        # Set initial content
        self.layout["header"].update(
            Text(f"SA Dataset Generation Pipeline - {machine_id}", style="bold cyan", justify="center")
        )
        self.layout["progress"].update(self.progress)
        
        # Initialize panels
        self._update_logs_panel()
        self._update_stats_panel()
        
        # Initialize live display
        self.live = Live(
            self.layout,
            console=console,
            refresh_per_second=4,
            screen=False,
            transient=False
        )
        
        self._shutdown = False
        self.stats_thread: Optional[threading.Thread] = None
    
    def _generate_logs_table(self) -> Table:
        """Generate logs table."""
        logs = list(self.shared_state['logs'])
        
        table = Table(
            title="[bold green]Activity Log",
            box=box.ROUNDED,
            border_style="green",
            expand=True,
            show_header=True,
            header_style="bold"
        )
        table.add_column("Time", style="dim", width=8)
        table.add_column("Status", width=3, justify="center")
        table.add_column("Task ID", style="white", min_width=30, max_width=40)
        table.add_column("Details", style="cyan")
        
        if logs:
            # Show last 15 log entries
            for log_entry in logs[-15:]:
                parts = log_entry.split("|", 3)
                if len(parts) >= 4:
                    table.add_row(parts[0], parts[1], parts[2], parts[3])
        else:
            table.add_row("", "", "[dim]Waiting for tasks...", "")
        
        return table
    
    def _generate_stats_panel(self) -> Panel:
        """Generate detailed statistics panel."""
        completed = self.shared_state['completed']
        failed = self.shared_state['failed']
        remaining = self.total_tasks - completed - failed
        # Calculate tasks completed THIS session (excluding resumed)
        completed_this_session = completed - self.already_completed
        elapsed = time.time() - self.start_time

        # Calculate statistics
        if completed_this_session > 0:
            avg_time = sum(self.task_durations) / len(self.task_durations) if self.task_durations else 0
            rate = completed_this_session / elapsed if elapsed > 0 else 0
            eta_seconds = remaining / rate if rate > 0 else float('inf')

            # Calculate crossings stats if available
            initial_list = list(self.shared_state['initial_crossings'])
            final_list = list(self.shared_state['final_crossings'])
            if initial_list and final_list:
                avg_initial = sum(initial_list) / len(initial_list)
                avg_final = sum(final_list) / len(final_list)
                best_final = min(final_list)  # Best (least) crossings achieved
            else:
                avg_initial = avg_final = best_final = 0
        else:
            avg_time = rate = eta_seconds = 0
            avg_initial = avg_final = best_final = 0

        # Create stats table
        stats_table = Table(box=None, show_header=False, padding=(0, 1))
        stats_table.add_column("Label", style="cyan", width=15)
        stats_table.add_column("Value", style="white")

        # Progress section
        stats_table.add_row("[bold yellow]Progress", "")
        stats_table.add_row("  Total:", f"[white]{format_number(self.total_tasks)}")
        if self.already_completed > 0:
            stats_table.add_row("  Resumed:", f"[dim]{format_number(self.already_completed)}")
        stats_table.add_row("  Completed:", f"[green]{format_number(completed)} ({completed/self.total_tasks*100:.1f}%)")
        stats_table.add_row("  Failed:", f"[red]{format_number(failed)}")
        stats_table.add_row("  Remaining:", f"[yellow]{format_number(remaining)}")
        stats_table.add_row("", "")

        # Time section
        stats_table.add_row("[bold yellow]Timing", "")
        stats_table.add_row("  Elapsed:", f"[white]{format_time(elapsed)}")
        stats_table.add_row("  ETA:", f"[white]{format_time(eta_seconds)}")
        stats_table.add_row("  Rate:", f"[green]{rate:.2f} tasks/sec")
        stats_table.add_row("  Avg Time:", f"[white]{avg_time:.1f}s per task")
        stats_table.add_row("", "")

        # Crossings section (only show if we have data from this session)
        if completed_this_session > 0 and avg_initial > 0:
            stats_table.add_row("[bold yellow]Crossings", "")
            stats_table.add_row("  Initial:", f"[white]{avg_initial:.1f} (avg)")
            stats_table.add_row("  Final:", f"[cyan]{avg_final:.1f} (avg)")
            stats_table.add_row("  Best:", f"[green]{best_final}")
            stats_table.add_row("", "")
        
        # Active tasks section
        current_tasks = dict(self.shared_state['current_tasks'])
        if current_tasks:
            stats_table.add_row(f"[bold yellow]Active ({len(current_tasks)})", "")
            now = time.time()
            for task_id, start_time in list(current_tasks.items())[:5]:
                duration = now - start_time
                short_id = task_id[:25] + "..." if len(task_id) > 25 else task_id
                stats_table.add_row("", f"[dim]• {short_id}[/] ([cyan]{duration:.1f}s[/])")
        
        return Panel(
            stats_table,
            title="[bold blue]Statistics",
            border_style="blue",
            box=box.ROUNDED
        )
    
    def _update_logs_panel(self):
        """Update the logs panel."""
        self.layout["logs"].update(self._generate_logs_table())
    
    def _update_stats_panel(self):
        """Update the stats panel."""
        self.layout["stats"].update(self._generate_stats_panel())
    
    def _update_loop(self):
        """Background thread to update the display."""
        while not self._shutdown:
            time.sleep(0.25)  # Update 4 times per second
            if not self._shutdown:
                # Update progress
                completed = self.shared_state['completed']
                elapsed = time.time() - self.start_time
                rate = completed / elapsed if elapsed > 0 else 0
                self.progress.update(self.main_task, completed=completed, rate=rate)
                
                # Update panels
                self._update_logs_panel()
                self._update_stats_panel()
    
    def start(self):
        """Start the live display."""
        self.live.start()
        self.stats_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.stats_thread.start()
        self.logger.info("Rich UI started")
    
    def log_task_completion(self, task_id: str, success: bool,
                           initial_crossings: Optional[int], final_crossings: Optional[int],
                           runtime_sec: float, error: Optional[str] = None):
        """Log task completion to both Rich UI and file."""
        timestamp = time.strftime("%H:%M:%S")

        if success:
            status = "[green]OK[/]"
            short_id = task_id[:35] + "..." if len(task_id) > 35 else task_id
            details = f"{initial_crossings}->{final_crossings}, {runtime_sec:.1f}s"
            log_entry = f"{timestamp}|OK|{short_id}|{details}"

            # Add to Rich UI logs
            logs = list(self.shared_state['logs'])
            logs.append(log_entry)
            if len(logs) > 100:
                logs = logs[-100:]
            self.shared_state['logs'] = logs

            # Log to file
            self.logger.info(f"Task completed: {task_id} | {initial_crossings}->{final_crossings} | time={runtime_sec:.2f}s")

            # Update crossings stats
            if initial_crossings is not None:
                initial_list = list(self.shared_state['initial_crossings'])
                initial_list.append(initial_crossings)
                self.shared_state['initial_crossings'] = initial_list
            if final_crossings is not None:
                final_list = list(self.shared_state['final_crossings'])
                final_list.append(final_crossings)
                self.shared_state['final_crossings'] = final_list
        else:
            status = "[red]X[/]"
            short_id = task_id[:35] + "..." if len(task_id) > 35 else task_id
            error_msg = error[:30] + "..." if error and len(error) > 30 else (error or "Unknown error")
            log_entry = f"{timestamp}|X|{short_id}|[red]{error_msg}[/]"

            # Add to Rich UI logs
            logs = list(self.shared_state['logs'])
            logs.append(log_entry)
            if len(logs) > 100:
                logs = logs[-100:]
            self.shared_state['logs'] = logs

            # Log to file
            self.logger.error(f"Task failed: {task_id} | error={error}")
    
    def on_task_start(self, task_id: str):
        """Called when a task starts."""
        current_tasks = dict(self.shared_state['current_tasks'])
        current_tasks[task_id] = time.time()
        self.shared_state['current_tasks'] = current_tasks
        self.logger.info(f"Task started: {task_id}")
    
    def on_task_complete(self, result: TaskResult):
        """Called when a task completes."""
        # Update current tasks
        current_tasks = dict(self.shared_state['current_tasks'])
        if result.task_id in current_tasks:
            duration = time.time() - current_tasks[result.task_id]
            self.task_durations.append(duration)
            if len(self.task_durations) > self.max_durations_history:
                self.task_durations.pop(0)
            del current_tasks[result.task_id]
        self.shared_state['current_tasks'] = current_tasks
        
        # Update counts
        if result.success:
            self.shared_state['completed'] = self.shared_state['completed'] + 1
        else:
            self.shared_state['failed'] = self.shared_state['failed'] + 1
        
        # Log completion
        self.log_task_completion(
            result.task_id,
            result.success,
            result.initial_crossings,
            result.final_crossings,
            result.runtime_sec or 0.0,
            result.error
        )
    
    def stop(self):
        """Stop the live display."""
        self._shutdown = True
        if self.stats_thread:
            self.stats_thread.join(timeout=2)
        self.live.stop()
        
        # Log final stats
        completed = self.shared_state['completed']
        failed = self.shared_state['failed']
        elapsed = time.time() - self.start_time
        self.logger.info(f"Pipeline finished: completed={completed}, failed={failed}, elapsed={elapsed:.1f}s")
        self.logger.info(f"Log file: {self.log_file}")
        
        self.manager.shutdown()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get final summary."""
        elapsed = time.time() - self.start_time
        completed = self.shared_state['completed']
        return {
            'total_tasks': self.total_tasks,
            'completed': completed,
            'failed': self.shared_state['failed'],
            'elapsed_sec': elapsed,
            'rate': completed / elapsed if elapsed > 0 else 0,
            'log_file': self.log_file
        }


class ParallelRunner:
    """Multiprocessing orchestrator with Rich UI."""

    def __init__(
        self,
        config_dir: str,
        machine_id: str,
        num_workers: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
    ):
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
        checkpoint_dir_str = str(checkpoint_dir) if checkpoint_dir else "."
        os.makedirs(checkpoint_dir_str, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir_str, f"checkpoint_{machine_id}.json")
        self.checkpoint = Checkpoint(checkpoint_path)

        # Generate all tasks
        self.all_tasks = generate_tasks(self.global_cfg, self.machine_cfg)

        # State
        self._pool: Optional[Pool] = None
        self._shutdown_requested = False
        self._results_callback: Optional[Callable] = None
        self._ui_manager: Optional[RichUIManager] = None

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def handler(signum, frame):
            if self._ui_manager:
                self._ui_manager.logger.info("Shutdown requested by user")
            self._shutdown_requested = True
            if self._pool is not None:
                self._pool.terminate()

        if sys.platform != "win32":
            signal.signal(signal.SIGINT, handler)
            signal.signal(signal.SIGTERM, handler)

    def _on_task_complete(self, result_dict: Dict[str, Any]) -> None:
        """Callback when a task completes."""
        result = TaskResult(
            task_id=result_dict["task_id"],
            success=result_dict["success"],
            initial_crossings=result_dict.get("initial_crossings"),
            final_crossings=result_dict.get("final_crossings"),
            runtime_sec=result_dict.get("runtime_sec"),
            error=result_dict.get("error"),
            timestamp=result_dict.get("timestamp", time.time()),
        )

        if result.success:
            self.checkpoint.mark_completed(result)
        else:
            self.checkpoint.mark_failed(result)

        if self._ui_manager:
            self._ui_manager.on_task_complete(result)

        if self._results_callback:
            self._results_callback(result)

    def _on_task_error(self, error: BaseException) -> None:
        """Callback when a task raises an exception."""
        if self._ui_manager:
            self._ui_manager.logger.error(f"Worker exception: {error}")

    def run(
        self,
        results_callback: Optional[Callable[[TaskResult], None]] = None,
        progress_interval: int = 60,
        retry_failed: bool = False,
    ) -> Dict[str, Any]:
        """Run the pipeline with Rich UI only."""
        self._results_callback = results_callback
        self._shutdown_requested = False

        # Get pending tasks
        if retry_failed:
            cleared = self.checkpoint.clear_failed()

        completed_ids = self.checkpoint.get_completed_ids()
        pending_tasks = filter_pending_tasks(self.all_tasks, completed_ids)

        total_all = len(self.all_tasks)
        total_pending = len(pending_tasks)

        if total_pending == 0:
            # Log to file only, no console output
            return self.checkpoint.get_stats()

        # Convert tasks to dicts
        task_dicts = [t.to_dict() for t in pending_tasks]

        # Set up signal handlers
        self._setup_signal_handlers()

        # Create output directory
        os.makedirs(self.machine_cfg.output_dir, exist_ok=True)

        # Initialize Rich UI Manager with total tasks and already completed count
        already_completed = len(completed_ids)
        self._ui_manager = RichUIManager(
            total_tasks=total_all,
            num_workers=self.num_workers,
            machine_id=self.machine_id,
            output_dir=self.machine_cfg.output_dir,
            already_completed=already_completed
        )

        # Run with multiprocessing pool
        start_time = time.time()

        try:
            # Start the Rich UI
            self._ui_manager.start()
            
            with Pool(processes=self.num_workers) as pool:
                self._pool = pool

                # Submit all tasks
                async_results = []
                for task_dict in task_dicts:
                    if self._shutdown_requested:
                        break
                    
                    self._ui_manager.on_task_start(task_dict['task_id'])
                    
                    result = pool.apply_async(
                        execute_task_dict,
                        (task_dict,),
                        callback=self._on_task_complete,
                        error_callback=self._on_task_error,
                    )
                    async_results.append(result)

                # Wait for completion
                completed = 0
                while completed < len(async_results) and not self._shutdown_requested:
                    new_completed = sum(1 for r in async_results if r.ready())
                    if new_completed > completed:
                        completed = new_completed
                    time.sleep(0.5)

                self._pool = None

        except KeyboardInterrupt:
            if self._ui_manager:
                self._ui_manager.logger.info("Interrupted by user")
            self._shutdown_requested = True

        # Stop Rich UI
        if self._ui_manager:
            self._ui_manager.stop()

        # Final stats - log to file only
        elapsed = time.time() - start_time
        stats = self.checkpoint.get_stats()
        stats["run_elapsed_sec"] = elapsed
        stats["shutdown_requested"] = self._shutdown_requested
        
        if self._ui_manager:
            self._ui_manager.logger.info("=" * 60)
            self._ui_manager.logger.info("Run Complete!")
            self._ui_manager.logger.info(f"Elapsed: {format_time(elapsed)}")
            self._ui_manager.logger.info(f"Completed this run: {stats['completed_count'] - len(completed_ids)}")
            self._ui_manager.logger.info(f"Total completed: {stats['completed_count']}/{total_all}")
            self._ui_manager.logger.info(f"Failed: {stats['failed_count']}")
            if stats['completed_count'] > 0:
                success_rate = (stats['completed_count'] / total_all) * 100
                self._ui_manager.logger.info(f"Success Rate: {success_rate:.1f}%")
            self._ui_manager.logger.info(f"Log File: {self._ui_manager.log_file}")
            self._ui_manager.logger.info("=" * 60)

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
    """Convenience function to run the pipeline."""
    runner = ParallelRunner(
        config_dir=config_dir,
        machine_id=machine_id,
        num_workers=num_workers,
    )

    return runner.run(
        retry_failed=retry_failed,
        progress_interval=progress_interval,
    )
