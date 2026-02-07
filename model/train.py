"""
CNN+RNN Training Script with Rich UI Dashboard
Consolidated training with beautiful terminal visualization
"""

import os
import sys
import yaml
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import json
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.models.cnn_rnn import CNNRNNHamiltonian
from model.training.loss import HamiltonianLoss
from model.data.dataset import HamiltonianDataset, collate_fn

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.columns import Columns
from rich import box
from rich.tree import Tree
import csv

console = Console()


class RichTrainer:
    """Trainer with Rich UI dashboard."""
    
    def __init__(self, config_path: str = "model/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.train_config = self.config['training']
        
        # GPU Setup
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available! This trainer requires GPU.")
        self.device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        console.print(Panel.fit(
            f"[bold green]GPU:[/bold green] {gpu_name}\n"
            f"[bold green]Memory:[/bold green] {gpu_mem:.1f} GB",
            title="[bold blue]System Info[/bold blue]",
            border_style="green"
        ))
        
        # Directories - configurable from config
        self.log_dir = Path(self.config['logging']['log_dir'])
        self.checkpoint_dir = Path(self.config['checkpointing']['checkpoint_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(f"[dim]Checkpoints will be saved to: {self.checkpoint_dir}[/dim]")
        console.print(f"[dim]Logs will be saved to: {self.log_dir}[/dim]")
        
        # Model
        self.model = CNNRNNHamiltonian(self.config).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        console.print(Panel.fit(
            f"[bold cyan]Total Parameters:[/bold cyan] {total_params:,}\n"
            f"[bold cyan]Trainable:[/bold cyan] {trainable_params:,}",
            title="[bold blue]Model[/bold blue]",
            border_style="cyan"
        ))
        
        # Optimizer - support AdamW
        optimizer_type = self.train_config.get('optimizer', 'Adam')
        if optimizer_type == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.train_config['learning_rate'],
                weight_decay=self.train_config['weight_decay'],
                betas=(0.9, 0.999)
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.train_config['learning_rate'],
                weight_decay=self.train_config['weight_decay']
            )
        
        # Scheduler - support CosineAnnealing
        scheduler_type = self.train_config.get('scheduler', 'ReduceLROnPlateau')
        if scheduler_type == 'CosineAnnealing':
            min_lr = self.train_config.get('min_learning_rate', 1e-6)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.train_config['epochs'],
                eta_min=min_lr
            )
            self.scheduler_type = 'cosine'
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.train_config['scheduler_patience']
            )
            self.scheduler_type = 'plateau'
        
        # Warmup settings
        self.warmup_epochs = self.train_config.get('warmup_epochs', 0)
        self.base_lr = self.train_config['learning_rate']
        
        # Teacher forcing decay
        self.tf_ratio = self.train_config.get('teacher_forcing_ratio', 0.5)
        self.tf_decay = self.train_config.get('teacher_forcing_decay', 1.0)
        
        # Loss with label smoothing
        max_pos = self.config['model']['predictor']['max_positions']
        label_smoothing = self.train_config.get('label_smoothing', 0.0)
        self.criterion = HamiltonianLoss(
            self.train_config['loss_weights'], 
            max_positions=max_pos,
            label_smoothing=label_smoothing
        )
        
        # State
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.start_time = time.time()
        
        # Logging setup
        self.log_file = self.log_dir / "training_log.csv"
        self._init_csv_log()
        
        # Batch sizes per grid - read from config
        batch_sizes_config = self.config['training'].get('batch_sizes', {
            "30x30": 32,
            "50x50": 16,
            "80x80": 8,
            "default": 8
        })
        self.batch_sizes = {}
        for key, value in batch_sizes_config.items():
            if key == "default":
                continue
            try:
                w, h = map(int, key.split('x'))
                self.batch_sizes[(w, h)] = value
            except:
                pass
        self.default_batch_size = batch_sizes_config.get("default", 8)
        
        # Validation settings - configurable
        self.validate_every = self.config['validation'].get('validate_every_n_epochs', 1)
        self.compute_perf_stats = self.config['validation'].get('compute_performance_stats', True)
        self.stats_sample_size = self.config['validation'].get('stats_sample_size', 10)
        self.detailed_stats_every = self.config['validation'].get('detailed_stats_every_n_epochs', 5)
        
        # Checkpointing settings - configurable
        self.save_best_only = self.config['checkpointing'].get('save_best_only', True)
        self.save_every_n_epochs = self.config['checkpointing'].get('save_every_n_epochs', 10)
        self.keep_last_n = self.config['checkpointing'].get('keep_last_n_checkpoints', 3)
        
        # Performance settings
        self.num_workers = self.config['hardware'].get('num_workers', 0)
        self.pin_memory = self.config['hardware'].get('pin_memory', True)
    
    def _init_csv_log(self):
        """Initialize CSV log file with headers."""
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'train_type_loss', 'train_x_loss', 
                'train_y_loss', 'train_variant_loss', 'val_loss', 
                'val_type_loss', 'val_x_loss', 'val_y_loss', 'val_variant_loss',
                'learning_rate', 'best_val_loss', 'time_elapsed'
            ])
    
    def log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log epoch metrics to CSV."""
        elapsed = time.time() - self.start_time
        lr = self.optimizer.param_groups[0]['lr']
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics.get('total', 0),
                train_metrics.get('type', 0),
                train_metrics.get('x', 0),
                train_metrics.get('y', 0),
                train_metrics.get('variant', 0),
                val_metrics.get('total', 0),
                val_metrics.get('type', 0),
                val_metrics.get('x', 0),
                val_metrics.get('y', 0),
                val_metrics.get('variant', 0),
                lr,
                self.best_val_loss,
                elapsed
            ])
    
    def save_training_summary(self):
        """Save final training summary as JSON."""
        summary = {
            'total_epochs': self.current_epoch,
            'best_val_loss': float(self.best_val_loss),
            'total_time_sec': time.time() - self.start_time,
            'train_losses': [float(x) for x in self.train_losses],
            'val_losses': [float(x) for x in self.val_losses],
            'config': self.config
        }
        
        summary_file = self.log_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        console.print(f"[dim]Training summary saved to: {summary_file}[/dim]")
    
    def group_by_grid_size(self, dataset: HamiltonianDataset) -> Dict[Tuple[int, int], List[int]]:
        """Group dataset indices by grid size."""
        groups = defaultdict(list)
        for idx in range(len(dataset)):
            record = dataset.records[idx]
            grid_size = (record['grid_W'], record['grid_H'])
            groups[grid_size].append(idx)
        return dict(groups)
    
    def create_grid_specific_loader(self, dataset: HamiltonianDataset,
                                   indices: List[int],
                                   grid_size: Tuple[int, int],
                                   shuffle: bool = True) -> DataLoader:
        """Create a DataLoader for a specific grid size."""
        batch_size = self.batch_sizes.get(grid_size, self.default_batch_size)
        subset = torch.utils.data.Subset(dataset, indices)

        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory
        )
    
    def format_time(self, seconds: float) -> str:
        """Format seconds to readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def create_dashboard(self, epoch: int, train_metrics: Dict, val_metrics: Dict,
                        progress: float, grid_stats: Dict, status: str = "Training") -> Layout:
        """Create Rich dashboard layout."""
        layout = Layout()
        
        # Header
        elapsed = time.time() - self.start_time
        header = Panel(
            f"[bold blue]CNN+RNN Hamiltonian Path Optimizer[/bold blue]\n"
            f"Epoch [bold]{epoch}[/bold]/{self.train_config['epochs']} | "
            f"Status: [bold]{status}[/bold] | "
            f"Elapsed: [dim]{self.format_time(elapsed)}[/dim]",
            box=box.ROUNDED,
            border_style="blue"
        )
        
        # Progress bar
        progress_bar = f"\n[progress.description]{int(progress*100)}%"
        progress_display = Panel(
            f"[cyan]{'█' * int(progress * 30)}[/cyan][dim]{'░' * (30 - int(progress * 30))}[/dim] "
            f"[bold]{progress*100:.1f}%[/bold]",
            title="[bold]Progress[/bold]",
            box=box.MINIMAL,
            border_style="cyan"
        )
        
        # Grid stats table
        grid_table = Table(title="Grid Distribution", show_header=True, box=box.SIMPLE)
        grid_table.add_column("Size", style="cyan", justify="center")
        grid_table.add_column("Samples", style="green", justify="right")
        grid_table.add_column("Batch", style="yellow", justify="right")
        
        for (W, H), count in sorted(grid_stats.items()):
            bs = self.batch_sizes.get((W, H), 4)
            grid_table.add_row(f"{W}x{H}", str(count), str(bs))
        
        # Training metrics
        train_table = Table(title="Training Metrics", show_header=True, box=box.SIMPLE)
        train_table.add_column("Metric", style="cyan")
        train_table.add_column("Value", style="green", justify="right")
        
        train_table.add_row("Total Loss", f"{train_metrics.get('total', 0):.4f}")
        train_table.add_row("Type Loss", f"{train_metrics.get('type', 0):.4f}")
        train_table.add_row("Position X", f"{train_metrics.get('x', 0):.4f}")
        train_table.add_row("Position Y", f"{train_metrics.get('y', 0):.4f}")
        train_table.add_row("Variant", f"{train_metrics.get('variant', 0):.4f}")
        
        # Validation metrics
        val_table = Table(title="Validation Metrics", show_header=True, box=box.SIMPLE)
        val_table.add_column("Metric", style="cyan")
        val_table.add_column("Value", style="magenta", justify="right")
        
        val_table.add_row("Total Loss", f"{val_metrics.get('total', 0):.4f}")
        val_table.add_row("Type Loss", f"{val_metrics.get('type', 0):.4f}")
        val_table.add_row("Position X", f"{val_metrics.get('x', 0):.4f}")
        val_table.add_row("Position Y", f"{val_metrics.get('y', 0):.4f}")
        val_table.add_row("Variant", f"{val_metrics.get('variant', 0):.4f}")
        
        # Status panel
        lr = self.optimizer.param_groups[0]['lr']
        best_loss = self.best_val_loss
        patience = self.train_config['early_stopping_patience']
        
        status_text = (
            f"[bold]Learning Rate:[/bold] {lr:.6f}\n"
            f"[bold]Best Val Loss:[/bold] {best_loss:.4f}\n"
            f"[bold]Patience:[/bold] {self.patience_counter}/{patience}\n"
        )
        
        if len(self.train_losses) > 1:
            improvement = self.train_losses[-2] - self.train_losses[-1]
            if improvement > 0:
                status_text += f"[green]↓ Train loss improved by {improvement:.4f}[/green]"
            else:
                status_text += f"[red]↑ Train loss increased by {abs(improvement):.4f}[/red]"
        
        status_panel = Panel(status_text, title="[bold]Status[/bold]", box=box.ROUNDED, border_style="yellow")
        
        # Recent history
        history_text = "[bold]Recent Epochs:[/bold]\n"
        recent_epochs = min(5, len(self.train_losses))
        for i in range(recent_epochs):
            idx = len(self.train_losses) - recent_epochs + i
            if idx < len(self.val_losses):
                history_text += f"  E{idx+1}: Train={self.train_losses[idx]:.4f} Val={self.val_losses[idx]:.4f}\n"
            else:
                history_text += f"  E{idx+1}: Train={self.train_losses[idx]:.4f}\n"
        
        history_panel = Panel(history_text, title="[bold]History[/bold]", box=box.ROUNDED, border_style="dim")
        
        # Layout structure
        layout.split_column(
            Layout(header, size=3),
            Layout(progress_display, size=5),
            Layout(name="main")
        )
        
        layout["main"].split_row(
            Layout(grid_table, size=25),
            Layout(name="metrics"),
            Layout(name="info", size=35)
        )
        
        layout["metrics"].split_column(
            Layout(train_table),
            Layout(val_table)
        )
        
        layout["info"].split_column(
            Layout(status_panel, size=8),
            Layout(history_panel)
        )
        
        return layout
    
    def train_epoch(self, train_dataset: HamiltonianDataset, epoch: int) -> Tuple[float, Dict]:
        """Train one epoch with progress tracking and timing diagnostics."""
        self.model.train()
        groups = self.group_by_grid_size(train_dataset)

        total_loss = 0.0
        total_batches = 0
        all_components = {'type': [], 'x': [], 'y': [], 'variant': [], 'total': []}

        # Calculate total batches
        total_batches_estimated = 0
        for grid_size, indices in groups.items():
            if len(indices) > 0:
                batch_size = self.batch_sizes.get(grid_size, self.default_batch_size)
                total_batches_estimated += (len(indices) + batch_size - 1) // batch_size

        # Timing diagnostics
        import time
        batch_times = []
        data_load_times = []
        forward_times = []
        backward_times = []
        last_batch_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True
        ) as progress:

            task = progress.add_task(f"Epoch {epoch:3d}", total=total_batches_estimated)

            for grid_size, indices in sorted(groups.items()):
                if len(indices) == 0:
                    continue

                batch_size = self.batch_sizes.get(grid_size, self.default_batch_size)
                loader = self.create_grid_specific_loader(
                    train_dataset, indices, grid_size, shuffle=True
                )

                W, H = grid_size
                seq_len = self.config['model']['predictor']['sequence_length']

                for batch in loader:
                    batch_start = time.time()

                    # Data loading
                    data_start = time.time()
                    grid_states = batch['grid_states'].to(self.device)
                    global_features = batch['global_features'].to(self.device)
                    targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
                    data_load_times.append(time.time() - data_start)

                    # Forward pass
                    forward_start = time.time()
                    B = grid_states.size(0)
                    grid_states = grid_states.unsqueeze(1).expand(B, seq_len, 4, H, W)
                    global_features = global_features.unsqueeze(1).expand(B, seq_len, 3)

                    predictions = self.model(grid_states, global_features)
                    loss, components = self.criterion(
                        predictions, targets, batch['seq_lens'], self.device
                    )
                    forward_times.append(time.time() - forward_start)

                    # Backward pass
                    backward_start = time.time()
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.train_config['gradient_clip']
                    )
                    self.optimizer.step()
                    backward_times.append(time.time() - backward_start)

                    total_loss += loss.item()
                    total_batches += 1

                    for key in all_components:
                        all_components[key].append(components[key])

                    batch_time = time.time() - batch_start
                    batch_times.append(batch_time)

                    # Show timing every 10 batches
                    if total_batches % 10 == 0:
                        avg_batch = sum(batch_times[-10:]) / len(batch_times[-10:])
                        avg_data = sum(data_load_times[-10:]) / len(data_load_times[-10:])
                        avg_forward = sum(forward_times[-10:]) / len(forward_times[-10:])
                        avg_backward = sum(backward_times[-10:]) / len(backward_times[-10:])
                        progress.console.print(
                            f"[dim]Batch {total_batches}: {avg_batch:.2f}s "
                            f"(data:{avg_data:.2f}s forward:{avg_forward:.2f}s backward:{avg_backward:.2f}s)[/dim]"
                        )

                    progress.update(task, advance=1)
        
        avg_loss = total_loss / max(total_batches, 1)
        avg_components = {k: sum(v) / max(len(v), 1) for k, v in all_components.items()}
        
        return avg_loss, avg_components
    
    def validate(self, val_dataset: HamiltonianDataset) -> Tuple[float, Dict]:
        """Validate model."""
        self.model.eval()
        groups = self.group_by_grid_size(val_dataset)
        
        total_loss = 0.0
        total_batches = 0
        all_components = {'type': [], 'x': [], 'y': [], 'variant': [], 'total': []}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold magenta]Validating"),
            BarColumn(bar_width=40),
            console=console,
            transient=True
        ) as progress:
            
            task = progress.add_task("Validation", total=sum(len(indices) for indices in groups.values()))
            
            with torch.no_grad():
                for grid_size, indices in sorted(groups.items()):
                    if len(indices) == 0:
                        continue
                    
                    batch_size = self.batch_sizes.get(grid_size, 4)
                    loader = self.create_grid_specific_loader(
                        val_dataset, indices, grid_size, shuffle=False
                    )
                    
                    W, H = grid_size
                    seq_len = self.config['model']['predictor']['sequence_length']
                    
                    for batch in loader:
                        grid_states = batch['grid_states'].to(self.device)
                        global_features = batch['global_features'].to(self.device)
                        targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
                        
                        B = grid_states.size(0)
                        grid_states = grid_states.unsqueeze(1).expand(B, seq_len, 4, H, W)
                        global_features = global_features.unsqueeze(1).expand(B, seq_len, 3)
                        
                        predictions = self.model(grid_states, global_features)
                        loss, components = self.criterion(
                            predictions, targets, batch['seq_lens'], self.device
                        )
                        
                        total_loss += loss.item()
                        total_batches += 1
                        
                        for key in all_components:
                            all_components[key].append(components[key])
                        
                        progress.update(task, advance=batch_size)
        
        avg_loss = total_loss / max(total_batches, 1)
        avg_components = {k: sum(v) / max(len(v), 1) for k, v in all_components.items()}
        
        return avg_loss, avg_components
    
    def train(self, train_file: str, val_file: str):
        """Main training loop with Rich UI."""
        console.print("\n[bold]Loading datasets...[/bold]\n")
        
        max_pos = self.config['model']['predictor']['max_positions']
        seq_len = self.config['model']['predictor']['sequence_length']
        
        train_dataset = HamiltonianDataset(train_file, max_seq_len=seq_len, max_grid_size=max_pos)
        val_dataset = HamiltonianDataset(val_file, max_seq_len=seq_len, max_grid_size=max_pos)
        
        train_groups = self.group_by_grid_size(train_dataset)
        val_groups = self.group_by_grid_size(val_dataset)
        
        # Display dataset info
        dataset_table = Table(title="Dataset Information", show_header=True, box=box.ROUNDED)
        dataset_table.add_column("Grid Size", style="cyan")
        dataset_table.add_column("Train Samples", justify="right", style="green")
        dataset_table.add_column("Val Samples", justify="right", style="magenta")
        dataset_table.add_column("Batch Size", justify="right", style="yellow")
        
        for (W, H) in sorted(set(list(train_groups.keys()) + list(val_groups.keys()))):
            train_count = len(train_groups.get((W, H), []))
            val_count = len(val_groups.get((W, H), []))
            bs = self.batch_sizes.get((W, H), 4)
            dataset_table.add_row(f"{W}x{H}", str(train_count), str(val_count), str(bs))
        
        total_train = sum(len(v) for v in train_groups.values())
        total_val = sum(len(v) for v in val_groups.values())
        dataset_table.add_row("[bold]Total[/bold]", f"[bold]{total_train}[/bold]", f"[bold]{total_val}[/bold]", "")
        
        console.print(dataset_table)
        console.print()
        
        # Training loop with Live display
        validate_every = self.config['validation']['validate_every_n_epochs']
        
        for epoch in range(1, self.train_config['epochs'] + 1):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Apply warmup learning rate
            if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
                warmup_lr = self.base_lr * (epoch / self.warmup_epochs)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            # Decay teacher forcing ratio
            current_tf = self.tf_ratio * (self.tf_decay ** (epoch - 1))
            
            # Train
            train_loss, train_comp = self.train_epoch(train_dataset, epoch)
            self.train_losses.append(train_loss)
            
            # Validate every epoch
            val_loss, val_comp = self.validate(val_dataset)
            self.val_losses.append(val_loss)
            
            epoch_time = time.time() - epoch_start_time
            
            # Step scheduler (different for cosine vs plateau)
            if hasattr(self, 'scheduler_type') and self.scheduler_type == 'cosine':
                if epoch > self.warmup_epochs:  # Don't step during warmup
                    self.scheduler.step()
            else:
                self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                if self.save_best_only:
                    self.save_checkpoint("best_model.pt")
                    console.print(f"[bold green]✓ New best model saved! (val_loss: {val_loss:.4f})[/bold green]")
            else:
                self.patience_counter += 1

            # Log metrics to CSV (if enabled)
            if self.config['logging'].get('log_to_csv', True):
                self.log_epoch(epoch, train_comp, val_comp)

            # Save periodic checkpoint (configurable frequency)
            if self.save_every_n_epochs > 0 and (epoch % self.save_every_n_epochs == 0 or epoch == 1):
                checkpoint_name = f"checkpoint_epoch_{epoch}.pt"
                self.save_checkpoint(checkpoint_name)
                console.print(f"[dim]✓ Checkpoint saved: {checkpoint_name}[/dim]")

            # Display dashboard
            progress = epoch / self.train_config['epochs']
            dashboard = self.create_dashboard(
                epoch, train_comp, val_comp, progress, train_groups, status="Validating"
            )
            console.print(dashboard)

            # Compute and display performance statistics (configurable)
            if self.compute_perf_stats:
                # Detailed stats every N epochs, fast stats otherwise
                is_detailed = (epoch % self.detailed_stats_every == 0)
                sample_size = self.stats_sample_size if not is_detailed else min(50, self.stats_sample_size * 5)

                console.print(f"[dim]Computing performance statistics (sample_size={sample_size})...[/dim]")
                import random
                train_idx = random.sample(range(len(train_dataset)), min(sample_size, len(train_dataset)))
                val_idx = random.sample(range(len(val_dataset)), min(sample_size, len(val_dataset)))
                train_perf_stats = self.compute_performance_stats_fast(train_dataset, train_idx)
                val_perf_stats = self.compute_performance_stats_fast(val_dataset, val_idx)
                self.display_performance_stats(epoch, epoch_time, train_perf_stats, val_perf_stats)

            if self.patience_counter >= self.train_config['early_stopping_patience']:
                console.print(f"\n[bold red]Early stopping at epoch {epoch}[/bold red]")
                break
        
        # Final summary
        elapsed = time.time() - self.start_time
        
        # Save final training summary
        self.save_training_summary()
        
        console.print(Panel.fit(
            f"[bold green]Training Complete![/bold green]\n\n"
            f"Total Epochs: [bold]{self.current_epoch}[/bold]\n"
            f"Best Val Loss: [bold green]{self.best_val_loss:.4f}[/bold green]\n"
            f"Total Time: [dim]{self.format_time(elapsed)}[/dim]\n\n"
            f"Checkpoint: [cyan]{self.checkpoint_dir / 'best_model.pt'}[/cyan]\n"
            f"Logs: [cyan]{self.log_dir}[/cyan]",
            title="[bold blue]Summary[/bold blue]",
            border_style="green"
        ))
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
    
    def compute_performance_stats(self, dataset: HamiltonianDataset, max_samples: int = 100) -> Dict:
        """Compute detailed performance statistics on a subset of data."""
        self.model.eval()
        
        # Operation type accuracy
        type_correct = 0
        type_total = 0
        
        # Position accuracy (within tolerance)
        pos_x_correct = 0
        pos_y_correct = 0
        pos_total = 0
        
        # Variant accuracy (for each operation type)
        flip_var_correct = 0
        flip_var_total = 0
        trans_var_correct = 0
        trans_var_total = 0
        
        # Confusion matrix for operation types
        type_confusion = {'T': {'T': 0, 'F': 0, 'N': 0},
                         'F': {'T': 0, 'F': 0, 'N': 0},
                         'N': {'T': 0, 'F': 0, 'N': 0}}
        
        # Sample predictions vs actual
        samples = []
        
        seq_len = self.config['model']['predictor']['sequence_length']
        max_pos = self.config['model']['predictor']['max_positions']
        
        with torch.no_grad():
            for idx in range(min(max_samples, len(dataset))):
                sample = dataset[idx]
                grid_state = sample['grid_state'].unsqueeze(0).to(self.device)
                global_features = sample['global_features'].unsqueeze(0).to(self.device)
                targets = sample['target']
                
                W, H = sample['grid_W'], sample['grid_H']
                
                # Expand for sequence
                grid_state = grid_state.unsqueeze(1).expand(1, seq_len, 4, H, W)
                global_features = global_features.unsqueeze(1).expand(1, seq_len, 3)
                
                predictions = self.model(grid_state, global_features)
                
                # Get first actual operation (ignore padding)
                actual_seq_len = min(sample['seq_len'], seq_len)
                if actual_seq_len == 0:
                    continue
                
                for t in range(actual_seq_len):
                    # Operation type
                    pred_type = predictions['operation_type'][0, t].argmax().item()
                    actual_type = targets['operation_type'][t].item()
                    
                    type_total += 1
                    if pred_type == actual_type:
                        type_correct += 1
                    
                    # Map to labels for confusion matrix
                    type_map = {0: 'T', 1: 'F', 2: 'N'}
                    pred_type_label = type_map[pred_type]
                    actual_type_label = type_map[actual_type]
                    type_confusion[actual_type_label][pred_type_label] += 1
                    
                    # Position (only for non-NOP)
                    if actual_type != 2:  # Not NOP
                        pred_x = predictions['position_x'][0, t].argmax().item()
                        pred_y = predictions['position_y'][0, t].argmax().item()
                        actual_x = targets['position_x'][t].item()
                        actual_y = targets['position_y'][t].item()
                        
                        pos_total += 1
                        
                        # Check if within tolerance (1 grid cell)
                        if abs(pred_x - actual_x) <= 1:
                            pos_x_correct += 1
                        if abs(pred_y - actual_y) <= 1:
                            pos_y_correct += 1
                        
                        # Variant accuracy
                        if actual_type == 0:  # Transpose
                            pred_var = predictions['transpose_variant'][0, t].argmax().item()
                            actual_var = targets['transpose_variant'][t].item()
                            trans_var_total += 1
                            if pred_var == actual_var:
                                trans_var_correct += 1
                        elif actual_type == 1:  # Flip
                            pred_var = predictions['flip_variant'][0, t].argmax().item()
                            actual_var = targets['flip_variant'][t].item()
                            flip_var_total += 1
                            if pred_var == actual_var:
                                flip_var_correct += 1
                
                # Save first few samples
                if len(samples) < 3 and actual_seq_len > 0:
                    t = 0
                    pred_type = predictions['operation_type'][0, t].argmax().item()
                    actual_type = targets['operation_type'][t].item()
                    type_map = {0: 'T', 1: 'F', 2: 'N'}
                    
                    samples.append({
                        'predicted': type_map[pred_type],
                        'actual': type_map[actual_type],
                        'correct': pred_type == actual_type
                    })
        
        stats = {
            'type_accuracy': type_correct / max(type_total, 1) * 100,
            'pos_x_accuracy': pos_x_correct / max(pos_total, 1) * 100,
            'pos_y_accuracy': pos_y_correct / max(pos_total, 1) * 100,
            'flip_var_accuracy': flip_var_correct / max(flip_var_total, 1) * 100,
            'trans_var_accuracy': trans_var_correct / max(trans_var_total, 1) * 100,
            'confusion': type_confusion,
            'samples': samples,
            'total_ops': type_total
        }
        
        return stats
    
    def compute_performance_stats_fast(self, dataset: HamiltonianDataset, indices: List[int]) -> Dict:
        """Fast performance stats using pre-selected indices (no random sampling overhead)."""
        self.model.eval()
        
        type_correct = type_total = 0
        pos_x_correct = pos_y_correct = pos_total = 0
        flip_var_correct = flip_var_total = 0
        trans_var_correct = trans_var_total = 0
        type_confusion = {'T': {'T': 0, 'F': 0, 'N': 0},
                         'F': {'T': 0, 'F': 0, 'N': 0},
                         'N': {'T': 0, 'F': 0, 'N': 0}}
        samples = []
        
        seq_len = self.config['model']['predictor']['sequence_length']
        type_map = {0: 'T', 1: 'F', 2: 'N'}
        
        with torch.no_grad():
            for idx in indices:
                sample = dataset[idx]
                grid_state = sample['grid_state'].unsqueeze(0).to(self.device)
                global_features = sample['global_features'].unsqueeze(0).to(self.device)
                targets = sample['target']
                W, H = sample['grid_W'], sample['grid_H']
                
                # Single forward pass
                grid_state = grid_state.unsqueeze(1).expand(1, seq_len, 4, H, W)
                global_features = global_features.unsqueeze(1).expand(1, seq_len, 3)
                predictions = self.model(grid_state, global_features)
                
                actual_seq_len = min(sample['seq_len'], seq_len)
                if actual_seq_len == 0:
                    continue
                
                # Process first operation only per sample (much faster, still representative)
                t = 0
                pred_type = predictions['operation_type'][0, t].argmax().item()
                actual_type = targets['operation_type'][t].item()
                
                type_total += 1
                if pred_type == actual_type:
                    type_correct += 1
                
                type_confusion[type_map[actual_type]][type_map[pred_type]] += 1
                
                # Positions and variants (only for non-NOP)
                if actual_type != 2:
                    pred_x = predictions['position_x'][0, t].argmax().item()
                    pred_y = predictions['position_y'][0, t].argmax().item()
                    actual_x = targets['position_x'][t].item()
                    actual_y = targets['position_y'][t].item()
                    
                    pos_total += 1
                    if abs(pred_x - actual_x) <= 1:
                        pos_x_correct += 1
                    if abs(pred_y - actual_y) <= 1:
                        pos_y_correct += 1
                    
                    if actual_type == 0:  # Transpose
                        pred_var = predictions['transpose_variant'][0, t].argmax().item()
                        actual_var = targets['transpose_variant'][t].item()
                        trans_var_total += 1
                        if pred_var == actual_var:
                            trans_var_correct += 1
                    elif actual_type == 1:  # Flip
                        pred_var = predictions['flip_variant'][0, t].argmax().item()
                        actual_var = targets['flip_variant'][t].item()
                        flip_var_total += 1
                        if pred_var == actual_var:
                            flip_var_correct += 1
                
                # Save samples
                if len(samples) < 3:
                    samples.append({
                        'predicted': type_map[pred_type],
                        'actual': type_map[actual_type],
                        'correct': pred_type == actual_type
                    })
        
        return {
            'type_accuracy': type_correct / max(type_total, 1) * 100,
            'pos_x_accuracy': pos_x_correct / max(pos_total, 1) * 100,
            'pos_y_accuracy': pos_y_correct / max(pos_total, 1) * 100,
            'flip_var_accuracy': flip_var_correct / max(flip_var_total, 1) * 100,
            'trans_var_accuracy': trans_var_correct / max(trans_var_total, 1) * 100,
            'confusion': type_confusion,
            'samples': samples,
            'total_ops': type_total
        }
    
    def get_gpu_stats(self) -> Dict:
        """Get GPU memory and utilization stats."""
        if not torch.cuda.is_available():
            return {}
        
        return {
            'allocated': torch.cuda.memory_allocated() / 1e9,
            'reserved': torch.cuda.memory_reserved() / 1e9,
            'max_allocated': torch.cuda.max_memory_allocated() / 1e9,
        }
    
    def display_performance_stats(self, epoch: int, epoch_time: float, 
                                  train_stats: Dict, val_stats: Dict = {}):
        """Display comprehensive performance statistics."""
        
        # Create main stats panel
        stats_table = Table(title=f"[bold cyan]Epoch {epoch} Performance Statistics[/bold cyan]", 
                           show_header=True, box=box.ROUNDED)
        stats_table.add_column("Metric", style="cyan", width=25)
        stats_table.add_column("Training", style="green", justify="right")
        if val_stats:
            stats_table.add_column("Validation", style="magenta", justify="right")
        
        # Accuracy metrics
        stats_table.add_row(
            "Operation Type Accuracy",
            f"{train_stats['type_accuracy']:.1f}%",
            f"{val_stats['type_accuracy']:.1f}%" if val_stats else "-"
        )
        stats_table.add_row(
            "Position X Accuracy (±1)",
            f"{train_stats['pos_x_accuracy']:.1f}%",
            f"{val_stats['pos_x_accuracy']:.1f}%" if val_stats else "-"
        )
        stats_table.add_row(
            "Position Y Accuracy (±1)",
            f"{train_stats['pos_y_accuracy']:.1f}%",
            f"{val_stats['pos_y_accuracy']:.1f}%" if val_stats else "-"
        )
        stats_table.add_row(
            "Flip Variant Accuracy",
            f"{train_stats['flip_var_accuracy']:.1f}%",
            f"{val_stats['flip_var_accuracy']:.1f}%" if val_stats else "-"
        )
        stats_table.add_row(
            "Transpose Variant Accuracy",
            f"{train_stats['trans_var_accuracy']:.1f}%",
            f"{val_stats['trans_var_accuracy']:.1f}%" if val_stats else "-"
        )
        stats_table.add_row(
            "Total Operations Evaluated",
            str(train_stats['total_ops']),
            str(val_stats['total_ops']) if val_stats else "-"
        )
        
        # Confusion matrix
        confusion_table = Table(title="[bold cyan]Operation Type Confusion Matrix[/bold cyan]", 
                               show_header=True, box=box.SIMPLE)
        confusion_table.add_column("Actual \\ Predicted", style="cyan")
        confusion_table.add_column("T (Transpose)", justify="right")
        confusion_table.add_column("F (Flip)", justify="right")
        confusion_table.add_column("N (No-op)", justify="right")
        
        confusion = train_stats['confusion']
        for actual in ['T', 'F', 'N']:
            row = [f"[bold]{actual}[/bold]"]
            for pred in ['T', 'F', 'N']:
                count = confusion[actual][pred]
                if actual == pred:
                    row.append(f"[green]{count}[/green]")
                elif count > 0:
                    row.append(f"[red]{count}[/red]")
                else:
                    row.append(str(count))
            confusion_table.add_row(*row)
        
        # Sample predictions
        samples_text = "[bold cyan]Sample Predictions (First 3 samples):[/bold cyan]\n"
        for i, sample in enumerate(train_stats['samples']):
            status = "[green]✓[/green]" if sample['correct'] else "[red]✗[/red]"
            samples_text += f"  Sample {i+1}: Predicted={sample['predicted']}, Actual={sample['actual']} {status}\n"
        
        samples_panel = Panel(samples_text, box=box.ROUNDED, border_style="dim")
        
        # System stats
        gpu_stats = self.get_gpu_stats()
        lr = self.optimizer.param_groups[0]['lr']
        
        system_text = f"[bold]Learning Rate:[/bold] {lr:.6f}\n"
        system_text += f"[bold]Epoch Time:[/bold] {epoch_time:.1f}s\n"
        system_text += f"[bold]Throughput:[/bold] {train_stats['total_ops'] / max(epoch_time, 0.1):.1f} ops/sec\n"
        
        if gpu_stats:
            system_text += f"\n[bold]GPU Memory:[/bold]\n"
            system_text += f"  Allocated: {gpu_stats['allocated']:.2f} GB\n"
            system_text += f"  Reserved: {gpu_stats['reserved']:.2f} GB\n"
            system_text += f"  Peak: {gpu_stats['max_allocated']:.2f} GB"
        
        system_panel = Panel(system_text, title="[bold]System Stats[/bold]", 
                            box=box.ROUNDED, border_style="yellow")
        
        # Display everything
        console.print("\n")
        console.print(stats_table)
        console.print("\n")
        
        # Side by side confusion matrix and samples
        from rich.columns import Columns
        console.print(Columns([confusion_table, samples_panel]))
        console.print("\n")
        console.print(system_panel)
        console.print("\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CNN+RNN with Rich UI')
    parser.add_argument('--train-file', default='nn_data/train_all.jsonl',
                       help='Training data file')
    parser.add_argument('--val-file', default='nn_data/val_all.jsonl',
                       help='Validation data file')
    parser.add_argument('--config', default='model/config.yaml',
                       help='Configuration file')
    
    args = parser.parse_args()
    
    trainer = RichTrainer(args.config)
    trainer.train(args.train_file, args.val_file)


if __name__ == "__main__":
    main()
