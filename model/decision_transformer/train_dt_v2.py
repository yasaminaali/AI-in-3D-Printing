"""
Decision Transformer v2 Training Script with Rich UI Dashboard.

Features:
- Rich live progress tracking with metrics
- Mixed precision (AMP) training on GPU
- Variant-aware loss with position emphasis
- Trajectory-level train/val split
- Cosine annealing with warmup
- Early stopping
- Checkpoint saving (best + periodic)

Usage:
    python model/decision_transformer/train_dt_v2.py
    python model/decision_transformer/train_dt_v2.py --epochs 100 --batch_size 64
"""

import os
import sys
import time
import csv
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dt_model import DecisionTransformer, compute_loss
from dt_dataset_v2 import create_train_val_split

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.layout import Layout
from rich.columns import Columns
from rich import box

console = Console()


def create_dashboard(epoch, epochs, metrics, best_metrics, lr, gpu_mem_used, gpu_mem_total):
    """Create Rich dashboard layout."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="train", ratio=1),
        Layout(name="val", ratio=1),
        Layout(name="best", ratio=1),
    )

    # Header
    header = Table(box=None, show_header=False)
    header.add_row(
        f"[bold cyan]Decision Transformer v2[/bold cyan] | "
        f"Epoch [bold]{epoch}[/bold]/{epochs} | "
        f"LR: {lr:.2e} | "
        f"GPU: {gpu_mem_used:.1f}/{gpu_mem_total:.1f} GB"
    )
    layout["header"].update(header)

    # Train metrics
    train_table = Table(title="[bold green]Training[/bold green]", box=box.SIMPLE)
    train_table.add_column("Metric", style="dim")
    train_table.add_column("Value", justify="right")
    for k, v in metrics.get('train', {}).items():
        train_table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    layout["train"].update(Panel(train_table))

    # Val metrics
    val_table = Table(title="[bold yellow]Validation[/bold yellow]", box=box.SIMPLE)
    val_table.add_column("Metric", style="dim")
    val_table.add_column("Value", justify="right")
    for k, v in metrics.get('val', {}).items():
        val_table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    layout["val"].update(Panel(val_table))

    # Best metrics
    best_table = Table(title="[bold magenta]Best[/bold magenta]", box=box.SIMPLE)
    best_table.add_column("Metric", style="dim")
    best_table.add_column("Value", justify="right")
    for k, v in best_metrics.items():
        best_table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    layout["best"].update(Panel(best_table))

    # Footer
    footer = Table(box=None, show_header=False)
    footer.add_row(f"[dim]Patience: {metrics.get('patience_left', '?')} | Time/epoch: {metrics.get('epoch_time', '?')}s[/dim]")
    layout["footer"].update(footer)

    return layout


def validate(model, val_loader, device, use_amp):
    """Run validation and return metrics."""
    model.eval()
    total_loss = 0
    total_losses = {'op_type': 0, 'x': 0, 'y': 0, 'variant': 0}
    correct = {'op_type': 0, 'x': 0, 'y': 0, 'variant': 0}
    total_tokens = 0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            rtg = batch['returns_to_go'].to(device)
            timesteps = batch['timesteps'].to(device)
            mask = batch['attention_mask'].to(device)

            if use_amp:
                with autocast():
                    logits = model(states, actions, rtg, timesteps, mask)
                    loss, loss_dict = compute_loss(logits, actions, mask)
            else:
                logits = model(states, actions, rtg, timesteps, mask)
                loss, loss_dict = compute_loss(logits, actions, mask)

            total_loss += loss.item()
            for k in total_losses:
                total_losses[k] += loss_dict[k].item()
            n_batches += 1

            # Accuracy
            valid_mask = mask.bool()
            n_valid = valid_mask.sum().item()
            total_tokens += n_valid

            correct['op_type'] += ((logits['op_type'].argmax(-1) == actions[:, :, 0]) & valid_mask).sum().item()
            correct['x'] += ((logits['x'].argmax(-1) == actions[:, :, 1]) & valid_mask).sum().item()
            correct['y'] += ((logits['y'].argmax(-1) == actions[:, :, 2]) & valid_mask).sum().item()
            correct['variant'] += ((logits['variant'].argmax(-1) == actions[:, :, 3]) & valid_mask).sum().item()

    avg_loss = total_loss / max(n_batches, 1)
    avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
    accuracy = {k: v / max(total_tokens, 1) for k, v in correct.items()}

    return avg_loss, avg_losses, accuracy


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # System info
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        gpu_name = 'CPU'
        gpu_mem_total = 0

    console.print(Panel.fit(
        f"[bold cyan]Decision Transformer v2 Training[/bold cyan]\n"
        f"Device: {device} ({gpu_name})\n"
        f"Data: {args.data_path}\n"
        f"Checkpoints: {args.checkpoint_dir}",
        border_style="cyan"
    ))

    # Load dataset
    console.print("\n[bold]Loading dataset...[/bold]")
    train_ds, val_ds = create_train_val_split(
        args.data_path,
        val_ratio=args.val_split,
        context_len=args.context_len,
        seed=42
    )
    console.print(f"  Train: [green]{len(train_ds)}[/green] samples")
    console.print(f"  Val: [green]{len(val_ds)}[/green] samples")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Create model
    console.print("\n[bold]Creating model...[/bold]")
    model = DecisionTransformer(
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        context_len=args.context_len,
        max_timestep=args.max_timestep,
        dropout=args.dropout,
        max_grid_size=args.max_grid_size,
        n_variants=args.n_variants,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"  Parameters: [green]{n_params:,}[/green]")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )

    # Scheduler: warmup + cosine annealing
    total_steps = len(train_loader) * args.epochs

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        return max(0.01, 0.5 * (1 + __import__('math').cos(__import__('math').pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    scaler = GradScaler() if args.use_amp else None

    # Checkpoint dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # CSV logger
    log_path = os.path.join(args.checkpoint_dir, 'training_log.csv')
    csv_file = open(log_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'epoch', 'train_loss', 'val_loss',
        'val_acc_op', 'val_acc_x', 'val_acc_y', 'val_acc_var',
        'learning_rate', 'epoch_time'
    ])

    # Training state
    best_val_loss = float('inf')
    best_metrics = {'val_loss': float('inf'), 'epoch': 0, 'acc_op': 0, 'acc_x': 0, 'acc_y': 0}
    patience_counter = 0
    global_step = 0

    console.print(f"\n[bold]Starting training for {args.epochs} epochs...[/bold]")
    console.print(f"  Batches/epoch: {len(train_loader)}")
    console.print(f"  Total steps: {total_steps:,}")
    console.print()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        model.train()
        epoch_loss = 0
        epoch_losses = {'op_type': 0, 'x': 0, 'y': 0, 'variant': 0}
        n_batches = 0

        with Progress(
            SpinnerColumn(),
            TextColumn(f"Epoch {epoch}/{args.epochs}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TextColumn("[cyan]{task.fields[loss]:.4f}[/cyan]"),
            TimeElapsedColumn(),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Training", total=len(train_loader), loss=0.0)

            for batch in train_loader:
                states = batch['states'].to(device)
                actions = batch['actions'].to(device)
                rtg = batch['returns_to_go'].to(device)
                timesteps = batch['timesteps'].to(device)
                mask = batch['attention_mask'].to(device)

                optimizer.zero_grad()

                if args.use_amp:
                    with autocast():
                        logits = model(states, actions, rtg, timesteps, mask)
                        loss, loss_dict = compute_loss(logits, actions, mask)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(states, actions, rtg, timesteps, mask)
                    loss, loss_dict = compute_loss(logits, actions, mask)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                global_step += 1

                epoch_loss += loss.item()
                for k in epoch_losses:
                    epoch_losses[k] += loss_dict[k].item()
                n_batches += 1

                progress.update(task, advance=1, loss=epoch_loss / n_batches)

        avg_train_loss = epoch_loss / max(n_batches, 1)
        epoch_time = time.time() - epoch_start

        # Validate
        val_loss, val_losses, val_acc = validate(model, val_loader, device, args.use_amp)

        # GPU memory
        gpu_mem_used = torch.cuda.memory_allocated(0) / 1e9 if device.type == 'cuda' else 0
        lr = optimizer.param_groups[0]['lr']

        # Check best
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            best_metrics = {
                'val_loss': val_loss,
                'epoch': epoch,
                'acc_op': val_acc['op_type'],
                'acc_x': val_acc['x'],
                'acc_y': val_acc['y'],
                'acc_var': val_acc['variant'],
            }
        else:
            patience_counter += 1

        # Display
        metrics = {
            'train': {
                'loss': avg_train_loss,
                'op_type': epoch_losses['op_type'] / max(n_batches, 1),
                'x': epoch_losses['x'] / max(n_batches, 1),
                'y': epoch_losses['y'] / max(n_batches, 1),
                'variant': epoch_losses['variant'] / max(n_batches, 1),
            },
            'val': {
                'loss': val_loss,
                'acc_op': val_acc['op_type'],
                'acc_x': val_acc['x'],
                'acc_y': val_acc['y'],
                'acc_var': val_acc['variant'],
            },
            'patience_left': args.patience - patience_counter,
            'epoch_time': f"{epoch_time:.1f}",
        }

        dashboard = create_dashboard(epoch, args.epochs, metrics, best_metrics, lr, gpu_mem_used, gpu_mem_total)
        console.print(dashboard)

        best_marker = " [bold green]** BEST **[/bold green]" if is_best else ""
        console.print(
            f"  Epoch {epoch}: train={avg_train_loss:.4f} val={val_loss:.4f} | "
            f"acc: op={val_acc['op_type']:.1%} x={val_acc['x']:.1%} "
            f"y={val_acc['y']:.1%} var={val_acc['variant']:.1%} | "
            f"{epoch_time:.1f}s{best_marker}"
        )

        # Log to CSV
        csv_writer.writerow([
            epoch, avg_train_loss, val_loss,
            val_acc['op_type'], val_acc['x'], val_acc['y'], val_acc['variant'],
            lr, epoch_time
        ])
        csv_file.flush()

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'config': {
                'embed_dim': args.embed_dim,
                'n_heads': args.n_heads,
                'n_layers': args.n_layers,
                'context_len': args.context_len,
                'max_timestep': args.max_timestep,
                'max_grid_size': args.max_grid_size,
                'n_variants': args.n_variants,
                'dropout': args.dropout,
            }
        }

        if is_best:
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best.pt'))

        if epoch % args.save_every == 0:
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'epoch_{epoch}.pt'))

        # Always save latest
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'latest.pt'))

        # Early stopping
        if patience_counter >= args.patience:
            console.print(f"\n[bold red]Early stopping at epoch {epoch} (patience={args.patience})[/bold red]")
            break

    csv_file.close()

    # Final summary
    console.print(Panel.fit(
        f"[bold green]Training Complete![/bold green]\n\n"
        f"Best validation loss: {best_metrics['val_loss']:.4f} (epoch {best_metrics['epoch']})\n"
        f"Best accuracies:\n"
        f"  op_type: {best_metrics['acc_op']:.1%}\n"
        f"  x: {best_metrics['acc_x']:.1%}\n"
        f"  y: {best_metrics['acc_y']:.1%}\n"
        f"  variant: {best_metrics.get('acc_var', 0):.1%}\n\n"
        f"Checkpoints: {args.checkpoint_dir}\n"
        f"Training log: {log_path}",
        border_style="green"
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Decision Transformer v2')

    # Data
    parser.add_argument('--data_path', default='model/decision_transformer/effective_dt_data.pkl')
    parser.add_argument('--checkpoint_dir', default='nn_checkpoints/dt_v2')
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)

    # Model
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--context_len', type=int, default=50)
    parser.add_argument('--max_timestep', type=int, default=500)
    parser.add_argument('--max_grid_size', type=int, default=32)
    parser.add_argument('--n_variants', type=int, default=12)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--no_amp', action='store_true')

    args = parser.parse_args()
    if args.no_amp:
        args.use_amp = False

    train(args)
