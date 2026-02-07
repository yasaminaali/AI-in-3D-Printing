"""
OperationNet (U-Net) Training Script with Rich UI Dashboard.

Features:
- Rich live progress tracking with metrics
- Factored loss (position + action) with boundary masking
- Trajectory-level train/val split
- Cosine annealing with warmup
- Checkpoint saving (best + periodic)

Usage:
    PYTHONPATH=$(pwd):$PYTHONPATH python model/unet/train_unet.py
    PYTHONPATH=$(pwd):$PYTHONPATH python model/unet/train_unet.py --epochs 100 --batch_size 64
"""

import os
import sys
import time
import csv
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unet_model import OperationNet, compute_loss, count_parameters
from unet_dataset import create_train_val_split

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from tqdm import tqdm
from rich.layout import Layout
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

    header = Table(box=None, show_header=False)
    header.add_row(
        f"[bold cyan]OperationNet (U-Net)[/bold cyan] | "
        f"Epoch [bold]{epoch}[/bold]/{epochs} | "
        f"LR: {lr:.2e} | "
        f"GPU: {gpu_mem_used:.1f}/{gpu_mem_total:.1f} GB"
    )
    layout["header"].update(header)

    train_table = Table(title="[bold green]Training[/bold green]", box=box.SIMPLE)
    train_table.add_column("Metric", style="dim")
    train_table.add_column("Value", justify="right")
    for k, v in metrics.get('train', {}).items():
        train_table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    layout["train"].update(Panel(train_table))

    val_table = Table(title="[bold yellow]Validation[/bold yellow]", box=box.SIMPLE)
    val_table.add_column("Metric", style="dim")
    val_table.add_column("Value", justify="right")
    for k, v in metrics.get('val', {}).items():
        val_table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    layout["val"].update(Panel(val_table))

    best_table = Table(title="[bold magenta]Best[/bold magenta]", box=box.SIMPLE)
    best_table.add_column("Metric", style="dim")
    best_table.add_column("Value", justify="right")
    for k, v in best_metrics.items():
        best_table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    layout["best"].update(Panel(best_table))

    footer = Table(box=None, show_header=False)
    footer.add_row(
        f"[dim]Time/epoch: {metrics.get('epoch_time', '?')}s[/dim]"
    )
    layout["footer"].update(footer)

    return layout


def validate(model, val_loader, device):
    """Run validation and return metrics."""
    model.eval()
    total_loss = 0
    total_pos_loss = 0
    total_act_loss = 0
    n_batches = 0

    # Accuracy tracking
    pos_correct_top1 = 0
    pos_correct_top5 = 0
    act_correct_at_gt = 0  # action accuracy at ground-truth position
    act_correct_at_pred = 0  # action accuracy at predicted position (end-to-end)
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            states = batch['state'].to(device)
            target_x = batch['target_x'].to(device)
            target_y = batch['target_y'].to(device)
            target_action = batch['target_action'].to(device)
            boundary_masks = batch['boundary_mask'].to(device)

            pos_logits, act_logits = model(states)
            loss, pos_loss, act_loss = compute_loss(
                pos_logits, act_logits, target_y, target_x, target_action, boundary_masks
            )

            total_loss += loss.item()
            total_pos_loss += pos_loss.item()
            total_act_loss += act_loss.item()
            n_batches += 1

            # Per-sample accuracy
            B = states.size(0)
            for i in range(B):
                mask = boundary_masks[i]
                bpos = mask.nonzero(as_tuple=False)
                if len(bpos) == 0:
                    continue

                total_samples += 1
                ty, tx = target_y[i].item(), target_x[i].item()

                # Position accuracy (top-1 and top-5 among boundary positions)
                pos_scores = pos_logits[i, 0, bpos[:, 0], bpos[:, 1]]
                topk_k = min(5, len(pos_scores))

                top1_idx = pos_scores.argmax()
                pred_y1, pred_x1 = bpos[top1_idx, 0].item(), bpos[top1_idx, 1].item()
                if pred_y1 == ty and pred_x1 == tx:
                    pos_correct_top1 += 1

                topk_indices = pos_scores.topk(topk_k).indices
                for ki in topk_indices:
                    py, px = bpos[ki, 0].item(), bpos[ki, 1].item()
                    if py == ty and px == tx:
                        pos_correct_top5 += 1
                        break

                # Action accuracy at ground-truth position
                act_at_gt = act_logits[i, :, ty, tx].argmax().item()
                if act_at_gt == target_action[i].item():
                    act_correct_at_gt += 1

                # Action accuracy at predicted position (end-to-end metric)
                act_at_pred = act_logits[i, :, pred_y1, pred_x1].argmax().item()
                if act_at_pred == target_action[i].item() and pred_y1 == ty and pred_x1 == tx:
                    act_correct_at_pred += 1

    n = max(n_batches, 1)
    s = max(total_samples, 1)
    return {
        'loss': total_loss / n,
        'pos_loss': total_pos_loss / n,
        'act_loss': total_act_loss / n,
        'pos_acc_top1': pos_correct_top1 / s,
        'pos_acc_top5': pos_correct_top5 / s,
        'act_acc_at_gt': act_correct_at_gt / s,
        'act_acc_e2e': act_correct_at_pred / s,
    }


def train(args):
    assert torch.cuda.is_available(), "CUDA is required. No GPU detected."
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9

    console.print(Panel.fit(
        f"[bold cyan]OperationNet (U-Net) Training[/bold cyan]\n"
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
        boundary_dilation=args.boundary_dilation,
        seed=42,
    )
    console.print(f"  Train: [green]{len(train_ds)}[/green] effective operation samples")
    console.print(f"  Val: [green]{len(val_ds)}[/green] effective operation samples")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Create model
    console.print("\n[bold]Creating OperationNet...[/bold]")
    model = OperationNet(in_channels=5, base_features=args.base_features).to(device)
    n_params = count_parameters(model)
    console.print(f"  Parameters: [green]{n_params:,}[/green]")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # Scheduler: warmup + cosine annealing
    total_steps = len(train_loader) * args.epochs

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Checkpoint dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # CSV logger
    log_path = os.path.join(args.checkpoint_dir, 'training_log.csv')
    csv_file = open(log_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'epoch', 'train_loss', 'train_pos_loss', 'train_act_loss',
        'val_loss', 'val_pos_loss', 'val_act_loss',
        'val_pos_acc_top1', 'val_pos_acc_top5',
        'val_act_acc_at_gt', 'val_act_acc_e2e',
        'learning_rate', 'epoch_time'
    ])

    # Training state
    best_val_loss = float('inf')
    best_metrics = {
        'val_loss': float('inf'), 'epoch': 0,
        'pos_top1': 0, 'pos_top5': 0,
        'act_at_gt': 0, 'act_e2e': 0,
    }
    global_step = 0

    console.print(f"\n[bold]Starting training for {args.epochs} epochs...[/bold]")
    console.print(f"  Batches/epoch: {len(train_loader)}")
    console.print(f"  Total steps: {total_steps:,}")
    console.print()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # ---- Train ----
        model.train()
        epoch_loss = 0
        epoch_pos_loss = 0
        epoch_act_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            states = batch['state'].to(device)
            target_x = batch['target_x'].to(device)
            target_y = batch['target_y'].to(device)
            target_action = batch['target_action'].to(device)
            boundary_masks = batch['boundary_mask'].to(device)

            optimizer.zero_grad()

            pos_logits, act_logits = model(states)
            loss, pos_loss, act_loss = compute_loss(
                pos_logits, act_logits, target_y, target_x,
                target_action, boundary_masks
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            scheduler.step()
            global_step += 1

            epoch_loss += loss.item()
            epoch_pos_loss += pos_loss.item()
            epoch_act_loss += act_loss.item()
            n_batches += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ---- Validate ----
        val_metrics = validate(model, val_loader, device)

        epoch_time = time.time() - epoch_start
        avg_train_loss = epoch_loss / max(n_batches, 1)
        avg_train_pos = epoch_pos_loss / max(n_batches, 1)
        avg_train_act = epoch_act_loss / max(n_batches, 1)
        current_lr = scheduler.get_last_lr()[0]

        # GPU memory
        gpu_mem_used = torch.cuda.memory_allocated(0) / 1e9 if device.type == 'cuda' else 0

        # Checkpointing
        val_loss = val_metrics['loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                'val_loss': val_loss,
                'epoch': epoch,
                'pos_top1': val_metrics['pos_acc_top1'],
                'pos_top5': val_metrics['pos_acc_top5'],
                'act_at_gt': val_metrics['act_acc_at_gt'],
                'act_e2e': val_metrics['act_acc_e2e'],
            }
            # Save best checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'args': vars(args),
            }, os.path.join(args.checkpoint_dir, 'best.pt'))

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
            }, os.path.join(args.checkpoint_dir, f'epoch_{epoch}.pt'))

        # Dashboard
        metrics = {
            'train': {
                'loss': avg_train_loss,
                'pos_loss': avg_train_pos,
                'act_loss': avg_train_act,
            },
            'val': {
                'loss': val_loss,
                'pos_loss': val_metrics['pos_loss'],
                'act_loss': val_metrics['act_loss'],
                'pos_acc_top1': val_metrics['pos_acc_top1'],
                'pos_acc_top5': val_metrics['pos_acc_top5'],
                'act_acc@gt': val_metrics['act_acc_at_gt'],
                'act_acc_e2e': val_metrics['act_acc_e2e'],
            },
            'epoch_time': f'{epoch_time:.1f}',
        }
        dashboard = create_dashboard(
            epoch, args.epochs, metrics, best_metrics,
            current_lr, gpu_mem_used, gpu_mem_total
        )
        console.print(dashboard)

        # CSV log
        csv_writer.writerow([
            epoch, avg_train_loss, avg_train_pos, avg_train_act,
            val_loss, val_metrics['pos_loss'], val_metrics['act_loss'],
            val_metrics['pos_acc_top1'], val_metrics['pos_acc_top5'],
            val_metrics['act_acc_at_gt'], val_metrics['act_acc_e2e'],
            current_lr, epoch_time,
        ])
        csv_file.flush()

    csv_file.close()

    # Final summary
    console.print(Panel.fit(
        f"[bold green]Training Complete[/bold green]\n"
        f"Best epoch: {best_metrics['epoch']}\n"
        f"Best val loss: {best_metrics['val_loss']:.4f}\n"
        f"Position accuracy (top-1): {best_metrics['pos_top1']:.2%}\n"
        f"Position accuracy (top-5): {best_metrics['pos_top5']:.2%}\n"
        f"Action accuracy @ GT pos: {best_metrics['act_at_gt']:.2%}\n"
        f"End-to-end accuracy: {best_metrics['act_e2e']:.2%}\n"
        f"Checkpoint: {args.checkpoint_dir}/best.pt",
        border_style="green"
    ))


def main():
    parser = argparse.ArgumentParser(description='OperationNet (U-Net) Training')
    parser.add_argument('--data_path', type=str,
                        default='model/unet/unet_data.pt')
    parser.add_argument('--checkpoint_dir', type=str, default='nn_checkpoints/unet')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--base_features', type=int, default=24)
    parser.add_argument('--boundary_dilation', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
