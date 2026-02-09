"""
FusionNet (CNN+RNN) Training Script with Rich UI Dashboard.

Features:
- Rich live dashboard with 3-column layout (Train / Validation / Best)
- tqdm progress bar per epoch for batch-level progress
- FiLM-conditioned fusion of CNN spatial + RNN temporal branches
- AdamW optimizer, cosine annealing with warmup, gradient clipping
- Early stopping on val_loss (patience=15)
- Trajectory-level train/val split with augmentation
- Saves best model + periodic checkpoints
- CSV logging

Usage:
    PYTHONPATH=$(pwd):$PYTHONPATH python model/fusion/train_fusion.py
    PYTHONPATH=$(pwd):$PYTHONPATH python model/fusion/train_fusion.py --epochs 30 --batch_size 128
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

from fusion_model import FusionNet, compute_loss, count_parameters
from fusion_dataset import create_train_val_split

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich import box
from tqdm import tqdm

console = Console()


def create_dashboard(epoch, epochs, metrics, best_metrics, lr, gpu_mem_used, gpu_mem_total,
                     patience_counter, patience_max):
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
        f"[bold cyan]FusionNet (CNN+RNN+FiLM)[/bold cyan] | "
        f"Epoch [bold]{epoch}[/bold]/{epochs} | "
        f"LR: {lr:.2e} | "
        f"GPU: {gpu_mem_used:.1f}/{gpu_mem_total:.1f} GB | "
        f"Patience: {patience_counter}/{patience_max}"
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

    # Train/val gap
    train_loss = metrics.get('train', {}).get('loss', 0)
    val_loss = metrics.get('val', {}).get('loss', 0)
    gap = val_loss - train_loss if train_loss > 0 else 0
    gap_style = "[green]" if gap < 0.5 else "[yellow]" if gap < 1.0 else "[red]"

    footer = Table(box=None, show_header=False)
    footer.add_row(
        f"[dim]Time/epoch: {metrics.get('epoch_time', '?')}s | "
        f"Train/Val gap: {gap_style}{gap:.4f}[/] | "
        f"Overfit: {gap_style}{'No' if gap < 1.0 else 'Risk'}[/][/dim]"
    )
    layout["footer"].update(footer)

    return layout


def validate(model, val_loader, device, pos_weight=10.0, diversity_weight=0.5):
    """Validation with oracle + mean-softmax pooling metrics."""
    model.eval()
    total_loss = 0
    total_pos_loss = 0
    total_act_loss = 0
    total_div_loss = 0
    n_batches = 0

    pos_correct_top1 = 0
    pos_correct_top5 = 0
    oracle_correct_top1 = 0
    act_correct_at_gt = 0
    act_correct_at_pred = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            states = batch['state'].to(device)
            target_x = batch['target_x'].to(device)
            target_y = batch['target_y'].to(device)
            target_action = batch['target_action'].to(device)
            boundary_masks = batch['boundary_mask'].to(device)
            hist_act = batch['history_actions'].to(device)
            hist_py = batch['history_positions_y'].to(device)
            hist_px = batch['history_positions_x'].to(device)
            hist_cb = batch['history_crossings_before'].to(device)
            hist_ca = batch['history_crossings_after'].to(device)
            hist_mask = batch['history_mask'].to(device)

            pos_logits, act_logits = model(
                states, hist_act, hist_py, hist_px, hist_cb, hist_ca, hist_mask
            )
            loss, pos_loss, act_loss, div_loss = compute_loss(
                pos_logits, act_logits, target_y, target_x, target_action,
                boundary_masks, pos_weight=pos_weight, diversity_weight=diversity_weight,
            )

            total_loss += loss.item()
            total_pos_loss += pos_loss.item()
            total_act_loss += act_loss.item()
            total_div_loss += div_loss.item()
            n_batches += 1

            B, K, H, W = pos_logits.shape
            batch_idx = torch.arange(B, device=device)
            mask_flat = boundary_masks.reshape(B, -1).bool()
            has_boundary = mask_flat.any(dim=1)

            if not has_boundary.any():
                continue

            # Mean-softmax pooling across K hypotheses
            pos_flat = pos_logits.reshape(B, K, -1)
            mask_k = mask_flat.unsqueeze(1).expand_as(pos_flat)
            pos_masked_k = pos_flat.masked_fill(~mask_k, float('-inf'))
            probs_k = torch.softmax(pos_masked_k, dim=-1)
            pos_pooled = probs_k.mean(dim=1)

            # Top-1
            top1_flat = pos_pooled.argmax(dim=1)
            pred_y1 = top1_flat // W
            pred_x1 = top1_flat % W
            pos_top1 = (pred_y1 == target_y) & (pred_x1 == target_x) & has_boundary

            # Top-5
            topk_flat = pos_pooled.topk(5, dim=1).indices
            topk_y = topk_flat // W
            topk_x = topk_flat % W
            pos_top5 = (
                (topk_y == target_y.unsqueeze(1)) & (topk_x == target_x.unsqueeze(1))
            ).any(dim=1) & has_boundary

            # Oracle: any hypothesis argmax matches GT
            per_hyp_argmax = pos_masked_k.argmax(dim=-1)
            per_hyp_y = per_hyp_argmax // W
            per_hyp_x = per_hyp_argmax % W
            oracle_hit = (
                (per_hyp_y == target_y.unsqueeze(1)) & (per_hyp_x == target_x.unsqueeze(1))
            ).any(dim=1) & has_boundary

            # Action @ GT position
            act_at_gt = act_logits[batch_idx, :, target_y, target_x].argmax(dim=1)
            act_gt_ok = (act_at_gt == target_action) & has_boundary

            # End-to-end: action @ predicted position
            act_at_pred = act_logits[batch_idx, :, pred_y1, pred_x1].argmax(dim=1)
            e2e_ok = (act_at_pred == target_action) & pos_top1

            pos_correct_top1 += pos_top1.sum().item()
            pos_correct_top5 += pos_top5.sum().item()
            oracle_correct_top1 += oracle_hit.sum().item()
            act_correct_at_gt += act_gt_ok.sum().item()
            act_correct_at_pred += e2e_ok.sum().item()
            total_samples += has_boundary.sum().item()

    n = max(n_batches, 1)
    s = max(total_samples, 1)
    return {
        'loss': total_loss / n,
        'pos_loss': total_pos_loss / n,
        'act_loss': total_act_loss / n,
        'div_loss': total_div_loss / n,
        'pos_acc_top1': pos_correct_top1 / s,
        'pos_acc_top5': pos_correct_top5 / s,
        'oracle_acc_top1': oracle_correct_top1 / s,
        'act_acc_at_gt': act_correct_at_gt / s,
        'act_acc_e2e': act_correct_at_pred / s,
    }


def train(args):
    assert torch.cuda.is_available(), "CUDA is required. No GPU detected."
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9

    console.print(Panel.fit(
        f"[bold cyan]FusionNet (CNN+RNN+FiLM) Training[/bold cyan]\n"
        f"Device: {device} ({gpu_name})\n"
        f"Data: {args.data_path}\n"
        f"Checkpoints: {args.checkpoint_dir}\n"
        f"Early stopping patience: {args.patience}",
        border_style="cyan"
    ))

    # Load dataset
    console.print("\n[bold]Loading dataset...[/bold]")
    train_ds, val_ds = create_train_val_split(
        args.data_path,
        val_ratio=args.val_split,
        boundary_dilation=args.boundary_dilation,
        seed=42,
        augment=not args.no_augment,
    )
    console.print(f"  Train: [green]{len(train_ds)}[/green] effective operation samples")
    console.print(f"  Val:   [green]{len(val_ds)}[/green] effective operation samples")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Create model
    console.print("\n[bold]Creating FusionNet...[/bold]")
    model = FusionNet(
        in_channels=5, base_features=args.base_features,
        n_hypotheses=args.n_hypotheses,
        max_history=train_ds.max_history,
        rnn_hidden=args.rnn_hidden,
        rnn_layers=args.rnn_layers,
        rnn_dropout=args.rnn_dropout,
    ).to(device)
    n_params = count_parameters(model)
    console.print(f"  Parameters: [green]{n_params:,}[/green]")
    console.print(f"  CNN base features: {args.base_features}")
    console.print(f"  RNN hidden: {args.rnn_hidden}, layers: {args.rnn_layers}")
    console.print(f"  Position hypotheses (K): {args.n_hypotheses}")

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
        'val_loss', 'val_pos_loss', 'val_act_loss', 'val_div_loss',
        'val_pos_acc_top1', 'val_pos_acc_top5', 'val_oracle_acc_top1',
        'val_act_acc_at_gt', 'val_act_acc_e2e',
        'learning_rate', 'epoch_time'
    ])

    # Training state
    best_val_loss = float('inf')
    best_metrics = {
        'val_loss': float('inf'), 'epoch': 0,
        'pos_top1': 0, 'pos_top5': 0, 'oracle_top1': 0,
        'act_at_gt': 0, 'act_e2e': 0,
    }
    global_step = 0
    patience_counter = 0

    console.print(f"\n[bold]Starting training for {args.epochs} epochs...[/bold]")
    console.print(f"  Batches/epoch: {len(train_loader)}")
    console.print(f"  Total steps: {total_steps:,}")
    console.print(f"  Early stopping patience: {args.patience}")
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
            hist_act = batch['history_actions'].to(device)
            hist_py = batch['history_positions_y'].to(device)
            hist_px = batch['history_positions_x'].to(device)
            hist_cb = batch['history_crossings_before'].to(device)
            hist_ca = batch['history_crossings_after'].to(device)
            hist_mask = batch['history_mask'].to(device)

            optimizer.zero_grad()

            pos_logits, act_logits = model(
                states, hist_act, hist_py, hist_px, hist_cb, hist_ca, hist_mask
            )
            loss, pos_loss, act_loss, div_loss = compute_loss(
                pos_logits, act_logits, target_y, target_x,
                target_action, boundary_masks,
                pos_weight=args.pos_weight,
                diversity_weight=args.diversity_weight,
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
        val_metrics = validate(model, val_loader, device,
                               pos_weight=args.pos_weight,
                               diversity_weight=args.diversity_weight)

        epoch_time = time.time() - epoch_start
        avg_train_loss = epoch_loss / max(n_batches, 1)
        avg_train_pos = epoch_pos_loss / max(n_batches, 1)
        avg_train_act = epoch_act_loss / max(n_batches, 1)
        current_lr = scheduler.get_last_lr()[0]

        # GPU memory
        gpu_mem_used = torch.cuda.memory_allocated(0) / 1e9

        # Checkpointing
        val_loss = val_metrics['loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_metrics = {
                'val_loss': val_loss,
                'epoch': epoch,
                'pos_top1': val_metrics['pos_acc_top1'],
                'pos_top5': val_metrics['pos_acc_top5'],
                'oracle_top1': val_metrics['oracle_acc_top1'],
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
            console.print(f"  [bold green]Best model saved at epoch {epoch} (val_loss={val_loss:.4f})[/bold green]")
        else:
            patience_counter += 1

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'args': vars(args),
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
                'div_loss': val_metrics['div_loss'],
                'pos_acc_top1': val_metrics['pos_acc_top1'],
                'pos_acc_top5': val_metrics['pos_acc_top5'],
                'oracle_top1': val_metrics['oracle_acc_top1'],
                'act_acc@gt': val_metrics['act_acc_at_gt'],
                'act_acc_e2e': val_metrics['act_acc_e2e'],
            },
            'epoch_time': f'{epoch_time:.1f}',
        }
        dashboard = create_dashboard(
            epoch, args.epochs, metrics, best_metrics,
            current_lr, gpu_mem_used, gpu_mem_total,
            patience_counter, args.patience,
        )
        console.print(dashboard)

        # CSV log
        csv_writer.writerow([
            epoch, avg_train_loss, avg_train_pos, avg_train_act,
            val_loss, val_metrics['pos_loss'], val_metrics['act_loss'],
            val_metrics['div_loss'],
            val_metrics['pos_acc_top1'], val_metrics['pos_acc_top5'],
            val_metrics['oracle_acc_top1'],
            val_metrics['act_acc_at_gt'], val_metrics['act_acc_e2e'],
            current_lr, epoch_time,
        ])
        csv_file.flush()

        # Early stopping
        if patience_counter >= args.patience:
            console.print(f"\n[bold yellow]Early stopping triggered at epoch {epoch} "
                          f"(no improvement for {args.patience} epochs)[/bold yellow]")
            break

    csv_file.close()

    # Final summary
    console.print(Panel.fit(
        f"[bold green]Training Complete[/bold green]\n"
        f"Best epoch: {best_metrics['epoch']}\n"
        f"Best val loss: {best_metrics['val_loss']:.4f}\n"
        f"Position accuracy (top-1): {best_metrics['pos_top1']:.2%}\n"
        f"Position accuracy (top-5): {best_metrics['pos_top5']:.2%}\n"
        f"Oracle accuracy (top-1): {best_metrics['oracle_top1']:.2%}\n"
        f"Action accuracy @ GT pos: {best_metrics['act_at_gt']:.2%}\n"
        f"End-to-end accuracy: {best_metrics['act_e2e']:.2%}\n"
        f"Checkpoint: {args.checkpoint_dir}/best.pt",
        border_style="green"
    ))


def main():
    parser = argparse.ArgumentParser(description='FusionNet (CNN+RNN) Training')
    parser.add_argument('--data_path', type=str,
                        default='model/fusion/fusion_data.pt')
    parser.add_argument('--checkpoint_dir', type=str, default='nn_checkpoints/fusion')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--base_features', type=int, default=48)
    parser.add_argument('--boundary_dilation', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pos_weight', type=float, default=10.0,
                        help='Position loss multiplier')
    parser.add_argument('--n_hypotheses', type=int, default=4,
                        help='Number of WTA position hypotheses')
    parser.add_argument('--diversity_weight', type=float, default=0.5,
                        help='Weight for winner-assignment entropy regularizer')
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable training augmentation')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (epochs without val_loss improvement)')
    parser.add_argument('--rnn_hidden', type=int, default=192,
                        help='RNN hidden size')
    parser.add_argument('--rnn_layers', type=int, default=2,
                        help='Number of GRU layers')
    parser.add_argument('--rnn_dropout', type=float, default=0.15,
                        help='RNN dropout between layers')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
