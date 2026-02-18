"""
FusionNet v2 (CNN+RNN) Training Script with DDP + bf16.

Change 18: Full training configuration update:
- DistributedDataParallel for 4x H100 GPUs
- bf16 autocast for H100 native performance
- Updated hyperparameters (lr=4e-4, warmup=2000, patience=40, epochs=200)
- Checkpoint saving with optimizer/scheduler state for resume
- Staged training support (smoke test, arch validation, short run, full)

Usage (single GPU):
    PYTHONPATH=$(pwd)/src:$PYTHONPATH python src/model/train_fusion.py

Usage (4x GPU DDP via torchrun):
    PYTHONPATH=$(pwd)/src:$PYTHONPATH torchrun --nproc_per_node=4 src/model/train_fusion.py
"""

import os
import sys
import time
import csv
import math
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from fusion_model import FusionNet, compute_ranking_loss, count_parameters
from fusion_dataset import create_train_val_split, build_9ch_batch_gpu

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from tqdm import tqdm

console = Console()


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def is_main_process():
    return get_rank() == 0


def setup_distributed():
    """Initialize DDP if launched via torchrun."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        return local_rank
    return 0


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def create_dashboard_str(epoch, epochs, metrics, best_metrics, lr, gpu_mem_used, gpu_mem_total,
                         patience_counter, patience_max, world_size):
    """Create simple text dashboard (Rich Live doesn't work well with DDP)."""
    train = metrics.get('train', {})
    val = metrics.get('val', {})
    lines = [
        f"{'='*70}",
        f"  FusionNet v4 (Ranking) | Epoch {epoch}/{epochs} | "
        f"LR: {lr:.2e} | GPUs: {world_size}",
        f"  GPU: {gpu_mem_used:.1f}/{gpu_mem_total:.1f} GB | "
        f"Patience: {patience_counter}/{patience_max}",
        f"{'='*70}",
        f"  TRAIN | loss={train.get('loss', 0):.4f} "
        f"pos={train.get('pos_loss', 0):.4f} "
        f"act={train.get('act_loss', 0):.4f}",
        f"  VAL   | loss={val.get('loss', 0):.4f} "
        f"recall@5={val.get('recall_at_5', 0):.4f} "
        f"recall@10={val.get('recall_at_10', 0):.4f} "
        f"rank_med={val.get('pos_rank_median', 0):.1f}",
        f"        | act@gt={val.get('act_acc_at_gt', 0):.4f} "
        f"violation={val.get('margin_violation_rate', 0):.4f}",
        f"  BEST  | epoch={best_metrics.get('epoch', 0)} "
        f"loss={best_metrics.get('val_loss', float('inf')):.4f} "
        f"recall@5={best_metrics.get('recall_at_5', 0):.4f} "
        f"recall@10={best_metrics.get('recall_at_10', 0):.4f}",
        f"  Time: {metrics.get('epoch_time', '?')}s",
        f"{'='*70}",
    ]
    return '\n'.join(lines)


def validate(model, val_loader, device, margin=1.0, n_hard_neg=32, use_amp=True, boundary_dilation=1):
    model.eval()
    total_loss = 0
    total_pos_loss = 0
    total_act_loss = 0
    n_batches = 0

    recall_at_5 = 0
    recall_at_10 = 0
    pos_rank_sum = 0
    act_correct_at_gt = 0
    violation_count = 0
    neg_count = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            state_4ch = batch['state_4ch'].to(device, non_blocking=True)
            states, boundary_masks = build_9ch_batch_gpu(
                state_4ch,
                batch['grid_h'].to(device, non_blocking=True),
                batch['grid_w'].to(device, non_blocking=True),
                batch['initial_crossings'].to(device, non_blocking=True),
                boundary_dilation=boundary_dilation)
            target_x = batch['target_x'].to(device, non_blocking=True)
            target_y = batch['target_y'].to(device, non_blocking=True)
            target_action = batch['target_action'].to(device, non_blocking=True)
            hist_act = batch['history_actions'].to(device, non_blocking=True)
            hist_py = batch['history_positions_y'].to(device, non_blocking=True)
            hist_px = batch['history_positions_x'].to(device, non_blocking=True)
            hist_cb = batch['history_crossings_before'].to(device, non_blocking=True)
            hist_ca = batch['history_crossings_after'].to(device, non_blocking=True)
            hist_mask = batch['history_mask'].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_amp):
                pos_logits, act_logits = model(
                    states, hist_act, hist_py, hist_px, hist_cb, hist_ca, hist_mask
                )
                pos_logits_f = pos_logits.float()
                act_logits_f = act_logits.float()
                loss, pos_loss, act_loss = compute_ranking_loss(
                    pos_logits_f, act_logits_f, target_y, target_x, target_action,
                    boundary_masks, margin=margin, n_hard_neg=n_hard_neg,
                )

            total_loss += loss.item()
            total_pos_loss += pos_loss.item()
            total_act_loss += act_loss.item()
            n_batches += 1

            B, K, H, W = pos_logits.shape
            batch_idx = torch.arange(B, device=device)
            mask_flat = boundary_masks.reshape(B, -1).bool()
            has_boundary = mask_flat.any(dim=1)

            if not has_boundary.any():
                continue

            # --- Ranking metrics (K=1) ---
            scores = pos_logits_f[:, 0].reshape(B, -1)  # [B, H*W]
            scores_masked = scores.masked_fill(~mask_flat, float('-inf'))

            target_flat = target_y * W + target_x
            s_pos = scores[batch_idx, target_flat]  # [B]

            # Rank of positive among boundary positions (1-indexed, lower=better)
            rank = (scores_masked > s_pos.unsqueeze(1)).float().sum(1) + 1  # [B]

            valid = has_boundary
            recall_at_5 += ((rank <= 5) & valid).sum().item()
            recall_at_10 += ((rank <= 10) & valid).sum().item()
            pos_rank_sum += (rank * valid.float()).sum().item()

            # Action accuracy at GT position
            act_at_gt = act_logits_f[batch_idx, :, target_y, target_x].argmax(dim=1)
            act_gt_ok = (act_at_gt == target_action) & valid
            act_correct_at_gt += act_gt_ok.sum().item()

            # Margin violation rate
            neg_mask = mask_flat.clone()
            neg_mask[batch_idx, target_flat] = False
            violations = ((scores - s_pos.unsqueeze(1)) > 0) & neg_mask
            violation_count += violations.float().sum().item()
            neg_count += neg_mask.float().sum().item()

            total_samples += valid.sum().item()

    n = max(n_batches, 1)
    s = max(total_samples, 1)
    return {
        'loss': total_loss / n,
        'pos_loss': total_pos_loss / n,
        'act_loss': total_act_loss / n,
        'recall_at_5': recall_at_5 / s,
        'recall_at_10': recall_at_10 / s,
        'pos_rank_median': pos_rank_sum / s,
        'act_acc_at_gt': act_correct_at_gt / s,
        'margin_violation_rate': violation_count / max(neg_count, 1),
    }


def train(args):
    # Setup DDP
    local_rank = setup_distributed()
    world_size = get_world_size()
    rank = get_rank()

    if world_size > 1:
        device = torch.device(f'cuda:{local_rank}')
    else:
        assert torch.cuda.is_available(), "CUDA required"
        device = torch.device('cuda')

    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem_total = torch.cuda.get_device_properties(device).total_memory / 1e9

    if is_main_process():
        console.print(Panel.fit(
            f"[bold cyan]FusionNet v2 (CNN+RNN+FiLM) Training[/bold cyan]\n"
            f"Device: {device} ({gpu_name})\n"
            f"World size: {world_size}\n"
            f"Data: {args.data_path}\n"
            f"Checkpoints: {args.checkpoint_dir}\n"
            f"bf16: {args.use_amp}\n"
            f"Epochs: {args.epochs} | Patience: {args.patience}",
            border_style="cyan"
        ))

    # Load dataset
    if is_main_process():
        console.print("\n[bold]Loading dataset...[/bold]")

    train_ds, val_ds = create_train_val_split(
        args.data_path,
        val_ratio=args.val_split,
        seed=42,
        augment=not args.no_augment,
    )

    if is_main_process():
        console.print(f"  Train: [green]{len(train_ds)}[/green] samples")
        console.print(f"  Val:   [green]{len(val_ds)}[/green] samples")

    # Samplers for DDP
    train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    # Create model
    model = FusionNet(
        in_channels=args.in_channels,
        base_features=args.base_features,
        n_hypotheses=args.n_hypotheses,
        max_history=train_ds.max_history,
        rnn_hidden=args.rnn_hidden,
        rnn_layers=args.rnn_layers,
        max_grid_size=args.max_grid_size,
        rnn_dropout=args.rnn_dropout,
    ).to(device)

    if is_main_process():
        n_params = count_parameters(model)
        console.print(f"\n[bold]FusionNet v2[/bold]")
        console.print(f"  Parameters: [green]{n_params:,}[/green]")
        console.print(f"  in_channels: {args.in_channels}")
        console.print(f"  base_features: {args.base_features}")
        console.print(f"  max_grid_size: {args.max_grid_size}")

    # Wrap in DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Resume from checkpoint
    start_epoch = 1
    global_step = 0
    best_val_loss = float('inf')
    best_metrics = {
        'val_loss': float('inf'), 'epoch': 0,
        'recall_at_5': 0, 'recall_at_10': 0,
        'pos_rank_median': 999, 'act_at_gt': 0,
    }

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    total_steps = len(train_loader) * args.epochs

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume
    if args.resume and os.path.exists(args.resume):
        if is_main_process():
            console.print(f"\n[bold]Resuming from {args.resume}...[/bold]")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict']
        if world_size > 1 and not any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        elif world_size == 1 and any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        global_step = ckpt.get('global_step', 0)
        best_val_loss = ckpt.get('val_loss', float('inf'))
        if 'val_metrics' in ckpt:
            vm = ckpt['val_metrics']
            best_metrics = {
                'val_loss': best_val_loss,
                'epoch': ckpt.get('epoch', 0),
                'recall_at_5': vm.get('recall_at_5', 0),
                'recall_at_10': vm.get('recall_at_10', 0),
                'pos_rank_median': vm.get('pos_rank_median', 999),
                'act_at_gt': vm.get('act_at_gt', vm.get('act_acc_at_gt', 0)),
            }
        if is_main_process():
            console.print(f"  Resumed from epoch {start_epoch - 1}, best_val_loss={best_val_loss:.4f}")

    # GradScaler not needed for bf16 on H100, but we keep autocast
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # CSV logger
    csv_file = None
    csv_writer = None
    if is_main_process():
        log_path = os.path.join(args.checkpoint_dir, 'training_log.csv')
        csv_file = open(log_path, 'a' if args.resume else 'w', newline='')
        csv_writer = csv.writer(csv_file)
        if not args.resume:
            csv_writer.writerow([
                'epoch', 'train_loss', 'train_pos_loss', 'train_act_loss',
                'val_loss', 'val_pos_loss', 'val_act_loss',
                'val_recall_at_5', 'val_recall_at_10', 'val_pos_rank_median',
                'val_act_acc_at_gt', 'val_margin_violation_rate',
                'learning_rate', 'epoch_time'
            ])

    patience_counter = 0

    if is_main_process():
        console.print(f"\n[bold]Starting training for {args.epochs} epochs...[/bold]")
        console.print(f"  Batches/epoch: {len(train_loader)}")
        console.print(f"  Total steps: {total_steps:,}")
        console.print(f"  Effective batch size: {args.batch_size * world_size}")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # ---- Train ----
        model.train()
        epoch_loss = 0
        epoch_pos_loss = 0
        epoch_act_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}",
                     leave=True, disable=not is_main_process(),
                     file=sys.stdout, mininterval=5.0)
        for batch in pbar:
            # GPU preprocessing: build 9ch + boundary mask from 4ch
            state_4ch = batch['state_4ch'].to(device, non_blocking=True)
            states, boundary_masks = build_9ch_batch_gpu(
                state_4ch,
                batch['grid_h'].to(device, non_blocking=True),
                batch['grid_w'].to(device, non_blocking=True),
                batch['initial_crossings'].to(device, non_blocking=True),
                boundary_dilation=args.boundary_dilation)
            target_x = batch['target_x'].to(device, non_blocking=True)
            target_y = batch['target_y'].to(device, non_blocking=True)
            target_action = batch['target_action'].to(device, non_blocking=True)
            hist_act = batch['history_actions'].to(device, non_blocking=True)
            hist_py = batch['history_positions_y'].to(device, non_blocking=True)
            hist_px = batch['history_positions_x'].to(device, non_blocking=True)
            hist_cb = batch['history_crossings_before'].to(device, non_blocking=True)
            hist_ca = batch['history_crossings_after'].to(device, non_blocking=True)
            hist_mask = batch['history_mask'].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_amp):
                pos_logits, act_logits = model(
                    states, hist_act, hist_py, hist_px, hist_cb, hist_ca, hist_mask
                )
                # Loss in float32
                loss, pos_loss, act_loss = compute_ranking_loss(
                    pos_logits.float(), act_logits.float(),
                    target_y, target_x, target_action, boundary_masks,
                    margin=args.margin,
                    n_hard_neg=args.n_hard_neg,
                )

            if torch.isnan(loss) or torch.isinf(loss):
                if is_main_process():
                    console.print(f"[red]WARNING: NaN/Inf loss at step {global_step}, skipping[/red]")
                    console.print(f"  pos={pos_loss.item()}, act={act_loss.item()}")
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_loss += loss.item()
            epoch_pos_loss += pos_loss.item()
            epoch_act_loss += act_loss.item()
            n_batches += 1

            avg_loss = epoch_loss / n_batches
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}",
                             pos=f"{pos_loss.item():.3f}", act=f"{act_loss.item():.3f}")

            # Mid-epoch log every 500 batches
            if is_main_process() and n_batches % 500 == 0:
                elapsed = time.time() - epoch_start
                it_per_sec = n_batches / elapsed
                print(f"  [{epoch}/{args.epochs}] batch {n_batches}/{len(train_loader)} | "
                      f"loss={avg_loss:.4f} pos={epoch_pos_loss/n_batches:.3f} "
                      f"act={epoch_act_loss/n_batches:.3f} | "
                      f"{it_per_sec:.1f} it/s | {elapsed:.0f}s elapsed",
                      flush=True)

        # ---- Validate ----
        val_metrics = validate(model, val_loader, device,
                               margin=args.margin,
                               n_hard_neg=args.n_hard_neg,
                               use_amp=args.use_amp,
                               boundary_dilation=args.boundary_dilation)

        epoch_time = time.time() - epoch_start
        avg_train_loss = epoch_loss / max(n_batches, 1)
        avg_train_pos = epoch_pos_loss / max(n_batches, 1)
        avg_train_act = epoch_act_loss / max(n_batches, 1)
        current_lr = scheduler.get_last_lr()[0]
        gpu_mem_used = torch.cuda.memory_allocated(device) / 1e9

        if is_main_process():
            val_loss = val_metrics['loss']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_metrics = {
                    'val_loss': val_loss, 'epoch': epoch,
                    'recall_at_5': val_metrics['recall_at_5'],
                    'recall_at_10': val_metrics['recall_at_10'],
                    'pos_rank_median': val_metrics['pos_rank_median'],
                    'act_at_gt': val_metrics['act_acc_at_gt'],
                }
                # Save best
                raw_model = model.module if hasattr(model, 'module') else model
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                    'args': vars(args),
                }, os.path.join(args.checkpoint_dir, 'best.pt'))
                console.print(f"  [bold green]Best model saved at epoch {epoch} "
                              f"(val_loss={val_loss:.4f})[/bold green]")
            else:
                patience_counter += 1

            # Periodic checkpoint
            if epoch % 10 == 0:
                raw_model = model.module if hasattr(model, 'module') else model
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_metrics': val_metrics,
                    'args': vars(args),
                }, os.path.join(args.checkpoint_dir, f'epoch_{epoch}.pt'))

            metrics = {
                'train': {'loss': avg_train_loss, 'pos_loss': avg_train_pos, 'act_loss': avg_train_act},
                'val': {
                    'loss': val_loss, 'pos_loss': val_metrics['pos_loss'],
                    'act_loss': val_metrics['act_loss'],
                    'recall_at_5': val_metrics['recall_at_5'],
                    'recall_at_10': val_metrics['recall_at_10'],
                    'pos_rank_median': val_metrics['pos_rank_median'],
                    'act_acc_at_gt': val_metrics['act_acc_at_gt'],
                    'margin_violation_rate': val_metrics['margin_violation_rate'],
                },
                'epoch_time': f'{epoch_time:.1f}',
            }

            dashboard = create_dashboard_str(
                epoch, args.epochs, metrics, best_metrics,
                current_lr, gpu_mem_used, gpu_mem_total,
                patience_counter, args.patience, world_size,
            )
            console.print(dashboard)

            csv_writer.writerow([
                epoch, avg_train_loss, avg_train_pos, avg_train_act,
                val_loss, val_metrics['pos_loss'], val_metrics['act_loss'],
                val_metrics['recall_at_5'], val_metrics['recall_at_10'],
                val_metrics['pos_rank_median'],
                val_metrics['act_acc_at_gt'], val_metrics['margin_violation_rate'],
                current_lr, epoch_time,
            ])
            csv_file.flush()

            if patience_counter >= args.patience:
                console.print(f"\n[bold yellow]Early stopping at epoch {epoch} "
                              f"(no improvement for {args.patience} epochs)[/bold yellow]")
                break

    if is_main_process():
        csv_file.close()
        console.print(Panel.fit(
            f"[bold green]Training Complete[/bold green]\n"
            f"Best epoch: {best_metrics['epoch']}\n"
            f"Best val loss: {best_metrics['val_loss']:.4f}\n"
            f"Recall@5: {best_metrics['recall_at_5']:.2%}\n"
            f"Recall@10: {best_metrics['recall_at_10']:.2%}\n"
            f"Position rank (avg): {best_metrics['pos_rank_median']:.1f}\n"
            f"Action accuracy @ GT pos: {best_metrics['act_at_gt']:.2%}\n"
            f"Checkpoint: {args.checkpoint_dir}/best.pt",
            border_style="green"
        ))

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description='FusionNet v2 Training')
    parser.add_argument('--data_path', type=str, default='checkpoints/fusion_data.pt')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64, help='Per-GPU batch size')
    parser.add_argument('--learning_rate', type=float, default=4e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--in_channels', type=int, default=9)
    parser.add_argument('--base_features', type=int, default=48)
    parser.add_argument('--max_grid_size', type=int, default=128)
    parser.add_argument('--boundary_dilation', type=int, default=2,
                        help='Dilation radius for boundary mask. 2 covers all valid '
                             'operation positions (3x3 subgrid reach). Was 1 before v5.')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for ranking hinge loss')
    parser.add_argument('--n_hard_neg', type=int, default=32, help='Number of hard negatives per sample')
    parser.add_argument('--n_hypotheses', type=int, default=1)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--rnn_hidden', type=int, default=192)
    parser.add_argument('--rnn_layers', type=int, default=2)
    parser.add_argument('--rnn_dropout', type=float, default=0.15)
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use bf16 autocast (default: True)')
    parser.add_argument('--no_amp', action='store_true', help='Disable bf16 autocast')

    args = parser.parse_args()
    if args.no_amp:
        args.use_amp = False

    train(args)


if __name__ == '__main__':
    main()
