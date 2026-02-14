"""
FusionNet v2 (CNN+RNN) Training Script with DDP + bf16.

Change 18: Full training configuration update:
- DistributedDataParallel for 4x H100 GPUs
- bf16 autocast for H100 native performance
- Updated hyperparameters (lr=4e-4, warmup=2000, patience=40, epochs=200)
- Checkpoint saving with optimizer/scheduler state for resume
- Staged training support (smoke test, arch validation, short run, full)

Usage (single GPU):
    PYTHONPATH=$(pwd):$PYTHONPATH python FusionModel/fusion/train_fusion.py

Usage (4x GPU DDP via torchrun):
    PYTHONPATH=$(pwd):$PYTHONPATH torchrun --nproc_per_node=4 FusionModel/fusion/train_fusion.py
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fusion_model import FusionNet, compute_loss, count_parameters
from fusion_dataset import create_train_val_split

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
        f"  FusionNet v2 (CNN+RNN+FiLM) | Epoch {epoch}/{epochs} | "
        f"LR: {lr:.2e} | GPUs: {world_size}",
        f"  GPU: {gpu_mem_used:.1f}/{gpu_mem_total:.1f} GB | "
        f"Patience: {patience_counter}/{patience_max}",
        f"{'='*70}",
        f"  TRAIN | loss={train.get('loss', 0):.4f} "
        f"pos={train.get('pos_loss', 0):.4f} "
        f"act={train.get('act_loss', 0):.4f}",
        f"  VAL   | loss={val.get('loss', 0):.4f} "
        f"pos_top1={val.get('pos_acc_top1', 0):.4f} "
        f"pos_top5={val.get('pos_acc_top5', 0):.4f} "
        f"oracle={val.get('oracle_acc_top1', 0):.4f}",
        f"        | act@gt={val.get('act_acc_at_gt', 0):.4f} "
        f"e2e={val.get('act_acc_e2e', 0):.4f}",
        f"  BEST  | epoch={best_metrics.get('epoch', 0)} "
        f"loss={best_metrics.get('val_loss', float('inf')):.4f} "
        f"pos_top1={best_metrics.get('pos_top1', 0):.4f} "
        f"oracle={best_metrics.get('oracle_top1', 0):.4f}",
        f"  Time: {metrics.get('epoch_time', '?')}s",
        f"{'='*70}",
    ]
    return '\n'.join(lines)


def validate(model, val_loader, device, pos_weight=5.0, diversity_weight=0.5, use_amp=True):
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

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_amp):
                pos_logits, act_logits = model(
                    states, hist_act, hist_py, hist_px, hist_cb, hist_ca, hist_mask
                )
                # Compute loss in float32 for stability
                pos_logits_f = pos_logits.float()
                act_logits_f = act_logits.float()
                loss, pos_loss, act_loss, div_loss = compute_loss(
                    pos_logits_f, act_logits_f, target_y, target_x, target_action,
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

            # Per-hypothesis: use float32 logits
            pos_flat = pos_logits_f.reshape(B, K, -1)
            mask_k = mask_flat.unsqueeze(1).expand_as(pos_flat)
            pos_masked_k = pos_flat.masked_fill(~mask_k, -1e9)

            # Per-hypothesis candidates (match inference approach)
            probs_k = torch.softmax(pos_masked_k, dim=-1)
            pos_pooled = probs_k.mean(dim=1)

            top1_flat = pos_pooled.argmax(dim=1)
            pred_y1 = top1_flat // W
            pred_x1 = top1_flat % W
            pos_top1 = (pred_y1 == target_y) & (pred_x1 == target_x) & has_boundary

            topk_flat = pos_pooled.topk(min(5, pos_pooled.shape[1]), dim=1).indices
            topk_y = topk_flat // W
            topk_x = topk_flat % W
            pos_top5 = (
                (topk_y == target_y.unsqueeze(1)) & (topk_x == target_x.unsqueeze(1))
            ).any(dim=1) & has_boundary

            per_hyp_argmax = pos_masked_k.argmax(dim=-1)
            per_hyp_y = per_hyp_argmax // W
            per_hyp_x = per_hyp_argmax % W
            oracle_hit = (
                (per_hyp_y == target_y.unsqueeze(1)) & (per_hyp_x == target_x.unsqueeze(1))
            ).any(dim=1) & has_boundary

            act_at_gt = act_logits_f[batch_idx, :, target_y, target_x].argmax(dim=1)
            act_gt_ok = (act_at_gt == target_action) & has_boundary

            act_at_pred = act_logits_f[batch_idx, :, pred_y1, pred_x1].argmax(dim=1)
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
        boundary_dilation=args.boundary_dilation,
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
        'pos_top1': 0, 'pos_top5': 0, 'oracle_top1': 0,
        'act_at_gt': 0, 'act_e2e': 0,
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
                'pos_top1': vm.get('pos_top1', 0),
                'pos_top5': vm.get('pos_top5', 0),
                'oracle_top1': vm.get('oracle_top1', 0),
                'act_at_gt': vm.get('act_at_gt', 0),
                'act_e2e': vm.get('act_e2e', 0),
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
                'val_loss', 'val_pos_loss', 'val_act_loss', 'val_div_loss',
                'val_pos_acc_top1', 'val_pos_acc_top5', 'val_oracle_acc_top1',
                'val_act_acc_at_gt', 'val_act_acc_e2e',
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

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_amp):
                pos_logits, act_logits = model(
                    states, hist_act, hist_py, hist_px, hist_cb, hist_ca, hist_mask
                )
                # Loss in float32
                loss, pos_loss, act_loss, div_loss = compute_loss(
                    pos_logits.float(), act_logits.float(),
                    target_y, target_x, target_action, boundary_masks,
                    pos_weight=args.pos_weight,
                    diversity_weight=args.diversity_weight,
                )

            if torch.isnan(loss) or torch.isinf(loss):
                if is_main_process():
                    console.print(f"[red]WARNING: NaN/Inf loss at step {global_step}, skipping batch[/red]")
                    console.print(f"  pos_loss={pos_loss.item()}, act_loss={act_loss.item()}")
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
                               pos_weight=args.pos_weight,
                               diversity_weight=args.diversity_weight,
                               use_amp=args.use_amp)

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
                    'pos_top1': val_metrics['pos_acc_top1'],
                    'pos_top5': val_metrics['pos_acc_top5'],
                    'oracle_top1': val_metrics['oracle_acc_top1'],
                    'act_at_gt': val_metrics['act_acc_at_gt'],
                    'act_e2e': val_metrics['act_acc_e2e'],
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
                    'act_loss': val_metrics['act_loss'], 'div_loss': val_metrics['div_loss'],
                    'pos_acc_top1': val_metrics['pos_acc_top1'],
                    'pos_acc_top5': val_metrics['pos_acc_top5'],
                    'oracle_acc_top1': val_metrics['oracle_acc_top1'],
                    'act_acc_at_gt': val_metrics['act_acc_at_gt'],
                    'act_acc_e2e': val_metrics['act_acc_e2e'],
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
                val_metrics['div_loss'],
                val_metrics['pos_acc_top1'], val_metrics['pos_acc_top5'],
                val_metrics['oracle_acc_top1'],
                val_metrics['act_acc_at_gt'], val_metrics['act_acc_e2e'],
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
            f"Position accuracy (top-1): {best_metrics['pos_top1']:.2%}\n"
            f"Position accuracy (top-5): {best_metrics['pos_top5']:.2%}\n"
            f"Oracle accuracy (top-1): {best_metrics['oracle_top1']:.2%}\n"
            f"Action accuracy @ GT pos: {best_metrics['act_at_gt']:.2%}\n"
            f"End-to-end accuracy: {best_metrics['act_e2e']:.2%}\n"
            f"Checkpoint: {args.checkpoint_dir}/best.pt",
            border_style="green"
        ))

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description='FusionNet v2 Training')
    parser.add_argument('--data_path', type=str, default='FusionModel/fusion/fusion_data.pt')
    parser.add_argument('--checkpoint_dir', type=str, default='FusionModel/nn_checkpoints/fusion')
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
    parser.add_argument('--boundary_dilation', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--pos_weight', type=float, default=5.0)
    parser.add_argument('--n_hypotheses', type=int, default=4)
    parser.add_argument('--diversity_weight', type=float, default=0.5)
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
