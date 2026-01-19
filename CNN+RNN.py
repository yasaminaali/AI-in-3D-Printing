import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
from typing import List, Dict, Any, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 


# Positive / negative labels
# Orientations, Subgrid_kinds, Ops
def scan_action_metadata(actions_csv: str):
    orientations = set()
    subgrid_kinds = set()
    ops = set()
    num_pos = 0
    num_neg = 0

    with open(actions_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            orientations.add(row['orientation'])
            subgrid_kinds.add(row['subgrid_kind'])
            ops.add(row['op'])

            y = int(row['best_in_state'])
            if y == 1:
                num_pos += 1
            else:
                num_neg += 1

    return {
        "orientations": sorted(list(orientations)),
        "subgrid_kinds": sorted(list(subgrid_kinds)),
        "ops": sorted(list(ops)),
        "num_pos": num_pos,
        "num_neg": num_neg
    }


def get_pattern_list(states_csv: str) -> List[str]:
    patterns = set()
    with open(states_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            patterns.add(row['zone_pattern'])
    return sorted(list(patterns))

# Split instance_id by seed 
# Seeds 0-6 -> Train
# 7-8 -> Validation
# 9 -> Test

def split_instances_by_seed(states_csv: str):
    train_insts: Set[str] = set()
    val_insts: Set[str] = set()
    test_insts: Set[str] = set()

    with open(states_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            inst = row["instance_id"]

            seed_val = None
            if "seed" in inst:
                try:
                    seed_str = inst.split("seed")[-1]
                    seed_val = int(seed_str)
                except ValueError:
                    seed_val = None

            if seed_val is None:
                train_insts.add(inst)
                continue

            if 0 <= seed_val <= 6:
                train_insts.add(inst)
            elif 7 <= seed_val <= 8:
                val_insts.add(inst)
            elif seed_val == 9:
                test_insts.add(inst)
            else:
                train_insts.add(inst)

    print(
        f"[Split by instance+seed]: "
        f"train instances={len(train_insts)}, "
        f"val={len(val_insts)}, test={len(test_insts)}"
    )
    return train_insts, val_insts, test_insts


# Sequence Dataset (global, all sizes/patterns)
class SequenceActionDataset(Dataset):
    def __init__(
        self,
        states_csv: str,
        actions_csv: str,
        allowed_instance_ids: Set[str],
        orientation_list: List[str],
        subgrid_kind_list: List[str],
        op_list: List[str],
        pattern_list: List[str],
        T_seq: int = 10,
    ):
        super().__init__()
        self.T_seq = T_seq

        # Convert strings -> numbers
        self.orient2idx = {o: i for i, o in enumerate(orientation_list)}
        self.subgrid2idx = {k: i for i, k in enumerate(subgrid_kind_list)}
        self.op2idx = {o: i for i, o in enumerate(op_list)}
        self.pattern2idx = {p: i for i, p in enumerate(pattern_list)}

        # Load states
        states_by_sid: Dict[str, Dict[str, Any]] = {}
        instance_to_sids: Dict[str, List[str]] = {}

        with open(states_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row["sample_id"]
                inst = row["instance_id"]
                states_by_sid[sid] = row
                instance_to_sids.setdefault(inst, []).append(sid)

        # Load actions grouped by sample_id
        actions_by_sid: Dict[str, List[Dict[str, Any]]] = {}
        with open(actions_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row["sample_id"]
                actions_by_sid.setdefault(sid, []).append(row)

        self.seq_data: List[Dict[str, Any]] = []

        # Build sequences per instance
        for inst_id, sid_list in instance_to_sids.items():
            if inst_id not in allowed_instance_ids:
                continue

            # Sort states by step_t
            sid_list_sorted = sorted(
                sid_list,
                key=lambda s: int(states_by_sid[s]["step_t"])
            )

            seq_steps: List[Dict[str, Any]] = []

            for sid in sid_list_sorted:
                srow = states_by_sid[sid]

                # All valid actions for this state
                arows = actions_by_sid.get(sid, [])
                arows_valid = [a for a in arows if int(a["valid"]) == 1]
                if not arows_valid:
                    continue

                # Pick the best_in_state == 1
                best_rows = [a for a in arows_valid if int(a["best_in_state"]) == 1]
                if not best_rows:
                    # No label for this state
                    continue
                # If multiple, keep the one with largest reward
                best = max(best_rows, key=lambda r: float(r["reward"]))

                grid_w = int(srow["grid_w"])
                grid_h = int(srow["grid_h"])
                num_zones = int(srow["num_zones"])
                zone_pattern = srow["zone_pattern"]
                feat_path = srow["features_file"]

                step_info = dict(
                    features_file=feat_path,
                    grid_w=grid_w,
                    grid_h=grid_h,
                    num_zones=num_zones,
                    zone_pattern=zone_pattern,
                    x=int(best["x"]),
                    y=int(best["y"]),
                    op=self.op2idx[best["op"]],
                    sg=self.subgrid2idx[best["subgrid_kind"]],
                    ori=self.orient2idx[best["orientation"]],
                )
                seq_steps.append(step_info)

            # A least T_seq steps to build a fixed-length sequence
            if len(seq_steps) >= self.T_seq:
                # Take the First T_seq steps
                seq_steps = seq_steps[:self.T_seq]
                self.seq_data.append(dict(
                    instance_id=inst_id,
                    steps=seq_steps,
                ))

        print(f"[SequenceDataset] Instances kept = {len(self.seq_data)} (T_seq={self.T_seq})")

        # (w_norm, h_norm, k_norm) + one-hot pattern
        self.cfg_dim = 3 + len(self.pattern2idx)

    def __len__(self):
        return len(self.seq_data)

    # Global Description
    def _build_cfg_vec(self, step0: Dict[str, Any]) -> np.ndarray:
        # Build configuration vector from the first step in the sequence.
        W = step0["grid_w"]
        H = step0["grid_h"]
        k = step0["num_zones"]
        pattern = step0["zone_pattern"]

        # Normalization (max 48x48, k<=6)
        w_norm = W / 48.0
        h_norm = H / 48.0
        k_norm = k / 6.0

        pat_idx = self.pattern2idx.get(pattern, 0)
        pat_onehot = np.zeros(len(self.pattern2idx), dtype=np.float32)
        pat_onehot[pat_idx] = 1.0

        cfg_vec = np.concatenate(
            [np.array([w_norm, h_norm, k_norm], dtype=np.float32), pat_onehot],
            axis=0,
        )
        return cfg_vec

    def __getitem__(self, idx: int):
        seq_entry = self.seq_data[idx]
        steps = seq_entry["steps"]

        # T_seq
        T = len(steps)
        assert T > 0

        # Same config for all steps
        cfg_vec = self._build_cfg_vec(steps[0]) 

        x_states_list = []
        y_op_list = []
        y_sg_list = []
        y_ori_list = []

        for s in steps:
            feat_path = s["features_file"]
            data = np.load(feat_path)
            x_state = data["x"].astype(np.float32)

            x_states_list.append(x_state)
            y_op_list.append(s["op"])
            y_sg_list.append(s["sg"])
            y_ori_list.append(s["ori"])

        # Stack along time -> [T,C,H,W]
        x_states = np.stack(x_states_list, axis=0)
        y_op = np.array(y_op_list, dtype=np.int64)
        y_sg = np.array(y_sg_list, dtype=np.int64)
        y_ori = np.array(y_ori_list, dtype=np.int64)

        # to torch
        x_states_t = torch.from_numpy(x_states)      
        cfg_t = torch.from_numpy(cfg_vec)           
        y_op_t = torch.from_numpy(y_op)              
        y_sg_t = torch.from_numpy(y_sg)              
        y_ori_t = torch.from_numpy(y_ori)           

        return x_states_t, cfg_t, y_op_t, y_sg_t, y_ori_t


# 3) Padding different grid sizes in batch
def sequence_batch(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    x_list, cfg_list, y_op_list, y_sg_list, y_ori_list = zip(*batch)

    T = x_list[0].shape[0]
    C = x_list[0].shape[1]

    # Find max H,W in this batch
    max_H = max(x.shape[2] for x in x_list)
    max_W = max(x.shape[3] for x in x_list)

    B = len(x_list)
    x_batch = torch.zeros(B, T, C, max_H, max_W, dtype=x_list[0].dtype)
    y_op_batch = torch.zeros(B, T, dtype=y_op_list[0].dtype)
    y_sg_batch = torch.zeros(B, T, dtype=y_sg_list[0].dtype)
    y_ori_batch = torch.zeros(B, T, dtype=y_ori_list[0].dtype)
    cfg_batch = torch.stack(cfg_list, dim=0)  

    for i in range(B):
        x = x_list[i]
        _, _, H, W = x.shape
        x_batch[i, :, :, :H, :W] = x

        y_op_batch[i, :] = y_op_list[i]
        y_sg_batch[i, :] = y_sg_list[i]
        y_ori_batch[i, :] = y_ori_list[i]

    return x_batch, cfg_batch, y_op_batch, y_sg_batch, y_ori_batch


# 4) Model: CNN + RNN
# CNN → processes each state
# RNN(GRU) → processes the sequence over time

class CNN_RNN(nn.Module):
    """
      Input:
        - x_states: [B,T,C,H,W]  (grid state sequence)
        B = batch size, T = time steps, C = channels (edges, zones, ...), 
        HxW = grid size
        - cfg_vec:  [B,cfg_dim]  (grid+zone pattern info)
        Grid size, number of zones, zone pattern

      Output:
        - logits_op:  [B,T,N_ops]
        - logits_sg:  [B,T,N_sg]
        - logits_ori: [B,T,N_ori]
    """

    def __init__(
        self,
        in_channels: int,
        cfg_dim: int,
        n_ops: int,
        n_sg: int,
        n_ori: int,
        cnn_hidden: int = 64,
        rnn_hidden: int = 128,
        cfg_hidden: int = 32,
    ):
        super().__init__()
        self.n_ops = n_ops
        self.n_sg = n_sg
        self.n_ori = n_ori

        # CNN over each state
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, cnn_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), 
        )

        # Converts spatial info into a 128-dimensional vector
        self.fc_state = nn.Sequential(
            nn.Flatten(),                
            nn.Linear(cnn_hidden * 4 * 4, 128),
            nn.ReLU(),
        )

        # Embedding
        self.fc_cfg = nn.Sequential(
            nn.Linear(cfg_dim, cfg_hidden),
            nn.ReLU(),
        )

        # RNN input dimension = state_emb + cfg_emb
        self.rnn_input_dim = 128 + cfg_hidden
        self.rnn_hidden = rnn_hidden

        self.rnn = nn.GRU(
            input_size=self.rnn_input_dim,
            hidden_size=self.rnn_hidden,
            batch_first=True,
        )

        self.head_op = nn.Linear(self.rnn_hidden, n_ops)
        self.head_sg = nn.Linear(self.rnn_hidden, n_sg)
        self.head_ori = nn.Linear(self.rnn_hidden, n_ori)

    def forward(self, x_states: torch.Tensor, cfg_vec: torch.Tensor):
        B, T, C, H, W = x_states.shape

        # merge batch+time for CNN
        x_flat = x_states.view(B * T, C, H, W) 
        s = self.conv(x_flat)              
        s = self.fc_state(s)                 

        s = s.view(B, T, -1)                 
        cfg_emb = self.fc_cfg(cfg_vec)  
        cfg_seq = cfg_emb.unsqueeze(1).expand(B, T, -1)

        rnn_input = torch.cat([s, cfg_seq], dim=-1) 

        rnn_out, _ = self.rnn(rnn_input)      

        logits_op = self.head_op(rnn_out) 
        logits_sg = self.head_sg(rnn_out)     
        logits_ori = self.head_ori(rnn_out)   

        return logits_op, logits_sg, logits_ori


# Training 

def train_seq_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cpu",
    epochs: int = 10,
    lr: float = 1e-3,
):
    model.to(device)

    ce_op = nn.CrossEntropyLoss()
    ce_sg = nn.CrossEntropyLoss()
    ce_ori = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_steps = 0

        for x_states, cfg_vec, y_op, y_sg, y_ori in train_loader:
            x_states = x_states.to(device)   
            cfg_vec = cfg_vec.to(device)    
            y_op = y_op.to(device)          
            y_sg = y_sg.to(device)           
            y_ori = y_ori.to(device)         

            optimizer.zero_grad()
            logits_op, logits_sg, logits_ori = model(x_states, cfg_vec)

            B, T = y_op.shape

            # flatten
            loss_op = ce_op(
                logits_op.view(B * T, -1),
                y_op.view(-1)
            )
            loss_sg = ce_sg(
                logits_sg.view(B * T, -1),
                y_sg.view(-1)
            )
            loss_ori = ce_ori(
                logits_ori.view(B * T, -1),
                y_ori.view(-1)
            )

            loss = loss_op + loss_sg + loss_ori
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            total_steps += B

        avg_train_loss = total_loss / max(1, total_steps)

        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for x_states, cfg_vec, y_op, y_sg, y_ori in val_loader:
                x_states = x_states.to(device)
                cfg_vec = cfg_vec.to(device)
                y_op = y_op.to(device)
                y_sg = y_sg.to(device)
                y_ori = y_ori.to(device)

                logits_op, logits_sg, logits_ori = model(x_states, cfg_vec)
                B, T = y_op.shape

                loss_op = ce_op(
                    logits_op.view(B * T, -1),
                    y_op.view(-1)
                )
                loss_sg = ce_sg(
                    logits_sg.view(B * T, -1),
                    y_sg.view(-1)
                )
                loss_ori = ce_ori(
                    logits_ori.view(B * T, -1),
                    y_ori.view(-1)
                )

                loss = loss_op + loss_sg + loss_ori
                val_loss += loss.item() * B
                val_steps += B

        avg_val_loss = val_loss / max(1, val_steps)
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f}"
        )

# Evaluation

def evaluate_seq_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
):
    model.to(device)
    model.eval()

    correct_op = 0
    correct_sg = 0
    correct_ori = 0
    total = 0

    with torch.no_grad():
        for x_states, cfg_vec, y_op, y_sg, y_ori in test_loader:
            x_states = x_states.to(device)
            cfg_vec = cfg_vec.to(device)
            y_op = y_op.to(device)
            y_sg = y_sg.to(device)
            y_ori = y_ori.to(device)

            logits_op, logits_sg, logits_ori = model(x_states, cfg_vec)

            # Predictions
            pred_op = torch.argmax(logits_op, dim=-1)  
            pred_sg = torch.argmax(logits_sg, dim=-1)
            pred_ori = torch.argmax(logits_ori, dim=-1)

            B, T = y_op.shape
            total += B * T

            correct_op += (pred_op == y_op).sum().item()
            correct_sg += (pred_sg == y_sg).sum().item()
            correct_ori += (pred_ori == y_ori).sum().item()

    acc_op = correct_op / total if total > 0 else 0.0
    acc_sg = correct_sg / total if total > 0 else 0.0
    acc_ori = correct_ori / total if total > 0 else 0.0

    print(f"[TEST] op  accuracy  = {acc_op:.4f}")
    print(f"[TEST] sg  accuracy  = {acc_sg:.4f}")
    print(f"[TEST] ori accuracy  = {acc_ori:.4f}")


# Main

def main():
    # GPU detection and setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("Device Configuration")
    print("=" * 60)
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Using CPU (GPU not available)")
    print("=" * 60)
    print()
    
    states_csv = os.path.join("Dataset", "states.csv")
    actions_csv = os.path.join("Dataset", "actions.csv")

    # Action metadata
    meta = scan_action_metadata(actions_csv)
    orientations = meta["orientations"]
    subgrid_kinds = meta["subgrid_kinds"]
    ops = meta["ops"]
    num_pos = meta["num_pos"]
    num_neg = meta["num_neg"]

    print("[Meta] orientations:", orientations)
    print("[Meta] subgrid_kinds:", subgrid_kinds)
    print("[Meta] ops:", ops)
    print(f"[Meta] num_pos={num_pos}, num_neg={num_neg}")

    # Patterns from states
    patterns = get_pattern_list(states_csv)
    print("[Meta] zone_patterns:", patterns)

    # Instance-level splits
    train_insts, val_insts, test_insts = split_instances_by_seed(states_csv)

    # Build sequence datasets
    T_SEQ = 10 

    train_dataset = SequenceActionDataset(
        states_csv=states_csv,
        actions_csv=actions_csv,
        allowed_instance_ids=train_insts,
        orientation_list=orientations,
        subgrid_kind_list=subgrid_kinds,
        op_list=ops,
        pattern_list=patterns,
        T_seq=T_SEQ,
    )

    val_dataset = SequenceActionDataset(
        states_csv=states_csv,
        actions_csv=actions_csv,
        allowed_instance_ids=val_insts,
        orientation_list=orientations,
        subgrid_kind_list=subgrid_kinds,
        op_list=ops,
        pattern_list=patterns,
        T_seq=T_SEQ,
    )

    test_dataset = SequenceActionDataset(
        states_csv=states_csv,
        actions_csv=actions_csv,
        allowed_instance_ids=test_insts,
        orientation_list=orientations,
        subgrid_kind_list=subgrid_kinds,
        op_list=ops,
        pattern_list=patterns,
        T_seq=T_SEQ,
    )

    print(f"[Dataset] train sequences = {len(train_dataset)}")
    print(f"[Dataset] val   sequences = {len(val_dataset)}")
    print(f"[Dataset] test  sequences = {len(test_dataset)}")

    if len(train_dataset) == 0:
        print("Train dataset is empty!")
        return

    # Infer channels & cfg_dim
    x_states0, cfg0, _, _, _ = train_dataset[0]
    in_channels = x_states0.shape[1]
    cfg_dim = cfg0.shape[0]
    print(f"[Shapes] in_channels={in_channels}, cfg_dim={cfg_dim}, T_seq={T_SEQ}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True,
        collate_fn=sequence_batch
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False,
        collate_fn=sequence_batch
    )
    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False,
        collate_fn=sequence_batch
    )

    # Build model (device already set earlier)
    print(f"\n[Model] Building CNN+RNN on {device.upper()}...")
    model = CNN_RNN(
        in_channels=in_channels,
        cfg_dim=cfg_dim,
        n_ops=len(ops),
        n_sg=len(subgrid_kinds),
        n_ori=len(orientations),
    )
    print(f"[Model] Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    train_seq_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=10,
        lr=1e-3,
    )

    # Save model
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "global_seq_policy.pt")
    torch.save(model.state_dict(), save_path)
    print("Model saved to:", save_path)

    # Evaluate on test
    evaluate_seq_model(model, test_loader, device=device)


if __name__ == "__main__":
    main()
