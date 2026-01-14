import os, csv, json, time, hashlib, copy
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List
import numpy as np

# Run-level

@dataclass
class RunMeta:
    run_id: str                 # Unique Identifier
    timestamp: str              # Time
    algorithm: str              # Paterns
    init_path: str              
    grid_w: int                 
    grid_h: int                 
    random_seed: int            
    zones_max: int              # Z_MAX used in feature builder (fixed channel count)


# State-level 
# A snapshot of the grid

@dataclass
class StateRow:
    sample_id: str              # State ID
    run_id: str                 # Unique Identifier(RunMeta)
    instance_id: str            # Identifier for this grid instance (size/pattern/seed)
    step_t: int                 # Steps
    layer_id: int               # Layers
    grid_w: int                
    grid_h: int               
    num_zones: int              # Number of Zone
    zone_pattern: str           # Zone Patterns
    zone_params: str            # JSON string of generator parameters
    crossings_before: int       # Crossings before Applyting Operations
    features_file: str          # Path to .npz



# Action-trial 

@dataclass
class ActionRow:
    sample_id: str              # State ID(StateRow)
    x: int                      # Top-left x of the subgrid 
    y: int                      # Top-left y of the subgrid 
    subgrid_kind: str           # "3x3", "3x2" or "2x3"
    orientation: str            # Operation Variant
    op: str                     # "transpose" or "flip"
    valid: int                  # 1 if Hamiltonicity preserved, else 0
    crossings_before: int       # Crossings before 
    crossings_after: int        # Crossings after 
    delta_cross: int            # crossings_before - crossings_after (positive = improvement)
    reward: float               # Scoring for learning: alpha*delta_cross - gamma*(invalid)
    best_in_state: int          # 1 if this is the argmax-reward action among trials in the state


# Collector Class

class ZoningCollector:
    def __init__(self, out_dir: str = 'Dataset', alpha: float = 1.0, gamma: float = 10.0):
        
        self.out_dir = out_dir
        self.alpha, self.gamma = alpha, gamma

        # Creat subfolder for feature tensors
        self.features_dir = os.path.join(out_dir, "features")
        os.makedirs(self.features_dir, exist_ok=True)

        # Define file paths
        self.actions_csv = os.path.join(out_dir, "actions.csv")
        self.states_csv = os.path.join(out_dir, "states.csv")
        self.runmeta_json = os.path.join(out_dir, "runmeta.json")

        # Create without multiples
        if not os.path.exists(self.actions_csv):
            with open(self.actions_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([f.name for f in ActionRow.__dataclass_fields__.keys()])
        if not os.path.exists(self.states_csv):
            with open(self.states_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([f.name for f in StateRow.__dataclass_fields__.keys()])       

    # helpers
    def IDGenerator(self, b: bytes) -> str:
        return hashlib.sha1(b).hexdigest()[:16]
    
    def reward(self, delta_cross: int, valid: bool) -> float:
        return float(delta_cross) - (self.gamma if not valid else 0.0)
    
    # run-level
    def write_run_meta(self, run: RunMeta):
        with open(self.runmeta_json, "w") as f:
            json.dump(asdict(run), f, indent=2)

    # state-level
    def save_features(self, features: np.ndarray) -> Tuple[str, str]:
        # Save tensor [C,H,W] as compressed NPZ in features.
        assert features.ndim == 3
        sample_id = self.IDGenerator(features.tobytes())
        path = os.path.join(self.features_dir, f"{sample_id}.npz")
        if not os.path.exists(path):
            np.savez_compressed(path, x=features.astype(np.float32))
        return sample_id, path
    
    def log_state(self, row: StateRow):
        with open(self.states_csv, "a", newline="") as f:
            csv.writer(f).writerow([getattr(row, k) for k in StateRow.__dataclass_fields__.keys()])

    def log_actions(self, rows: List[ActionRow]):
        if rows:
            best_idx = max(range(len(rows)), key=lambda i: rows[i].reward)
            for i in range(len(rows)):
                rows[i].best_in_state = int(i == best_idx)
        with open(self.actions_csv, "a", newline="") as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow([getattr(r, k) for k in ActionRow.__dataclass_fields__.keys()])


# Feature Builder
# Converts a Hamiltonian grid + zones into a fixed-channel tensor.

def build_features_multizone(
        h,
        zones: Dict[Tuple[int,int], int],
        layer_id: int,
        Z_MAX: int = 6,
        add_dist: bool = False
) -> np.ndarray:
    
    H, W = h.height, h.width
    out: list[np.ndarray] = []

    # Zone one-hot
    Z = [np.zeros((H, W), np.float32) for _ in range(Z_MAX)]
    for y in range(H):
        for x in range(W):
            z = zones[(x,y)]
            ch = min(z, Z_MAX - 1)
            Z[ch][y, x] = 1.0
    out.extend(Z)

    # Edge Maps
    E_right = np.zeros((H, W), np.float32)
    for y in range(H):
        for x in range(W - 1):
            if h.H[y][x]:
                E_right[y, x] = 1.0
    E_down = np.zeros((H, W), np.float32)
    for y in range(H - 1):
        for x in range(W):
            if h.V[y][x]:
                E_down[y, x] = 1.0
    out += [E_right, E_down]

    # Boundry map (1 if neighbor zone differs)
    B = np.zeros((H, W), np.float32)
    for y in range(H):
        for x in range(W):
            z = zones[(x, y)]
            if (x>0 and zones[(x-1,y)]!=z) or (x<W-1 and zones[(x+1,y)]!=z) or (y>0 and zones[(x,y-1)]!=z) or (y<H-1 and zones[(x,y+1)]!=z):
                B[y, x] = 1.0
    out.append(B)        

    # Layer 
    L = np.full((H, W), float(layer_id), np.float32)
    max_layer = min(W, H) // 4
    if max_layer > 0:
        L /= float(max_layer)
    out.append(L)

    # Distance Boundry
    if add_dist:
        D = np.ones((H, W), np.float32) * (W + H)
        frontier = [(x, y) for y in range(H) for x in range(W) if B[y, x] == 1.0]
        for (x, y) in frontier:
            D[y, x] = 0.0
        for _ in range(W + H):
            for y in range(H):
                for x in range(W):
                    d = D[y, x]
                    if x>0:   d = min(d, D[y, x-1]+1)
                    if x<W-1: d = min(d, D[y, x+1]+1)
                    if y>0:   d = min(d, D[y-1, x]+1)
                    if y<H-1: d = min(d, D[y+1, x]+1)
                    D[y, x] = d
        D /= (W + H)
        out.append(D)
    return np.stack(out, axis=0)

# Operation Trial

def try_op(
    h,
    compute_crossings_fn,
    op: str,
    x: int,
    y: int,
    subgrid_kind: str,
    variant: str
) -> Tuple[bool, int, int, int]:
    
    beforeC = compute_crossings_fn()
    clone = copy.deepcopy(h)

    if op == "transpose":
        if x + 2 >= h.width or y + 2 >= h.height:
            return False, beforeC, beforeC, beforeC, 0
        sub = clone.get_subgrid((x,y), (x + 2, y + 2))
        _, res = clone.transpose_subgrid(sub, variant)
        valid = isinstance(res, str) and res.startswith("transposed")

    elif op == "flip":
        if subgrid_kind == "3x2":
            if x + 2 >= h.width or y + 1 >= h.height:
                return False, beforeC, beforeC, beforeC, 0
            sub = clone.get_subgrid((x, y), (x + 2, y + 1))
        elif subgrid_kind == "2x3":
            if x + 1 >= h.width or y + 2 >= h.height:
                return False, beforeC, beforeC, beforeC, 0
            sub = clone.get_subgrid((x, y), (x + 1, y + 2))
        else:
            return False, beforeC, beforeC, 0
        _, res = clone.flip_subgrid(sub, variant)
        valid = isinstance(res, str) and res.startswith("flipped")
    
    else:
        return False, beforeC, beforeC, 0
    
    if valid:
        H0, V0 = h.H, h.V
        h.H, h.V = [row[:] for row in clone.H], [row[:] for row in clone.V]
        afterC = compute_crossings_fn()
        h.H, h.V = H0, V0
    else:
        afterC = beforeC

    deltaC = beforeC - afterC
    return valid, beforeC, afterC, deltaC