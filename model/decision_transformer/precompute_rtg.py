"""Pre-compute returns-to-go and save to disk for fast loading."""
import json
import pickle
import numpy as np
from pathlib import Path
import sys
import time
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from operations import HamiltonianSTL

GRID_SIZE = 30

def compute_crossings(H, V, zones):
    """Fast crossing computation using numpy."""
    crossings = 0
    # Horizontal edges
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE - 1):
            if H[y][x]:
                if zones[y, x] != zones[y, x + 1]:
                    crossings += 1
    # Vertical edges
    for y in range(GRID_SIZE - 1):
        for x in range(GRID_SIZE):
            if V[y][x]:
                if zones[y, x] != zones[y + 1, x]:
                    crossings += 1
    return crossings

def apply_op(h, op):
    kind = op['kind']
    x, y = op['x'], op['y']
    variant = op['variant']
    if kind == 'N':
        return
    try:
        if kind == 'T':
            sub = h.get_subgrid((x, y), (x + 2, y + 2))
            h.transpose_subgrid(sub, variant)
        elif kind == 'F':
            if variant in ['n', 's']:
                sub = h.get_subgrid((x, y), (x + 1, y + 2))
            else:
                sub = h.get_subgrid((x, y), (x + 2, y + 1))
            h.flip_subgrid(sub, variant)
    except:
        pass

def process_trajectory(args):
    idx, traj = args
    zone_grid = traj['zone_grid']
    if isinstance(zone_grid[0], int):
        zones = np.array(zone_grid).reshape(GRID_SIZE, GRID_SIZE)
    else:
        zones = np.array(zone_grid)
    
    h = HamiltonianSTL(GRID_SIZE, GRID_SIZE, init_pattern='zigzag')
    crossings_seq = [compute_crossings(h.H, h.V, zones)]
    
    for op in traj['sequence_ops']:
        apply_op(h, op)
        crossings_seq.append(compute_crossings(h.H, h.V, zones))
    
    # Rewards and RTG
    rewards = [crossings_seq[i] - crossings_seq[i+1] for i in range(len(crossings_seq)-1)]
    rtg = []
    running = 0
    for r in reversed(rewards):
        running += r
        rtg.insert(0, running)
    
    return idx, crossings_seq, rewards, rtg

def main():
    data_path = '/workspace/AI-in-3D-Printing/combined_dataset.jsonl'
    output_path = '/workspace/AI-in-3D-Printing/model/decision_transformer/rtg_cache.pkl'
    
    print('Loading trajectories...')
    trajectories = []
    with open(data_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            if len(record.get('sequence_ops', [])) > 0:
                trajectories.append(record)
    
    print(f'Loaded {len(trajectories)} trajectories')
    print(f'Computing RTG using {cpu_count()} workers...')
    
    start = time.time()
    
    # Process in parallel
    with Pool(cpu_count()) as pool:
        results = pool.map(process_trajectory, enumerate(trajectories))
    
    # Sort by index and attach to trajectories
    results.sort(key=lambda x: x[0])
    for idx, crossings_seq, rewards, rtg in results:
        trajectories[idx]['crossings_sequence'] = crossings_seq
        trajectories[idx]['rewards'] = rewards
        trajectories[idx]['returns_to_go'] = rtg
    
    print(f'Computed in {time.time()-start:.1f}s')
    
    # Save
    print(f'Saving to {output_path}...')
    with open(output_path, 'wb') as f:
        pickle.dump(trajectories, f)
    
    print('Done!')

if __name__ == '__main__':
    main()
