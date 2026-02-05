"""
Visualization utilities for CNN+RNN model solutions
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch


def visualize_solution(grid_W: int, grid_H: int, zone_grid: np.ndarray,
                      h_edges: np.ndarray, v_edges: np.ndarray,
                      initial_crossings: int, final_crossings: int,
                      operation_sequence: List[Dict],
                      output_path: str, title: str = "Optimized Path"):
    """
    Visualize a Hamiltonian path solution with zone crossings highlighted.
    
    Args:
        grid_W, grid_H: Grid dimensions
        zone_grid: 2D array of zone IDs
        h_edges: Horizontal edge matrix
        v_edges: Vertical edge matrix  
        initial_crossings: Starting crossing count
        final_crossings: Optimized crossing count
        operation_sequence: List of operations applied
        output_path: Where to save visualization
        title: Plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Draw zone backgrounds with distinct colors
    zone_colors = ['#E8F4F8', '#FFE4E1', '#F0E68C', '#DDA0DD', '#98FB98', '#FFB6C1']
    
    for y in range(grid_H):
        for x in range(grid_W):
            zone_id = int(zone_grid[y, x]) if y < zone_grid.shape[0] and x < zone_grid.shape[1] else 0
            color = zone_colors[zone_id % len(zone_colors)]
            rect = patches.Rectangle((x, y), 1, 1, 
                                   linewidth=0.5, 
                                   edgecolor='gray',
                                   facecolor=color,
                                   alpha=0.3)
            ax.add_patch(rect)
    
    # Build adjacency and trace path
    adj = {}
    for y in range(grid_H):
        for x in range(grid_W):
            adj[(x, y)] = []
    
    # Add edges from h_edges and v_edges
    for y in range(grid_H):
        for x in range(grid_W - 1):
            if y < h_edges.shape[0] and x < h_edges.shape[1] and h_edges[y][x]:
                adj[(x, y)].append((x + 1, y))
                adj[(x + 1, y)].append((x, y))
    
    for y in range(grid_H - 1):
        for x in range(grid_W):
            if y < v_edges.shape[0] and x < v_edges.shape[1] and v_edges[y][x]:
                adj[(x, y)].append((x, y + 1))
                adj[(x, y + 1)].append((x, y))
    
    # Find path
    start = None
    for (x, y), neighbors in adj.items():
        if len(neighbors) == 1:
            start = (x, y)
            break
    
    if start is None:
        start = (0, 0)
    
    # Trace Hamiltonian path
    path = [start]
    visited = {start}
    current = start
    
    while True:
        next_nodes = [n for n in adj[current] if n not in visited]
        if not next_nodes:
            break
        current = next_nodes[0]
        path.append(current)
        visited.add(current)
    
    # Draw path with crossing highlighting
    for i in range(len(path) - 1):
        x1, y1 = path[i][0] + 0.5, path[i][1] + 0.5
        x2, y2 = path[i + 1][0] + 0.5, path[i + 1][1] + 0.5
        
        # Check if crossing zone boundary
        z1 = zone_grid[path[i][1], path[i][0]] if (path[i][0] < zone_grid.shape[1] and path[i][1] < zone_grid.shape[0]) else 0
        z2 = zone_grid[path[i + 1][1], path[i + 1][0]] if (path[i + 1][0] < zone_grid.shape[1] and path[i + 1][1] < zone_grid.shape[0]) else 0
        
        is_crossing = z1 != z2
        color = '#FF4444' if is_crossing else '#4444FF'  # Red for crossings, blue for normal
        linewidth = 3 if is_crossing else 2
        alpha = 0.9 if is_crossing else 0.6
        
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=alpha)
    
    # Mark start and end
    if path:
        ax.plot(path[0][0] + 0.5, path[0][1] + 0.5, 'go', markersize=12, label='Start', zorder=5)
        ax.plot(path[-1][0] + 0.5, path[-1][1] + 0.5, 'ro', markersize=12, label='End', zorder=5)
    
    # Mark crossing points with X
    crossing_count = 0
    for i in range(len(path) - 1):
        z1 = zone_grid[path[i][1], path[i][0]] if (path[i][0] < zone_grid.shape[1] and path[i][1] < zone_grid.shape[0]) else 0
        z2 = zone_grid[path[i + 1][1], path[i + 1][0]] if (path[i + 1][0] < zone_grid.shape[1] and path[i + 1][1] < zone_grid.shape[0]) else 0
        if z1 != z2:
            crossing_count += 1
            mid_x = (path[i][0] + path[i + 1][0]) / 2 + 0.5
            mid_y = (path[i][1] + path[i + 1][1]) / 2 + 0.5
            ax.plot(mid_x, mid_y, 'rx', markersize=10, markeredgewidth=2, zorder=4)
    
    # Settings
    ax.set_xlim(-0.5, grid_W + 0.5)
    ax.set_ylim(-0.5, grid_H + 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10)
    
    # Title with stats
    improvement = initial_crossings - final_crossings
    improvement_pct = (improvement / initial_crossings * 100) if initial_crossings > 0 else 0
    
    full_title = f"{title}\nGrid: {grid_W}×{grid_H} | "
    full_title += f"Initial: {initial_crossings} crossings → Final: {final_crossings} crossings | "
    full_title += f"Improvement: {improvement} ({improvement_pct:.1f}%) | "
    full_title += f"Operations: {len([op for op in operation_sequence if op['kind'] != 'N'])}"
    
    ax.set_title(full_title, fontsize=11, fontweight='bold', pad=20)
    ax.set_xlabel('X coordinate', fontsize=10)
    ax.set_ylabel('Y coordinate', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return crossing_count


def create_comparison_visualization(grid_W: int, grid_H: int, zone_grid: np.ndarray,
                                   initial_h_edges: np.ndarray, initial_v_edges: np.ndarray,
                                   final_h_edges: np.ndarray, final_v_edges: np.ndarray,
                                   initial_crossings: int, final_crossings: int,
                                   operation_sequence: List[Dict],
                                   output_path: str):
    """
    Create side-by-side comparison of initial vs optimized path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    zone_colors = ['#E8F4F8', '#FFE4E1', '#F0E68C', '#DDA0DD', '#98FB98', '#FFB6C1']
    
    def draw_path(ax, h_edges, v_edges, title, crossings):
        # Draw zones
        for y in range(grid_H):
            for x in range(grid_W):
                zone_id = int(zone_grid[y, x]) if y < zone_grid.shape[0] and x < zone_grid.shape[1] else 0
                color = zone_colors[zone_id % len(zone_colors)]
                rect = patches.Rectangle((x, y), 1, 1, 
                                       linewidth=0.5, 
                                       edgecolor='gray',
                                       facecolor=color,
                                       alpha=0.3)
                ax.add_patch(rect)
        
        # Build adjacency
        adj = {}
        for y in range(grid_H):
            for x in range(grid_W):
                adj[(x, y)] = []
        
        for y in range(min(grid_H, h_edges.shape[0])):
            for x in range(min(grid_W - 1, h_edges.shape[1])):
                if h_edges[y][x]:
                    adj[(x, y)].append((x + 1, y))
                    adj[(x + 1, y)].append((x, y))
        
        for y in range(min(grid_H - 1, v_edges.shape[0])):
            for x in range(min(grid_W, v_edges.shape[1])):
                if v_edges[y][x]:
                    adj[(x, y)].append((x, y + 1))
                    adj[(x, y + 1)].append((x, y))
        
        # Find and trace path
        start = None
        for (x, y), neighbors in adj.items():
            if len(neighbors) == 1:
                start = (x, y)
                break
        if start is None:
            start = (0, 0)
        
        path = [start]
        visited = {start}
        current = start
        
        while True:
            next_nodes = [n for n in adj[current] if n not in visited]
            if not next_nodes:
                break
            current = next_nodes[0]
            path.append(current)
            visited.add(current)
        
        # Draw path
        for i in range(len(path) - 1):
            x1, y1 = path[i][0] + 0.5, path[i][1] + 0.5
            x2, y2 = path[i + 1][0] + 0.5, path[i + 1][1] + 0.5
            
            z1 = zone_grid[path[i][1], path[i][0]] if (path[i][0] < zone_grid.shape[1] and path[i][1] < zone_grid.shape[0]) else 0
            z2 = zone_grid[path[i + 1][1], path[i + 1][0]] if (path[i + 1][0] < zone_grid.shape[1] and path[i + 1][1] < zone_grid.shape[0]) else 0
            
            is_crossing = z1 != z2
            color = '#FF4444' if is_crossing else '#4444FF'
            linewidth = 3 if is_crossing else 2
            
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=0.7)
        
        # Mark start/end
        if path:
            ax.plot(path[0][0] + 0.5, path[0][1] + 0.5, 'go', markersize=10)
            ax.plot(path[-1][0] + 0.5, path[-1][1] + 0.5, 'ro', markersize=10)
        
        ax.set_xlim(-0.5, grid_W + 0.5)
        ax.set_ylim(-0.5, grid_H + 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{title}\n{crossings} crossings", fontsize=12, fontweight='bold')
    
    # Draw initial and final
    draw_path(ax1, initial_h_edges, initial_v_edges, "Initial Path (Zigzag)", initial_crossings)
    draw_path(ax2, final_h_edges, final_v_edges, "Optimized Path (NN)", final_crossings)
    
    # Overall title
    improvement = initial_crossings - final_crossings
    improvement_pct = (improvement / initial_crossings * 100) if initial_crossings > 0 else 0
    fig.suptitle(f"CNN+RNN Optimization Results - {grid_W}×{grid_H} Grid\n"
                f"Improvement: {improvement} crossings ({improvement_pct:.1f}%) | "
                f"Operations Applied: {len([op for op in operation_sequence if op['kind'] != 'N'])}",
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def visualize_training_progress(train_losses: List[float], val_losses: List[float],
                                output_path: str):
    """Visualize training loss curves."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('CNN+RNN Training Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
