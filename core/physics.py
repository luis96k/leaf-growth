"""
Physics engine with scipy cKDTree for efficient spatial queries.

Key optimization: Using C-implemented KD-tree instead of Python dict-based grid.
This gives O(n log n) tree construction and O(log n + k) queries.
"""

import numpy as np
from typing import TYPE_CHECKING, List, Tuple, Set, Dict
from dataclasses import dataclass

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not found. Install with 'pip install scipy' for better performance.")

if TYPE_CHECKING:
    from .tissue import Tissue
    from .cell import Cell


class SpatialGrid:
    """
    Spatial partitioning grid for neighbor queries.
    Fallback when scipy is not available, also used for get_cell_at_position.
    """
    
    def __init__(self, cell_size: float = 2.5):
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int], Set[str]] = {}
        self.cell_positions: Dict[str, Tuple[int, int]] = {}
    
    def _get_grid_coords(self, position: np.ndarray) -> Tuple[int, int]:
        return (
            int(np.floor(position[0] / self.cell_size)),
            int(np.floor(position[1] / self.cell_size))
        )
    
    def clear(self):
        self.grid.clear()
        self.cell_positions.clear()
    
    def insert(self, cell_id: str, position: np.ndarray):
        new_grid_pos = self._get_grid_coords(position)
        
        if cell_id in self.cell_positions:
            old_pos = self.cell_positions[cell_id]
            if old_pos != new_grid_pos and old_pos in self.grid:
                self.grid[old_pos].discard(cell_id)
        
        if new_grid_pos not in self.grid:
            self.grid[new_grid_pos] = set()
        self.grid[new_grid_pos].add(cell_id)
        self.cell_positions[cell_id] = new_grid_pos
    
    def remove(self, cell_id: str):
        if cell_id in self.cell_positions:
            grid_pos = self.cell_positions[cell_id]
            if grid_pos in self.grid:
                self.grid[grid_pos].discard(cell_id)
            del self.cell_positions[cell_id]
    
    def get_nearby_cell_ids(self, position: np.ndarray, radius: float = None) -> Set[str]:
        grid_pos = self._get_grid_coords(position)
        search_range = 2 if radius is None else max(1, int(np.ceil(radius / self.cell_size)))
        
        nearby = set()
        for dx in range(-search_range, search_range + 1):
            for dy in range(-search_range, search_range + 1):
                check_pos = (grid_pos[0] + dx, grid_pos[1] + dy)
                if check_pos in self.grid:
                    nearby.update(self.grid[check_pos])
        
        return nearby


class PhysicsEngine:
    """
    Physics engine using scipy cKDTree for O(n log n) neighbor finding.
    
    Major optimizations over basic approach:
    1. cKDTree.query_pairs finds ALL interacting pairs in one C call
    2. Vectorized force computation using numpy broadcasting
    3. Batch position updates
    """
    
    def __init__(self):
        self.repulsion_constant: float = 2.5
        self.damping: float = 0.55
        self.default_iterations: int = 15
        self.convergence_threshold: float = 0.008
        
        # Keep spatial grid for non-physics queries (cell selection, etc.)
        self.spatial_grid: SpatialGrid = SpatialGrid(cell_size=2.5)
    
    def rebuild_spatial_grid(self, tissue: 'Tissue'):
        """Rebuild grid from current cell positions."""
        self.spatial_grid.clear()
        for cell in tissue.cells.values():
            self.spatial_grid.insert(cell.id, cell.center)
    
    def settle(self, tissue: 'Tissue', iterations: int = None):
        """
        Run physics simulation using vectorized operations.
        """
        if iterations is None:
            iterations = self.default_iterations
        
        cells = list(tissue.cells.values())
        n_cells = len(cells)
        
        if n_cells <= 1:
            for cell in cells:
                if cell.is_anchor:
                    cell.center = np.array([0.0, 0.0])
            self.rebuild_spatial_grid(tissue)
            return
        
        # Extract data to contiguous numpy arrays
        positions = np.array([c.center for c in cells], dtype=np.float64)
        radii = np.array([c.radius for c in cells], dtype=np.float64)
        is_anchor = np.array([c.is_anchor for c in cells], dtype=bool)
        masses = np.array([self._cell_mass(c) for c in cells], dtype=np.float64)
        
        velocities = np.zeros((n_cells, 2), dtype=np.float64)
        
        # Maximum interaction distance (cells interact within 4x max radius)
        max_radius = np.max(radii)
        interaction_dist = max_radius * 3.5
        
        anchor_indices = np.where(is_anchor)[0]
        
        for iteration in range(iterations):
            if HAS_SCIPY:
                forces, max_force = self._compute_forces_kdtree(
                    positions, radii, is_anchor, interaction_dist, n_cells
                )
            else:
                forces, max_force = self._compute_forces_grid(
                    positions, radii, is_anchor, n_cells, tissue, cells
                )
            
            if max_force < self.convergence_threshold:
                break
            
            # Update velocities: v = damping * v + (force / mass) * dt
            # Using mass-weighted update for stability
            inv_mass = 0.12 / masses
            velocities = velocities * self.damping + forces * inv_mass[:, np.newaxis]
            
            # Velocity cap for stability
            vel_mags = np.linalg.norm(velocities, axis=1)
            too_fast = vel_mags > 0.35
            if np.any(too_fast):
                velocities[too_fast] = (velocities[too_fast].T / vel_mags[too_fast] * 0.35).T
            
            # Update positions
            positions += velocities
            
            # Reset anchor positions
            if len(anchor_indices) > 0:
                positions[anchor_indices] = 0.0
                velocities[anchor_indices] = 0.0
        
        # Final overlap resolution pass
        self._resolve_overlaps_vectorized(positions, radii, is_anchor, masses, n_cells)
        
        # Write positions back to cells
        for i, cell in enumerate(cells):
            cell.center = positions[i].copy()
        
        self.rebuild_spatial_grid(tissue)
    
    def _compute_forces_kdtree(self, positions: np.ndarray, radii: np.ndarray,
                                is_anchor: np.ndarray, interaction_dist: float,
                                n_cells: int) -> Tuple[np.ndarray, float]:
        """Compute forces using scipy cKDTree - fully vectorized."""
        
        # Build KD-tree (O(n log n) in C)
        tree = cKDTree(positions)
        
        # Find all pairs within interaction distance (O(n log n + k) in C)
        pairs = tree.query_pairs(interaction_dist, output_type='ndarray')
        
        forces = np.zeros((n_cells, 2), dtype=np.float64)
        
        if len(pairs) == 0:
            return forces, 0.0
        
        # Extract pair indices
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        
        # Compute deltas and distances (vectorized)
        deltas = positions[j_idx] - positions[i_idx]
        dists = np.linalg.norm(deltas, axis=1)
        
        # Avoid division by zero
        near_zero = dists < 0.0001
        if np.any(near_zero):
            deltas[near_zero] = np.random.randn(np.sum(near_zero), 2) * 0.1
            dists[near_zero] = np.linalg.norm(deltas[near_zero], axis=1)
        
        # Compute sum of radii for each pair
        sum_radii = radii[i_idx] + radii[j_idx]
        soft_margin = sum_radii * 0.05
        
        # Find overlapping pairs
        overlap_threshold = sum_radii - soft_margin
        overlapping = dists < overlap_threshold
        
        if not np.any(overlapping):
            return forces, 0.0
        
        # Compute overlap amounts (only for overlapping pairs)
        overlaps = np.zeros(len(pairs), dtype=np.float64)
        overlaps[overlapping] = overlap_threshold[overlapping] - dists[overlapping]
        
        # Force magnitude: F = k * overlap * (1 + overlap * 0.5)
        force_mags = self.repulsion_constant * overlaps * (1 + overlaps * 0.5)
        
        # Direction vectors (normalized)
        directions = deltas / dists[:, np.newaxis]
        
        # Force vectors
        force_vectors = force_mags[:, np.newaxis] * directions
        
        # Accumulate forces using np.add.at (handles duplicate indices)
        # Cell i feels repulsion FROM j, so force is in -direction
        # Cell j feels repulsion FROM i, so force is in +direction
        np.add.at(forces, i_idx, -force_vectors)
        np.add.at(forces, j_idx, force_vectors)
        
        # Zero out forces on anchor cells
        forces[is_anchor] = 0.0
        
        max_force = np.max(np.linalg.norm(forces, axis=1))
        
        return forces, max_force
    
    def _compute_forces_grid(self, positions: np.ndarray, radii: np.ndarray,
                              is_anchor: np.ndarray, n_cells: int,
                              tissue: 'Tissue', cells: list) -> Tuple[np.ndarray, float]:
        """Fallback force computation using spatial grid (when scipy unavailable)."""
        
        forces = np.zeros((n_cells, 2), dtype=np.float64)
        cell_ids = [c.id for c in cells]
        id_to_idx = {cid: i for i, cid in enumerate(cell_ids)}
        
        # Rebuild grid for this iteration
        for i, cell in enumerate(cells):
            self.spatial_grid.insert(cell.id, positions[i])
        
        for i in range(n_cells):
            if is_anchor[i]:
                continue
            
            nearby_ids = self.spatial_grid.get_nearby_cell_ids(
                positions[i], radius=radii[i] * 4
            )
            
            for other_id in nearby_ids:
                j = id_to_idx.get(other_id)
                if j is None or j <= i:  # Avoid double-counting pairs
                    continue
                
                delta = positions[j] - positions[i]
                dist = np.linalg.norm(delta)
                
                if dist < 0.0001:
                    delta = np.random.randn(2) * 0.1
                    dist = np.linalg.norm(delta)
                
                direction = delta / dist
                sum_radii = radii[i] + radii[j]
                soft_margin = sum_radii * 0.05
                
                if dist < sum_radii - soft_margin:
                    overlap = (sum_radii - soft_margin) - dist
                    force_mag = self.repulsion_constant * overlap * (1 + overlap * 0.5)
                    
                    force_vec = force_mag * direction
                    if not is_anchor[i]:
                        forces[i] -= force_vec
                    if not is_anchor[j]:
                        forces[j] += force_vec
        
        max_force = np.max(np.linalg.norm(forces, axis=1))
        return forces, max_force
    
    def _resolve_overlaps_vectorized(self, positions: np.ndarray, radii: np.ndarray,
                                      is_anchor: np.ndarray, masses: np.ndarray,
                                      n_cells: int):
        """Direct geometric overlap resolution - vectorized version."""
        
        max_radius = np.max(radii)
        
        for _ in range(3):
            if HAS_SCIPY:
                tree = cKDTree(positions)
                pairs = tree.query_pairs(max_radius * 2.2, output_type='ndarray')
                
                if len(pairs) == 0:
                    break
                
                i_idx = pairs[:, 0]
                j_idx = pairs[:, 1]
                
                deltas = positions[j_idx] - positions[i_idx]
                dists = np.linalg.norm(deltas, axis=1)
                
                near_zero = dists < 0.0001
                if np.any(near_zero):
                    deltas[near_zero] = np.random.randn(np.sum(near_zero), 2) * 0.1
                    dists[near_zero] = np.linalg.norm(deltas[near_zero], axis=1)
                
                min_dists = (radii[i_idx] + radii[j_idx]) * 0.92
                overlapping = dists < min_dists
                
                if not np.any(overlapping):
                    break
                
                # Process overlapping pairs
                for idx in np.where(overlapping)[0]:
                    i, j = i_idx[idx], j_idx[idx]
                    overlap = min_dists[idx] - dists[idx]
                    direction = deltas[idx] / dists[idx]
                    
                    if is_anchor[i] and not is_anchor[j]:
                        positions[j] += direction * overlap
                    elif is_anchor[j] and not is_anchor[i]:
                        positions[i] -= direction * overlap
                    elif not is_anchor[i] and not is_anchor[j]:
                        total_mass = masses[i] + masses[j]
                        share_i = masses[j] / total_mass
                        share_j = masses[i] / total_mass
                        max_step = min(radii[i], radii[j]) * 0.25
                        move_i = min(overlap * share_i, max_step)
                        move_j = min(overlap * share_j, max_step)
                        positions[i] -= direction * move_i
                        positions[j] += direction * move_j
            else:
                break  # Skip detailed resolution without scipy
        
        # Ensure anchors at origin
        positions[is_anchor] = 0.0
    
    def _cell_mass(self, cell: 'Cell') -> float:
        """Effective mass for displacement sharing."""
        if cell.age == 0:
            return 0.15
        return 1.0 + min(cell.age - 1, 8) * 0.25
    
    def get_neighbors(self, cell: 'Cell', tissue: 'Tissue',
                      distance_factor: float = 2.5) -> List['Cell']:
        """Get cells within distance_factor × (sum of radii)."""
        neighbors = []
        max_radius = cell.radius * distance_factor * 2
        
        nearby_ids = self.spatial_grid.get_nearby_cell_ids(cell.center, radius=max_radius)
        
        for other_id in nearby_ids:
            if other_id == cell.id:
                continue
            
            other = tissue.get_cell(other_id)
            if other is None:
                continue
            
            dist = np.linalg.norm(other.center - cell.center)
            threshold = (cell.radius + other.radius) * distance_factor
            
            if dist < threshold:
                neighbors.append(other)
        
        return neighbors
    
    def get_touching_neighbors(self, cell: 'Cell', tissue: 'Tissue') -> List['Cell']:
        """Get cells that are actually touching."""
        neighbors = []
        nearby_ids = self.spatial_grid.get_nearby_cell_ids(cell.center, radius=cell.radius * 3)
        
        for other_id in nearby_ids:
            if other_id == cell.id:
                continue
            
            other = tissue.get_cell(other_id)
            if other is None:
                continue
            
            dist = np.linalg.norm(other.center - cell.center)
            if dist < (cell.radius + other.radius) * 1.15:
                neighbors.append(other)
        
        return neighbors