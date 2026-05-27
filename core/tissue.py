"""
Tissue management: orchestrates cells, rules, physics, and gradients.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any

from .cell import Cell
from .gradients import GradientField, Gradient
from .rules import DivisionRule, ApoptosisRule, DifferentiationRule, perform_division, perform_differentiation
from .physics import PhysicsEngine


class Tissue:
    """
    The main simulation container.
    
    Manages:
    - Collection of cells
    - Gradient fields (static and cell-attached)
    - Division and apoptosis rules
    - Physics simulation
    """
    
    def __init__(self):
        self.cells: Dict[str, Cell] = {}
        self.gradient_field: GradientField = GradientField()
        self.division_rules: Dict[str, List[DivisionRule]] = {}
        self.apoptosis_rules: Dict[str, List[ApoptosisRule]] = {}
        self.differentiation_rules: Dict[str, List[DifferentiationRule]] = {}
        self.physics: PhysicsEngine = PhysicsEngine()
        self.cell_colors: Dict[str, str] = {}
        self.simulation_step: int = 0
        
        # Link gradient field to tissue for cell-attached gradients
        self.gradient_field.set_tissue_reference(self)
        
        # Callbacks for external notification
        self.on_cell_added: Optional[Callable[[Cell], None]] = None
        self.on_cell_removed: Optional[Callable[[str], None]] = None
        self.on_cell_divided: Optional[Callable[[Cell, Cell, Cell], None]] = None
        self.on_cell_died: Optional[Callable[[Cell], None]] = None
        self.on_cell_differentiated: Optional[Callable[[Cell, str, str], None]] = None  # cell, old_type, new_type
        self.on_step_complete: Optional[Callable[[int], None]] = None

        self._stats_dirty: bool = True
        self._cached_stats: Optional[Dict] = None

        # Neighbor cache (cleared each step)
        self._neighbor_cache: Dict[str, List[Cell]] = {}
        self._neighbor_cache_factor: float = 0.0
    
    def add_cell(self, cell: Cell):
        """Add a cell to the tissue."""
        self.cells[cell.id] = cell
        self.physics.spatial_grid.insert(cell.id, cell.center)
        if self.on_cell_added:
            self.on_cell_added(cell)
    
    def remove_cell(self, cell_id: str):
        """Remove a cell and clean up associated data."""
        if cell_id not in self.cells:
            return
        
        # Remove any gradients this cell was generating
        self.gradient_field.remove_gradients_attached_to(cell_id)
        
        # Remove from spatial grid
        self.physics.spatial_grid.remove(cell_id)
        
        del self.cells[cell_id]
        
        if self.on_cell_removed:
            self.on_cell_removed(cell_id)
    
    def get_cell(self, cell_id: str) -> Optional[Cell]:
        """Get a cell by ID."""
        return self.cells.get(cell_id)
    
    def add_gradient(self, gradient: Gradient):
        """Add a gradient to the environment."""
        self.gradient_field.add_gradient(gradient)
    
    def add_division_rule(self, rule: DivisionRule):
        """Add a division rule (sorted by priority)."""
        if rule.cell_type not in self.division_rules:
            self.division_rules[rule.cell_type] = []
        
        self.division_rules[rule.cell_type].append(rule)
        self.division_rules[rule.cell_type].sort(key=lambda r: -r.priority)
    
    def add_apoptosis_rule(self, rule: ApoptosisRule):
        """Add an apoptosis rule."""
        if rule.cell_type not in self.apoptosis_rules:
            self.apoptosis_rules[rule.cell_type] = []
        
        self.apoptosis_rules[rule.cell_type].append(rule)
        self.apoptosis_rules[rule.cell_type].sort(key=lambda r: -r.priority)

    def add_differentiation_rule(self, rule: 'DifferentiationRule'):
        """Add a differentiation rule (sorted by priority)."""
        if rule.source_type not in self.differentiation_rules:
            self.differentiation_rules[rule.source_type] = []
        
        self.differentiation_rules[rule.source_type].append(rule)
        self.differentiation_rules[rule.source_type].sort(key=lambda r: -r.priority)
    
    def set_cell_color(self, cell_type: str, color: str):
        """Set display color for a cell type."""
        self.cell_colors[cell_type] = color
    
    def get_cell_color(self, cell_type: str) -> str:
        """Get display color for a cell type."""
        return self.cell_colors.get(cell_type, "#808080")
    
    def create_initial_cell(self, cell_type: str = "default") -> Cell:
        """Create the first cell at origin (anchored)."""
        cell = Cell(
            cell_type=cell_type,
            center=np.array([0.0, 0.0]),
            area=1.0,
            is_anchor=True
        )
        self.add_cell(cell)
        return cell
    
    # =========================================================================
    # Neighbor queries (delegated to physics engine)
    # =========================================================================
    
    def get_neighbors(self, cell: Cell, distance_factor: float = 2.5) -> List[Cell]:
        """Get cells within distance_factor × sum_of_radii (cached within step)."""
        # Check cache
        cache_key = f"{cell.id}_{distance_factor}"
        if cache_key in self._neighbor_cache:
            return self._neighbor_cache[cache_key]
        
        result = self.physics.get_neighbors(cell, self, distance_factor)
        self._neighbor_cache[cache_key] = result
        return result
    
    def count_neighbors(self, cell: Cell, distance_factor: float = 2.5) -> int:
        """Count neighboring cells."""
        return len(self.get_neighbors(cell, distance_factor))
    
    def get_neighbors_of_type(self, cell: Cell, cell_type: str, 
                               distance_factor: float = 2.5) -> List[Cell]:
        """Get neighboring cells of a specific type."""
        return [n for n in self.get_neighbors(cell, distance_factor) 
                if n.cell_type == cell_type]
    
    def count_neighbors_of_type(self, cell: Cell, cell_type: str,
                                 distance_factor: float = 2.5) -> int:
        """Count neighboring cells of a specific type."""
        return len(self.get_neighbors_of_type(cell, cell_type, distance_factor))
    
    def get_touching_neighbors(self, cell: Cell) -> List[Cell]:
        """Get cells that are physically touching (for rendering)."""
        return self.physics.get_touching_neighbors(cell, self)
    
    # =========================================================================
    # Simulation step
    # =========================================================================
    
    def step(self) -> Tuple[List[Tuple[Cell, Tuple[Cell, Cell]]], List[Cell], List[Tuple[Cell, str, str]]]:
        """
        Perform one simulation step.
        
        Order of operations:
        1. Age all cells
        2. Check and apply apoptosis rules
        3. Check and apply differentiation rules (NEW)
        4. Check and apply division rules
        5. Run physics to settle positions
        
        Returns:
            (list of (parent, (daughter1, daughter2)) for divisions,
            list of cells that died,
            list of (cell, old_type, new_type) for differentiations)
        """
        
        # Clear neighbor cache
        self._neighbor_cache.clear()
        
        # 1. Age all cells
        for cell in self.cells.values():
            cell.age += 1
        
        # 2. Apoptosis
        cells_to_die = []
        for cell in list(self.cells.values()):
            rules = self.apoptosis_rules.get(cell.cell_type, [])
            for rule in rules:
                if rule.can_apply(cell, self, self.gradient_field):
                    cells_to_die.append(cell)
                    break
        
        dead_cells = []
        for cell in cells_to_die:
            dead_cells.append(cell)
            if self.on_cell_died:
                self.on_cell_died(cell)
            self.remove_cell(cell.id)
        
        # 3. Differentiation (NEW - before division)
        differentiation_results = []
        for cell in list(self.cells.values()):
            rules = self.differentiation_rules.get(cell.cell_type, [])
            for rule in rules:
                if rule.can_apply(cell, self, self.gradient_field):
                    old_type = cell.cell_type
                    perform_differentiation(cell, rule)
                    differentiation_results.append((cell, old_type, rule.target_type))
                    
                    if self.on_cell_differentiated:
                        self.on_cell_differentiated(cell, old_type, rule.target_type)
                    break  # Only one differentiation per step per cell
        
        # 4. Division
        divisions_to_perform = []
        for cell in list(self.cells.values()):
            if not cell.can_divide:
                continue
            
            rules = self.division_rules.get(cell.cell_type, [])
            for rule in rules:
                if rule.can_apply(cell, self, self.gradient_field):
                    divisions_to_perform.append((cell, rule))
                    break
        
        division_results = []
        for cell, rule in divisions_to_perform:
            daughter1, daughter2 = perform_division(
                cell, rule, self, self.gradient_field
            )
            
            self.remove_cell(cell.id)
            self.add_cell(daughter1)
            self.add_cell(daughter2)
            
            division_results.append((cell, (daughter1, daughter2)))
            
            if self.on_cell_divided:
                self.on_cell_divided(cell, daughter1, daughter2)
        
        # 5. Physics
        self.physics.settle(self)
        
        self.simulation_step += 1
        self._stats_dirty = True
        
        if self.on_step_complete:
            self.on_step_complete(self.simulation_step)
        
        return division_results, dead_cells, differentiation_results
    
    def run_steps(self, n: int) -> Tuple[int, int, int]:
        """Run multiple steps. Returns (total_divisions, total_deaths, total_differentiations)."""
        total_divisions = 0
        total_deaths = 0
        total_differentiations = 0
        for _ in range(n):
            divisions, deaths, differentiations = self.step()
            total_divisions += len(divisions)
            total_deaths += len(deaths)
            total_differentiations += len(differentiations)
        return total_divisions, total_deaths, total_differentiations
    
    # =========================================================================
    # Utility methods
    # =========================================================================
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box (min_x, min_y, max_x, max_y)."""
        if not self.cells:
            return (-5, -5, 5, 5)
        
        min_x = min(c.center[0] - c.radius * 1.5 for c in self.cells.values())
        max_x = max(c.center[0] + c.radius * 1.5 for c in self.cells.values())
        min_y = min(c.center[1] - c.radius * 1.5 for c in self.cells.values())
        max_y = max(c.center[1] + c.radius * 1.5 for c in self.cells.values())
        
        return (min_x - 1, min_y - 1, max_x + 1, max_y + 1)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tissue statistics (cached)."""
        if not self._stats_dirty and self._cached_stats is not None:
            return self._cached_stats
        
        type_counts = {}
        for cell in self.cells.values():
            type_counts[cell.cell_type] = type_counts.get(cell.cell_type, 0) + 1
        
        self._cached_stats = {
            "total_cells": len(self.cells),
            "type_counts": type_counts,
            "simulation_step": self.simulation_step,
            "bounds": self.get_bounds(),
            "num_division_rules": sum(len(rules) for rules in self.division_rules.values()),
            "num_apoptosis_rules": sum(len(rules) for rules in self.apoptosis_rules.values()),
            "num_differentiation_rules": sum(len(rules) for rules in self.differentiation_rules.values()),
        }
        self._stats_dirty = False
        return self._cached_stats
    
    def get_cell_at_position(self, pos: np.ndarray) -> Optional[Cell]:
        """Get cell at world position (for click detection)."""
        nearby_ids = self.physics.spatial_grid.get_nearby_cell_ids(pos, radius=2.0)
        
        for cell_id in nearby_ids:
            cell = self.cells.get(cell_id)
            if cell and np.linalg.norm(pos - cell.center) < cell.radius:
                return cell
        
        return None