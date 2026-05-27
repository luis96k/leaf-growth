"""
Gradient field definitions with support for cell-generated micro-gradients.

Mathematical models:
- Radial decreasing: G(r) = e^(-s·r) where r = distance from center
- Radial increasing: G(r) = 1 - e^(-s·r)  
- Asymmetric linear: G(x) = 0.5 + 0.5·tanh(s·projection) along axis
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Callable, Any, List


class GradientType(Enum):
    RADIAL_DECREASING = "radial_decreasing"
    RADIAL_INCREASING = "radial_increasing"
    SYMMETRIC_LINEAR_INCREASING = "symmetric_linear_increasing"
    SYMMETRIC_LINEAR_DECREASING = "symmetric_linear_decreasing"
    ASYMMETRIC_LINEAR = "asymmetric_linear"


@dataclass
class Gradient:
    name: str
    gradient_type: GradientType
    axis_angle: float = 0.0
    scale: float = 1.0
    min_value: float = 0.0
    max_value: float = 1.0
    attached_to_cell_id: Optional[str] = None
    _position_getter: Optional[Callable[[], np.ndarray]] = field(default=None, repr=False)
    fixed_position: Optional[np.ndarray] = None  # ADD THIS
    weight: float = 1.0  # ADD THIS
    
    @property
    def center(self) -> np.ndarray:
        """Get the center point of this gradient."""
        if self.fixed_position is not None:
            return self.fixed_position
        if self._position_getter is not None:
            return self._position_getter()
        return np.array([0.0, 0.0])
    
    def get_value(self, position: np.ndarray) -> float:
        """
        Get gradient concentration at a position.
        
        For radial_decreasing centered at C with scale s:
            G(P) = e^(-s · ||P - C||)
        
        This gives 1.0 at center, approaching 0 far away.
        """
        rel_pos = position - self.center
        x, y = rel_pos[0], rel_pos[1]
        distance = np.sqrt(x**2 + y**2)
        
        if self.gradient_type == GradientType.RADIAL_DECREASING:
            # Maximum at center, decays exponentially outward
            raw_value = np.exp(-distance * self.scale)
            
        elif self.gradient_type == GradientType.RADIAL_INCREASING:
            # Minimum at center, rises outward
            raw_value = 1 - np.exp(-distance * self.scale)
            
        elif self.gradient_type == GradientType.SYMMETRIC_LINEAR_INCREASING:
            # Minimum at axis center, increases along axis in both directions
            axis_rad = np.radians(self.axis_angle)
            projection = abs(x * np.cos(axis_rad) + y * np.sin(axis_rad))
            raw_value = 1 - np.exp(-projection * self.scale)
            
        elif self.gradient_type == GradientType.SYMMETRIC_LINEAR_DECREASING:
            axis_rad = np.radians(self.axis_angle)
            projection = abs(x * np.cos(axis_rad) + y * np.sin(axis_rad))
            raw_value = np.exp(-projection * self.scale)
            
        elif self.gradient_type == GradientType.ASYMMETRIC_LINEAR:
            # Smooth transition from 0 to 1 along axis direction
            # Uses tanh for bounded, smooth transition
            axis_rad = np.radians(self.axis_angle)
            projection = x * np.cos(axis_rad) + y * np.sin(axis_rad)
            raw_value = 0.5 + 0.5 * np.tanh(projection * self.scale)
            
        else:
            raw_value = 0.5
        
        # Map [0,1] to [min_value, max_value]
        return self.min_value + raw_value * (self.max_value - self.min_value)
    
    def get_direction(self, position: np.ndarray) -> float:
        """Get direction of gradient increase (with analytical shortcuts)."""
        rel_pos = position - self.center
        
        # Analytical solutions for common types
        if self.gradient_type == GradientType.RADIAL_DECREASING:
            # Points toward center (direction of increase is inward)
            if np.linalg.norm(rel_pos) < 0.001:
                return 0.0
            return (np.degrees(np.arctan2(-rel_pos[1], -rel_pos[0]))) % 360
        
        elif self.gradient_type == GradientType.RADIAL_INCREASING:
            # Points away from center
            if np.linalg.norm(rel_pos) < 0.001:
                return 0.0
            return (np.degrees(np.arctan2(rel_pos[1], rel_pos[0]))) % 360
        
        elif self.gradient_type == GradientType.ASYMMETRIC_LINEAR:
            # Direction is just the axis angle
            return self.axis_angle
        
        # Fall back to numerical for symmetric linear types
        epsilon = 0.01
        val_px = self.get_value(position + np.array([epsilon, 0]))
        val_mx = self.get_value(position + np.array([-epsilon, 0]))
        val_py = self.get_value(position + np.array([0, epsilon]))
        val_my = self.get_value(position + np.array([0, -epsilon]))
        
        grad_x = (val_px - val_mx) / (2 * epsilon)
        grad_y = (val_py - val_my) / (2 * epsilon)
        
        if abs(grad_x) < 1e-10 and abs(grad_y) < 1e-10:
            return self.axis_angle if self.gradient_type == GradientType.ASYMMETRIC_LINEAR else 0.0
        
        return np.degrees(np.arctan2(grad_y, grad_x)) % 360


@dataclass
class GradientField:
    """
    Collection of all gradients in the simulation environment.
    
    Supports both static gradients and cell-attached micro-gradients.
    """
    
    gradients: Dict[str, Gradient] = field(default_factory=dict)
    _tissue_ref: Any = field(default=None, repr=False)
    
    def set_tissue_reference(self, tissue):
        """Set reference to tissue for looking up cell positions."""
        self._tissue_ref = tissue
    
    def add_gradient(self, gradient: Gradient):
        """Add a gradient. If cell-attached, sets up position tracking."""
        if gradient.attached_to_cell_id is not None and self._tissue_ref is not None:
            cell_id = gradient.attached_to_cell_id
            # Create closure to get cell position dynamically
            gradient._position_getter = lambda cid=cell_id: self._get_cell_position(cid)
        
        self.gradients[gradient.name] = gradient
    
    def _get_cell_position(self, cell_id: str) -> np.ndarray:
        """Get current position of a cell by ID."""
        if self._tissue_ref is None:
            return np.array([0.0, 0.0])
        cell = self._tissue_ref.get_cell(cell_id)
        return cell.center if cell else np.array([0.0, 0.0])
    
    def remove_gradient(self, name: str):
        """Remove a gradient by name."""
        if name in self.gradients:
            del self.gradients[name]
    
    def remove_gradients_attached_to(self, cell_id: str):
        """Remove all gradients attached to a cell (when cell dies)."""
        to_remove = [
            name for name, g in self.gradients.items()
            if g.attached_to_cell_id == cell_id
        ]
        for name in to_remove:
            del self.gradients[name]
    
    def get_gradient(self, name: str) -> Optional[Gradient]:
        return self.gradients.get(name)
    
    def get_value(self, name: str, position: np.ndarray) -> float:
        """Get value of a specific gradient at position."""
        gradient = self.gradients.get(name)
        return gradient.get_value(position) if gradient else 0.0
    
    def get_direction(self, name: str, position: np.ndarray) -> float:
        """Get direction of gradient increase."""
        gradient = self.gradients.get(name)
        return gradient.get_direction(position) if gradient else 0.0
    
    def get_all_values(self, position: np.ndarray) -> Dict[str, float]:
        """Get all gradient values at a position."""
        return {name: g.get_value(position) for name, g in self.gradients.items()}
    
    def __contains__(self, name: str) -> bool:
        return name in self.gradients

    def get_summed_value(self, prefix: str, position: np.ndarray) -> float:
        """
        Get SUMMED value of all gradients whose names start with prefix.
        Multiple micro-gradients of same type add together.
        Result is clamped to [0, 1].
        """
        matching = [g for name, g in self.gradients.items() if name.startswith(prefix)]
        if not matching:
            return 0.0
        
        total = sum(g.get_value(position) * g.weight for g in matching)
        return min(1.0, total)

    def create_persistent_gradient(self, name: str, position: np.ndarray,
                                gradient_type: GradientType = GradientType.RADIAL_DECREASING,
                                scale: float = 1.0) -> Gradient:
        """
        Create a gradient fixed at a position (doesn't move with cells).
        """
        gradient = Gradient(
            name=name,
            gradient_type=gradient_type,
            scale=scale,
            fixed_position=position.copy()
        )
        self.gradients[name] = gradient
        return gradient