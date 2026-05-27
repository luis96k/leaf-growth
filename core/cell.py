"""
Cell representation - simplified without explicit contacts.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import uuid
import time


@dataclass 
class Cell:
    """
    A single cell in the tissue.
    """
    
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    cell_type: str = "default"
    center: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    area: float = 1.0
    is_anchor: bool = False
    can_divide: bool = True
    division_count: int = 0
    age: int = 0
    creation_time: float = field(default_factory=time.time)
    parent_id: Optional[str] = None
    sibling_id: Optional[str] = None
    generated_gradient_name: Optional[str] = None
    facing_direction: float = 90.0  # NEW: direction cell is "facing"
    
    @property
    def radius(self) -> float:
        """Equivalent radius for a circle of this area: r = √(A/π)"""
        return np.sqrt(self.area / np.pi)
    
    @property
    def visual_radius(self) -> float:
        """Extended radius for visual 'jelly' layer (40% larger)."""
        return self.radius * 1.4
    
    @property
    def physical_radius(self) -> float:
        """Alias for actual collision radius."""
        return self.radius
    
    def distance_to(self, other: 'Cell') -> float:
        """Euclidean distance to another cell's center."""
        return np.linalg.norm(self.center - other.center)
    
    def direction_to(self, other: 'Cell') -> float:
        """Angle (degrees) from this cell toward another. 0=East, 90=North."""
        delta = other.center - self.center
        if np.linalg.norm(delta) < 0.0001:
            return 0.0
        return np.degrees(np.arctan2(delta[1], delta[0])) % 360
    
    def __repr__(self):
        return f"Cell({self.id}, type={self.cell_type}, pos=({self.center[0]:.2f}, {self.center[1]:.2f}))"