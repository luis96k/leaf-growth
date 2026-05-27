"""
Cell division and apoptosis rules with boolean condition logic.
"""

import numpy as np
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TYPE_CHECKING, Any
from enum import Enum
import uuid

if TYPE_CHECKING:
    from .cell import Cell
    from .tissue import Tissue
    from .gradients import GradientField


class ComparisonOp(Enum):
    GREATER = ">"
    LESS = "<"
    GREATER_EQ = ">="
    LESS_EQ = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="


def _compare(value: float, op: ComparisonOp, threshold: float) -> bool:
    if op == ComparisonOp.GREATER:
        return value > threshold
    elif op == ComparisonOp.LESS:
        return value < threshold
    elif op == ComparisonOp.GREATER_EQ:
        return value >= threshold
    elif op == ComparisonOp.LESS_EQ:
        return value <= threshold
    elif op == ComparisonOp.EQUAL:
        return abs(value - threshold) < 0.001
    elif op == ComparisonOp.NOT_EQUAL:
        return abs(value - threshold) >= 0.001
    return False

# =============================================================================
# NUMERIC EXPRESSIONS (for Random probability arguments)
# =============================================================================

@dataclass
class NumericExpression:
    """
    Parsed arithmetic expression tree. Evaluates to a float at runtime
    using the cell's current position to sample gradients.

    Supported ops: CONST, VAR, ADD, SUB, MUL, DIV, NEG
    Supported variables: gradient names (with or without * suffix), 'age'
    """
    op: str
    value: Optional[float] = None              # for CONST
    var_name: Optional[str] = None             # for VAR
    left: Optional['NumericExpression'] = None
    right: Optional['NumericExpression'] = None

    def evaluate(self, cell: 'Cell', gradient_field: 'GradientField') -> float:
        if self.op == 'CONST':
            return self.value

        elif self.op == 'VAR':
            name = self.var_name
            if name == 'age':
                return float(cell.age)
            if name.endswith('*'):
                return gradient_field.get_summed_value(name[:-1], cell.center)
            if name in gradient_field:
                return gradient_field.get_value(name, cell.center)
            # Fallback: treat as micro-gradient prefix (matches GradientCondition behavior)
            return gradient_field.get_summed_value(name, cell.center)

        elif self.op == 'ADD':
            return self.left.evaluate(cell, gradient_field) + self.right.evaluate(cell, gradient_field)
        elif self.op == 'SUB':
            return self.left.evaluate(cell, gradient_field) - self.right.evaluate(cell, gradient_field)
        elif self.op == 'MUL':
            return self.left.evaluate(cell, gradient_field) * self.right.evaluate(cell, gradient_field)
        elif self.op == 'DIV':
            denom = self.right.evaluate(cell, gradient_field)
            if abs(denom) < 1e-12:
                return 0.0  # safe divide-by-zero
            return self.left.evaluate(cell, gradient_field) / denom
        elif self.op == 'NEG':
            return -self.left.evaluate(cell, gradient_field)

        return 0.0


@dataclass
class RandomCondition:
    """
    Stochastic condition. Rolls a uniform random [0,1) each time it's
    evaluated and returns True if the roll is below the probability.

    probability_expr=None → uses 0.5 (fair coin flip).
    Result is clamped to [0,1], so expressions can safely overshoot.
    """
    probability_expr: Optional[NumericExpression] = None

    def evaluate(self, cell: 'Cell', gradient_field: 'GradientField') -> bool:
        if self.probability_expr is None:
            prob = 0.5
        else:
            prob = self.probability_expr.evaluate(cell, gradient_field)
        prob = max(0.0, min(1.0, prob))
        return np.random.random() < prob

# =============================================================================
# LEAF CONDITIONS
# =============================================================================

@dataclass
class GradientCondition:
    gradient_name: str
    operator: ComparisonOp
    threshold: float
    
    def evaluate(self, cell: 'Cell', gradient_field: 'GradientField') -> bool:
        if self.gradient_name in gradient_field:
            value = gradient_field.get_value(self.gradient_name, cell.center)
        else:
            value = gradient_field.get_summed_value(self.gradient_name, cell.center)
        return _compare(value, self.operator, self.threshold)


@dataclass
class GradientPatternCondition:
    pattern: str
    operator: ComparisonOp
    threshold: float
    
    def evaluate(self, cell: 'Cell', gradient_field: 'GradientField') -> bool:
        if self.pattern.endswith('*'):
            prefix = self.pattern[:-1]
            value = gradient_field.get_summed_value(prefix, cell.center)
        else:
            value = gradient_field.get_value(self.pattern, cell.center)
        return _compare(value, self.operator, self.threshold)


@dataclass
class AgeCondition:
    operator: ComparisonOp
    threshold: int
    
    def evaluate(self, cell: 'Cell') -> bool:
        return _compare(cell.age, self.operator, self.threshold)


@dataclass
class NeighborCountCondition:
    operator: ComparisonOp
    threshold: int
    distance_factor: float = 1.8
    
    def evaluate(self, cell: 'Cell', tissue: 'Tissue') -> bool:
        count = tissue.count_neighbors(cell, self.distance_factor)
        return _compare(count, self.operator, self.threshold)


@dataclass
class NeighborTypeCondition:
    cell_type: str
    operator: ComparisonOp
    threshold: int
    distance_factor: float = 1.8
    
    def evaluate(self, cell: 'Cell', tissue: 'Tissue') -> bool:
        count = tissue.count_neighbors_of_type(cell, self.cell_type, self.distance_factor)
        return _compare(count, self.operator, self.threshold)

@dataclass
class DirectionalNeighborCondition:
    """Check if there's a neighbor within a cone in a specific direction."""
    direction_mode: str = "facing"  # "facing" or a gradient name
    angle_tolerance: float = 45.0   # half-cone width in degrees
    distance_factor: float = 1.8    # how far to look (× sum of radii)
    
    def evaluate(self, cell: 'Cell', tissue: 'Tissue', gradient_field: 'GradientField') -> bool:
        if self.direction_mode == "facing":
            check_angle = cell.facing_direction
        else:
            check_angle = gradient_field.get_direction(self.direction_mode, cell.center)
        
        neighbors = tissue.get_neighbors(cell, self.distance_factor)
        for n in neighbors:
            delta = n.center - cell.center
            dist = np.linalg.norm(delta)
            if dist < 0.001:
                continue
            neighbor_angle = np.degrees(np.arctan2(delta[1], delta[0])) % 360
            diff = abs(neighbor_angle - check_angle)
            if diff > 180:
                diff = 360 - diff
            if diff <= self.angle_tolerance:
                return True
        return False

@dataclass
class SurroundedCondition:
    min_angle_difference: float = 120.0
    
    def evaluate(self, cell: 'Cell', tissue: 'Tissue') -> bool:
        neighbors = tissue.get_neighbors(cell, distance_factor=1.3)
        if len(neighbors) < 2:
            return False
        
        angles = []
        for n in neighbors:
            delta = n.center - cell.center
            if np.linalg.norm(delta) > 0.001:
                angles.append(np.degrees(np.arctan2(delta[1], delta[0])) % 360)
        
        if len(angles) < 2:
            return False
        
        for i, a1 in enumerate(angles):
            for a2 in angles[i+1:]:
                diff = abs(a1 - a2)
                if diff > 180:
                    diff = 360 - diff
                if diff >= self.min_angle_difference:
                    return True
        return False


# =============================================================================
# PARSED CONDITION (Boolean tree)
# =============================================================================

@dataclass
class ParsedCondition:
    op: str  # 'AND', 'OR', 'NOT', 'LEAF'
    left: Optional['ParsedCondition'] = None
    right: Optional['ParsedCondition'] = None
    leaf_condition: Optional[Any] = None
    
    def evaluate(self, cell: 'Cell', tissue: 'Tissue', gradient_field: 'GradientField') -> bool:
        if self.op == 'LEAF':
            return self._eval_leaf(cell, tissue, gradient_field)
        elif self.op == 'AND':
            return (self.left.evaluate(cell, tissue, gradient_field) and 
                    self.right.evaluate(cell, tissue, gradient_field))
        elif self.op == 'OR':
            return (self.left.evaluate(cell, tissue, gradient_field) or
                    self.right.evaluate(cell, tissue, gradient_field))
        elif self.op == 'NOT':
            return not self.left.evaluate(cell, tissue, gradient_field)
        return False
    
    def _eval_leaf(self, cell, tissue, gradient_field) -> bool:
        cond = self.leaf_condition
        if cond is None:
            return True
        
        if isinstance(cond, SurroundedCondition):
            return cond.evaluate(cell, tissue)
        elif isinstance(cond, AgeCondition):
            return cond.evaluate(cell)
        elif isinstance(cond, NeighborCountCondition):
            return cond.evaluate(cell, tissue)
        elif isinstance(cond, NeighborTypeCondition):
            return cond.evaluate(cell, tissue)
        elif isinstance(cond, GradientCondition):
            return cond.evaluate(cell, gradient_field)
        elif isinstance(cond, GradientPatternCondition):
            return cond.evaluate(cell, gradient_field)
        elif isinstance(cond, DirectionalNeighborCondition):
            return cond.evaluate(cell, tissue, gradient_field)
        elif isinstance(cond, RandomCondition):
            return cond.evaluate(cell, gradient_field)
        
        return True


# =============================================================================
# DIVISION AXIS
# =============================================================================

class DivisionAxisType(Enum):
    FIXED = "fixed"
    RANDOM = "random"
    GRADIENT = "gradient"
    TOWARD_GRADIENT = "toward_gradient"
    TOWARD_NEIGHBOR_TYPE = "toward_neighbor_type"
    AWAY_FROM_NEIGHBOR_TYPE = "away_from_neighbor_type"
    FACING_RELATIVE = "facing_relative"
    FACING_BENT = "facing_bent"      # NEW


@dataclass
class DivisionAxis:
    axis_type: DivisionAxisType
    fixed_angle: float = 0.0
    gradient_name: Optional[str] = None
    angle_offset: float = 0.0
    random_spread: float = 0.0
    target_cell_type: Optional[str] = None
    
    def compute_axis(self, cell: 'Cell', tissue: 'Tissue', 
                     gradient_field: 'GradientField') -> float:
        base_angle = 0.0
        
        if self.axis_type == DivisionAxisType.FIXED:
            base_angle = self.fixed_angle
        
        elif self.axis_type == DivisionAxisType.RANDOM:
            return np.random.uniform(0, 360)
        
        elif self.axis_type == DivisionAxisType.TOWARD_GRADIENT:
            if self.gradient_name and self.gradient_name in gradient_field:
                base_angle = gradient_field.get_direction(self.gradient_name, cell.center)
            else:
                base_angle = cell.facing_direction
        
        elif self.axis_type == DivisionAxisType.GRADIENT:
            if self.gradient_name and self.gradient_name in gradient_field:
                base_angle = gradient_field.get_direction(self.gradient_name, cell.center) + 90
            else:
                base_angle = cell.facing_direction + 90
        
        elif self.axis_type == DivisionAxisType.TOWARD_NEIGHBOR_TYPE:
            if self.target_cell_type:
                neighbors = tissue.get_neighbors_of_type(cell, self.target_cell_type)
                if neighbors:
                    angles = [cell.direction_to(n) for n in neighbors]
                    base_angle = self._circular_mean(angles)
        
        elif self.axis_type == DivisionAxisType.AWAY_FROM_NEIGHBOR_TYPE:
            if self.target_cell_type:
                neighbors = tissue.get_neighbors_of_type(cell, self.target_cell_type)
                if neighbors:
                    angles = [cell.direction_to(n) for n in neighbors]
                    base_angle = self._circular_mean(angles) + 180

        elif self.axis_type == DivisionAxisType.FACING_RELATIVE:
            base_angle = cell.facing_direction

        elif self.axis_type == DivisionAxisType.FACING_BENT:
            facing = cell.facing_direction
            if self.gradient_name and self.gradient_name in gradient_field:
                target = gradient_field.get_direction(self.gradient_name, cell.center)
                diff = ((target - facing + 180) % 360) - 180
                max_bend = abs(self.angle_offset)
                bend = max(-max_bend, min(max_bend, diff))
                base_angle = facing + bend
            else:
                base_angle = facing
            # Return early — angle_offset was already consumed as max_bend
            if self.random_spread > 0:
                sign = np.random.choice([-1, 1])
                base_angle += sign * self.random_spread
            return base_angle % 360
        
        result = base_angle + self.angle_offset
        
        if self.random_spread > 0:
            sign = np.random.choice([-1, 1])
            result += sign * self.random_spread
        
        return result % 360
    
    @staticmethod
    def _circular_mean(angles_deg: List[float]) -> float:
        rads = [np.radians(a) for a in angles_deg]
        sin_sum = sum(np.sin(r) for r in rads)
        cos_sum = sum(np.cos(r) for r in rads)
        return np.degrees(np.arctan2(sin_sum, cos_sum)) % 360


# =============================================================================
# GENERATED GRADIENT
# =============================================================================

@dataclass
class GeneratedGradient:
    name_prefix: str
    gradient_type: str = "radial_decreasing"
    scale: float = 1.0
    attach_to: str = "daughter1"


# =============================================================================
# DIVISION RULE
# =============================================================================

@dataclass
class DivisionRule:
    cell_type: str
    daughter1_type: Optional[str] = None
    daughter2_type: Optional[str] = None
    priority: int = 0
    
    condition: Optional[ParsedCondition] = None
    division_axis: DivisionAxis = field(default_factory=lambda: DivisionAxis(DivisionAxisType.FIXED))
    generated_gradient: Optional[GeneratedGradient] = None
    max_divisions: int = 1000
    
    # Legacy condition lists (for backwards compatibility)
    gradient_conditions: List[GradientCondition] = field(default_factory=list)
    gradient_pattern_conditions: List[GradientPatternCondition] = field(default_factory=list)
    age_conditions: List[AgeCondition] = field(default_factory=list)
    neighbor_conditions: List[NeighborCountCondition] = field(default_factory=list)
    neighbor_type_conditions: List[NeighborTypeCondition] = field(default_factory=list)
    
    def __post_init__(self):
        if self.daughter1_type is None:
            self.daughter1_type = self.cell_type
        if self.daughter2_type is None:
            self.daughter2_type = self.cell_type
    
    def can_apply(self, cell: 'Cell', tissue: 'Tissue', 
                  gradient_field: 'GradientField') -> bool:
        # Fast checks first (no computation needed)
        if cell.cell_type != self.cell_type:
            return False
        
        if not cell.can_divide:
            return False
        
        if cell.division_count >= self.max_divisions:
            return False
        
        # Then evaluate complex conditions
        if self.condition is not None:
            return self.condition.evaluate(cell, tissue, gradient_field)
        
        # Legacy support
        for cond in self.gradient_conditions:
            if not cond.evaluate(cell, gradient_field):
                return False
        for cond in self.gradient_pattern_conditions:
            if not cond.evaluate(cell, gradient_field):
                return False
        for cond in self.age_conditions:
            if not cond.evaluate(cell):
                return False
        for cond in self.neighbor_conditions:
            if not cond.evaluate(cell, tissue):
                return False
        for cond in self.neighbor_type_conditions:
            if not cond.evaluate(cell, tissue):
                return False
        
        return True
    
    def get_axis(self, cell: 'Cell', tissue: 'Tissue',
                 gradient_field: 'GradientField') -> float:
        return self.division_axis.compute_axis(cell, tissue, gradient_field)


# =============================================================================
# APOPTOSIS RULE
# =============================================================================

@dataclass
class ApoptosisRule:
    cell_type: str
    priority: int = 0
    condition: Optional[ParsedCondition] = None
    
    # Legacy
    gradient_conditions: List[GradientCondition] = field(default_factory=list)
    gradient_pattern_conditions: List[GradientPatternCondition] = field(default_factory=list)
    age_conditions: List[AgeCondition] = field(default_factory=list)
    neighbor_conditions: List[NeighborCountCondition] = field(default_factory=list)
    neighbor_type_conditions: List[NeighborTypeCondition] = field(default_factory=list)
    
    def can_apply(self, cell: 'Cell', tissue: 'Tissue',
                  gradient_field: 'GradientField') -> bool:
        if cell.cell_type != self.cell_type:
            return False
        
        if self.condition is not None:
            return self.condition.evaluate(cell, tissue, gradient_field)
        
        # Legacy
        for cond in self.gradient_conditions:
            if not cond.evaluate(cell, gradient_field):
                return False
        for cond in self.age_conditions:
            if not cond.evaluate(cell):
                return False
        for cond in self.neighbor_conditions:
            if not cond.evaluate(cell, tissue):
                return False
        
        return True

# =============================================================================
# DIFFERENTIATION RULE
# =============================================================================

@dataclass
class DifferentiationRule:
    """
    Rule for cell type transformation without division.
    
    When conditions are met, a cell changes from source_type to target_type
    in place, without creating new cells.
    """
    source_type: str
    target_type: str
    priority: int = 0
    condition: Optional[ParsedCondition] = None
    reset_age: bool = False
    reset_division_count: bool = False
    
    # Legacy conditions
    gradient_conditions: List[GradientCondition] = field(default_factory=list)
    gradient_pattern_conditions: List[GradientPatternCondition] = field(default_factory=list)
    age_conditions: List[AgeCondition] = field(default_factory=list)
    neighbor_conditions: List[NeighborCountCondition] = field(default_factory=list)
    neighbor_type_conditions: List[NeighborTypeCondition] = field(default_factory=list)
    
    def can_apply(self, cell: 'Cell', tissue: 'Tissue',
                  gradient_field: 'GradientField') -> bool:
        if cell.cell_type != self.source_type:
            return False
        
        if self.condition is not None:
            return self.condition.evaluate(cell, tissue, gradient_field)
        
        # Legacy support
        for cond in self.gradient_conditions:
            if not cond.evaluate(cell, gradient_field):
                return False
        for cond in self.gradient_pattern_conditions:
            if not cond.evaluate(cell, gradient_field):
                return False
        for cond in self.age_conditions:
            if not cond.evaluate(cell):
                return False
        for cond in self.neighbor_conditions:
            if not cond.evaluate(cell, tissue):
                return False
        for cond in self.neighbor_type_conditions:
            if not cond.evaluate(cell, tissue):
                return False
        
        return True


def perform_differentiation(cell: 'Cell', rule: DifferentiationRule) -> None:
    """
    Transform a cell's type in place.
    
    Modifies the cell object directly rather than creating new cells.
    """
    cell.cell_type = rule.target_type
    
    if rule.reset_age:
        cell.age = 0
    
    if rule.reset_division_count:
        cell.division_count = 0

# =============================================================================
# DIVISION EXECUTION
# =============================================================================

def perform_division(cell: 'Cell', rule: DivisionRule, tissue: 'Tissue',
                     gradient_field: 'GradientField') -> Tuple['Cell', 'Cell']:
    from .cell import Cell
    from .gradients import GradientType
    
    division_axis = rule.get_axis(cell, tissue, gradient_field)
    daughter_area = cell.area
    
    axis_rad = np.radians(division_axis)
    offset = cell.radius * 0.55
    offset_vec = np.array([np.cos(axis_rad), np.sin(axis_rad)]) * offset
    
    daughter1 = Cell(
        cell_type=rule.daughter1_type,
        center=cell.center.copy() + offset_vec,
        area=daughter_area,
        is_anchor=False,
        parent_id=cell.id,
        division_count=cell.division_count + 1,
        facing_direction=division_axis
    )
    
    daughter2 = Cell(
        cell_type=rule.daughter2_type,
        center=cell.center.copy() - offset_vec,
        area=daughter_area,
        is_anchor=False,
        parent_id=cell.id,
        division_count=cell.division_count + 1,
        facing_direction=(division_axis + 180) % 360
    )
    
    # Anchor stays at origin: assign to whichever daughter is closest to origin
    if cell.is_anchor:
        d1_dist = np.linalg.norm(daughter1.center)
        d2_dist = np.linalg.norm(daughter2.center)
        if d1_dist <= d2_dist:
            daughter1.is_anchor = True
        else:
            daughter2.is_anchor = True
    
    daughter1.sibling_id = daughter2.id
    daughter2.sibling_id = daughter1.id
    
    # Create persistent gradient
    if rule.generated_gradient:
        gen = rule.generated_gradient
        
        GRADIENT_TYPES = {
            "radial_decreasing": GradientType.RADIAL_DECREASING,
            "radial_increasing": GradientType.RADIAL_INCREASING,
        }
        grad_type = GRADIENT_TYPES.get(gen.gradient_type, GradientType.RADIAL_DECREASING)
        
        grad_name = f"{gen.name_prefix}_{uuid.uuid4().hex[:6]}"
        
        gradient_field.create_persistent_gradient(
            name=grad_name,
            position=daughter1.center,
            gradient_type=grad_type,
            scale=gen.scale
        )
        
        daughter1.generated_gradient_name = grad_name
    
    return daughter1, daughter2