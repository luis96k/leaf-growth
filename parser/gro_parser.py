"""
Parser for .gro v2 syntax with boolean conditions.
"""

import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from core.tissue import Tissue
from core.gradients import Gradient, GradientType
from core.rules import (
    DivisionRule, ApoptosisRule, DifferentiationRule, DivisionAxis, DivisionAxisType,
    GradientCondition, GradientPatternCondition, AgeCondition,
    NeighborCountCondition, NeighborTypeCondition, SurroundedCondition,
    ComparisonOp, GeneratedGradient, ParsedCondition,
    DirectionalNeighborCondition, NumericExpression, RandomCondition
)



GRADIENT_TYPES = {
    "r-": GradientType.RADIAL_DECREASING,
    "r+": GradientType.RADIAL_INCREASING,
    "l": GradientType.ASYMMETRIC_LINEAR,
    "l-": GradientType.SYMMETRIC_LINEAR_DECREASING,
    "l+": GradientType.SYMMETRIC_LINEAR_INCREASING,
}

COMPARISON_OPS = {
    ">": ComparisonOp.GREATER, "<": ComparisonOp.LESS,
    ">=": ComparisonOp.GREATER_EQ, "<=": ComparisonOp.LESS_EQ,
    "==": ComparisonOp.EQUAL, "!=": ComparisonOp.NOT_EQUAL,
}

class ExpressionParser:
    """
    Arithmetic expression parser for Random(...) arguments.

    Grammar (descending precedence):
        expr    := addsub
        addsub  := muldiv (('+' | '-') muldiv)*
        muldiv  := unary  (('*' | '/') unary)*
        unary   := '-' unary | atom
        atom    := '(' expr ')' | NUMBER | IDENT['*']
    """

    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def parse(self) -> NumericExpression:
        result = self._parse_addsub()
        self._skip_ws()
        if self.pos < len(self.text):
            raise ValueError(f"Unexpected text in expression: '{self.text[self.pos:]}'")
        return result

    def _skip_ws(self):
        while self.pos < len(self.text) and self.text[self.pos] in ' \t':
            self.pos += 1

    def _parse_addsub(self) -> NumericExpression:
        left = self._parse_muldiv()
        while True:
            self._skip_ws()
            if self.pos < len(self.text) and self.text[self.pos] in '+-':
                op_char = self.text[self.pos]
                self.pos += 1
                right = self._parse_muldiv()
                left = NumericExpression(
                    op='ADD' if op_char == '+' else 'SUB',
                    left=left, right=right
                )
            else:
                return left

    def _parse_muldiv(self) -> NumericExpression:
        left = self._parse_unary()
        while True:
            self._skip_ws()
            if self.pos < len(self.text) and self.text[self.pos] in '*/':
                op_char = self.text[self.pos]
                self.pos += 1
                right = self._parse_unary()
                left = NumericExpression(
                    op='MUL' if op_char == '*' else 'DIV',
                    left=left, right=right
                )
            else:
                return left

    def _parse_unary(self) -> NumericExpression:
        self._skip_ws()
        if self.pos < len(self.text) and self.text[self.pos] == '-':
            self.pos += 1
            return NumericExpression(op='NEG', left=self._parse_unary())
        return self._parse_atom()

    def _parse_atom(self) -> NumericExpression:
        self._skip_ws()

        # Parenthesized sub-expression
        if self.pos < len(self.text) and self.text[self.pos] == '(':
            self.pos += 1
            result = self._parse_addsub()
            self._skip_ws()
            if self.pos >= len(self.text) or self.text[self.pos] != ')':
                raise ValueError("Missing ')' in expression")
            self.pos += 1
            return result

        # Numeric literal
        num_match = re.match(r'\d+(?:\.\d+)?', self.text[self.pos:])
        if num_match:
            self.pos += num_match.end()
            return NumericExpression(op='CONST', value=float(num_match.group(0)))

        # Variable: gradient name, optionally with trailing * for patterns
        var_match = re.match(r'\w+\*?', self.text[self.pos:])
        if var_match:
            self.pos += var_match.end()
            return NumericExpression(op='VAR', var_name=var_match.group(0))

        raise ValueError(f"Invalid expression atom at: '{self.text[self.pos:]}'")

class ConditionParser:
    """Parse boolean condition expressions."""
    
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
    
    def parse(self) -> ParsedCondition:
        """Parse the full condition."""
        result = self._parse_or()
        self._skip_whitespace()
        if self.pos < len(self.text):
            raise ValueError(f"Unexpected text: {self.text[self.pos:]}")
        return result
    
    def _skip_whitespace(self):
        while self.pos < len(self.text) and self.text[self.pos] in ' \t':
            self.pos += 1
    
    def _parse_or(self) -> ParsedCondition:
        left = self._parse_and()
        self._skip_whitespace()
        
        while self.pos < len(self.text) and self.text[self.pos:self.pos+2].upper() == 'OR':
            self.pos += 2
            self._skip_whitespace()
            right = self._parse_and()
            left = ParsedCondition(op='OR', left=left, right=right)
        
        return left
    
    def _parse_and(self) -> ParsedCondition:
        left = self._parse_not()
        self._skip_whitespace()
        
        while self.pos < len(self.text):
            if self.text[self.pos:self.pos+3].upper() == 'AND':
                self.pos += 3
                self._skip_whitespace()
                right = self._parse_not()
                left = ParsedCondition(op='AND', left=left, right=right)
            else:
                break
        
        return left
    
    def _parse_not(self) -> ParsedCondition:
        self._skip_whitespace()
        
        if self.text[self.pos:self.pos+3].upper() == 'NOT':
            self.pos += 3
            self._skip_whitespace()
            inner = self._parse_not()
            return ParsedCondition(op='NOT', left=inner)
        
        return self._parse_atom()
    
    def _parse_atom(self) -> ParsedCondition:
        self._skip_whitespace()
        
        if self.pos < len(self.text) and self.text[self.pos] == '(':
            self.pos += 1
            result = self._parse_or()
            self._skip_whitespace()
            if self.pos >= len(self.text) or self.text[self.pos] != ')':
                raise ValueError("Missing closing parenthesis")
            self.pos += 1
            return result
        
        return self._parse_comparison()
    
    def _parse_comparison(self) -> ParsedCondition:
        self._skip_whitespace()

        # Special: "surrounded"
        if self.text[self.pos:self.pos+10].lower() == 'surrounded':
            self.pos += 10
            return ParsedCondition(op='LEAF', leaf_condition=SurroundedCondition())

        # Special: Random  or  Random(<expression>)
        # Peek ahead so that a gradient literally named 'random' used in a
        # comparison (e.g. `random > 0.5`) still parses as a comparison.
        rand_match = re.match(r'[Rr]andom\b', self.text[self.pos:])
        if rand_match:
            peek = self.pos + rand_match.end()
            while peek < len(self.text) and self.text[peek] in ' \t':
                peek += 1
            next2 = self.text[peek:peek+2]
            is_comparison_follow = (
                next2 in ('>=', '<=', '==', '!=')
                or (peek < len(self.text) and self.text[peek] in '><')
            )

            if not is_comparison_follow:
                self.pos += rand_match.end()
                self._skip_whitespace()

                prob_expr = None
                if self.pos < len(self.text) and self.text[self.pos] == '(':
                    # Extract balanced-paren contents (expression may nest parens)
                    self.pos += 1
                    start = self.pos
                    depth = 1
                    while self.pos < len(self.text) and depth > 0:
                        c = self.text[self.pos]
                        if c == '(':
                            depth += 1
                        elif c == ')':
                            depth -= 1
                        self.pos += 1
                    if depth != 0:
                        raise ValueError("Unbalanced parentheses in Random(...)")
                    inner = self.text[start:self.pos - 1].strip()
                    if inner:  # Random() with empty parens → default 0.5
                        prob_expr = ExpressionParser(inner).parse()

                return ParsedCondition(
                    op='LEAF',
                    leaf_condition=RandomCondition(probability_expr=prob_expr)
                )

        # Special: "blocked(GRADIENT)" or "blocked(facing)"
        blocked_match = re.match(r'blocked\((\w+)\)', self.text[self.pos:])
        if blocked_match:
            mode = blocked_match.group(1)
            self.pos += blocked_match.end()
            return ParsedCondition(op='LEAF', leaf_condition=DirectionalNeighborCondition(
                direction_mode=mode, angle_tolerance=45.0, distance_factor=1.8
            ))

        # Parse: variable OP value
        match = re.match(
            r'(\w+(?:\*)?(?:\([^)]+\))?)\s*(>=|<=|>|<|==|!=)\s*(\d+(?:\.\d+)?)',
            self.text[self.pos:]
        )

        if not match:
            raise ValueError(f"Invalid comparison at: {self.text[self.pos:]}")

        var_name = match.group(1)
        op_str = match.group(2)
        value = float(match.group(3))
        self.pos += match.end()

        op = COMPARISON_OPS[op_str]

        if var_name == 'age':
            cond = AgeCondition(op, int(value))
        elif var_name == 'neighbors':
            cond = NeighborCountCondition(op, int(value))
        elif var_name.startswith('neighbors('):
            type_match = re.match(r'neighbors\((\w+)\)', var_name)
            cell_type = type_match.group(1)
            cond = NeighborTypeCondition(cell_type, op, int(value))
        elif var_name.endswith('*'):
            cond = GradientPatternCondition(var_name, op, value)
        else:
            cond = GradientCondition(var_name, op, value)

        return ParsedCondition(op='LEAF', leaf_condition=cond)


def parse_axis(text: str) -> DivisionAxis:
    """Parse axis specification."""
    text = text.strip()
    
    # random
    if text == 'random':
        return DivisionAxis(axis_type=DivisionAxisType.RANDOM)
    
    # @gradient±spread or @gradient+offset
    gradient_match = re.match(r'@(\w+)(?:([+-])(\d+(?:\.\d+)?))?(?:±(\d+(?:\.\d+)?))?', text)
    if gradient_match:
        grad_name = gradient_match.group(1)
        sign = gradient_match.group(2)
        offset_val = float(gradient_match.group(3)) if gradient_match.group(3) else 0.0
        spread = float(gradient_match.group(4)) if gradient_match.group(4) else 0.0
        
        if sign == '-':
            offset_val = -offset_val
        
        return DivisionAxis(
            axis_type=DivisionAxisType.TOWARD_GRADIENT,
            gradient_name=grad_name,
            angle_offset=offset_val,
            random_spread=spread
        )
    
    # facing@gradient+N  — bend facing toward gradient by up to N degrees
    bent_match = re.match(r'facing@(\w+)\+(\d+(?:\.\d+)?)$', text)
    if bent_match:
        return DivisionAxis(
            axis_type=DivisionAxisType.FACING_BENT,
            gradient_name=bent_match.group(1),
            angle_offset=float(bent_match.group(2)),
        )

    # facing [ +N | -N ] [ ±M ]   — relative to cell's facing direction
    facing_match = re.match(r'facing(?:([+-])(\d+(?:\.\d+)?))?(?:±(\d+(?:\.\d+)?))?$', text)
    if facing_match:
        offset = 0.0
        if facing_match.group(2):
            offset = float(facing_match.group(2))
            if facing_match.group(1) == '-':
                offset = -offset
        spread = float(facing_match.group(3)) if facing_match.group(3) else 0.0
        return DivisionAxis(
            axis_type=DivisionAxisType.FACING_RELATIVE,
            angle_offset=offset,
            random_spread=spread,
        )

    # Fixed angle
    try:
        angle = float(text)
        return DivisionAxis(axis_type=DivisionAxisType.FIXED, fixed_angle=angle)
    except ValueError:
        pass
    
    raise ValueError(f"Unknown axis syntax: {text}")


class GroParser:
    """Parser for .gro v2 files."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.gradients: List[Gradient] = []
        self.cell_types: Dict[str, dict] = {}
        self.division_rules: List[DivisionRule] = []
        self.differentiation_rules: List[DifferentiationRule] = []
        self.apoptosis_rules: List[ApoptosisRule] = []
        self.initial_cell_type: str = "default"
        self.current_section: Optional[str] = None
        self.current_rule = None
    
    @staticmethod
    def strip_comments(line: str) -> str:
        i = 0
        while i < len(line):
            if line[i] == '#':
                remaining = line[i+1:i+7]
                if len(remaining) == 6 and all(c in '0123456789abcdefABCDEF' for c in remaining):
                    i += 7
                    continue
                else:
                    return line[:i].rstrip()
            i += 1
        return line
    
    def parse_file(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            return self.parse_string(f.read())
    
    def parse_string(self, content: str):
        self.reset()
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = self.strip_comments(line).rstrip()
            if not line.strip():
                continue
            
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            upper = stripped.upper()
            
            try:
                if upper == 'GRADIENTS:':
                    self._save_rule()
                    self.current_section = 'gradients'
                elif upper == 'CELLS:':
                    self._save_rule()
                    self.current_section = 'cells'
                elif upper == 'DIVIDE:':
                    self._save_rule()
                    self.current_section = 'divide'
                elif upper == 'DIFFERENTIATE:':
                    self._save_rule()
                    self.current_section = 'differentiate'
                elif upper == 'DIE:':
                    self._save_rule()
                    self.current_section = 'die'
                elif self.current_section == 'gradients' and indent > 0:
                    self._parse_gradient(stripped)
                elif self.current_section == 'cells' and indent > 0:
                    self._parse_cell_type(stripped)
                elif self.current_section == 'divide':
                    if '->' in stripped:
                        self._save_rule()
                        self._parse_division_header(stripped)
                    elif indent > 0 and self.current_rule:
                        self._parse_rule_body(stripped)
                elif self.current_section == 'differentiate':
                    if '->' in stripped:
                        self._save_rule()
                        self._parse_differentiation_header(stripped)
                    elif indent > 0 and self.current_rule:
                        self._parse_differentiation_body(stripped)
                elif self.current_section == 'die':
                    if stripped.endswith(':'):
                        self._save_rule()
                        self._parse_apoptosis_header(stripped)
                    elif indent > 0 and self.current_rule:
                        self._parse_apoptosis_body(stripped)
                        
            except Exception as e:
                raise ValueError(f"Line {line_num}: {e}\n  '{line}'")
        
        self._save_rule()
        return self
    
    def _save_rule(self):
        if self.current_rule:
            if isinstance(self.current_rule, DivisionRule):
                self.division_rules.append(self.current_rule)
            elif isinstance(self.current_rule, ApoptosisRule):
                self.apoptosis_rules.append(self.current_rule)
            elif isinstance(self.current_rule, DifferentiationRule):
                self.differentiation_rules.append(self.current_rule)
            self.current_rule = None
    
    def _parse_gradient(self, line: str):
        match = re.match(
            r'(\w+)\s*:\s*(\S+)'
            r'(?:\s+angle\s*=\s*(-?\d+(?:\.\d+)?))?'
            r'(?:\s+scale\s*=\s*(\d+(?:\.\d+)?))?',
            line
        )
        if not match:
            raise ValueError(f"Invalid gradient: {line}")
        
        name = match.group(1)
        type_str = match.group(2).lower()
        angle = float(match.group(3)) if match.group(3) else 0.0
        scale = float(match.group(4)) if match.group(4) else 1.0
        
        if type_str not in GRADIENT_TYPES:
            raise ValueError(f"Unknown gradient type: {type_str}")
        
        self.gradients.append(Gradient(
            name=name,
            gradient_type=GRADIENT_TYPES[type_str],
            axis_angle=angle,
            scale=scale
        ))
    
    def _parse_cell_type(self, line: str):
        match = re.match(r'(\w+)\s*:\s*(#[0-9a-fA-F]{6})(?:\s+(initial))?', line)
        if not match:
            raise ValueError(f"Invalid cell type: {line}")
        
        name = match.group(1)
        color = match.group(2)
        is_initial = match.group(3) is not None
        
        self.cell_types[name] = {"color": color, "initial": is_initial}
        if is_initial:
            self.initial_cell_type = name
    
    def _parse_division_header(self, line: str):
        match = re.match(r'(\w+)\s*->\s*(\w+)\s*,\s*(\w+)\s*:', line)
        if not match:
            raise ValueError(f"Invalid division rule: {line}")
        
        self.current_rule = DivisionRule(
            cell_type=match.group(1),
            daughter1_type=match.group(2),
            daughter2_type=match.group(3)
        )

    def _parse_differentiation_header(self, line: str):
        """Parse: source_type -> target_type:"""
        match = re.match(r'(\w+)\s*->\s*(\w+)\s*:', line)
        if not match:
            raise ValueError(f"Invalid differentiation rule: {line}")
        
        self.current_rule = DifferentiationRule(
            source_type=match.group(1),
            target_type=match.group(2)
        )

    def _parse_differentiation_body(self, line: str):
        """Parse differentiation rule body."""
        rule = self.current_rule
        line = line.strip()
        
        # when CONDITION
        if line.startswith('when '):
            cond_text = line[5:].strip()
            parser = ConditionParser(cond_text)
            rule.condition = parser.parse()
            return
        
        # priority = N
        priority_match = re.match(r'priority\s*=\s*(\d+)', line)
        if priority_match:
            rule.priority = int(priority_match.group(1))
            return
        
        # reset_age = true/false
        reset_age_match = re.match(r'reset_age\s*=\s*(true|false)', line, re.IGNORECASE)
        if reset_age_match:
            rule.reset_age = reset_age_match.group(1).lower() == 'true'
            return
        
        # reset_divisions = true/false
        reset_div_match = re.match(r'reset_divisions\s*=\s*(true|false)', line, re.IGNORECASE)
        if reset_div_match:
            rule.reset_division_count = reset_div_match.group(1).lower() == 'true'
            return
        
        raise ValueError(f"Unknown differentiation rule syntax: {line}")
    
    def _parse_rule_body(self, line: str):
        rule = self.current_rule
        line = line.strip()
        
        # when CONDITION
        if line.startswith('when '):
            cond_text = line[5:].strip()
            parser = ConditionParser(cond_text)
            rule.condition = parser.parse()
            return
        
        # axis = SPEC
        axis_match = re.match(r'axis\s*=\s*(.+)', line)
        if axis_match:
            rule.division_axis = parse_axis(axis_match.group(1))
            return
        
        # emit NAME [scale=N]
        emit_match = re.match(r'emit\s+(\w+)(?:\s+scale\s*=\s*(\d+(?:\.\d+)?))?', line)
        if emit_match:
            rule.generated_gradient = GeneratedGradient(
                name_prefix=emit_match.group(1),
                gradient_type="radial_decreasing",
                scale=float(emit_match.group(2)) if emit_match.group(2) else 0.8,
                attach_to="daughter1"
            )
            return

        # priority = N
        priority_match = re.match(r'priority\s*=\s*(\d+)', line)
        if priority_match:
            rule.priority = int(priority_match.group(1))
            return        

        # limit = N
        limit_match = re.match(r'limit\s*=\s*(\d+)', line)
        if limit_match:
            rule.max_divisions = int(limit_match.group(1))
            return
        
        raise ValueError(f"Unknown rule syntax: {line}")
    
    def _parse_apoptosis_header(self, line: str):
        match = re.match(r'(\w+)\s*:', line)
        if not match:
            raise ValueError(f"Invalid apoptosis rule: {line}")
        self.current_rule = ApoptosisRule(cell_type=match.group(1))
    
    def _parse_apoptosis_body(self, line: str):
        rule = self.current_rule
        line = line.strip()
        
        if line.startswith('when '):
            cond_text = line[5:].strip()
            parser = ConditionParser(cond_text)
            rule.condition = parser.parse()


def load_gro_file(filepath: str, tissue: Tissue) -> Tissue:
    parser = GroParser()
    parser.parse_file(filepath)
    
    for grad in parser.gradients:
        tissue.add_gradient(grad)
    
    for name, info in parser.cell_types.items():
        tissue.set_cell_color(name, info["color"])
    
    for rule in parser.division_rules:
        tissue.add_division_rule(rule)
    
    for rule in parser.apoptosis_rules:
        tissue.add_apoptosis_rule(rule)
    
    for rule in parser.differentiation_rules:
        tissue.add_differentiation_rule(rule)
    
    tissue.create_initial_cell(parser.initial_cell_type)
    return tissue