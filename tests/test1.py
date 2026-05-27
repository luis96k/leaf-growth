"""
Test suite for leaf-growth simulator.

Structure:
  - TestParser          : unit tests for the .gro file parser
  - TestConditionParser : unit tests for the boolean condition language
  - TestGradients       : unit tests for gradient math
  - TestSimulation      : integration tests for the simulation engine
  - TestGroFiles        : golden tests using the real .gro example files

Run with:
    pytest test_leaf_growth.py -v

To run a specific group:
    pytest test_leaf_growth.py::TestParser -v
"""

import textwrap
import numpy as np
import pytest

from parser.gro_parser import GroParser, parse_axis
from core.tissue import Tissue
from core.cell import Cell
from core.gradients import Gradient, GradientType, GradientField
from core.rules import (
    AgeCondition, ComparisonOp, DivisionRule,
    ApoptosisRule, DifferentiationRule, perform_differentiation,
)


# =============================================================================
# Helpers
# =============================================================================

def make_tissue_from_gro(text: str) -> Tissue:
    """Parse inline .gro text and return a fully initialised Tissue."""
    from parser.gro_parser import load_gro_file
    import tempfile, os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".gro", delete=False) as f:
        f.write(textwrap.dedent(text))
        path = f.name
    try:
        tissue = Tissue()
        load_gro_file(path, tissue)
    finally:
        os.unlink(path)
    return tissue


# =============================================================================
# TestParser – unit tests for the .gro parser
# =============================================================================

class TestParser:
    """Tests for GroParser: cell types, gradients, division/die/differentiate rules."""

    MINIMAL_GRO = """
        CELLS:
          stem: #44cc44 initial
    """

    def test_parses_initial_cell_type(self):
        """Parser identifies which cell type is marked 'initial'."""
        p = GroParser()
        p.parse_string(textwrap.dedent(self.MINIMAL_GRO))
        assert p.initial_cell_type == "stem"

    def test_parses_cell_color(self):
        """Parser captures hex color for each cell type."""
        p = GroParser()
        p.parse_string(textwrap.dedent(self.MINIMAL_GRO))
        assert p.cell_types["stem"]["color"] == "#44cc44"

    def test_parses_multiple_cell_types(self):
        gro = """
            CELLS:
              stem:  #44cc44 initial
              leaf:  #88cc66
              flower: #ff88aa
        """
        p = GroParser()
        p.parse_string(textwrap.dedent(gro))
        assert set(p.cell_types.keys()) == {"stem", "leaf", "flower"}
        assert p.initial_cell_type == "stem"

    def test_parses_gradient_radial_decreasing(self):
        gro = """
            GRADIENTS:
              auxin: r- scale=0.3
            CELLS:
              cell: #ffffff initial
        """
        p = GroParser()
        p.parse_string(textwrap.dedent(gro))
        assert len(p.gradients) == 1
        g = p.gradients[0]
        assert g.name == "auxin"
        assert g.gradient_type == GradientType.RADIAL_DECREASING
        assert abs(g.scale - 0.3) < 1e-9

    def test_parses_gradient_linear_with_angle(self):
        gro = """
            GRADIENTS:
              light: l angle=90 scale=0.2
            CELLS:
              cell: #ffffff initial
        """
        p = GroParser()
        p.parse_string(textwrap.dedent(gro))
        g = p.gradients[0]
        assert g.gradient_type == GradientType.ASYMMETRIC_LINEAR
        assert abs(g.axis_angle - 90.0) < 1e-9

    def test_parses_division_rule_types(self):
        gro = """
            CELLS:
              stem: #44cc44 initial
              leaf: #88cc66
            DIVIDE:
              stem -> stem, leaf:
                when age >= 3
                axis = random
                limit = 10
        """
        p = GroParser()
        p.parse_string(textwrap.dedent(gro))
        assert len(p.division_rules) == 1
        r = p.division_rules[0]
        assert r.cell_type == "stem"
        assert r.daughter1_type == "stem"
        assert r.daughter2_type == "leaf"
        assert r.max_divisions == 10

    def test_parses_apoptosis_rule(self):
        gro = """
            CELLS:
              leaf: #88cc66 initial
            DIE:
              leaf:
                when age > 25
        """
        p = GroParser()
        p.parse_string(textwrap.dedent(gro))
        assert len(p.apoptosis_rules) == 1
        assert p.apoptosis_rules[0].cell_type == "leaf"

    def test_parses_differentiation_rule(self):
        gro = """
            CELLS:
              stem:   #44cc44 initial
              mature: #228833
            DIFFERENTIATE:
              stem -> mature:
                when age >= 5
                priority = 10
                reset_age = true
        """
        p = GroParser()
        p.parse_string(textwrap.dedent(gro))
        assert len(p.differentiation_rules) == 1
        r = p.differentiation_rules[0]
        assert r.source_type == "stem"
        assert r.target_type == "mature"
        assert r.priority == 10
        assert r.reset_age is True

    def test_unknown_gradient_type_raises(self):
        gro = """
            GRADIENTS:
              bad: zzz
            CELLS:
              cell: #ffffff initial
        """
        with pytest.raises(ValueError):
            GroParser().parse_string(textwrap.dedent(gro))

    def test_malformed_cell_line_raises(self):
        gro = """
            CELLS:
              nodolor
        """
        with pytest.raises(ValueError):
            GroParser().parse_string(textwrap.dedent(gro))

    def test_emit_creates_generated_gradient(self):
        gro = """
            CELLS:
              stem: #44cc44 initial
            DIVIDE:
              stem -> stem, stem:
                emit hormone scale=0.4
        """
        p = GroParser()
        p.parse_string(textwrap.dedent(gro))
        r = p.division_rules[0]
        assert r.generated_gradient is not None
        assert r.generated_gradient.name_prefix == "hormone"
        assert abs(r.generated_gradient.scale - 0.4) < 1e-9


# =============================================================================
# TestConditionParser – unit tests for the boolean condition language
# =============================================================================

class TestConditionParser:
    """Tests for the ConditionParser embedded in GroParser."""

    def _parse_cond(self, text: str):
        from parser.gro_parser import ConditionParser
        return ConditionParser(text).parse()

    def test_simple_age_condition(self):
        cond = self._parse_cond("age >= 3")
        # Should parse without error and produce a leaf node
        assert cond is not None

    def test_and_condition(self):
        cond = self._parse_cond("(age >= 3) AND (neighbors < 6)")
        assert cond.op == "AND"

    def test_or_condition(self):
        cond = self._parse_cond("(age > 10) OR (neighbors == 0)")
        assert cond.op == "OR"

    def test_not_condition(self):
        cond = self._parse_cond("NOT surrounded")
        assert cond.op == "NOT"

    def test_nested_condition(self):
        cond = self._parse_cond("((age >= 3) AND (neighbors < 6)) OR (auxin > 0.5)")
        assert cond.op == "OR"
        assert cond.left.op == "AND"

    def test_random_no_args(self):
        cond = self._parse_cond("Random")
        assert cond is not None

    def test_random_with_probability(self):
        cond = self._parse_cond("Random(0.3)")
        assert cond is not None

    def test_random_with_expression(self):
        cond = self._parse_cond("Random((center + 0.05) / 4)")
        assert cond is not None

    def test_age_and_random(self):
        # From random.gro: real-world condition
        cond = self._parse_cond("age >= 3 AND Random(1 - center)")
        assert cond.op == "AND"


# =============================================================================
# TestAxisParser – unit tests for parse_axis
# =============================================================================

class TestAxisParser:
    """Tests for parse_axis() function."""

    def test_fixed_angle(self):
        axis = parse_axis("90")
        from core.rules import DivisionAxisType
        assert axis.axis_type == DivisionAxisType.FIXED
        assert abs(axis.fixed_angle - 90.0) < 1e-9

    def test_random_axis(self):
        from core.rules import DivisionAxisType
        axis = parse_axis("random")
        assert axis.axis_type == DivisionAxisType.RANDOM

    def test_facing_axis(self):
        from core.rules import DivisionAxisType
        axis = parse_axis("facing")
        assert axis.axis_type == DivisionAxisType.FACING

    def test_gradient_relative_axis(self):
        from core.rules import DivisionAxisType
        axis = parse_axis("@auxin+90")
        assert axis.axis_type == DivisionAxisType.GRADIENT_RELATIVE
        assert axis.gradient_name == "auxin"
        assert abs(axis.angle_offset - 90.0) < 1e-9


# =============================================================================
# TestGradients – unit tests for gradient math
# =============================================================================

class TestGradients:
    """Tests for Gradient.get_value() and GradientField."""

    def test_radial_decreasing_max_at_center(self):
        g = Gradient("g", GradientType.RADIAL_DECREASING, scale=1.0)
        val_center = g.get_value(np.array([0.0, 0.0]))
        val_far    = g.get_value(np.array([10.0, 0.0]))
        assert abs(val_center - 1.0) < 1e-6
        assert val_center > val_far

    def test_radial_increasing_min_at_center(self):
        g = Gradient("g", GradientType.RADIAL_INCREASING, scale=1.0)
        val_center = g.get_value(np.array([0.0, 0.0]))
        val_far    = g.get_value(np.array([10.0, 0.0]))
        assert abs(val_center) < 1e-6
        assert val_far > val_center

    def test_gradient_values_in_0_1_range(self):
        for gt in GradientType:
            g = Gradient("g", gt, scale=0.5)
            for pos in [np.array([0.0, 0.0]), np.array([5.0, 3.0]), np.array([-2.0, 7.0])]:
                v = g.get_value(pos)
                assert 0.0 <= v <= 1.0, f"{gt} gave {v} at {pos}"

    def test_gradient_field_get_value(self):
        field = GradientField()
        g = Gradient("auxin", GradientType.RADIAL_DECREASING, scale=1.0)
        field.add_gradient(g)
        assert "auxin" in field
        val = field.get_value("auxin", np.array([0.0, 0.0]))
        assert abs(val - 1.0) < 1e-6

    def test_summed_gradient_accumulates(self):
        """Multiple micro-gradients with same prefix should sum (clamped to 1)."""
        field = GradientField()
        for i in range(3):
            g = Gradient(
                f"hormone_{i}",
                GradientType.RADIAL_DECREASING,
                scale=10.0,
                fixed_position=np.array([0.0, 0.0])
            )
            field.gradients[f"hormone_{i}"] = g
        summed = field.get_summed_value("hormone", np.array([0.0, 0.0]))
        assert abs(summed - 1.0) < 1e-6  # clamped

    def test_missing_gradient_returns_zero(self):
        field = GradientField()
        assert field.get_value("nonexistent", np.array([0.0, 0.0])) == 0.0


# =============================================================================
# TestSimulation – integration tests for the Tissue engine
# =============================================================================

class TestSimulation:
    """Integration tests: tissue initialisation, division, death, differentiation."""

    ALWAYS_DIVIDE = """
        CELLS:
          cell: #44cc44 initial
        DIVIDE:
          cell -> cell, cell:
            limit = 3
    """

    def test_initial_state_one_cell(self):
        """After loading, tissue has exactly 1 cell."""
        tissue = make_tissue_from_gro(self.ALWAYS_DIVIDE)
        assert len(tissue.cells) == 1

    def test_initial_cell_at_origin(self):
        tissue = make_tissue_from_gro(self.ALWAYS_DIVIDE)
        cell = next(iter(tissue.cells.values()))
        assert np.allclose(cell.center, [0.0, 0.0], atol=0.01)

    def test_initial_cell_is_anchor(self):
        tissue = make_tissue_from_gro(self.ALWAYS_DIVIDE)
        cell = next(iter(tissue.cells.values()))
        assert cell.is_anchor is True

    def test_initial_cell_type_correct(self):
        tissue = make_tissue_from_gro(self.ALWAYS_DIVIDE)
        cell = next(iter(tissue.cells.values()))
        assert cell.cell_type == "cell"

    def test_division_increases_cell_count(self):
        """One step with always-divide rule should produce 2 cells."""
        tissue = make_tissue_from_gro(self.ALWAYS_DIVIDE)
        tissue.step()
        assert len(tissue.cells) == 2

    def test_division_limit_respected(self):
        """With limit=3, cell count must never exceed 2^3 = 8."""
        tissue = make_tissue_from_gro(self.ALWAYS_DIVIDE)
        for _ in range(20):
            tissue.step()
        assert len(tissue.cells) <= 8

    def test_age_increments_each_step(self):
        """Cells age by 1 per step."""
        tissue = make_tissue_from_gro("""
            CELLS:
              cell: #44cc44 initial
        """)
        cell = next(iter(tissue.cells.values()))
        tissue.step()
        # The original cell was removed by division; but if no division rule, it ages
        cell = next(iter(tissue.cells.values()))
        assert cell.age == 1

    def test_apoptosis_removes_cell(self):
        """A cell with age > 0 and die rule fires on step 1."""
        gro = """
            CELLS:
              cell: #44cc44 initial
            DIE:
              cell:
                when age >= 1
        """
        tissue = make_tissue_from_gro(gro)
        tissue.step()
        assert len(tissue.cells) == 0

    def test_differentiation_changes_type(self):
        """Cell differentiates from 'stem' to 'mature' when age >= 2."""
        gro = """
            CELLS:
              stem:   #44cc44 initial
              mature: #228833
            DIFFERENTIATE:
              stem -> mature:
                when age >= 2
        """
        tissue = make_tissue_from_gro(gro)
        tissue.step()  # age = 1, no change
        tissue.step()  # age = 2, differentiates
        cell = next(iter(tissue.cells.values()))
        assert cell.cell_type == "mature"

    def test_differentiation_reset_age(self):
        """reset_age=true resets the cell's age on differentiation."""
        gro = """
            CELLS:
              stem:   #44cc44 initial
              mature: #228833
            DIFFERENTIATE:
              stem -> mature:
                when age >= 2
                reset_age = true
        """
        tissue = make_tissue_from_gro(gro)
        tissue.step()
        tissue.step()
        cell = next(iter(tissue.cells.values()))
        assert cell.cell_type == "mature"
        assert cell.age == 0

    def test_simulation_step_counter(self):
        tissue = make_tissue_from_gro(self.ALWAYS_DIVIDE)
        assert tissue.simulation_step == 0
        tissue.step()
        assert tissue.simulation_step == 1
        tissue.run_steps(4)
        assert tissue.simulation_step == 5

    def test_daughters_inherit_division_count(self):
        """Daughter cells have division_count = parent + 1."""
        tissue = make_tissue_from_gro(self.ALWAYS_DIVIDE)
        tissue.step()
        for cell in tissue.cells.values():
            assert cell.division_count == 1

    def test_get_statistics_returns_expected_keys(self):
        tissue = make_tissue_from_gro(self.ALWAYS_DIVIDE)
        stats = tissue.get_statistics()
        for key in ("total_cells", "type_counts", "simulation_step", "bounds"):
            assert key in stats

    def test_tissue_always_has_anchor(self):
        """After any number of divisions, exactly one cell is anchor."""
        tissue = make_tissue_from_gro(self.ALWAYS_DIVIDE)
        for _ in range(5):
            tissue.step()
            anchors = [c for c in tissue.cells.values() if c.is_anchor]
            assert len(anchors) == 1


# =============================================================================
# TestGroFiles – golden tests using the real .gro example files
# =============================================================================

class TestGroFiles:
    """
    Parse each .gro example file and verify basic structural correctness.

    These tests do NOT check exact cell counts (which are stochastic), but
    verify that the file loads without error and produces a sensible tissue.

    Place your .gro files in an 'examples/' folder next to this test file.
    Tests are skipped automatically if the folder / file is missing.
    """

    import os
    EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")

    def _load(self, filename: str) -> Tissue:
        import os
        from parser.gro_parser import load_gro_file
        path = os.path.join(self.EXAMPLES_DIR, filename)
        if not os.path.exists(path):
            pytest.skip(f"Example file not found: {path}")
        tissue = Tissue()
        load_gro_file(path, tissue)
        return tissue

    def _run(self, filename: str, steps: int = 10) -> Tissue:
        tissue = self._load(filename)
        tissue.run_steps(steps)
        return tissue

    def test_random_gro_loads(self):
        """random.gro parses without error."""
        tissue = self._load("random.gro")
        assert len(tissue.cells) == 1

    def test_random_gro_grows(self):
        """random.gro produces more than 1 cell after 15 steps."""
        tissue = self._run("random.gro", steps=15)
        assert len(tissue.cells) > 1

    def test_random_gro_has_correct_cell_types(self):
        """random.gro tissue only contains cell types declared in the file."""
        tissue = self._run("random.gro", steps=20)
        valid_types = {"seed", "green", "blue", "red", "gold"}
        for cell in tissue.cells.values():
            assert cell.cell_type in valid_types, \
                f"Unexpected cell type: {cell.cell_type}"

    def test_all_example_files_load(self):
        """Every .gro file in the examples/ folder parses without error."""
        import os, glob
        if not os.path.isdir(self.EXAMPLES_DIR):
            pytest.skip("No examples/ directory found")
        files = glob.glob(os.path.join(self.EXAMPLES_DIR, "*.gro"))
        if not files:
            pytest.skip("No .gro files found in examples/")
        for path in files:
            tissue = Tissue()
            from parser.gro_parser import load_gro_file
            load_gro_file(path, tissue)  # must not raise
            assert len(tissue.cells) >= 1, f"{path} produced no initial cell"

    @pytest.mark.timeout(10)
    def test_all_example_files_run_10_steps(self):
        """Every .gro file runs 10 steps without hanging or crashing."""
        import os, glob
        if not os.path.isdir(self.EXAMPLES_DIR):
            pytest.skip("No examples/ directory found")
        files = glob.glob(os.path.join(self.EXAMPLES_DIR, "*.gro"))
        if not files:
            pytest.skip("No .gro files found in examples/")
        for path in files:
            tissue = Tissue()
            from parser.gro_parser import load_gro_file
            load_gro_file(path, tissue)
            tissue.run_steps(10)  # must finish in < 10s and not raise
