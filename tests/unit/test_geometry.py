"""Unit tests for core.geometry – GeometryDescriptor extraction."""

import math

import pytest

from core.types import NodeType, NodeRegistry
from core.ast_node import reset_id_counter
from core.geometry import (
    GeometryDescriptor,
    extract_geometry_descriptor,
    extract_geometry_descriptors,
    _dequantize,
    _distance,
    _circumradius,
)

from tests.helpers import (
    build_rectangle_solid,
    build_circle_solid,
    build_arc_sketch_solid,
    make_coord,
    make_line,
    make_arc,
    make_circle,
    make_scalar,
    make_edge,
    make_loop,
    make_face,
    make_sketch,
    make_solid,
    make_extrude,
    make_program,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_id_counter()
    NodeRegistry.reset()
    yield


# ═══════════════════════════════════════════════════════════════════
# Low-level helpers
# ═══════════════════════════════════════════════════════════════════

class TestDequantize:

    def test_min(self):
        assert _dequantize(0) == pytest.approx(-1.0)

    def test_max(self):
        assert _dequantize(255) == pytest.approx(1.0)

    def test_mid(self):
        # 127 → approx 0 (not exactly 0, but close)
        val = _dequantize(127)
        assert -0.01 < val < 0.01

    def test_quarter(self):
        val = _dequantize(64)
        assert val == pytest.approx(-1.0 + 2.0 * 64 / 255)


class TestDistance:

    def test_zero(self):
        assert _distance(0, 0, 0, 0) == 0.0

    def test_unit(self):
        assert _distance(0, 0, 1, 0) == pytest.approx(1.0)

    def test_diagonal(self):
        assert _distance(0, 0, 1, 1) == pytest.approx(math.sqrt(2))


class TestCircumradius:

    def test_equilateral_triangle(self):
        # equilateral with side 2 → R = 2/sqrt(3)
        r = _circumradius(0, 0, 2, 0, 1, math.sqrt(3))
        assert r == pytest.approx(2.0 / math.sqrt(3), abs=1e-6)

    def test_collinear_returns_inf(self):
        r = _circumradius(0, 0, 1, 0, 2, 0)
        assert r == float("inf")


# ═══════════════════════════════════════════════════════════════════
# Line curvature = 0
# ═══════════════════════════════════════════════════════════════════

class TestLineCurvature:

    def test_line_curvature_is_zero(self):
        line = make_line(0, 0, 128, 128)
        desc = extract_geometry_descriptor(line)
        assert desc.curvature == 0.0


# ═══════════════════════════════════════════════════════════════════
# Arc curvature = 1/R
# ═══════════════════════════════════════════════════════════════════

class TestArcCurvature:

    def test_arc_curvature_positive(self):
        # Three points on a circle of known radius
        arc = make_arc(0, 128, 64, 192, 128, 128)
        desc = extract_geometry_descriptor(arc)
        assert desc.curvature > 0

    def test_arc_curvature_matches_circumradius(self):
        arc = make_arc(0, 128, 64, 192, 128, 128)
        desc = extract_geometry_descriptor(arc)
        # Manually compute
        c0 = _dequantize(0), _dequantize(128)
        c1 = _dequantize(64), _dequantize(192)
        c2 = _dequantize(128), _dequantize(128)
        R = _circumradius(*c0, *c1, *c2)
        expected = 1.0 / R if R > 1e-12 else 0.0
        assert desc.curvature == pytest.approx(expected, rel=1e-6)


# ═══════════════════════════════════════════════════════════════════
# Circle curvature = 1/R
# ═══════════════════════════════════════════════════════════════════

class TestCircleCurvature:

    def test_circle_curvature_positive(self):
        circle = make_circle(128, 128, 64)
        desc = extract_geometry_descriptor(circle)
        assert desc.curvature > 0

    def test_circle_curvature_inversely_proportional_to_radius(self):
        # Q8 value 192 → dequantize ≈ 0.506 (larger physical radius)
        # Q8 value 160 → dequantize ≈ 0.255 (smaller physical radius)
        c_small = make_circle(128, 128, 160)
        c_large = make_circle(128, 128, 192)
        d_small = extract_geometry_descriptor(c_small)
        d_large = extract_geometry_descriptor(c_large)
        assert d_small.curvature > d_large.curvature


# ═══════════════════════════════════════════════════════════════════
# Scale
# ═══════════════════════════════════════════════════════════════════

class TestScale:

    def test_coord_scale_is_zero(self):
        coord = make_coord(128, 128)
        desc = extract_geometry_descriptor(coord)
        assert desc.scale == 0.0

    def test_line_scale_positive(self):
        line = make_line(0, 0, 255, 255)
        desc = extract_geometry_descriptor(line)
        assert desc.scale > 0

    def test_scale_normalised_to_01(self):
        """Maximum possible line (0,0)→(255,255) scale ≤ 1."""
        line = make_line(0, 0, 255, 255)
        desc = extract_geometry_descriptor(line)
        assert 0.0 <= desc.scale <= 1.0

    def test_circle_scale(self):
        circle = make_circle(128, 128, 128)
        desc = extract_geometry_descriptor(circle)
        assert desc.scale > 0


# ═══════════════════════════════════════════════════════════════════
# Depth Ratio
# ═══════════════════════════════════════════════════════════════════

class TestDepthRatio:

    def test_prog_depth_ratio_zero(self):
        ast = build_rectangle_solid()
        desc = extract_geometry_descriptor(ast)
        assert desc.depth_ratio == pytest.approx(0.0)

    def test_crd_depth_ratio_one(self):
        coord = make_coord(0, 0)
        desc = extract_geometry_descriptor(coord)
        assert desc.depth_ratio == pytest.approx(1.0)

    def test_face_depth_ratio(self):
        face = make_face([make_loop([make_edge(make_line(0, 0, 128, 128))])])
        desc = extract_geometry_descriptor(face)
        assert desc.depth_ratio == pytest.approx(3.0 / 5.0)


# ═══════════════════════════════════════════════════════════════════
# Subtree Size
# ═══════════════════════════════════════════════════════════════════

class TestSubtreeSize:

    def test_coord_subtree_size_1(self):
        coord = make_coord(0, 0)
        desc = extract_geometry_descriptor(coord)
        assert desc.subtree_size == 1

    def test_line_subtree_size_3(self):
        line = make_line(0, 0, 128, 128)
        desc = extract_geometry_descriptor(line)
        assert desc.subtree_size == 3  # LN + 2 CRD

    def test_arc_subtree_size_4(self):
        arc = make_arc(0, 0, 64, 64, 128, 0)
        desc = extract_geometry_descriptor(arc)
        assert desc.subtree_size == 4  # ARC + 3 CRD


# ═══════════════════════════════════════════════════════════════════
# extract_geometry_descriptors (whole tree)
# ═══════════════════════════════════════════════════════════════════

class TestExtractAll:

    def test_descriptor_count_equals_dfs_count(self):
        ast = build_rectangle_solid()
        descs = extract_geometry_descriptors(ast)
        dfs_count = ast.subtree_size
        assert len(descs) == dfs_count

    def test_first_is_prog(self):
        ast = build_rectangle_solid()
        descs = extract_geometry_descriptors(ast)
        assert descs[0].depth_ratio == 0.0

    def test_all_depths_valid(self):
        ast = build_rectangle_solid()
        descs = extract_geometry_descriptors(ast)
        for d in descs:
            assert 0.0 <= d.depth_ratio <= 1.0

    def test_to_list(self):
        desc = GeometryDescriptor(0.5, 0.1, 0.6, 10)
        lst = desc.to_list()
        assert len(lst) == 4
        assert lst == [0.5, 0.1, 0.6, 10.0]
