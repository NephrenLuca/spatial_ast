"""Unit tests for data.augmentation – coordinate transforms."""

import random

import pytest

from core.types import NodeType, NodeRegistry
from core.ast_node import reset_id_counter
from core.grammar import validate_ast

from data.augmentation import (
    ASTAugmentor,
    CoordinateMirror,
    CoordinateScale,
    CoordinateShift,
    EdgeOrderShuffle,
    SolidOrderShuffle,
)

from tests.helpers import (
    build_rectangle_solid,
    build_multi_solid,
    make_coord,
    make_edge,
    make_extrude,
    make_face,
    make_line,
    make_loop,
    make_program,
    make_sketch,
    make_solid,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_id_counter()
    NodeRegistry.reset()
    random.seed(42)
    yield


class TestCoordinateShift:

    def test_shift_changes_coords(self):
        ast = build_rectangle_solid()
        shifted = CoordinateShift(max_shift=10)(ast)
        orig_coords = ast.collect(lambda n: n.node_type == NodeType.CRD)
        new_coords = shifted.collect(lambda n: n.node_type == NodeType.CRD)
        changed = any(
            o.params != n.params
            for o, n in zip(orig_coords, new_coords)
        )
        assert changed

    def test_shift_preserves_validity(self):
        ast = build_rectangle_solid()
        shifted = CoordinateShift(max_shift=5)(ast)
        r = validate_ast(shifted)
        assert r.is_valid, r.errors

    def test_shift_clamps_to_q8(self):
        """Coords near boundary should clamp to [0, 255]."""
        ast = build_rectangle_solid(w=250, h=250)
        shifted = CoordinateShift(max_shift=20)(ast)
        for n in shifted.dfs():
            if n.node_type == NodeType.CRD:
                assert 0 <= n.params["x"] <= 255
                assert 0 <= n.params["y"] <= 255

    def test_zero_shift_is_identity(self):
        ast = build_rectangle_solid()
        shifted = CoordinateShift(max_shift=0)(ast)
        assert ast.structurally_equal(shifted)


class TestCoordinateScale:

    def test_scale_preserves_validity(self):
        ast = build_rectangle_solid()
        scaled = CoordinateScale(scale_range=(0.8, 1.2))(ast)
        r = validate_ast(scaled)
        assert r.is_valid, r.errors

    def test_scale_one_is_near_identity(self):
        ast = build_rectangle_solid()
        scaled = CoordinateScale(scale_range=(1.0, 1.0))(ast)
        orig_coords = ast.collect(lambda n: n.node_type == NodeType.CRD)
        new_coords = scaled.collect(lambda n: n.node_type == NodeType.CRD)
        for o, n in zip(orig_coords, new_coords):
            assert abs(o.params["x"] - n.params["x"]) <= 1
            assert abs(o.params["y"] - n.params["y"]) <= 1

    def test_scale_clamps_to_q8(self):
        ast = build_rectangle_solid()
        scaled = CoordinateScale(scale_range=(2.0, 2.0))(ast)
        for n in scaled.dfs():
            if n.node_type == NodeType.CRD:
                assert 0 <= n.params["x"] <= 255
                assert 0 <= n.params["y"] <= 255


class TestCoordinateMirror:

    def test_x_mirror(self):
        coord = make_coord(100, 50)
        line = make_line(100, 50, 200, 50)
        edge = make_edge(line)
        loop = make_loop([edge])
        face = make_face([loop])
        sketch = make_sketch([face])
        sol = make_solid(sketch, [make_extrude(64)])
        ast = make_program([sol])

        mirrored = CoordinateMirror(axes=["x"])(ast)
        coords = mirrored.collect(lambda n: n.node_type == NodeType.CRD)
        assert coords[0].params["x"] == 255 - 100
        assert coords[0].params["y"] == 50

    def test_y_mirror(self):
        ast = build_rectangle_solid()
        mirrored = CoordinateMirror(axes=["y"])(ast)
        r = validate_ast(mirrored)
        assert r.is_valid, r.errors

    def test_mirror_preserves_structure(self):
        ast = build_rectangle_solid()
        mirrored = CoordinateMirror(axes=["x"])(ast)
        assert mirrored.subtree_size == ast.subtree_size


class TestEdgeOrderShuffle:

    def test_shuffle_preserves_edge_count(self):
        ast = build_rectangle_solid()
        shuffled = EdgeOrderShuffle()(ast)
        orig_edges = ast.collect(lambda n: n.node_type == NodeType.EDGE)
        new_edges = shuffled.collect(lambda n: n.node_type == NodeType.EDGE)
        assert len(orig_edges) == len(new_edges)

    def test_shuffle_preserves_validity(self):
        ast = build_rectangle_solid()
        shuffled = EdgeOrderShuffle()(ast)
        r = validate_ast(shuffled)
        assert r.is_valid, r.errors


class TestSolidOrderShuffle:

    def test_shuffle_multi_solid(self):
        ast = build_multi_solid()
        random.seed(123)
        shuffled = SolidOrderShuffle()(ast)
        assert len(shuffled.children) == 2
        r = validate_ast(shuffled)
        assert r.is_valid, r.errors

    def test_single_solid_unchanged(self):
        ast = build_rectangle_solid()
        shuffled = SolidOrderShuffle()(ast)
        assert ast.structurally_equal(shuffled)


class TestASTAugmentor:

    def test_augmentor_preserves_validity(self):
        ast = build_rectangle_solid()
        augmentor = ASTAugmentor(p=1.0)
        for _ in range(10):
            augmented = augmentor(ast)
            r = validate_ast(augmented)
            assert r.is_valid, r.errors

    def test_augmentor_changes_tree(self):
        random.seed(99)
        ast = build_rectangle_solid()
        augmentor = ASTAugmentor(p=1.0)
        augmented = augmentor(ast)
        # With p=1.0 and seed=99 at least one transform should change something
        assert not ast.structurally_equal(augmented)

    def test_augmentor_p0_is_identity(self):
        ast = build_rectangle_solid()
        augmentor = ASTAugmentor(p=0.0)
        augmented = augmentor(ast)
        assert ast.structurally_equal(augmented)
