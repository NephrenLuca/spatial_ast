"""Unit tests for core.grammar – validation, parent-child matrix, cardinality."""

import pytest

from core.types import NodeType, NodeRegistry
from core.ast_node import ASTNode, reset_id_counter
from core.grammar import (
    CHILDREN_CARDINALITY,
    PARENT_CHILD_MATRIX,
    ValidationResult,
    get_allowed_children,
    is_valid_child,
    validate_ast,
)

from tests.helpers import (
    build_arc_sketch_solid,
    build_circle_solid,
    build_multi_solid,
    build_rectangle_solid,
    build_triangle_solid,
    make_coord,
    make_edge,
    make_extrude,
    make_face,
    make_line,
    make_loop,
    make_program,
    make_scalar,
    make_sketch,
    make_solid,
    make_revolve,
    make_boolean,
    make_circle,
    make_arc,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_id_counter()
    NodeRegistry.reset()
    yield


# ═══════════════════════════════════════════════════════════════════
# Parent-Child Matrix
# ═══════════════════════════════════════════════════════════════════

class TestParentChildMatrix:

    def test_covers_all_semantic_types(self):
        for nt in NodeType:
            if nt.value <= NodeType.SCL.value:
                assert nt in PARENT_CHILD_MATRIX

    def test_prog_allows_sol_and_bool(self):
        assert NodeType.SOL in PARENT_CHILD_MATRIX[NodeType.PROG]
        assert NodeType.BOOL in PARENT_CHILD_MATRIX[NodeType.PROG]

    def test_prog_disallows_face(self):
        assert NodeType.FACE not in PARENT_CHILD_MATRIX[NodeType.PROG]

    def test_edge_allows_curves(self):
        allowed = PARENT_CHILD_MATRIX[NodeType.EDGE]
        assert NodeType.LN in allowed
        assert NodeType.ARC in allowed
        assert NodeType.CIR in allowed

    def test_leaf_nodes_have_no_children(self):
        for leaf in (NodeType.CRD, NodeType.SCL, NodeType.EXT, NodeType.REV):
            assert len(PARENT_CHILD_MATRIX[leaf]) == 0

    def test_is_valid_child_helper(self):
        assert is_valid_child(NodeType.PROG, NodeType.SOL)
        assert not is_valid_child(NodeType.PROG, NodeType.FACE)

    def test_get_allowed_children(self):
        allowed = get_allowed_children(NodeType.LOOP)
        assert allowed == frozenset({NodeType.EDGE})


# ═══════════════════════════════════════════════════════════════════
# Cardinality Constraints
# ═══════════════════════════════════════════════════════════════════

class TestCardinality:

    def test_covers_all_semantic_types(self):
        for nt in NodeType:
            if nt.value <= NodeType.SCL.value:
                assert nt in CHILDREN_CARDINALITY

    def test_ln_needs_2_coords(self):
        cards = CHILDREN_CARDINALITY[NodeType.LN]
        assert len(cards) == 1
        assert cards[0].min_count == 2
        assert cards[0].max_count == 2

    def test_arc_needs_3_coords(self):
        cards = CHILDREN_CARDINALITY[NodeType.ARC]
        assert cards[0].min_count == 3
        assert cards[0].max_count == 3


# ═══════════════════════════════════════════════════════════════════
# Legal AST Validation (should pass)
# ═══════════════════════════════════════════════════════════════════

class TestValidLegalASTs:
    """10 hand-crafted legal ASTs that must pass validation."""

    def test_rectangle_solid(self):
        r = validate_ast(build_rectangle_solid())
        assert r.is_valid, r.errors

    def test_triangle_solid(self):
        r = validate_ast(build_triangle_solid())
        assert r.is_valid, r.errors

    def test_circle_solid(self):
        r = validate_ast(build_circle_solid())
        assert r.is_valid, r.errors

    def test_multi_solid(self):
        r = validate_ast(build_multi_solid())
        assert r.is_valid, r.errors

    def test_arc_sketch_solid(self):
        r = validate_ast(build_arc_sketch_solid())
        assert r.is_valid, r.errors

    def test_solid_with_two_extrudes(self):
        sketch = make_sketch([make_face([make_loop([
            make_edge(make_line(0, 0, 128, 0)),
            make_edge(make_line(128, 0, 0, 0)),
        ])])])
        sol = make_solid(sketch, [make_extrude(64), make_extrude(32, 16, "cut")])
        ast = make_program([sol])
        r = validate_ast(ast)
        assert r.is_valid, r.errors

    def test_solid_with_revolve(self):
        sketch = make_sketch([make_face([make_loop([
            make_edge(make_line(0, 0, 128, 0)),
            make_edge(make_line(128, 0, 0, 0)),
        ])])])
        sol = make_solid(sketch, [make_revolve(128)])
        ast = make_program([sol])
        r = validate_ast(ast)
        assert r.is_valid, r.errors

    def test_face_with_two_loops(self):
        """Outer loop + inner loop (hole)."""
        outer = make_loop([
            make_edge(make_line(0, 0, 255, 0)),
            make_edge(make_line(255, 0, 255, 255)),
            make_edge(make_line(255, 255, 0, 255)),
            make_edge(make_line(0, 255, 0, 0)),
        ])
        inner = make_loop([
            make_edge(make_line(64, 64, 192, 64)),
            make_edge(make_line(192, 64, 192, 192)),
            make_edge(make_line(192, 192, 64, 192)),
            make_edge(make_line(64, 192, 64, 64)),
        ])
        face = make_face([outer, inner])
        sketch = make_sketch([face])
        sol = make_solid(sketch, [make_extrude(128)])
        ast = make_program([sol])
        r = validate_ast(ast)
        assert r.is_valid, r.errors

    def test_min_legal_ast(self):
        """Smallest possible legal program: 1 solid, 1 sketch, 1 face,
        1 loop, 1 edge, 1 line (2 coords), 1 extrude."""
        edge = make_edge(make_line(0, 0, 128, 128))
        loop = make_loop([edge])
        face = make_face([loop])
        sketch = make_sketch([face])
        ext = make_extrude(64)
        sol = make_solid(sketch, [ext])
        ast = make_program([sol])
        r = validate_ast(ast)
        assert r.is_valid, r.errors

    def test_circle_edge_solid(self):
        edge = make_edge(make_circle(128, 128, 64))
        loop = make_loop([edge])
        face = make_face([loop])
        sketch = make_sketch([face])
        sol = make_solid(sketch, [make_extrude(80)])
        ast = make_program([sol])
        r = validate_ast(ast)
        assert r.is_valid, r.errors


# ═══════════════════════════════════════════════════════════════════
# Illegal AST Validation (should reject)
# ═══════════════════════════════════════════════════════════════════

class TestRejectIllegalASTs:
    """10 hand-crafted illegal ASTs that must be rejected."""

    def test_non_prog_root(self):
        """Root must be PROG."""
        sol = make_solid(
            make_sketch([make_face([make_loop([
                make_edge(make_line(0, 0, 128, 128)),
            ])])]),
            [make_extrude(64)],
        )
        r = validate_ast(sol)
        assert not r.is_valid

    def test_prog_with_no_children(self):
        ast = ASTNode(NodeType.PROG, 0, children=(), params={"version": "1.0"})
        r = validate_ast(ast)
        assert not r.is_valid

    def test_sol_missing_sketch(self):
        """SOL with only an extrude (no sketch)."""
        sol = ASTNode(
            NodeType.SOL, 1,
            children=(make_extrude(64),),
            params={},
        )
        ast = make_program([sol])
        r = validate_ast(ast)
        assert not r.is_valid

    def test_sol_sketch_not_first(self):
        """SOL with extrude before sketch."""
        sketch = make_sketch([make_face([make_loop([
            make_edge(make_line(0, 0, 128, 128)),
        ])])])
        ext = make_extrude(64)
        sol = ASTNode(NodeType.SOL, 1, children=(ext, sketch), params={})
        ast = make_program([sol])
        r = validate_ast(ast)
        assert not r.is_valid

    def test_ln_with_one_coord(self):
        """Line needs exactly 2 CRD children."""
        bad_line = ASTNode(
            NodeType.LN, 5,
            children=(make_coord(0, 0),),
            params={},
        )
        edge = make_edge(bad_line)
        loop = make_loop([edge])
        face = make_face([loop])
        sketch = make_sketch([face])
        sol = make_solid(sketch, [make_extrude(64)])
        ast = make_program([sol])
        r = validate_ast(ast)
        assert not r.is_valid

    def test_arc_with_two_coords(self):
        """Arc needs exactly 3 CRD children."""
        bad_arc = ASTNode(
            NodeType.ARC, 5,
            children=(make_coord(0, 0), make_coord(128, 128)),
            params={},
        )
        edge = make_edge(bad_arc)
        loop = make_loop([edge])
        face = make_face([loop])
        sketch = make_sketch([face])
        sol = make_solid(sketch, [make_extrude(64)])
        ast = make_program([sol])
        r = validate_ast(ast)
        assert not r.is_valid

    def test_wrong_depth(self):
        """PROG at depth=3 is invalid."""
        ast = ASTNode(NodeType.PROG, 3, children=(
            make_solid(
                make_sketch([make_face([make_loop([
                    make_edge(make_line(0, 0, 128, 128)),
                ])])]),
                [make_extrude(64)],
            ),
        ), params={"version": "1.0"})
        r = validate_ast(ast)
        assert not r.is_valid

    def test_face_under_prog(self):
        """FACE cannot be direct child of PROG."""
        face = make_face([make_loop([make_edge(make_line(0, 0, 128, 128))])])
        ast = ASTNode(NodeType.PROG, 0, children=(face,), params={"version": "1.0"})
        r = validate_ast(ast)
        assert not r.is_valid

    def test_crd_with_children(self):
        """CRD is a leaf node, cannot have children."""
        bad_crd = ASTNode(
            NodeType.CRD, 5,
            children=(make_coord(0, 0),),
            params={"x": 0, "y": 0},
        )
        line = ASTNode(NodeType.LN, 5, children=(bad_crd, make_coord(128, 128)), params={})
        edge = make_edge(line)
        loop = make_loop([edge])
        face = make_face([loop])
        sketch = make_sketch([face])
        sol = make_solid(sketch, [make_extrude(64)])
        ast = make_program([sol])
        r = validate_ast(ast)
        assert not r.is_valid

    def test_q8_out_of_range(self):
        """Q8 param value > 255 is invalid."""
        bad_coord = ASTNode(
            NodeType.CRD, 5, children=(), params={"x": 300, "y": 0}
        )
        line = ASTNode(NodeType.LN, 5, children=(bad_coord, make_coord(128, 128)), params={})
        edge = make_edge(line)
        loop = make_loop([edge])
        face = make_face([loop])
        sketch = make_sketch([face])
        sol = make_solid(sketch, [make_extrude(64)])
        ast = make_program([sol])
        r = validate_ast(ast)
        assert not r.is_valid


# ═══════════════════════════════════════════════════════════════════
# ValidationResult
# ═══════════════════════════════════════════════════════════════════

class TestValidationResult:

    def test_initially_valid(self):
        vr = ValidationResult()
        assert vr.is_valid
        assert vr.errors == []

    def test_add_error_makes_invalid(self):
        vr = ValidationResult()
        vr.add_error("something wrong")
        assert not vr.is_valid
        assert len(vr.errors) == 1

    def test_merge(self):
        a = ValidationResult()
        b = ValidationResult()
        b.add_error("err1")
        a.merge(b)
        assert not a.is_valid
        assert "err1" in a.errors
