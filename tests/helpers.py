"""
Shared test helpers – AST fixture builders for common geometries.

Every builder creates a valid, architecture-compliant AST tree and can be
used across unit, integration, and smoke tests.
"""

from core.types import NodeType
from core.ast_node import ASTNode, reset_id_counter


def make_coord(x: int, y: int) -> ASTNode:
    return ASTNode(
        node_type=NodeType.CRD,
        depth=5,
        children=(),
        params={"x": x, "y": y},
    )


def make_scalar(value: int) -> ASTNode:
    return ASTNode(
        node_type=NodeType.SCL,
        depth=5,
        children=(),
        params={"value": value},
    )


def make_line(x1: int, y1: int, x2: int, y2: int) -> ASTNode:
    return ASTNode(
        node_type=NodeType.LN,
        depth=5,
        children=(make_coord(x1, y1), make_coord(x2, y2)),
        params={},
    )


def make_arc(x1: int, y1: int, x2: int, y2: int, x3: int, y3: int) -> ASTNode:
    return ASTNode(
        node_type=NodeType.ARC,
        depth=5,
        children=(make_coord(x1, y1), make_coord(x2, y2), make_coord(x3, y3)),
        params={},
    )


def make_circle(cx: int, cy: int, r: int) -> ASTNode:
    return ASTNode(
        node_type=NodeType.CIR,
        depth=5,
        children=(make_coord(cx, cy), make_scalar(r)),
        params={},
    )


def make_edge(curve: ASTNode) -> ASTNode:
    return ASTNode(
        node_type=NodeType.EDGE,
        depth=4,
        children=(curve,),
        params={},
    )


def make_loop(edges: list) -> ASTNode:
    return ASTNode(
        node_type=NodeType.LOOP,
        depth=4,
        children=tuple(edges),
        params={},
    )


def make_face(loops: list) -> ASTNode:
    return ASTNode(
        node_type=NodeType.FACE,
        depth=3,
        children=tuple(loops),
        params={},
    )


def make_sketch(faces: list) -> ASTNode:
    return ASTNode(
        node_type=NodeType.SKT,
        depth=2,
        children=tuple(faces),
        params={},
    )


def make_extrude(distance_fwd: int = 64, distance_bwd: int = 0,
                 op_type: str = "new") -> ASTNode:
    return ASTNode(
        node_type=NodeType.EXT,
        depth=2,
        children=(),
        params={
            "distance_fwd": distance_fwd,
            "distance_bwd": distance_bwd,
            "op_type": op_type,
        },
    )


def make_revolve(angle: int = 128, op_type: str = "new") -> ASTNode:
    return ASTNode(
        node_type=NodeType.REV,
        depth=2,
        children=(),
        params={"angle": angle, "op_type": op_type},
    )


def make_solid(sketch: ASTNode, ops: list) -> ASTNode:
    return ASTNode(
        node_type=NodeType.SOL,
        depth=1,
        children=(sketch, *ops),
        params={},
    )


def make_boolean(left: ASTNode, right: ASTNode,
                 op_type: str = "union") -> ASTNode:
    return ASTNode(
        node_type=NodeType.BOOL,
        depth=1,
        children=(left, right),
        params={"op_type": op_type},
    )


def make_program(solids: list) -> ASTNode:
    return ASTNode(
        node_type=NodeType.PROG,
        depth=0,
        children=tuple(solids),
        params={"version": "1.0"},
    )


# ── Pre-built fixture ASTs ──────────────────────────────────────────

def build_rectangle_solid(
    w: int = 128, h: int = 128, depth: int = 64,
) -> ASTNode:
    """
    A rectangle sketch (4 lines) extruded by *depth*.
    Coords: (0,0)→(w,0)→(w,h)→(0,h)→(0,0).
    """
    edges = [
        make_edge(make_line(0, 0, w, 0)),
        make_edge(make_line(w, 0, w, h)),
        make_edge(make_line(w, h, 0, h)),
        make_edge(make_line(0, h, 0, 0)),
    ]
    loop = make_loop(edges)
    face = make_face([loop])
    sketch = make_sketch([face])
    ext = make_extrude(depth, 0, "new")
    solid = make_solid(sketch, [ext])
    return make_program([solid])


def build_triangle_solid(depth: int = 64) -> ASTNode:
    """Equilateral-ish triangle sketch extruded."""
    edges = [
        make_edge(make_line(0, 0, 128, 0)),
        make_edge(make_line(128, 0, 64, 111)),
        make_edge(make_line(64, 111, 0, 0)),
    ]
    loop = make_loop(edges)
    face = make_face([loop])
    sketch = make_sketch([face])
    ext = make_extrude(depth, 0, "new")
    solid = make_solid(sketch, [ext])
    return make_program([solid])


def build_circle_solid(cx: int = 128, cy: int = 128, r: int = 64,
                       depth: int = 64) -> ASTNode:
    """Single circle sketch extruded."""
    edge = make_edge(make_circle(cx, cy, r))
    loop = make_loop([edge])
    face = make_face([loop])
    sketch = make_sketch([face])
    ext = make_extrude(depth, 0, "new")
    solid = make_solid(sketch, [ext])
    return make_program([solid])


def build_multi_solid() -> ASTNode:
    """Program with two separate solids."""
    s1 = build_rectangle_solid(100, 100, 50).children[0]
    s2 = build_circle_solid(depth=30).children[0]
    return make_program([s1, s2])


def build_arc_sketch_solid() -> ASTNode:
    """Sketch with both lines and arcs."""
    edges = [
        make_edge(make_line(0, 0, 128, 0)),
        make_edge(make_arc(128, 0, 160, 64, 128, 128)),
        make_edge(make_line(128, 128, 0, 128)),
        make_edge(make_line(0, 128, 0, 0)),
    ]
    loop = make_loop(edges)
    face = make_face([loop])
    sketch = make_sketch([face])
    ext = make_extrude(64, 0, "new")
    solid = make_solid(sketch, [ext])
    return make_program([solid])
