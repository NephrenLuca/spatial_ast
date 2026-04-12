"""
DeepCAD command list → AST tree decompiler.

Converts the normalised command dicts (as produced by ``DeepCADParser`` or
the compiler backend) into a valid AST tree.  This is the inverse of
``IREmitter + DeepCADBackend``.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Tuple

from core.types import NodeType
from core.ast_node import ASTNode


class DeepCADDecompiler:
    """
    Convert a list of DeepCAD command dicts into an AST tree.

    The input format matches what ``DeepCADParser.parse_dict`` produces
    and what ``DeepCADBackend.ir_to_commands`` outputs.
    """

    def __init__(self) -> None:
        self._id_gen = itertools.count()

    def _nid(self) -> int:
        return next(self._id_gen)

    def decompile(self, commands: List[Dict[str, Any]]) -> ASTNode:
        """
        Build a complete AST from a flat command list.

        Groups consecutive (sketch, extrude/revolve…) sequences into
        Solid nodes under a single Program root.
        """
        solids: List[ASTNode] = []
        i = 0
        while i < len(commands):
            cmd = commands[i]
            if cmd["type"] == "sketch":
                solid, i = self._build_solid(commands, i)
                solids.append(solid)
            else:
                i += 1  # skip unexpected top-level command

        if not solids:
            raise ValueError("No valid solid found in command list")

        return ASTNode(
            node_type=NodeType.PROG,
            depth=0,
            children=tuple(solids),
            params={"version": "1.0"},
            node_id=self._nid(),
            span=(0, 0),
        )

    # ── solid-level ─────────────────────────────────────────────────

    def _build_solid(
        self, commands: List[Dict[str, Any]], start: int,
    ) -> Tuple[ASTNode, int]:
        sketch_cmd = commands[start]
        sketch = self._build_sketch(sketch_cmd)
        i = start + 1

        ops: List[ASTNode] = []
        while i < len(commands) and commands[i]["type"] in ("extrude", "revolve"):
            op = self._build_operation(commands[i])
            ops.append(op)
            i += 1

        if not ops:
            ops.append(self._default_extrude())

        return ASTNode(
            node_type=NodeType.SOL,
            depth=1,
            children=(sketch, *ops),
            params={},
            node_id=self._nid(),
            span=(0, 0),
        ), i

    # ── sketch / face / loop ────────────────────────────────────────

    def _build_sketch(self, cmd: Dict[str, Any]) -> ASTNode:
        faces: List[ASTNode] = []
        for loop_cmd in cmd.get("loops", []):
            loop = self._build_loop(loop_cmd)
            face = ASTNode(
                node_type=NodeType.FACE, depth=3,
                children=(loop,), params={},
                node_id=self._nid(), span=(0, 0),
            )
            faces.append(face)

        if not faces:
            raise ValueError("Sketch has no loops")

        return ASTNode(
            node_type=NodeType.SKT, depth=2,
            children=tuple(faces), params={},
            node_id=self._nid(), span=(0, 0),
        )

    def _build_loop(self, loop_cmd: Dict[str, Any]) -> ASTNode:
        edges: List[ASTNode] = []
        for curve_cmd in loop_cmd.get("curves", []):
            curve = self._build_curve(curve_cmd)
            edge = ASTNode(
                node_type=NodeType.EDGE, depth=4,
                children=(curve,), params={},
                node_id=self._nid(), span=(0, 0),
            )
            edges.append(edge)

        if not edges:
            raise ValueError("Loop has no curves")

        return ASTNode(
            node_type=NodeType.LOOP, depth=4,
            children=tuple(edges), params={},
            node_id=self._nid(), span=(0, 0),
        )

    # ── curves ──────────────────────────────────────────────────────

    def _build_curve(self, c: Dict[str, Any]) -> ASTNode:
        ct = c["type"]
        if ct == "line":
            return self._build_line(c)
        elif ct == "arc":
            return self._build_arc(c)
        elif ct == "circle":
            return self._build_circle(c)
        else:
            raise ValueError(f"Unknown curve type: {ct}")

    def _build_line(self, c: Dict[str, Any]) -> ASTNode:
        start = self._make_coord(c["start_x"], c["start_y"])
        end = self._make_coord(c["end_x"], c["end_y"])
        return ASTNode(
            node_type=NodeType.LN, depth=5,
            children=(start, end), params={},
            node_id=self._nid(), span=(0, 0),
        )

    def _build_arc(self, c: Dict[str, Any]) -> ASTNode:
        start = self._make_coord(c["start_x"], c["start_y"])
        mid = self._make_coord(c["mid_x"], c["mid_y"])
        end = self._make_coord(c["end_x"], c["end_y"])
        return ASTNode(
            node_type=NodeType.ARC, depth=5,
            children=(start, mid, end), params={},
            node_id=self._nid(), span=(0, 0),
        )

    def _build_circle(self, c: Dict[str, Any]) -> ASTNode:
        center = self._make_coord(c["center_x"], c["center_y"])
        radius = ASTNode(
            node_type=NodeType.SCL, depth=5,
            children=(), params={"value": int(c["radius"])},
            node_id=self._nid(), span=(0, 0),
        )
        return ASTNode(
            node_type=NodeType.CIR, depth=5,
            children=(center, radius), params={},
            node_id=self._nid(), span=(0, 0),
        )

    # ── operations ──────────────────────────────────────────────────

    def _build_operation(self, cmd: Dict[str, Any]) -> ASTNode:
        if cmd["type"] == "extrude":
            return ASTNode(
                node_type=NodeType.EXT, depth=2,
                children=(), params={
                    "distance_fwd": int(cmd["distance_fwd"]),
                    "distance_bwd": int(cmd["distance_bwd"]),
                    "op_type": cmd.get("op_type", "new"),
                },
                node_id=self._nid(), span=(0, 0),
            )
        elif cmd["type"] == "revolve":
            return ASTNode(
                node_type=NodeType.REV, depth=2,
                children=(), params={
                    "angle": int(cmd["angle"]),
                    "op_type": cmd.get("op_type", "new"),
                },
                node_id=self._nid(), span=(0, 0),
            )
        else:
            raise ValueError(f"Unknown operation type: {cmd['type']}")

    def _default_extrude(self) -> ASTNode:
        """Fallback extrude when sketch has no following operation."""
        return ASTNode(
            node_type=NodeType.EXT, depth=2,
            children=(), params={
                "distance_fwd": 128,
                "distance_bwd": 0,
                "op_type": "new",
            },
            node_id=self._nid(), span=(0, 0),
        )

    # ── helpers ─────────────────────────────────────────────────────

    def _make_coord(self, x: Any, y: Any) -> ASTNode:
        return ASTNode(
            node_type=NodeType.CRD, depth=5,
            children=(), params={"x": int(x), "y": int(y)},
            node_id=self._nid(), span=(0, 0),
        )
