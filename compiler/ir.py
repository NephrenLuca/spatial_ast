"""
Intermediate Representation (IR) for the deterministic compiler.

``IRInstruction`` is a flat, opcode-based representation that sits between
the tree-structured AST and the DeepCAD command sequence.  Each instruction
carries an opcode string and a list of float parameters.

Supported opcodes
-----------------
  sketch_start / sketch_end   – bracket a 2-D sketch
  loop_start   / loop_end     – bracket a closed loop
  line                        – 4 params: sx, sy, ex, ey
  arc                         – 6 params: sx, sy, mx, my, ex, ey
  circle                      – 3 params: cx, cy, r
  extrude                     – 4 params: dist_fwd, dist_bwd, op_code, direction
  revolve                     – 3 params: angle, op_code, axis
  boolean                     – 1 param:  op_code
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class IRInstruction:
    opcode: str
    params: List[float] = field(default_factory=list)

    def __repr__(self) -> str:
        p = ", ".join(f"{v:.4f}" for v in self.params)
        return f"IR({self.opcode}[{p}])"


# Opcode → human-readable description (for debugging / visualisation)
IR_OPCODE_DOC = {
    "sketch_start": "Begin 2-D sketch",
    "sketch_end":   "End 2-D sketch",
    "loop_start":   "Begin closed loop",
    "loop_end":     "End closed loop",
    "line":         "Line segment (sx, sy, ex, ey)",
    "arc":          "Three-point arc (sx, sy, mx, my, ex, ey)",
    "circle":       "Full circle (cx, cy, r)",
    "extrude":      "Extrude (dist_fwd, dist_bwd, op, direction)",
    "revolve":      "Revolve (angle, op, axis)",
    "boolean":      "Boolean operation (op)",
}

# Numeric codes for extrude / revolve operation types
EXTRUDE_OP_MAP = {"new": 0.0, "cut": 1.0, "join": 2.0}
EXTRUDE_OP_INV = {0.0: "new", 1.0: "cut", 2.0: "join"}

BOOLEAN_OP_MAP = {"union": 0.0, "intersect": 1.0, "subtract": 2.0}
BOOLEAN_OP_INV = {0.0: "union", 1.0: "intersect", 2.0: "subtract"}
