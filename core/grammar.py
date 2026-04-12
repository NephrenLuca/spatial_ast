"""
Context-free grammar rules, parent-child compatibility matrix,
children cardinality constraints, and AST validation.

Every validation check returns a ``ValidationResult`` so that callers
can inspect individual violations without exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from core.types import (
    DEPTH_OF,
    MAX_EDGES,
    MAX_FACES,
    MAX_LOOPS,
    MAX_OPS,
    MAX_SOLIDS,
    NodeType,
    NodeRegistry,
)
from core.ast_node import ASTNode

# ═══════════════════════════════════════════════════════════════════
# Parent-Child Compatibility Matrix
# ═══════════════════════════════════════════════════════════════════

PARENT_CHILD_MATRIX: Dict[NodeType, FrozenSet[NodeType]] = {
    NodeType.PROG: frozenset({NodeType.SOL, NodeType.BOOL}),
    NodeType.SOL:  frozenset({NodeType.SKT, NodeType.EXT, NodeType.REV}),
    NodeType.BOOL: frozenset({NodeType.SOL}),
    NodeType.SKT:  frozenset({NodeType.FACE}),
    NodeType.EXT:  frozenset(),
    NodeType.REV:  frozenset(),
    NodeType.FACE: frozenset({NodeType.LOOP}),
    NodeType.LOOP: frozenset({NodeType.EDGE}),
    NodeType.EDGE: frozenset({NodeType.LN, NodeType.ARC, NodeType.CIR}),
    NodeType.LN:   frozenset({NodeType.CRD}),
    NodeType.ARC:  frozenset({NodeType.CRD}),
    NodeType.CIR:  frozenset({NodeType.CRD, NodeType.SCL}),
    NodeType.CRD:  frozenset(),
    NodeType.SCL:  frozenset(),
}


# ═══════════════════════════════════════════════════════════════════
# Children Cardinality Constraints
# ═══════════════════════════════════════════════════════════════════
# Maps parent type to { child_type_or_group: (min, max) }
# A "group" collapses multiple allowed child types into one bucket.

@dataclass(frozen=True)
class _Card:
    """Min/max child count for a set of child types."""
    child_types: FrozenSet[NodeType]
    min_count: int
    max_count: int


CHILDREN_CARDINALITY: Dict[NodeType, List[_Card]] = {
    NodeType.PROG: [
        _Card(frozenset({NodeType.SOL, NodeType.BOOL}), 1, MAX_SOLIDS),
    ],
    NodeType.SOL: [
        _Card(frozenset({NodeType.SKT}), 1, 1),
        _Card(frozenset({NodeType.EXT, NodeType.REV}), 1, MAX_OPS),
    ],
    NodeType.BOOL: [
        _Card(frozenset({NodeType.SOL}), 2, 2),
    ],
    NodeType.SKT: [
        _Card(frozenset({NodeType.FACE}), 1, MAX_FACES),
    ],
    NodeType.FACE: [
        _Card(frozenset({NodeType.LOOP}), 1, MAX_LOOPS),
    ],
    NodeType.LOOP: [
        _Card(frozenset({NodeType.EDGE}), 1, MAX_EDGES),
    ],
    NodeType.EDGE: [
        _Card(frozenset({NodeType.LN, NodeType.ARC, NodeType.CIR}), 1, 1),
    ],
    NodeType.LN: [
        _Card(frozenset({NodeType.CRD}), 2, 2),
    ],
    NodeType.ARC: [
        _Card(frozenset({NodeType.CRD}), 3, 3),
    ],
    NodeType.CIR: [
        _Card(frozenset({NodeType.CRD}), 1, 1),
        _Card(frozenset({NodeType.SCL}), 1, 1),
    ],
    # Leaf nodes – no children
    NodeType.EXT: [],
    NodeType.REV: [],
    NodeType.CRD: [],
    NodeType.SCL: [],
}

# ═══════════════════════════════════════════════════════════════════
# SOL child ordering: first child must be SKT, rest must be EXT/REV.
# This is a structural constraint beyond simple cardinality.
# ═══════════════════════════════════════════════════════════════════

_SOL_SKETCH_TYPES = frozenset({NodeType.SKT})
_SOL_OP_TYPES = frozenset({NodeType.EXT, NodeType.REV})

# ═══════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.is_valid = False
        self.errors.append(msg)

    def merge(self, other: ValidationResult) -> None:
        if not other.is_valid:
            self.is_valid = False
            self.errors.extend(other.errors)


def validate_ast(root: ASTNode) -> ValidationResult:
    """
    Perform comprehensive AST validation:
      1. Root must be PROG
      2. Node depths match DEPTH_OF
      3. Parent-child type compatibility
      4. Children cardinality constraints
      5. SOL child ordering (sketch first, then ops)
      6. Q8 parameter value ranges
      7. Leaf nodes have no children
    """
    result = ValidationResult()

    if root.node_type != NodeType.PROG:
        result.add_error(
            f"Root must be PROG, got {root.node_type.name}"
        )
        return result

    _validate_node(root, parent_type=None, result=result)
    return result


def _validate_node(
    node: ASTNode,
    parent_type: Optional[NodeType],
    result: ValidationResult,
) -> None:
    nt = node.node_type

    # 1. Depth check
    expected_depth = DEPTH_OF.get(nt)
    if expected_depth is not None and node.depth != expected_depth:
        result.add_error(
            f"Node {nt.name} (id={node.node_id}): depth={node.depth}, "
            f"expected {expected_depth}"
        )

    # 2. Parent-child type compatibility
    if parent_type is not None:
        allowed = PARENT_CHILD_MATRIX.get(parent_type, frozenset())
        if nt not in allowed:
            result.add_error(
                f"Node {nt.name} (id={node.node_id}) is not a valid child "
                f"of {parent_type.name} (allowed: "
                f"{', '.join(t.name for t in allowed)})"
            )

    # 3. Cardinality constraints
    _validate_cardinality(node, result)

    # 4. SOL ordering: children[0] must be SKT
    if nt == NodeType.SOL and node.children:
        if node.children[0].node_type not in _SOL_SKETCH_TYPES:
            result.add_error(
                f"SOL (id={node.node_id}): first child must be SKT, "
                f"got {node.children[0].node_type.name}"
            )
        for i, child in enumerate(node.children[1:], 1):
            if child.node_type not in _SOL_OP_TYPES:
                result.add_error(
                    f"SOL (id={node.node_id}): child[{i}] must be an "
                    f"operation (EXT/REV), got {child.node_type.name}"
                )

    # 5. CIR ordering: first child CRD (center), then SCL (radius)
    if nt == NodeType.CIR and len(node.children) == 2:
        if node.children[0].node_type != NodeType.CRD:
            result.add_error(
                f"CIR (id={node.node_id}): first child must be CRD (center), "
                f"got {node.children[0].node_type.name}"
            )
        if node.children[1].node_type != NodeType.SCL:
            result.add_error(
                f"CIR (id={node.node_id}): second child must be SCL (radius), "
                f"got {node.children[1].node_type.name}"
            )

    # 6. Q8 param range check
    _validate_params(node, result)

    # 7. Leaf nodes must have no children
    if nt in (NodeType.CRD, NodeType.SCL, NodeType.EXT, NodeType.REV):
        if node.children:
            result.add_error(
                f"Leaf node {nt.name} (id={node.node_id}) must have no "
                f"children, but has {len(node.children)}"
            )

    # Recurse
    for child in node.children:
        _validate_node(child, parent_type=nt, result=result)


def _validate_cardinality(node: ASTNode, result: ValidationResult) -> None:
    nt = node.node_type
    constraints = CHILDREN_CARDINALITY.get(nt)
    if constraints is None:
        return

    child_types = [c.node_type for c in node.children]

    for card in constraints:
        count = sum(1 for ct in child_types if ct in card.child_types)
        types_str = "/".join(t.name for t in sorted(card.child_types, key=lambda x: x.value))
        if count < card.min_count:
            result.add_error(
                f"{nt.name} (id={node.node_id}): needs at least "
                f"{card.min_count} {types_str} child(ren), found {count}"
            )
        if count > card.max_count:
            result.add_error(
                f"{nt.name} (id={node.node_id}): allows at most "
                f"{card.max_count} {types_str} child(ren), found {count}"
            )


def _validate_params(node: ASTNode, result: ValidationResult) -> None:
    nt = node.node_type
    for pname, pval in node.params.items():
        if pname in ("version", "name"):
            continue
        # Q8 values must be 0-255
        if isinstance(pval, int) and not (0 <= pval <= 255):
            result.add_error(
                f"{nt.name} (id={node.node_id}): param '{pname}' = {pval} "
                f"out of Q8 range [0, 255]"
            )


# ═══════════════════════════════════════════════════════════════════
# Helper: check if a child type is valid under a parent type
# ═══════════════════════════════════════════════════════════════════

def is_valid_child(parent_type: NodeType, child_type: NodeType) -> bool:
    allowed = PARENT_CHILD_MATRIX.get(parent_type, frozenset())
    return child_type in allowed


def get_allowed_children(parent_type: NodeType) -> FrozenSet[NodeType]:
    return PARENT_CHILD_MATRIX.get(parent_type, frozenset())
