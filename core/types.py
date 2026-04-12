"""
AST type system: NodeType enum, NodeSpec, NodeRegistry, ChildSlot, ParamDef.

All 14 semantic + 3 special node types are registered here with their
child-slot constraints and parameter schemas. This module has zero
external dependencies (pure Python / stdlib only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, FrozenSet, List, Optional, Set

# ── Cardinality Limits ──────────────────────────────────────────────

MAX_SOLIDS = 16
MAX_OPS = 8
MAX_FACES = 32
MAX_LOOPS = 8
MAX_EDGES = 64
MAX_DEPTH = 5


# ── NodeType Enum ───────────────────────────────────────────────────

class NodeType(IntEnum):
    """AST node types spanning six depth levels (L0–L5) plus specials."""
    # L0
    PROG = 0
    # L1
    SOL = 1
    BOOL = 2
    # L2
    SKT = 3
    EXT = 4
    REV = 5
    # L3
    FACE = 6
    # L4
    LOOP = 7
    EDGE = 8
    # L5 – Curves
    LN = 9
    ARC = 10
    CIR = 11
    # L5 – Values
    CRD = 12
    SCL = 13
    # Special
    MASK = 14
    NOISE = 15
    NIL = 16


DEPTH_OF: Dict[NodeType, int] = {
    NodeType.PROG: 0,
    NodeType.SOL: 1,
    NodeType.BOOL: 1,
    NodeType.SKT: 2,
    NodeType.EXT: 2,
    NodeType.REV: 2,
    NodeType.FACE: 3,
    NodeType.LOOP: 4,
    NodeType.EDGE: 4,
    NodeType.LN: 5,
    NodeType.ARC: 5,
    NodeType.CIR: 5,
    NodeType.CRD: 5,
    NodeType.SCL: 5,
}

SEMANTIC_NODE_TYPES: List[NodeType] = [
    nt for nt in NodeType if nt.value <= NodeType.SCL.value
]


# ── Supporting Dataclasses ──────────────────────────────────────────

@dataclass(frozen=True)
class ParamDef:
    """Definition of a single node parameter."""
    name: str
    dtype: str  # "q8" | "enum" | "vec3" | "str"
    enum_values: Optional[List[str]] = None
    default: Optional[Any] = None


@dataclass(frozen=True)
class ChildSlot:
    """Defines one named child-slot of a node type."""
    name: str
    allowed_types: FrozenSet[str]
    min_count: int
    max_count: int


@dataclass(frozen=True)
class NodeSpec:
    """
    Complete specification of a node type:
    tag, depth, child slots, parameter schema, and token ID in vocabulary.
    """
    tag: str
    depth: int
    child_schema: List[ChildSlot] = field(default_factory=list)
    param_schema: Dict[str, ParamDef] = field(default_factory=dict)
    token_id: int = -1  # filled at registration time


# ── Node Registry ───────────────────────────────────────────────────

class NodeRegistry:
    """
    Global node-type registry.

    Maintains a mapping ``tag -> NodeSpec`` and supports runtime extension.
    Call ``NodeRegistry.bootstrap()`` to register the built-in 14 semantic
    node types defined by the architecture spec.
    """

    _registry: Dict[str, NodeSpec] = {}
    _bootstrapped: bool = False

    @classmethod
    def register(cls, spec: NodeSpec) -> None:
        if spec.tag in cls._registry:
            raise ValueError(f"Duplicate tag: {spec.tag}")
        cls._registry[spec.tag] = spec

    @classmethod
    def get(cls, tag: str) -> NodeSpec:
        cls._ensure_bootstrapped()
        if tag not in cls._registry:
            raise KeyError(f"Unknown node tag: {tag}")
        return cls._registry[tag]

    @classmethod
    def all_tags(cls) -> List[str]:
        cls._ensure_bootstrapped()
        return list(cls._registry.keys())

    @classmethod
    def all_specs(cls) -> List[NodeSpec]:
        cls._ensure_bootstrapped()
        return list(cls._registry.values())

    @classmethod
    def contains(cls, tag: str) -> bool:
        cls._ensure_bootstrapped()
        return tag in cls._registry

    @classmethod
    def reset(cls) -> None:
        """Clear registry (mainly for tests)."""
        cls._registry.clear()
        cls._bootstrapped = False

    @classmethod
    def _ensure_bootstrapped(cls) -> None:
        if not cls._bootstrapped:
            cls.bootstrap()

    @classmethod
    def bootstrap(cls) -> None:
        """Register all built-in node types per the architecture spec."""
        if cls._bootstrapped:
            return
        cls._bootstrapped = True

        from core.tokenizer import get_node_type_token

        _defs: List[tuple] = [
            # (tag, depth, child_schema, param_schema)
            ("PROG", 0,
             [ChildSlot("solids", frozenset({"SOL", "BOOL"}), 1, MAX_SOLIDS)],
             {"version": ParamDef("version", "str", default="1.0")}),

            ("SOL", 1,
             [ChildSlot("sketch", frozenset({"SKT"}), 1, 1),
              ChildSlot("ops", frozenset({"EXT", "REV"}), 1, MAX_OPS)],
             {}),

            ("BOOL", 1,
             [ChildSlot("operands", frozenset({"SOL"}), 2, 2)],
             {"op_type": ParamDef("op_type", "enum",
                                  enum_values=["union", "intersect", "subtract"])}),

            ("SKT", 2,
             [ChildSlot("faces", frozenset({"FACE"}), 1, MAX_FACES)],
             {}),

            ("EXT", 2, [],
             {"distance_fwd": ParamDef("distance_fwd", "q8"),
              "distance_bwd": ParamDef("distance_bwd", "q8"),
              "op_type": ParamDef("op_type", "enum",
                                  enum_values=["new", "cut", "join"])}),

            ("REV", 2, [],
             {"angle": ParamDef("angle", "q8"),
              "op_type": ParamDef("op_type", "enum",
                                  enum_values=["new", "cut", "join"])}),

            ("FACE", 3,
             [ChildSlot("loops", frozenset({"LOOP"}), 1, MAX_LOOPS)],
             {}),

            ("LOOP", 4,
             [ChildSlot("edges", frozenset({"EDGE"}), 1, MAX_EDGES)],
             {}),

            ("EDGE", 4,
             [ChildSlot("curve", frozenset({"LN", "ARC", "CIR"}), 1, 1)],
             {}),

            ("LN", 5,
             [ChildSlot("coords", frozenset({"CRD"}), 2, 2)],
             {}),

            ("ARC", 5,
             [ChildSlot("coords", frozenset({"CRD"}), 3, 3)],
             {}),

            ("CIR", 5,
             [ChildSlot("center", frozenset({"CRD"}), 1, 1),
              ChildSlot("radius", frozenset({"SCL"}), 1, 1)],
             {}),

            ("CRD", 5, [],
             {"x": ParamDef("x", "q8"),
              "y": ParamDef("y", "q8")}),

            ("SCL", 5, [],
             {"value": ParamDef("value", "q8")}),
        ]

        for tag, depth, child_schema, param_schema in _defs:
            token_id = get_node_type_token(tag)
            spec = NodeSpec(
                tag=tag,
                depth=depth,
                child_schema=list(child_schema),
                param_schema=dict(param_schema),
                token_id=token_id,
            )
            cls.register(spec)

    @classmethod
    def get_children_types(cls, parent_tag: str) -> Set[str]:
        """Return the set of all allowed child type tags for *parent_tag*."""
        spec = cls.get(parent_tag)
        result: Set[str] = set()
        for slot in spec.child_schema:
            result |= set(slot.allowed_types)
        return result
