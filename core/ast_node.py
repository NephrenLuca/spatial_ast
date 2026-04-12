"""
ASTNode – immutable dataclass representing a single node in the spatial AST.

Provides tree traversal helpers (DFS / BFS), depth calculation,
subtree slicing, equality checks, and pretty-printing.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from core.types import DEPTH_OF, NodeType


_id_counter = itertools.count()


def _next_node_id() -> int:
    return next(_id_counter)


def reset_id_counter(start: int = 0) -> None:
    """Reset the global node-id counter (useful in tests)."""
    global _id_counter
    _id_counter = itertools.count(start)


@dataclass(frozen=True)
class ASTNode:
    """
    Immutable AST node.

    Parameters
    ----------
    node_type : NodeType
        Enum tag for this node.
    depth : int
        Tree depth (0-5) matching the architecture spec.
    children : tuple[ASTNode, ...]
        Ordered child nodes (empty for leaves).
    params : dict[str, Any]
        Node-specific parameters (Q8, enum, etc.).
    node_id : int
        Unique identifier within a tree, auto-generated if -1.
    span : tuple[int, int]
        ``[start, end)`` position in the serialised token stream.
        Set to ``(0, 0)`` until serialisation fills it in.
    """
    node_type: NodeType
    depth: int
    children: Tuple["ASTNode", ...] = ()
    params: Dict[str, Any] = field(default_factory=dict)
    node_id: int = -1
    span: Tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if self.node_id == -1:
            object.__setattr__(self, "node_id", _next_node_id())
        if not isinstance(self.children, tuple):
            object.__setattr__(self, "children", tuple(self.children))

    # ── Traversal ───────────────────────────────────────────────────

    def dfs(self) -> Iterator[ASTNode]:
        """Pre-order depth-first traversal."""
        yield self
        for child in self.children:
            yield from child.dfs()

    def bfs(self) -> Iterator[ASTNode]:
        """Level-order breadth-first traversal."""
        queue: List[ASTNode] = [self]
        while queue:
            node = queue.pop(0)
            yield node
            queue.extend(node.children)

    # ── Tree Metrics ────────────────────────────────────────────────

    @property
    def subtree_size(self) -> int:
        """Total number of nodes in this subtree (including self)."""
        return 1 + sum(c.subtree_size for c in self.children)

    @property
    def max_depth(self) -> int:
        """Maximum depth among all descendant nodes."""
        if not self.children:
            return self.depth
        return max(c.max_depth for c in self.children)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    # ── Structural Equality ─────────────────────────────────────────

    def structurally_equal(self, other: ASTNode) -> bool:
        """
        Two AST subtrees are structurally equal when they share the same
        node_type, params, and recursively equal children (ignoring
        node_id and span).
        """
        if self.node_type != other.node_type:
            return False
        if self.params != other.params:
            return False
        if len(self.children) != len(other.children):
            return False
        return all(
            a.structurally_equal(b)
            for a, b in zip(self.children, other.children)
        )

    # ── Subtree Manipulation (returns new trees) ────────────────────

    def replace_child(self, index: int, new_child: ASTNode) -> ASTNode:
        """Return a copy of this node with ``children[index]`` replaced."""
        children = list(self.children)
        children[index] = new_child
        return replace(self, children=tuple(children))

    def map(self, fn: Callable[[ASTNode], ASTNode]) -> ASTNode:
        """Apply *fn* to every node (bottom-up) and return the new tree."""
        new_children = tuple(c.map(fn) for c in self.children)
        return fn(replace(self, children=new_children))

    def find(self, predicate: Callable[[ASTNode], bool]) -> Optional[ASTNode]:
        """Return the first DFS node satisfying *predicate*, or None."""
        for node in self.dfs():
            if predicate(node):
                return node
        return None

    def collect(self, predicate: Callable[[ASTNode], bool]) -> List[ASTNode]:
        """Collect all DFS nodes satisfying *predicate*."""
        return [n for n in self.dfs() if predicate(n)]

    def deepcopy(self) -> ASTNode:
        """Return a deep copy with freshly assigned node_ids."""
        new_children = tuple(c.deepcopy() for c in self.children)
        return replace(self, children=new_children, node_id=_next_node_id())

    # ── Pretty Printing ─────────────────────────────────────────────

    def pretty(self, indent: int = 0) -> str:
        """Human-readable tree representation."""
        pad = "  " * indent
        parts = [f"{pad}{self.node_type.name}"]
        if self.params:
            param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
            parts[0] += f"({param_str})"
        lines = [parts[0]]
        for child in self.children:
            lines.append(child.pretty(indent + 1))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ASTNode({self.node_type.name}, depth={self.depth}, "
            f"#children={len(self.children)}, params={self.params})"
        )
