"""
Online data augmentation for AST trees.

All transforms operate on the immutable ``ASTNode`` tree, returning a
new tree with modified Q8 coordinate parameters.  They are designed to
be composed and applied on-the-fly inside the DataLoader.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import List, Optional

from core.types import NodeType
from core.ast_node import ASTNode


def _clamp_q8(v: int) -> int:
    return max(0, min(255, v))


# ═══════════════════════════════════════════════════════════════════
# Individual transforms
# ═══════════════════════════════════════════════════════════════════

class CoordinateShift:
    """Shift all CRD (x, y) values by a random offset in [-max_shift, +max_shift]."""

    def __init__(self, max_shift: int = 10) -> None:
        self.max_shift = max_shift

    def __call__(self, root: ASTNode) -> ASTNode:
        dx = random.randint(-self.max_shift, self.max_shift)
        dy = random.randint(-self.max_shift, self.max_shift)
        return self._apply(root, dx, dy)

    def _apply(self, node: ASTNode, dx: int, dy: int) -> ASTNode:
        if node.node_type == NodeType.CRD:
            new_params = {
                "x": _clamp_q8(node.params["x"] + dx),
                "y": _clamp_q8(node.params["y"] + dy),
            }
            return replace(node, params=new_params)

        if node.children:
            new_children = tuple(self._apply(c, dx, dy) for c in node.children)
            return replace(node, children=new_children)

        return node


class CoordinateScale:
    """Scale all CRD values around the centroid by a random factor."""

    def __init__(self, scale_range: tuple = (0.8, 1.2)) -> None:
        self.lo, self.hi = scale_range

    def __call__(self, root: ASTNode) -> ASTNode:
        factor = random.uniform(self.lo, self.hi)
        coords = self._collect_coords(root)
        if not coords:
            return root
        cx = sum(c[0] for c in coords) / len(coords)
        cy = sum(c[1] for c in coords) / len(coords)
        return self._apply(root, factor, cx, cy)

    def _apply(self, node: ASTNode, factor: float, cx: float, cy: float) -> ASTNode:
        if node.node_type == NodeType.CRD:
            nx = _clamp_q8(int(round(cx + (node.params["x"] - cx) * factor)))
            ny = _clamp_q8(int(round(cy + (node.params["y"] - cy) * factor)))
            return replace(node, params={"x": nx, "y": ny})

        if node.children:
            new_children = tuple(self._apply(c, factor, cx, cy) for c in node.children)
            return replace(node, children=new_children)

        return node

    @staticmethod
    def _collect_coords(node: ASTNode) -> List[tuple]:
        return [
            (n.params["x"], n.params["y"])
            for n in node.dfs()
            if n.node_type == NodeType.CRD
        ]


class CoordinateMirror:
    """Mirror all CRD values along the chosen axis."""

    def __init__(self, axes: Optional[List[str]] = None) -> None:
        self.axes = axes or ["x", "y"]

    def __call__(self, root: ASTNode) -> ASTNode:
        axis = random.choice(self.axes)
        return self._apply(root, axis)

    def _apply(self, node: ASTNode, axis: str) -> ASTNode:
        if node.node_type == NodeType.CRD:
            params = dict(node.params)
            if axis == "x":
                params["x"] = 255 - params["x"]
            elif axis == "y":
                params["y"] = 255 - params["y"]
            return replace(node, params=params)

        if node.children:
            new_children = tuple(self._apply(c, axis) for c in node.children)
            return replace(node, children=new_children)

        return node


class EdgeOrderShuffle:
    """
    Cyclically rotate edges within each LOOP.

    This preserves loop closure (same set of edges, just a different
    starting point).
    """

    def __call__(self, root: ASTNode) -> ASTNode:
        return self._apply(root)

    def _apply(self, node: ASTNode) -> ASTNode:
        if node.node_type == NodeType.LOOP and len(node.children) > 1:
            edges = list(node.children)
            k = random.randint(0, len(edges) - 1)
            rotated = edges[k:] + edges[:k]
            return replace(node, children=tuple(rotated))

        if node.children:
            new_children = tuple(self._apply(c) for c in node.children)
            return replace(node, children=new_children)

        return node


class SolidOrderShuffle:
    """Randomly permute the order of Solid children under Program."""

    def __call__(self, root: ASTNode) -> ASTNode:
        if root.node_type == NodeType.PROG and len(root.children) > 1:
            children = list(root.children)
            random.shuffle(children)
            return replace(root, children=tuple(children))
        return root


# ═══════════════════════════════════════════════════════════════════
# Composed augmentor
# ═══════════════════════════════════════════════════════════════════

class ASTAugmentor:
    """
    Apply a random subset of augmentations to an AST tree.

    Each transform is applied independently with probability ``p``.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p
        self.transforms = [
            CoordinateShift(max_shift=10),
            CoordinateScale(scale_range=(0.8, 1.2)),
            CoordinateMirror(axes=["x", "y"]),
            EdgeOrderShuffle(),
            SolidOrderShuffle(),
        ]

    def __call__(self, root: ASTNode) -> ASTNode:
        for t in self.transforms:
            if random.random() < self.p:
                root = t(root)
        return root
