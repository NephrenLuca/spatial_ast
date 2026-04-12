"""
GeometryDescriptor extraction from an AST tree.

Each token in the serialised sequence gets a 4-dim continuous descriptor:
  (scale, curvature, depth_ratio, subtree_size)

These descriptors are fed to the model's ``geom_proj`` layer and also used
to condition the GC-Mamba SSM kernels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from core.types import DEPTH_OF, MAX_DEPTH, NodeType
from core.ast_node import ASTNode


@dataclass
class GeometryDescriptor:
    """4-dim continuous geometry descriptor for a single token."""
    scale: float           # local geometric scale, normalised to [0, 1]
    curvature: float       # Line=0, Arc≈1/R, Circle=1/R
    depth_ratio: float     # depth / max_depth ∈ [0, 1]
    subtree_size: int      # token count of the node's subtree

    def to_list(self) -> List[float]:
        return [self.scale, self.curvature, self.depth_ratio, float(self.subtree_size)]


def _dequantize(q8: int, q_min: float = -1.0, q_max: float = 1.0) -> float:
    return q_min + (q_max - q_min) * q8 / 255.0


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _circumradius(
    x1: float, y1: float,
    x2: float, y2: float,
    x3: float, y3: float,
) -> float:
    """Radius of the circumscribed circle through three points.
    Returns inf if the points are collinear."""
    a = _distance(x1, y1, x2, y2)
    b = _distance(x2, y2, x3, y3)
    c = _distance(x3, y3, x1, y1)
    denom = 2 * abs(
        x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    )
    if denom < 1e-12:
        return float("inf")
    return (a * b * c) / denom


# ═══════════════════════════════════════════════════════════════════
# Per-node descriptor extraction
# ═══════════════════════════════════════════════════════════════════

def _extract_coord_values(node: ASTNode) -> Optional[tuple]:
    """Extract (x, y) from a CRD node's params, returning dequantised floats."""
    if node.node_type != NodeType.CRD:
        return None
    x = _dequantize(node.params.get("x", 0))
    y = _dequantize(node.params.get("y", 0))
    return (x, y)


def _node_scale(node: ASTNode) -> float:
    """
    Compute a local geometric scale for *node*.

    - For curves (LN/ARC/CIR): bounding diameter of control points.
    - For containers (LOOP/FACE/SKT/SOL/PROG): max child scale.
    - For value nodes (CRD/SCL): 0.
    """
    nt = node.node_type

    if nt == NodeType.CRD or nt == NodeType.SCL:
        return 0.0

    if nt == NodeType.LN:
        coords = [_extract_coord_values(c) for c in node.children]
        coords = [c for c in coords if c is not None]
        if len(coords) == 2:
            return _distance(*coords[0], *coords[1])
        return 0.0

    if nt == NodeType.ARC:
        coords = [_extract_coord_values(c) for c in node.children]
        coords = [c for c in coords if c is not None]
        if len(coords) >= 2:
            dists = [
                _distance(*coords[i], *coords[j])
                for i in range(len(coords))
                for j in range(i + 1, len(coords))
            ]
            return max(dists) if dists else 0.0
        return 0.0

    if nt == NodeType.CIR:
        for child in node.children:
            if child.node_type == NodeType.SCL:
                r = _dequantize(child.params.get("value", 0))
                return abs(2.0 * r)
        return 0.0

    # Containers: propagate max child scale
    child_scales = [_node_scale(c) for c in node.children]
    return max(child_scales) if child_scales else 0.0


def _node_curvature(node: ASTNode) -> float:
    """
    Curvature descriptor:
      Line → 0
      Arc  → 1/R (R from circumradius)
      Circle → 1/R
    Non-curve nodes inherit 0.
    """
    nt = node.node_type

    if nt == NodeType.LN:
        return 0.0

    if nt == NodeType.ARC:
        coords = [_extract_coord_values(c) for c in node.children]
        coords = [c for c in coords if c is not None]
        if len(coords) == 3:
            r = _circumradius(*coords[0], *coords[1], *coords[2])
            if r > 1e-12 and r != float("inf"):
                return 1.0 / r
        return 0.0

    if nt == NodeType.CIR:
        for child in node.children:
            if child.node_type == NodeType.SCL:
                r = abs(_dequantize(child.params.get("value", 0)))
                if r > 1e-12:
                    return 1.0 / r
        return 0.0

    return 0.0


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════

def extract_geometry_descriptor(node: ASTNode) -> GeometryDescriptor:
    """
    Compute the ``GeometryDescriptor`` for a single AST node.
    """
    raw_scale = _node_scale(node)
    # Normalise scale to [0,1] using the theoretical max extent
    # (diagonal of the [-1,1]^2 bounding box = 2*sqrt(2) ≈ 2.83).
    norm_scale = min(1.0, raw_scale / (2.0 * math.sqrt(2)))

    curvature = _node_curvature(node)
    depth_ratio = node.depth / MAX_DEPTH if MAX_DEPTH > 0 else 0.0
    subtree_sz = node.subtree_size

    return GeometryDescriptor(
        scale=norm_scale,
        curvature=curvature,
        depth_ratio=depth_ratio,
        subtree_size=subtree_sz,
    )


def extract_geometry_descriptors(root: ASTNode) -> List[GeometryDescriptor]:
    """
    Compute one ``GeometryDescriptor`` per DFS-visited node of *root*.

    The returned list is in DFS pre-order, matching the token order
    produced by ``ASTSerializer.serialize``.
    """
    return [extract_geometry_descriptor(node) for node in root.dfs()]
