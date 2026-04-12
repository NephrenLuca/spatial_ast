"""
Meta-annotator: AST → per-token metadata arrays.

Takes an AST tree and its serialised token sequence, then produces
parallel arrays of depth, node-type, role, parent-type, sibling-index,
and geometry descriptors.  These are the auxiliary inputs consumed by
``SpatialASTEmbedding``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any

from core.types import NodeType
from core.ast_node import ASTNode
from core.tokenizer import TokenMeta, TokenRole, TOKEN_PAD
from core.serializer import ASTSerializer
from core.geometry import (
    GeometryDescriptor,
    extract_geometry_descriptor,
    extract_geometry_descriptors,
)


@dataclass
class AnnotatedSample:
    """Complete annotated sample ready for the dataset / dataloader."""
    tokens:     List[int]
    depths:     List[int]
    types:      List[int]
    roles:      List[int]
    parents:    List[int]
    siblings:   List[int]
    geom_desc:  List[List[float]]   # [L, 4]
    seq_len:    int                  # unpadded length (excl. PAD)


class MetaAnnotator:
    """
    Produces a fully annotated sample from an AST tree.

    Combines ``ASTSerializer.serialize`` (which already generates
    ``TokenMeta``) with geometry descriptor extraction to build
    every auxiliary array the model needs.
    """

    def __init__(self, max_seq_len: int = 512) -> None:
        self._serializer = ASTSerializer(max_seq_len=max_seq_len)
        self._max_seq_len = max_seq_len

    def annotate(self, root: ASTNode) -> AnnotatedSample:
        tokens, metas = self._serializer.serialize(root, pad=True)

        # Build node-level geometry descriptors indexed by DFS order
        node_geom = {id(n): extract_geometry_descriptor(n) for n in root.dfs()}

        # Default descriptor for special / padding tokens
        default_geom = GeometryDescriptor(0.0, 0.0, 0.0, 0)

        depths:    List[int] = []
        types:     List[int] = []
        roles:     List[int] = []
        parents:   List[int] = []
        siblings:  List[int] = []
        geom_desc: List[List[float]] = []

        # Map node_type to the closest real node for geometry lookup
        dfs_nodes = list(root.dfs())
        node_type_to_node = {}
        for n in dfs_nodes:
            if n.node_type not in node_type_to_node:
                node_type_to_node[n.node_type] = n

        for meta in metas:
            depths.append(meta.depth)
            types.append(meta.node_type.value)
            roles.append(meta.role.value)
            parents.append(meta.parent_type.value)
            siblings.append(min(meta.sibling_idx, 63))  # cap at max_siblings

            # Geometry: use the token's node-type for lookup
            if meta.role == TokenRole.SPECIAL:
                geom_desc.append(default_geom.to_list())
            else:
                gd = self._find_geom(meta, dfs_nodes, default_geom)
                geom_desc.append(gd.to_list())

        seq_len = sum(1 for t in tokens if t != TOKEN_PAD)

        return AnnotatedSample(
            tokens=tokens,
            depths=depths,
            types=types,
            roles=roles,
            parents=parents,
            siblings=siblings,
            geom_desc=geom_desc,
            seq_len=seq_len,
        )

    @staticmethod
    def _find_geom(
        meta: TokenMeta,
        dfs_nodes: List[ASTNode],
        default: GeometryDescriptor,
    ) -> GeometryDescriptor:
        """Best-effort geometry descriptor for a token based on its metadata."""
        for node in dfs_nodes:
            if (node.node_type == meta.node_type
                    and node.depth == meta.depth):
                return extract_geometry_descriptor(node)
        return default
