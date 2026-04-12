"""
Dataset distribution statistics and summary utilities.

Computes aggregate statistics over a collection of AST trees or
serialised token sequences: sequence length distribution, node-type
frequencies, depth histograms, etc.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.types import NodeType, SEMANTIC_NODE_TYPES
from core.ast_node import ASTNode


@dataclass
class DatasetStatistics:
    """Aggregate statistics over a set of AST samples."""
    num_samples: int = 0
    seq_lengths: List[int] = field(default_factory=list)
    node_type_counts: Dict[str, int] = field(default_factory=lambda: Counter())
    depth_counts: Dict[int, int] = field(default_factory=lambda: Counter())
    num_solids_per_prog: List[int] = field(default_factory=list)
    num_edges_per_loop: List[int] = field(default_factory=list)
    num_ops_per_solid: List[int] = field(default_factory=list)

    # ── accumulation ────────────────────────────────────────────────

    def add_ast(self, root: ASTNode, seq_len: Optional[int] = None) -> None:
        """Accumulate statistics from a single AST tree."""
        self.num_samples += 1
        if seq_len is not None:
            self.seq_lengths.append(seq_len)

        for node in root.dfs():
            self.node_type_counts[node.node_type.name] += 1
            self.depth_counts[node.depth] += 1

        if root.node_type == NodeType.PROG:
            self.num_solids_per_prog.append(len(root.children))

        for node in root.dfs():
            if node.node_type == NodeType.LOOP:
                edge_count = sum(
                    1 for c in node.children if c.node_type == NodeType.EDGE
                )
                self.num_edges_per_loop.append(edge_count)

            if node.node_type == NodeType.SOL:
                op_count = sum(
                    1 for c in node.children
                    if c.node_type in (NodeType.EXT, NodeType.REV)
                )
                self.num_ops_per_solid.append(op_count)

    # ── summaries ───────────────────────────────────────────────────

    @property
    def mean_seq_length(self) -> float:
        if not self.seq_lengths:
            return 0.0
        return sum(self.seq_lengths) / len(self.seq_lengths)

    @property
    def max_seq_length(self) -> int:
        return max(self.seq_lengths) if self.seq_lengths else 0

    @property
    def min_seq_length(self) -> int:
        return min(self.seq_lengths) if self.seq_lengths else 0

    def percentile_seq_length(self, pct: float) -> int:
        """Return the *pct*-th percentile of sequence lengths."""
        if not self.seq_lengths:
            return 0
        sorted_lens = sorted(self.seq_lengths)
        idx = int(len(sorted_lens) * pct / 100.0)
        idx = min(idx, len(sorted_lens) - 1)
        return sorted_lens[idx]

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            f"Dataset Statistics ({self.num_samples} samples)",
            f"  Seq length: mean={self.mean_seq_length:.1f}, "
            f"min={self.min_seq_length}, max={self.max_seq_length}",
        ]
        if self.seq_lengths:
            lines.append(
                f"  P50={self.percentile_seq_length(50)}, "
                f"P95={self.percentile_seq_length(95)}, "
                f"P99={self.percentile_seq_length(99)}"
            )
        lines.append("  Node type distribution:")
        for nt in SEMANTIC_NODE_TYPES:
            count = self.node_type_counts.get(nt.name, 0)
            lines.append(f"    {nt.name:5s}: {count}")
        return "\n".join(lines)
