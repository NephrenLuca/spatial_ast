"""
Depth-aware noise schedule for hierarchical corruption diffusion.

Implements the corruption probability function ``p(d, t)`` from
architecture.md Section 5.2: deeper AST nodes are corrupted earlier
(lower t), shallower nodes only at high noise levels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List


@dataclass
class DiffusionConfig:
    """Configuration for the hierarchical corruption engine."""
    T: int = 1000
    max_depth: int = 5
    tau_scale: float = 0.9
    beta_scale: float = 0.1
    corruption_modes: List[str] = field(
        default_factory=lambda: ["mask", "noise", "shuffle", "resample"]
    )


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def corruption_probability(
    depth: int,
    t: int,
    T: int,
    config: DiffusionConfig,
) -> float:
    """
    Probability that a node at *depth* is corrupted at timestep *t*.

    Properties:
      - t -> 0: p -> 0 for all depths
      - t -> T: shallow nodes (d=0) p -> 1
      - Fixed t: p increases with depth (deeper = corrupted earlier)

    The threshold tau(d) determines when each depth level starts
    experiencing significant corruption.
    """
    tau_d = (1.0 - depth / config.max_depth) * config.tau_scale * T
    beta = config.beta_scale * T
    if beta < 1e-12:
        return 1.0 if t > tau_d else 0.0
    return _sigmoid((t - tau_d) / beta)
