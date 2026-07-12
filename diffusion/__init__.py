"""
diffusion package — hierarchical corruption engine for SpatialAST.
"""

from diffusion.schedule import DiffusionConfig, corruption_probability
from diffusion.corruption import (
    CorruptionMode,
    SubtreeCorruptor,
    hierarchical_corrupt,
    batch_corrupt,
)
from diffusion.loss import LossConfig, compute_loss

__all__ = [
    "DiffusionConfig",
    "corruption_probability",
    "CorruptionMode",
    "SubtreeCorruptor",
    "hierarchical_corrupt",
    "batch_corrupt",
    "LossConfig",
    "compute_loss",
]
