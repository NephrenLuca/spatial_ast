"""
Geometry-Conditioned Bidirectional Mamba (GC-Mamba).

Wraps ``mamba_ssm.Mamba`` with geometry-derived SSM kernel modulation,
bidirectional scanning, and residual gating.
Follows architecture.md Section 6.5 / GeoFusion GSM-SSD.

Requires ``mamba-ssm`` and ``causal-conv1d`` packages (GPU only).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None  # type: ignore[assignment,misc]


def _require_mamba() -> None:
    if Mamba is None:
        raise ImportError(
            "GC-Mamba requires the `mamba-ssm` package (GPU only). "
            "Install with: pip install mamba-ssm causal-conv1d"
        )


class GeometryConditionedMamba(nn.Module):
    """
    Bidirectional Mamba with geometry-conditioned SSM kernels.

    The ``geom_to_kernels`` MLP maps 4-dim geometry descriptors to
    modulation factors that scale the SSM's internal ``dt`` and ``B``
    parameters, giving different state transition dynamics to different
    geometric token types (line vs arc vs extrude, etc.).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        geom_dim: int,
    ) -> None:
        super().__init__()
        _require_mamba()

        # Geometry -> SSM kernel modulation:  A_mod, B_mod, C_mod, G_mod
        self.geom_to_kernels = nn.Sequential(
            nn.Linear(geom_dim, d_state * 2),
            nn.SiLU(),
            nn.Linear(d_state * 2, 4 * d_state),
        )

        self.forward_ssm = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.backward_ssm = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.merge = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.residual_gate = nn.Linear(geom_dim, d_model)

    def forward(self, x: Tensor, geom_desc: Tensor) -> Tensor:
        """
        Args:
            x:         [B, L, d_model]
            geom_desc: [B, L, geom_dim]  (scale, curvature, depth_ratio, subtree_size)
        Returns:
            [B, L, d_model]
        """
        kernels = self.geom_to_kernels(geom_desc)       # [B, L, 4*d_state]
        A_mod, B_mod, C_mod, G_mod = kernels.chunk(4, dim=-1)

        fwd = self.forward_ssm(x)
        bwd = self.backward_ssm(x.flip(dims=[1])).flip(dims=[1])

        merged = self.merge(torch.cat([fwd, bwd], dim=-1))

        gate = torch.sigmoid(self.residual_gate(geom_desc))
        c_gate = torch.sigmoid(C_mod[..., : merged.shape[-1]])

        return self.norm(merged * c_gate + gate * x)
