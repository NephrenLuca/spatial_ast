"""
Iterative masked-diffusion sampler (MaskGIT-style).

Generation starts from an all-``[MASK]`` body and progressively unmasks the
most-confident positions over ``num_steps`` iterations.  Because the denoiser
consumes per-token structural metadata (depth / type / role / parent /
sibling), that metadata is recomputed from the current partial sequence at
every step via :func:`recompute_metadata` (a bracket-nesting scan).

Note
----
Geometry descriptors cannot be recovered for masked positions, so they are
zeroed during sampling — this is a known simplification of the naive design
(the model is trained with ground-truth geometry).  Free-form generation
quality is therefore expected to trail the reconstruction/denoising metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from core.types import NodeType, DEPTH_OF
from core.tokenizer import (
    TOKEN_BOS,
    TOKEN_EOS,
    TOKEN_MASK,
    TOKEN_PAD,
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TokenRole,
    get_node_type_from_token,
    is_q8_token,
    is_enum_token,
    is_node_tag_token,
)

GEOM_DIM = 4


# ═══════════════════════════════════════════════════════════════════════
# Metadata recomputation from a (partial) token sequence
# ═══════════════════════════════════════════════════════════════════════

def recompute_metadata(tokens: List[int]) -> dict:
    """
    Best-effort per-token metadata mirroring ``ASTSerializer`` semantics.

    Depth is the *semantic* AST depth (``DEPTH_OF`` — note Loop and Edge share
    depth 4), attributed to the owning node rather than to bracket nesting.
    Returns dict with lists (same length as *tokens*):
    ``depth, type, role, parent, sibling``.  Masked / unknown node tags get
    type ``NIL`` and an approximate depth; geometry is handled by the caller.
    """
    n = len(tokens)
    max_depth = 5
    nil = NodeType.NIL.value
    prog = NodeType.PROG.value

    depth = [0] * n
    types = [nil] * n
    roles = [int(TokenRole.SPECIAL)] * n
    parents = [nil] * n
    siblings = [0] * n

    # Each open scope: [owner_type, child_count, owner_depth, owner_sibling]
    stack: List[List[int]] = []
    last_node_type = nil
    last_node_depth = 0
    last_node_sibling = 0

    for i, tok in enumerate(tokens):
        parent_type = stack[-1][0] if stack else nil

        if tok in (TOKEN_BOS, TOKEN_EOS):
            roles[i] = int(TokenRole.SPECIAL)
            types[i] = prog
            depth[i] = 0
            parents[i] = nil
            continue
        if tok == TOKEN_PAD:
            roles[i] = int(TokenRole.SPECIAL)
            types[i] = nil
            depth[i] = 0
            parents[i] = nil
            continue

        if tok == TOKEN_LPAREN:
            roles[i] = int(TokenRole.OPEN_PAREN)
            types[i] = last_node_type
            depth[i] = last_node_depth
            parents[i] = parent_type
            siblings[i] = last_node_sibling
            stack.append([last_node_type, 0, last_node_depth, last_node_sibling])
            continue

        if tok == TOKEN_RPAREN:
            if stack:
                owner_type, _, owner_depth, owner_sibling = stack.pop()
            else:
                owner_type, owner_depth, owner_sibling = nil, 0, 0
            roles[i] = int(TokenRole.CLOSE_PAREN)
            types[i] = owner_type
            depth[i] = owner_depth
            parents[i] = stack[-1][0] if stack else nil
            siblings[i] = owner_sibling
            continue

        # A node tag (or a masked slot standing in for one)
        if is_node_tag_token(tok) or tok == TOKEN_MASK:
            nt = get_node_type_from_token(tok)
            if nt is not None:
                nt_val = nt.value
                d = DEPTH_OF.get(nt, 0)
            else:  # masked: approximate depth from the enclosing scope
                nt_val = nil
                parent_depth = stack[-1][2] if stack else -1
                d = min(parent_depth + 1, max_depth)
            sib = stack[-1][1] if stack else 0
            if stack:
                stack[-1][1] += 1
            sib = min(sib, 63)
            roles[i] = int(TokenRole.NODE_TAG)
            types[i] = nt_val
            depth[i] = d
            parents[i] = parent_type
            siblings[i] = sib
            last_node_type = nt_val
            last_node_depth = d
            last_node_sibling = sib
            continue

        # Params (Q8 / enum) belong to the most recent node tag on this level.
        if is_q8_token(tok) or is_enum_token(tok):
            roles[i] = int(TokenRole.PARAM_VALUE)
            types[i] = last_node_type
            depth[i] = last_node_depth
            parents[i] = parent_type
            siblings[i] = last_node_sibling
            continue

        # Fallback (e.g. NOISE / stray special tokens mid-sequence)
        roles[i] = int(TokenRole.SPECIAL)
        depth[i] = last_node_depth
        parents[i] = parent_type

    return {
        "depth": depth,
        "type": types,
        "role": roles,
        "parent": parents,
        "sibling": siblings,
    }


# ═══════════════════════════════════════════════════════════════════════
# Sampler
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SamplerConfig:
    num_steps: int = 20
    temperature: float = 1.0
    cfg_scale: float = 1.0          # >1 enables classifier-free guidance
    forward_t_start: int = 1000     # timestep fed on the first (fully masked) step


class SpatialASTSampler:
    """Iterative confidence-based unmasking sampler for the denoiser."""

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device

    @torch.no_grad()
    def sample(
        self,
        lengths: List[int],
        text_tokens: Optional[Tensor] = None,
        uncond_text_tokens: Optional[Tensor] = None,
        config: Optional[SamplerConfig] = None,
    ) -> List[List[int]]:
        """
        Generate a batch of token sequences.

        Parameters
        ----------
        lengths : list[int]
            Target sequence length per sample (including BOS/EOS).
        text_tokens : [B, L_text], optional
            Conditioning text token IDs.
        uncond_text_tokens : [B, L_text], optional
            Unconditional (empty-prompt) tokens for classifier-free guidance.
        config : SamplerConfig

        Returns
        -------
        list of token-id lists (unpadded, per requested length).
        """
        cfg = config or SamplerConfig()
        self.model.eval()
        B = len(lengths)
        max_len = max(lengths)

        x = torch.full((B, max_len), TOKEN_PAD, dtype=torch.long, device=self.device)
        active = torch.zeros(B, max_len, dtype=torch.bool, device=self.device)
        for b, L in enumerate(lengths):
            x[b, 0] = TOKEN_BOS
            x[b, 1:L - 1] = TOKEN_MASK
            x[b, L - 1] = TOKEN_EOS
            active[b, :L] = True

        body_mask = active & (x == TOKEN_MASK)  # positions we still need to fill

        for step in range(cfg.num_steps):
            meta = self._batch_metadata(x, active)
            t_val = int(cfg.forward_t_start * (1.0 - step / max(1, cfg.num_steps)))
            t = torch.full((B,), max(1, t_val), dtype=torch.long, device=self.device)

            logits = self._forward(x, meta, t, active, text_tokens,
                                    uncond_text_tokens, cfg.cfg_scale)
            logits = logits / max(1e-6, cfg.temperature)
            probs = F.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)  # [B, L]

            still_masked = active & (x == TOKEN_MASK)
            num_masked = still_masked.sum(dim=1)  # [B]

            # cosine unmask schedule: how many should REMAIN masked after this step
            ratio = math.cos(0.5 * math.pi * (step + 1) / cfg.num_steps)
            for b in range(B):
                nm = int(num_masked[b].item())
                if nm == 0:
                    continue
                keep_masked = int(math.floor(ratio * nm)) if step + 1 < cfg.num_steps else 0
                n_unmask = max(1, nm - keep_masked)

                pos = still_masked[b].nonzero(as_tuple=True)[0]
                c = conf[b, pos]
                top = torch.topk(c, min(n_unmask, pos.numel())).indices
                chosen = pos[top]
                x[b, chosen] = pred[b, chosen]

            if not (active & (x == TOKEN_MASK)).any():
                break

        # any leftover masks -> EOS-safe fallback to PAD-free tokens
        leftover = active & (x == TOKEN_MASK)
        x[leftover] = TOKEN_RPAREN

        out: List[List[int]] = []
        for b, L in enumerate(lengths):
            out.append(x[b, :L].tolist())
        return out

    # ── internals ───────────────────────────────────────────────────
    def _batch_metadata(self, x: Tensor, active: Tensor) -> dict:
        B, L = x.shape
        depth = torch.zeros(B, L, dtype=torch.long, device=self.device)
        types = torch.zeros(B, L, dtype=torch.long, device=self.device)
        roles = torch.zeros(B, L, dtype=torch.long, device=self.device)
        parents = torch.zeros(B, L, dtype=torch.long, device=self.device)
        siblings = torch.zeros(B, L, dtype=torch.long, device=self.device)
        geom = torch.zeros(B, L, GEOM_DIM, dtype=torch.float32, device=self.device)

        for b in range(B):
            n = int(active[b].sum().item())
            md = recompute_metadata(x[b, :n].tolist())
            depth[b, :n] = torch.tensor(md["depth"], device=self.device)
            types[b, :n] = torch.tensor(md["type"], device=self.device)
            roles[b, :n] = torch.tensor(md["role"], device=self.device)
            parents[b, :n] = torch.tensor(md["parent"], device=self.device)
            siblings[b, :n] = torch.tensor(md["sibling"], device=self.device)

        return {
            "depth_ids": depth,
            "type_ids": types,
            "role_ids": roles,
            "parent_ids": parents,
            "sibling_ids": siblings,
            "geom_desc": geom,
        }

    def _forward(
        self,
        x: Tensor,
        meta: dict,
        t: Tensor,
        active: Tensor,
        text_tokens: Optional[Tensor],
        uncond_text_tokens: Optional[Tensor],
        cfg_scale: float,
    ) -> Tensor:
        def run(tt: Optional[Tensor]) -> Tensor:
            return self.model(
                token_ids=x,
                depth_ids=meta["depth_ids"],
                type_ids=meta["type_ids"],
                role_ids=meta["role_ids"],
                parent_ids=meta["parent_ids"],
                sibling_ids=meta["sibling_ids"],
                geom_desc=meta["geom_desc"],
                t=t,
                text_tokens=tt,
                mask=active,
            )

        cond = run(text_tokens)
        if cfg_scale > 1.0 and uncond_text_tokens is not None:
            uncond = run(uncond_text_tokens)
            return uncond + cfg_scale * (cond - uncond)
        return cond
