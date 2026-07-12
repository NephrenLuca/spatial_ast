"""
Evaluation: validation loss + structural generation metrics.

* :meth:`Evaluator.evaluate_loss` reuses the training corruption + composite
  loss on the validation set (cheap, position-aligned).
* :meth:`Evaluator.evaluate_generation` samples sequences with the iterative
  sampler, runs constraint decoding, and reports structural quality
  (validity, bracket match, compile success, uniqueness).

Heavy geometric metrics (Chamfer / MMD / COV / JSD) require a CAD kernel and
are intentionally deferred — see the experiment plan / architecture risk R8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from core.grammar import validate_ast
from decoder.bracket_balancer import BracketBalancer
from decoder.pipeline import ConstraintDecoderPipeline
from compiler.emitter import IREmitter
from compiler.backend import DeepCADBackend
from diffusion.schedule import DiffusionConfig
from diffusion.corruption import batch_corrupt
from diffusion.loss import LossConfig, compute_loss
from inference.sampler import SpatialASTSampler, SamplerConfig


@dataclass
class EvalResult:
    metrics: Dict[str, float]

    def __getitem__(self, k: str) -> float:
        return self.metrics[k]


class Evaluator:
    def __init__(
        self,
        diffusion_cfg: DiffusionConfig,
        loss_cfg: LossConfig,
        device: torch.device,
    ) -> None:
        self.diffusion_cfg = diffusion_cfg
        self.loss_cfg = loss_cfg
        self.device = device
        self.bracket = BracketBalancer()
        self.pipeline = ConstraintDecoderPipeline()

    # ── validation loss ──────────────────────────────────────────────
    @torch.no_grad()
    def evaluate_loss(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        max_batches: int = 0,
    ) -> Dict[str, float]:
        model.eval()
        totals: Dict[str, float] = {}
        count = 0

        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            batch = _to_device(batch, self.device)
            B = batch["token_ids"].shape[0]
            t = torch.randint(
                1, self.diffusion_cfg.T + 1, (B,), device=self.device
            )
            corrupted, corruption_mask = batch_corrupt(
                batch["token_ids"], batch["depth_ids"], batch["role_ids"],
                t, self.diffusion_cfg,
            )
            logits = model(
                token_ids=corrupted,
                depth_ids=batch["depth_ids"],
                type_ids=batch["type_ids"],
                role_ids=batch["role_ids"],
                parent_ids=batch["parent_ids"],
                sibling_ids=batch["sibling_ids"],
                geom_desc=batch["geom_desc"],
                t=t,
                text_tokens=batch.get("text_tokens"),
                mask=batch["attention_mask"],
            )
            losses = compute_loss(
                logits, batch["token_ids"], corruption_mask,
                batch["depth_ids"], batch["role_ids"], self.loss_cfg,
            )
            for k, v in losses.items():
                totals[k] = totals.get(k, 0.0) + float(v.item())
            count += 1

        return {f"val/{k}": v / max(1, count) for k, v in totals.items()}

    # ── generation / structural metrics ─────────────────────────────
    @torch.no_grad()
    def evaluate_generation(
        self,
        model: torch.nn.Module,
        prompts: List[Tensor],           # unused placeholder kept for API symmetry
        text_tokens: Optional[Tensor],
        lengths: List[int],
        num_steps: int = 20,
    ) -> Dict[str, float]:
        sampler = SpatialASTSampler(model, self.device)
        seqs = sampler.sample(
            lengths=lengths,
            text_tokens=text_tokens,
            config=SamplerConfig(num_steps=num_steps),
        )
        return self.structural_metrics(seqs)

    def structural_metrics(self, sequences: List[List[int]]) -> Dict[str, float]:
        n = len(sequences)
        if n == 0:
            return {}

        n_valid = 0
        n_bracket = 0
        n_compile = 0
        unique = set()

        for tokens in sequences:
            unique.add(tuple(tokens))
            balanced = self.bracket.check_and_repair(list(tokens))
            if balanced == list(tokens):
                n_bracket += 1
            try:
                ast = self.pipeline.decode_to_ast(list(tokens))
            except Exception:
                continue
            vr = validate_ast(ast)
            if vr.is_valid:
                n_valid += 1
            try:
                ir = IREmitter().emit(ast)
                DeepCADBackend().ir_to_commands(ir)
                n_compile += 1
            except Exception:
                pass

        return {
            "gen/valid_rate": n_valid / n,
            "gen/bracket_match_rate": n_bracket / n,
            "gen/compile_rate": n_compile / n,
            "gen/unique_rate": len(unique) / n,
        }


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out
