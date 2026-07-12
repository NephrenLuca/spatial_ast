"""
Training callbacks: LR schedule, checkpointing, and optional W&B logging.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# ═══════════════════════════════════════════════════════════════════════
# LR schedule: linear warmup -> cosine decay
# ═══════════════════════════════════════════════════════════════════════

def build_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.05,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


# ═══════════════════════════════════════════════════════════════════════
# Checkpointing
# ═══════════════════════════════════════════════════════════════════════

class CheckpointManager:
    """
    Saves / loads training state and prunes old step-checkpoints.

    Layout under ``ckpt_dir``::
        latest.ckpt              # overwritten every save (fast resume)
        step_000005000.ckpt      # rolling, keeps last k
        best_<metric>.ckpt       # best-so-far per tracked metric
    """

    def __init__(self, ckpt_dir: str | Path, keep_last_k: int = 3) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_k = keep_last_k
        self._best: Dict[str, float] = {}

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: Any,
        scaler: Any,
        config: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        state = {
            "step": step,
            "model": _unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "config": config,
            "best": self._best,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None
                ),
            },
        }
        if extra:
            state.update(extra)

        step_path = self.ckpt_dir / f"step_{step:09d}.ckpt"
        torch.save(state, step_path)
        torch.save(state, self.ckpt_dir / "latest.ckpt")
        self._prune()
        return step_path

    def save_best(
        self,
        metric: str,
        value: float,
        step: int,
        model: torch.nn.Module,
        config: Dict[str, Any],
        higher_is_better: bool = False,
    ) -> bool:
        """Save a best-metric checkpoint. Returns True if this is a new best."""
        prev = self._best.get(metric)
        improved = (
            prev is None
            or (value > prev if higher_is_better else value < prev)
        )
        if improved:
            self._best[metric] = value
            torch.save(
                {
                    "step": step,
                    "model": _unwrap(model).state_dict(),
                    "config": config,
                    "metric": metric,
                    "value": value,
                },
                self.ckpt_dir / f"best_{metric}.ckpt",
            )
        return improved

    def _prune(self) -> None:
        step_ckpts = sorted(
            self.ckpt_dir.glob("step_*.ckpt"),
            key=lambda p: int(re.findall(r"\d+", p.stem)[0]),
        )
        for p in step_ckpts[: -self.keep_last_k] if self.keep_last_k > 0 else []:
            try:
                p.unlink()
            except OSError:
                pass

    def resolve(self, resume: str) -> Optional[Path]:
        """Resolve a resume spec ('latest' or a path) to a checkpoint path."""
        if resume == "latest":
            p = self.ckpt_dir / "latest.ckpt"
            return p if p.exists() else None
        p = Path(resume)
        return p if p.exists() else None

    @staticmethod
    def load(
        path: str | Path,
        model: torch.nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Any = None,
        scaler: Any = None,
        map_location: Any = "cpu",
    ) -> Dict[str, Any]:
        state = torch.load(path, map_location=map_location, weights_only=False)
        _unwrap(model).load_state_dict(state["model"])
        if optimizer is not None and state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        return state


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


# ═══════════════════════════════════════════════════════════════════════
# W&B logger (optional)
# ═══════════════════════════════════════════════════════════════════════

class WandbLogger:
    """Thin optional wrapper around Weights & Biases; a no-op when disabled."""

    def __init__(
        self,
        enabled: bool,
        project: str,
        group: str,
        run_name: str,
        config: Dict[str, Any],
        is_main: bool,
    ) -> None:
        self.active = enabled and is_main
        self._run = None
        if self.active:
            import wandb

            self._wandb = wandb
            self._run = wandb.init(
                project=project, group=group, name=run_name, config=config
            )

    def log(self, metrics: Dict[str, float], step: int) -> None:
        if self.active and self._run is not None:
            self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.active and self._run is not None:
            self._wandb.finish()
