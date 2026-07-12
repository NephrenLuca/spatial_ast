"""
Trainer: the hierarchical-corruption diffusion training loop.

Supports single-GPU and multi-GPU (DDP via ``torchrun``), bf16/fp16/fp32
precision, gradient accumulation, cosine LR schedule with warmup,
checkpointing / resume, periodic evaluation, and optional W&B logging.
"""

from __future__ import annotations

import itertools
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from diffusion.corruption import batch_corrupt
from diffusion.loss import compute_loss
from training.config import ExperimentConfig, experiment_to_dict
from training.distributed import DistInfo, all_reduce_mean, barrier
from training.callbacks import build_scheduler, CheckpointManager, WandbLogger
from training.evaluator import Evaluator

log = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        cfg: ExperimentConfig,
        dist_info: DistInfo,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        raw_model: Optional[torch.nn.Module] = None,
    ) -> None:
        self.model = model
        self.raw_model = raw_model if raw_model is not None else model
        self.cfg = cfg
        self.tc = cfg.train
        self.dist = dist_info
        self.device = dist_info.device
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.tc.lr,
            weight_decay=self.tc.weight_decay,
            betas=(self.tc.beta1, self.tc.beta2),
        )
        self.scheduler = build_scheduler(
            self.optimizer, self.tc.warmup_steps, self.tc.max_steps
        )

        self.precision = self.tc.precision
        self.use_fp16 = self.precision == "fp16"
        self.amp_dtype = {
            "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32
        }[self.precision]
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_fp16)

        self.out_dir = Path(self.tc.output_dir) / self.tc.run_name
        self.ckpt_mgr = CheckpointManager(
            self.out_dir / "checkpoints", keep_last_k=self.tc.keep_last_k
        )
        self.evaluator = Evaluator(cfg.diffusion, cfg.loss, self.device)
        self.logger = WandbLogger(
            enabled=self.tc.wandb,
            project=self.tc.wandb_project,
            group=self.tc.wandb_group,
            run_name=self.tc.run_name,
            config=experiment_to_dict(cfg),
            is_main=self.dist.is_main,
        )

        self.step = 0
        self._maybe_resume()

    # ── resume ───────────────────────────────────────────────────────
    def _maybe_resume(self) -> None:
        if not self.tc.resume:
            return
        path = self.ckpt_mgr.resolve(self.tc.resume)
        if path is None:
            log.warning("Resume requested but no checkpoint found: %s", self.tc.resume)
            return
        state = CheckpointManager.load(
            path, self.raw_model, self.optimizer, self.scheduler,
            self.scaler if self.use_fp16 else None, map_location=self.device,
        )
        self.step = int(state.get("step", 0))
        if "best" in state:
            self.ckpt_mgr._best = state["best"]
        log.info("Resumed from %s at step %d", path, self.step)

    # ── one optimisation step (with accumulation) ────────────────────
    def _train_micro_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        batch = self._to_device(batch)
        B = batch["token_ids"].shape[0]
        t = torch.randint(1, self.cfg.diffusion.T + 1, (B,), device=self.device)

        corrupted, corruption_mask = batch_corrupt(
            batch["token_ids"], batch["depth_ids"], batch["role_ids"],
            t, self.cfg.diffusion,
        )

        autocast = torch.autocast(
            device_type="cuda" if self.device.type == "cuda" else "cpu",
            dtype=self.amp_dtype,
            enabled=self.precision != "fp32",
        )
        with autocast:
            logits = self.model(
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
                batch["depth_ids"], batch["role_ids"], self.cfg.loss,
            )
        loss = losses["total"] / self.tc.accumulation_steps
        self.scaler.scale(loss).backward()
        return {k: float(v.item()) for k, v in losses.items()}

    def train(self) -> None:
        self.model.train()
        data_iter = _infinite(self.train_loader)
        accum = self.tc.accumulation_steps
        t0 = time.time()
        running: Dict[str, float] = {}

        while self.step < self.tc.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            micro_losses: Dict[str, float] = {}

            for micro in range(accum):
                batch = next(data_iter)
                sync = (micro == accum - 1)
                ctx = _maybe_no_sync(self.model, sync)
                with ctx:
                    ls = self._train_micro_step(batch)
                for k, v in ls.items():
                    micro_losses[k] = micro_losses.get(k, 0.0) + v / accum

            if self.tc.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.tc.grad_clip
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.step += 1

            for k, v in micro_losses.items():
                running[k] = running.get(k, 0.0) + v

            if self.step % self.tc.log_every == 0:
                self._log_train(running, t0)
                running = {}
                t0 = time.time()

            if self.val_loader is not None and self.step % self.tc.eval_every == 0:
                self._run_eval()
                self.model.train()

            if self.step % self.tc.save_every == 0 and self.dist.is_main:
                self.ckpt_mgr.save(
                    self.step, self.raw_model, self.optimizer, self.scheduler,
                    self.scaler if self.use_fp16 else None,
                    experiment_to_dict(self.cfg),
                )

        if self.dist.is_main:
            self.ckpt_mgr.save(
                self.step, self.raw_model, self.optimizer, self.scheduler,
                self.scaler if self.use_fp16 else None,
                experiment_to_dict(self.cfg),
            )
        self.logger.finish()

    # ── logging / eval ───────────────────────────────────────────────
    def _log_train(self, running: Dict[str, float], t0: float) -> None:
        n = self.tc.log_every
        lr = self.scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        sps = n / elapsed if elapsed > 0 else 0.0
        metrics = {f"train/{k}": v / n for k, v in running.items()}
        metrics["train/lr"] = lr
        metrics["train/steps_per_sec"] = sps
        if self.dist.is_main:
            log.info(
                "step %d | loss %.4f | ce %.4f | reg %.4f | lr %.2e | %.2f it/s",
                self.step, metrics.get("train/total", 0.0),
                metrics.get("train/ce", 0.0), metrics.get("train/reg", 0.0),
                lr, sps,
            )
            self.logger.log(metrics, self.step)

    def _run_eval(self) -> None:
        val_metrics = self.evaluator.evaluate_loss(
            self.model, self.val_loader, max_batches=50
        )
        # reduce across ranks
        reduced = {}
        for k, v in val_metrics.items():
            t = torch.tensor(v, device=self.device)
            reduced[k] = float(all_reduce_mean(t).item())

        if self.dist.is_main:
            log.info("[eval] step %d | %s", self.step,
                     " ".join(f"{k}={v:.4f}" for k, v in reduced.items()))
            self.logger.log(reduced, self.step)
            self.ckpt_mgr.save_best(
                "val_total", reduced.get("val/total", float("inf")),
                self.step, self.raw_model, experiment_to_dict(self.cfg),
            )
        barrier()

    # ── utils ────────────────────────────────────────────────────────
    def _to_device(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(self.device, non_blocking=True) if isinstance(v, Tensor) else v
        return out


def _infinite(loader: DataLoader):
    epoch = 0
    while True:
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _maybe_no_sync(model: torch.nn.Module, sync: bool):
    """Return model.no_sync() during accumulation micro-steps (DDP only)."""
    if not sync and hasattr(model, "no_sync"):
        return model.no_sync()
    return _NullCtx()
