"""
Training entry point.

Single GPU::

    python scripts/train.py --config configs/naive_text.yaml

Multi-GPU (1-2x A100 on one node) via torchrun::

    torchrun --nproc_per_node=2 scripts/train.py \
        --config configs/naive_text.yaml \
        train.per_gpu_batch_size=16 train.accumulation_steps=4

Any ``section.key=value`` argument after ``--config`` overrides the YAML.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import ModelConfig
from model.denoiser import SpatialASTDenoiser
from data.dataset import SpatialASTDataset, Collator, build_text_tokenizer
from training.config import load_experiment
from training.distributed import setup_distributed, cleanup_distributed
from training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")


def seed_everything(seed: int, rank: int) -> None:
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_loader(dataset, collator, batch_size, num_workers, dist_info, shuffle):
    sampler = None
    if dist_info.distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=dist_info.world_size,
            rank=dist_info.rank, shuffle=shuffle, drop_last=True,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SpatialAST (naive text)")
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    args, overrides = parser.parse_known_args()

    cfg = load_experiment(args.config, overrides=overrides)
    dist_info = setup_distributed()
    seed_everything(cfg.train.seed, dist_info.rank)

    if dist_info.is_main:
        log.info("Experiment: %s | world_size=%d | device=%s",
                 cfg.train.run_name, dist_info.world_size, dist_info.device)

    # ── model ────────────────────────────────────────────────────────
    model = SpatialASTDenoiser(cfg.model).to(dist_info.device)
    raw_model = model
    if dist_info.distributed:
        model = DDP(
            model,
            device_ids=[dist_info.local_rank] if dist_info.device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    if dist_info.is_main:
        log.info("Trainable parameters: %.1fM", n_params / 1e6)

    # ── data ─────────────────────────────────────────────────────────
    text_tok = build_text_tokenizer(
        use_real_encoder=cfg.model.use_real_encoder,
        model_name=cfg.model.text_encoder_name,
        vocab_size=cfg.model.text_vocab_size,
        max_len=cfg.train.max_text_len,
    ) if cfg.model.use_text else None

    train_collator = Collator(
        text_tokenizer=text_tok, cfg_dropout=cfg.train.cfg_dropout
    )
    val_collator = Collator(text_tokenizer=text_tok, cfg_dropout=0.0)

    train_ds = SpatialASTDataset(
        cfg.train.train_path,
        text_annotations=cfg.train.text_annotations,
        max_samples=cfg.train.max_train_samples,
    )
    train_loader = build_loader(
        train_ds, train_collator, cfg.train.per_gpu_batch_size,
        cfg.train.num_workers, dist_info, shuffle=True,
    )

    val_loader = None
    if os.path.exists(cfg.train.val_path):
        val_ds = SpatialASTDataset(
            cfg.train.val_path,
            text_annotations=cfg.train.text_annotations,
            max_samples=cfg.train.max_val_samples,
        )
        val_loader = build_loader(
            val_ds, val_collator, cfg.train.per_gpu_batch_size,
            cfg.train.num_workers, dist_info, shuffle=False,
        )

    if dist_info.is_main:
        log.info("Train samples: %d | Val samples: %d",
                 len(train_ds), len(val_ds) if val_loader else 0)

    # ── train ────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model, cfg=cfg, dist_info=dist_info,
        train_loader=train_loader, val_loader=val_loader, raw_model=raw_model,
    )
    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
