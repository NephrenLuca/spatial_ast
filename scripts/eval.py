"""
Evaluation entry point.

Two modes:

1. ``--generated results/generation.json`` — compute structural metrics over
   a previously generated file (from ``scripts/generate.py``).
2. ``--checkpoint ... --val_path ...`` — compute validation loss on a
   preprocessed val split.

Geometric metrics (Chamfer / MMD / COV / JSD) require a CAD kernel and are
not computed here (see the experiment plan).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import ModelConfig
from model.denoiser import SpatialASTDenoiser
from data.dataset import SpatialASTDataset, Collator, build_text_tokenizer
from diffusion.schedule import DiffusionConfig
from diffusion.loss import LossConfig
from training.evaluator import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval")


def eval_generated(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    seqs = [r["tokens"] for r in records]
    evaluator = Evaluator(DiffusionConfig(), LossConfig(), torch.device("cpu"))
    metrics = evaluator.structural_metrics(seqs)
    log.info("Structural metrics over %d generated samples:", len(seqs))
    for k, v in metrics.items():
        log.info("  %-28s %.4f", k, v)


def eval_val_loss(checkpoint: str, val_path: str, text_annotations, batch_size: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**state["config"]["model"])
    diff_cfg = DiffusionConfig(**state["config"]["diffusion"])
    loss_cfg = LossConfig(**state["config"]["loss"])

    model = SpatialASTDenoiser(cfg)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    tok = build_text_tokenizer(
        cfg.use_real_encoder, cfg.text_encoder_name, cfg.text_vocab_size, 64
    ) if cfg.use_text else None
    ds = SpatialASTDataset(val_path, text_annotations=text_annotations)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=Collator(text_tokenizer=tok, cfg_dropout=0.0),
    )

    evaluator = Evaluator(diff_cfg, loss_cfg, device)
    metrics = evaluator.evaluate_loss(model, loader, max_batches=0)
    log.info("Validation loss over %d samples:", len(ds))
    for k, v in metrics.items():
        log.info("  %-16s %.4f", k, v)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate SpatialAST")
    ap.add_argument("--generated", default=None, help="generation JSON to score")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--val_path", default="data/processed/val.parquet")
    ap.add_argument("--text_annotations", default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    if args.generated:
        eval_generated(args.generated)
    elif args.checkpoint:
        eval_val_loss(args.checkpoint, args.val_path,
                      args.text_annotations, args.batch_size)
    else:
        ap.error("Provide either --generated or --checkpoint")


if __name__ == "__main__":
    main()
