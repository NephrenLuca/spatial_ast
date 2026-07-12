"""
Generation entry point.

Loads a trained checkpoint and samples token sequences (optionally text
conditioned), decodes them to ASTs with the constraint decoder, compiles to
DeepCAD commands, and writes everything to a JSON file.

Example::

    python scripts/generate.py \
        --checkpoint outputs/naive_text/checkpoints/best_val_total.ckpt \
        --prompts prompts.json --num_steps 20 --out results/gen.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import ModelConfig
from model.denoiser import SpatialASTDenoiser
from data.dataset import build_text_tokenizer
from inference.sampler import SpatialASTSampler, SamplerConfig
from decoder.pipeline import ConstraintDecoderPipeline
from compiler.emitter import IREmitter
from compiler.backend import DeepCADBackend
from core.grammar import validate_ast

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("generate")


def load_model(ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg_dict = state["config"]["model"]
    cfg = ModelConfig(**model_cfg_dict)
    model = SpatialASTDenoiser(cfg)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate CAD ASTs from a checkpoint")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompts", default=None, help="JSON list of text prompts")
    ap.add_argument("--num_samples", type=int, default=16)
    ap.add_argument("--length", type=int, default=120, help="target seq length")
    ap.add_argument("--num_steps", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--cfg_scale", type=float, default=1.0)
    ap.add_argument("--max_text_len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/generation.json")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg = load_model(args.checkpoint, device)
    log.info("Loaded checkpoint %s (d_model=%d, blocks=%d)",
             args.checkpoint, cfg.d_model, cfg.num_blocks)

    prompts: List[str]
    if args.prompts:
        with open(args.prompts, encoding="utf-8") as f:
            prompts = list(json.load(f))
    else:
        prompts = [""] * args.num_samples

    text_tokens: Optional[torch.Tensor] = None
    uncond_tokens: Optional[torch.Tensor] = None
    if cfg.use_text:
        tok = build_text_tokenizer(
            cfg.use_real_encoder, cfg.text_encoder_name,
            cfg.text_vocab_size, args.max_text_len,
        )
        enc = tok(prompts)
        text_tokens = enc["input_ids"].to(device)
        if args.cfg_scale > 1.0:
            uenc = tok([""] * len(prompts))
            uncond_tokens = uenc["input_ids"].to(device)

    lengths = [args.length] * len(prompts)
    sampler = SpatialASTSampler(model, device)
    seqs = sampler.sample(
        lengths=lengths,
        text_tokens=text_tokens,
        uncond_text_tokens=uncond_tokens,
        config=SamplerConfig(
            num_steps=args.num_steps,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
        ),
    )

    pipeline = ConstraintDecoderPipeline()
    records = []
    n_valid = n_compile = 0
    for prompt, tokens in zip(prompts, seqs):
        rec = {"prompt": prompt, "tokens": tokens, "valid": False, "compiled": False}
        try:
            ast = pipeline.decode_to_ast(list(tokens))
            rec["valid"] = bool(validate_ast(ast).is_valid)
            n_valid += int(rec["valid"])
            try:
                ir = IREmitter().emit(ast)
                cmds = DeepCADBackend().ir_to_commands(ir)
                rec["commands"] = cmds
                rec["compiled"] = True
                n_compile += 1
            except Exception as e:
                rec["compile_error"] = str(e)[:200]
        except Exception as e:
            rec["decode_error"] = str(e)[:200]
        records.append(rec)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    n = len(records)
    log.info("Generated %d samples -> %s", n, out_path)
    log.info("valid_rate=%.3f  compile_rate=%.3f", n_valid / n, n_compile / n)


if __name__ == "__main__":
    main()
