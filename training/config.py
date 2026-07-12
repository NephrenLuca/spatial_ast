"""
Experiment configuration loading.

A single YAML file describes an experiment with four sections:
``model``, ``diffusion``, ``loss`` and ``train``.  The YAML is loaded with
OmegaConf (so command-line dotlist overrides work) and materialised into
the concrete dataclasses used throughout the codebase.

Example::

    cfg = load_experiment("configs/naive_text.yaml",
                          overrides=["train.per_gpu_batch_size=8"])
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

from model.config import ModelConfig
from diffusion.schedule import DiffusionConfig
from diffusion.loss import LossConfig


@dataclass
class TrainConfig:
    """Optimisation, data, distributed and I/O settings for a training run."""

    # ── run identity ──────────────────────────────────────────────
    run_name: str = "naive_text"
    seed: int = 42
    output_dir: str = "outputs"

    # ── data ──────────────────────────────────────────────────────
    train_path: str = "data/processed/train.parquet"
    val_path: str = "data/processed/val.parquet"
    text_annotations: Optional[str] = None       # JSON {file_id: prompt}
    max_train_samples: int = 0                    # 0 = all
    max_val_samples: int = 0
    num_workers: int = 8
    max_text_len: int = 64
    cfg_dropout: float = 0.1                      # classifier-free-guidance dropout

    # ── optimisation ──────────────────────────────────────────────
    lr: float = 1e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    warmup_steps: int = 2000
    max_steps: int = 200_000
    per_gpu_batch_size: int = 16
    accumulation_steps: int = 2
    precision: str = "bf16"                       # bf16 | fp16 | fp32

    # ── checkpoint / logging / eval ───────────────────────────────
    log_every: int = 100
    eval_every: int = 5000
    save_every: int = 5000
    keep_last_k: int = 3
    resume: Optional[str] = None                  # path to checkpoint or "latest"

    # ── weights & biases ──────────────────────────────────────────
    wandb: bool = False
    wandb_project: str = "spatial-ast"
    wandb_group: str = "main"

    # ── evaluation sampling (structural metrics on val) ───────────
    eval_num_samples: int = 64
    eval_sample_steps: int = 20


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _filter_known(dc_cls, data: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only keys that are valid fields of *dc_cls* (raise on unknown)."""
    valid = {f.name for f in fields(dc_cls)}
    unknown = set(data) - valid
    if unknown:
        raise KeyError(
            f"Unknown config keys for {dc_cls.__name__}: {sorted(unknown)}"
        )
    return {k: v for k, v in data.items() if k in valid}


def load_experiment(
    path: str,
    overrides: Optional[List[str]] = None,
) -> ExperimentConfig:
    """
    Load an experiment YAML and apply dotlist *overrides*.

    Returns a fully materialised :class:`ExperimentConfig`.
    """
    base = OmegaConf.structured(ExperimentConfig())
    file_cfg = OmegaConf.load(path)
    merged = OmegaConf.merge(base, file_cfg)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))

    raw = OmegaConf.to_container(merged, resolve=True)

    model = ModelConfig(**_filter_known(ModelConfig, raw.get("model", {})))
    diffusion = DiffusionConfig(
        **_filter_known(DiffusionConfig, raw.get("diffusion", {}))
    )
    loss = LossConfig(**_filter_known(LossConfig, raw.get("loss", {})))
    train = TrainConfig(**_filter_known(TrainConfig, raw.get("train", {})))

    return ExperimentConfig(model=model, diffusion=diffusion, loss=loss, train=train)


def experiment_to_dict(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Flatten an ExperimentConfig into a plain dict (for logging / saving)."""
    return {
        "model": asdict(cfg.model),
        "diffusion": asdict(cfg.diffusion),
        "loss": asdict(cfg.loss),
        "train": asdict(cfg.train),
    }
