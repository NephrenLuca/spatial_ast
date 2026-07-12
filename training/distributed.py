"""
Distributed training helpers (DDP over ``torchrun``).

All functions are safe to call in a single-process (non-distributed) run:
``setup_distributed`` detects the ``torchrun`` environment variables and
falls back to a single-device configuration when they are absent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    distributed: bool = False
    device: torch.device = torch.device("cpu")

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed() -> DistInfo:
    """
    Initialise the process group if launched under ``torchrun``.

    Returns a :class:`DistInfo` describing the current process.  When the
    distributed env vars are absent, returns a single-process config using
    ``cuda:0`` if available, else CPU.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        return DistInfo(rank, local_rank, world_size, True, device)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return DistInfo(0, 0, 1, False, device)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    """Average a tensor across all ranks (no-op if not distributed)."""
    if dist.is_available() and dist.is_initialized():
        value = value.clone()
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= dist.get_world_size()
    return value
