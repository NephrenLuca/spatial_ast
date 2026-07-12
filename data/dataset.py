"""
SpatialASTDataset and dynamic-padding collator.

Loads the Parquet files produced by ``scripts/preprocess.py`` (see the
schema in ``results_to_arrow``) and serves per-sample, natural-length
tensors.  Padding to a uniform length is deferred to :class:`Collator`,
which pads each batch to its own longest sequence (dynamic padding).

Text conditioning
------------------
DeepCAD itself carries no natural-language prompts, so text annotations
are supplied separately via a JSON mapping ``{file_id: prompt}`` (e.g. the
Text2CAD subset).  When a ``file_id`` has no annotation the prompt falls
back to the empty string.  The collator turns prompts into token IDs for
the condition encoder using either a HuggingFace tokenizer (real encoder)
or a lightweight deterministic hash tokenizer (stub encoder).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset

from core.tokenizer import TOKEN_PAD

GEOM_DIM = 4


# ═══════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════

class SpatialASTDataset(Dataset):
    """
    Variable-length dataset backed by a preprocessed Parquet file.

    Each item is a dict of natural-length tensors plus the raw text prompt
    (a string) and ``file_id``.  Dynamic padding happens in :class:`Collator`.

    Parameters
    ----------
    data_path : str | Path
        Path to a ``*.parquet`` file written by ``scripts/preprocess.py``.
    text_annotations : str | Path | dict, optional
        JSON file (or already-loaded dict) mapping ``file_id -> prompt``.
        Missing IDs fall back to ``""``.
    max_samples : int, optional
        Truncate the dataset (useful for debugging / smoke tests).
    """

    def __init__(
        self,
        data_path: str | Path,
        text_annotations: Optional[str | Path | Dict[str, str]] = None,
        max_samples: int = 0,
    ) -> None:
        import pyarrow.parquet as pq  # lazy: only needed to load data

        self.data_path = str(data_path)
        table = pq.read_table(self.data_path)
        cols = table.to_pydict()

        self.file_ids: List[str] = list(cols["file_id"])
        self.tokens: List[List[int]] = [list(x) for x in cols["tokens"]]
        self.depths: List[List[int]] = [list(x) for x in cols["depths"]]
        self.types: List[List[int]] = [list(x) for x in cols["types"]]
        self.roles: List[List[int]] = [list(x) for x in cols["roles"]]
        self.parents: List[List[int]] = [list(x) for x in cols["parents"]]
        self.siblings: List[List[int]] = [list(x) for x in cols["siblings"]]
        self._geom_flat: List[List[float]] = [list(x) for x in cols["geom_desc"]]

        if max_samples and max_samples > 0:
            self._truncate(max_samples)

        self.text_map: Dict[str, str] = self._load_text_map(text_annotations)

    # ── helpers ─────────────────────────────────────────────────────
    def _truncate(self, n: int) -> None:
        self.file_ids = self.file_ids[:n]
        self.tokens = self.tokens[:n]
        self.depths = self.depths[:n]
        self.types = self.types[:n]
        self.roles = self.roles[:n]
        self.parents = self.parents[:n]
        self.siblings = self.siblings[:n]
        self._geom_flat = self._geom_flat[:n]

    @staticmethod
    def _load_text_map(
        source: Optional[str | Path | Dict[str, str]],
    ) -> Dict[str, str]:
        if source is None:
            return {}
        if isinstance(source, dict):
            return {str(k): str(v) for k, v in source.items()}
        with open(source, encoding="utf-8") as f:
            raw = json.load(f)
        out: Dict[str, str] = {}
        for k, v in raw.items():
            if isinstance(v, list):  # allow list of prompts -> pick first
                out[str(k)] = str(v[0]) if v else ""
            elif isinstance(v, dict):  # allow {"text": ...}
                out[str(k)] = str(v.get("text", ""))
            else:
                out[str(k)] = str(v)
        return out

    # ── Dataset protocol ────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tokens = self.tokens[idx]
        L = len(tokens)
        geom_flat = self._geom_flat[idx]
        geom = [
            geom_flat[i * GEOM_DIM:(i + 1) * GEOM_DIM] for i in range(L)
        ]

        return {
            "token_ids": torch.tensor(tokens, dtype=torch.long),
            "depth_ids": torch.tensor(self.depths[idx], dtype=torch.long),
            "type_ids": torch.tensor(self.types[idx], dtype=torch.long),
            "role_ids": torch.tensor(self.roles[idx], dtype=torch.long),
            "parent_ids": torch.tensor(self.parents[idx], dtype=torch.long),
            "sibling_ids": torch.tensor(self.siblings[idx], dtype=torch.long),
            "geom_desc": torch.tensor(geom, dtype=torch.float32),
            "seq_len": L,
            "text": self.text_map.get(self.file_ids[idx], ""),
            "file_id": self.file_ids[idx],
        }


# ═══════════════════════════════════════════════════════════════════════
# Text tokenisation
# ═══════════════════════════════════════════════════════════════════════

class HashTextTokenizer:
    """
    Deterministic fallback tokenizer for the *stub* text encoder.

    Maps whitespace-delimited words to IDs in ``[1, vocab_size)`` via a
    stable hash.  Not semantically meaningful — use a real HF tokenizer
    (``use_real_encoder=True``) for actual text conditioning.  Kept only so
    the training loop is runnable without downloading pretrained weights.
    """

    def __init__(self, vocab_size: int = 32128, max_len: int = 64) -> None:
        self.vocab_size = vocab_size
        self.max_len = max_len

    def __call__(self, texts: List[str]) -> Dict[str, Tensor]:
        ids: List[List[int]] = []
        for t in texts:
            words = t.split()[: self.max_len]
            row = [1 + (hash(w) % (self.vocab_size - 1)) for w in words]
            if not row:
                row = [0]  # empty prompt -> single PAD-like id
            ids.append(row)
        L = max(len(r) for r in ids)
        input_ids = torch.zeros(len(ids), L, dtype=torch.long)
        attn = torch.zeros(len(ids), L, dtype=torch.long)
        for i, r in enumerate(ids):
            input_ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
            attn[i, : len(r)] = 1
        return {"input_ids": input_ids, "attention_mask": attn}


class HFTextTokenizer:
    """Thin wrapper over a HuggingFace tokenizer (e.g. Flan-T5)."""

    def __init__(self, model_name: str, max_len: int = 64) -> None:
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len

    def __call__(self, texts: List[str]) -> Dict[str, Tensor]:
        out = self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": out["input_ids"],
            "attention_mask": out["attention_mask"],
        }


def build_text_tokenizer(
    use_real_encoder: bool,
    model_name: str,
    vocab_size: int,
    max_len: int,
) -> Any:
    if use_real_encoder:
        return HFTextTokenizer(model_name, max_len=max_len)
    return HashTextTokenizer(vocab_size=vocab_size, max_len=max_len)


# ═══════════════════════════════════════════════════════════════════════
# Collator
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Collator:
    """
    Dynamic-padding collate function.

    Pads all per-token arrays to the batch's longest sequence and builds a
    boolean ``attention_mask`` (True = real token).  Optionally tokenises
    text prompts, with classifier-free-guidance dropout (replace prompt
    with empty string with probability ``cfg_dropout``).

    Attributes
    ----------
    pad_id : int
        Padding token id for ``token_ids``.
    text_tokenizer : callable, optional
        ``list[str] -> {"input_ids", "attention_mask"}``.
    cfg_dropout : float
        Probability of dropping a prompt (for classifier-free guidance).
    """

    pad_id: int = TOKEN_PAD
    text_tokenizer: Optional[Any] = None
    cfg_dropout: float = 0.0

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        B = len(batch)
        max_len = max(b["seq_len"] for b in batch)

        token_ids = torch.full((B, max_len), self.pad_id, dtype=torch.long)
        depth_ids = torch.zeros(B, max_len, dtype=torch.long)
        type_ids = torch.zeros(B, max_len, dtype=torch.long)
        role_ids = torch.zeros(B, max_len, dtype=torch.long)
        parent_ids = torch.zeros(B, max_len, dtype=torch.long)
        sibling_ids = torch.zeros(B, max_len, dtype=torch.long)
        geom_desc = torch.zeros(B, max_len, GEOM_DIM, dtype=torch.float32)
        attention_mask = torch.zeros(B, max_len, dtype=torch.bool)

        for i, b in enumerate(batch):
            n = b["seq_len"]
            token_ids[i, :n] = b["token_ids"]
            depth_ids[i, :n] = b["depth_ids"]
            type_ids[i, :n] = b["type_ids"]
            role_ids[i, :n] = b["role_ids"]
            parent_ids[i, :n] = b["parent_ids"]
            sibling_ids[i, :n] = b["sibling_ids"]
            geom_desc[i, :n] = b["geom_desc"]
            attention_mask[i, :n] = True

        out: Dict[str, Any] = {
            "token_ids": token_ids,
            "depth_ids": depth_ids,
            "type_ids": type_ids,
            "role_ids": role_ids,
            "parent_ids": parent_ids,
            "sibling_ids": sibling_ids,
            "geom_desc": geom_desc,
            "attention_mask": attention_mask,
            "file_ids": [b["file_id"] for b in batch],
        }

        texts = [b["text"] for b in batch]
        if self.cfg_dropout > 0.0:
            texts = [
                "" if random.random() < self.cfg_dropout else t for t in texts
            ]
        out["texts"] = texts

        if self.text_tokenizer is not None:
            tok = self.text_tokenizer(texts)
            out["text_tokens"] = tok["input_ids"]
            out["text_attention_mask"] = tok["attention_mask"]

        return out
