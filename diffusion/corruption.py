"""
Subtree-level corruption for hierarchical diffusion training.

Operates on token sequences (List[int]) produced by ``ASTSerializer``,
applying depth-aware stochastic corruption at each timestep.

Follows architecture.md Section 5.3–5.4.
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor

from core.tokenizer import (
    TOKEN_MASK,
    TOKEN_NOISE,
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_PAD,
    TOKEN_BOS,
    TOKEN_EOS,
    Q8_OFFSET,
    TokenRole,
    is_q8_token,
    is_node_tag_token,
)
from diffusion.schedule import DiffusionConfig, corruption_probability


class CorruptionMode(Enum):
    MASK = "mask"
    NOISE = "noise"
    SHUFFLE = "shuffle"
    RESAMPLE = "resample"


def _find_subtree_spans(
    tokens: List[int],
    depth_map: List[int],
    role_map: List[int],
) -> List[Tuple[int, int, int]]:
    """
    Identify subtree spans from a token sequence.

    Returns a list of ``(start, end, depth)`` tuples where ``[start, end)``
    is the span of each node-tag and its associated params/children.
    Spans are in DFS order (outermost first).
    """
    spans: List[Tuple[int, int, int]] = []
    stack: List[int] = []
    i = 0
    n = len(tokens)

    while i < n:
        tok = tokens[i]
        if tok in (TOKEN_PAD, TOKEN_BOS, TOKEN_EOS):
            i += 1
            continue

        if is_node_tag_token(tok):
            start = i
            depth = depth_map[i]
            i += 1
            while i < n and role_map[i] == TokenRole.PARAM_VALUE:
                i += 1
            if i < n and tokens[i] == TOKEN_LPAREN:
                bracket_depth = 1
                i += 1
                while i < n and bracket_depth > 0:
                    if tokens[i] == TOKEN_LPAREN:
                        bracket_depth += 1
                    elif tokens[i] == TOKEN_RPAREN:
                        bracket_depth -= 1
                    i += 1
            spans.append((start, i, depth))
        else:
            i += 1

    return spans


def _get_children_spans_inside(
    tokens: List[int],
    role_map: List[int],
    start: int,
    end: int,
) -> List[Tuple[int, int]]:
    """
    Find direct-child node spans inside a parent span ``[start, end)``.
    Used by SHUFFLE corruption mode.
    """
    i = start
    i += 1
    while i < end and role_map[i] == TokenRole.PARAM_VALUE:
        i += 1
    if i >= end or tokens[i] != TOKEN_LPAREN:
        return []
    i += 1

    children: List[Tuple[int, int]] = []
    while i < end - 1:
        if not is_node_tag_token(tokens[i]):
            i += 1
            continue
        child_start = i
        i += 1
        while i < end - 1 and role_map[i] == TokenRole.PARAM_VALUE:
            i += 1
        if i < end - 1 and tokens[i] == TOKEN_LPAREN:
            bracket_depth = 1
            i += 1
            while i < end - 1 and bracket_depth > 0:
                if tokens[i] == TOKEN_LPAREN:
                    bracket_depth += 1
                elif tokens[i] == TOKEN_RPAREN:
                    bracket_depth -= 1
                i += 1
        children.append((child_start, i))
    return children


class SubtreeCorruptor:
    """Applies a single corruption mode to a node span within a token list."""

    def corrupt_span(
        self,
        tokens: List[int],
        role_map: List[int],
        start: int,
        end: int,
        mode: CorruptionMode,
    ) -> List[int]:
        """
        Corrupt ``tokens[start:end]`` in-place according to *mode*.
        Returns the (potentially resized) token list.
        """
        if mode == CorruptionMode.MASK:
            tokens[start] = TOKEN_MASK
        elif mode == CorruptionMode.NOISE:
            tokens[start:end] = [TOKEN_NOISE]
            return tokens
        elif mode == CorruptionMode.SHUFFLE:
            children = _get_children_spans_inside(tokens, role_map, start, end)
            if len(children) > 1:
                child_contents = [tokens[s:e] for s, e in children]
                random.shuffle(child_contents)
                flat: List[int] = []
                for cc in child_contents:
                    flat.extend(cc)
                first_child_start = children[0][0]
                last_child_end = children[-1][1]
                tokens[first_child_start:last_child_end] = flat
        elif mode == CorruptionMode.RESAMPLE:
            for idx in range(start, min(end, len(tokens))):
                if is_q8_token(tokens[idx]):
                    tokens[idx] = random.randint(Q8_OFFSET, Q8_OFFSET + 255)
        return tokens


def hierarchical_corrupt(
    tokens: List[int],
    depth_map: List[int],
    role_map: List[int],
    t: int,
    config: DiffusionConfig,
    rng: random.Random | None = None,
) -> Tuple[List[int], List[bool]]:
    """
    Apply hierarchical corruption to a token sequence.

    Parameters
    ----------
    tokens : list[int]
        Clean token sequence (will be copied, not modified in-place).
    depth_map : list[int]
        Per-token AST depth.
    role_map : list[int]
        Per-token role (TokenRole int values).
    t : int
        Current diffusion timestep (0 = clean, T = max noise).
    config : DiffusionConfig
        Diffusion hyperparameters.
    rng : random.Random, optional
        RNG for reproducibility.

    Returns
    -------
    corrupted : list[int]
        Corrupted token sequence (may differ in length from input
        due to NOISE mode collapsing subtrees).
    mask : list[bool]
        Per-token corruption mask (True = corrupted). Length matches
        ``corrupted``.
    """
    if rng is None:
        rng = random.Random()

    corrupted = list(tokens)
    mask = [False] * len(corrupted)
    role_copy = list(role_map)

    spans = _find_subtree_spans(corrupted, depth_map, role_copy)
    corrupted_positions: Set[int] = set()
    corruptor = SubtreeCorruptor()

    for start, end, depth in spans:
        if start in corrupted_positions:
            continue

        p = corruption_probability(depth, t, config.T, config)
        if rng.random() >= p:
            continue

        available_modes = list(config.corruption_modes)
        mode_str = rng.choice(available_modes)
        mode = CorruptionMode(mode_str)

        old_len = len(corrupted)
        corrupted = corruptor.corrupt_span(corrupted, role_copy, start, end, mode)
        new_len = len(corrupted)

        delta = new_len - old_len
        if mode == CorruptionMode.NOISE:
            mask = mask[:start] + [True] + mask[end:]
            role_copy = role_copy[:start] + [TokenRole.SPECIAL] + role_copy[end:]
            for pos in range(start, start + 1):
                corrupted_positions.add(pos)
        else:
            actual_end = end + delta
            for pos in range(start, min(actual_end, len(corrupted))):
                mask[pos] = True
                corrupted_positions.add(pos)

    return corrupted, mask


def batch_corrupt(
    token_batch: Tensor,
    depth_batch: Tensor,
    role_batch: Tensor,
    t_batch: Tensor,
    config: DiffusionConfig,
    pad_id: int = 0,
) -> Tuple[Tensor, Tensor]:
    """
    Vectorised wrapper: corrupt a batch of padded token sequences.

    Parameters
    ----------
    token_batch : [B, L]  int tensor
    depth_batch : [B, L]  int tensor
    role_batch  : [B, L]  int tensor
    t_batch     : [B]     int tensor (one timestep per sample)
    config      : DiffusionConfig
    pad_id      : padding token id

    Returns
    -------
    corrupted_batch : [B, L] int tensor (re-padded to max length)
    mask_batch      : [B, L] bool tensor
    """
    B, L = token_batch.shape
    device = token_batch.device
    results_tokens: List[List[int]] = []
    results_masks: List[List[bool]] = []

    for b in range(B):
        seq_len = (token_batch[b] != pad_id).sum().item()
        tokens = token_batch[b, :seq_len].tolist()
        depths = depth_batch[b, :seq_len].tolist()
        roles = role_batch[b, :seq_len].tolist()
        t_val = t_batch[b].item()

        corrupted, mask = hierarchical_corrupt(tokens, depths, roles, t_val, config)
        results_tokens.append(corrupted)
        results_masks.append(mask)

    max_len = max(len(s) for s in results_tokens)
    max_len = max(max_len, L)

    out_tokens = torch.full((B, max_len), pad_id, dtype=token_batch.dtype, device=device)
    out_mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)

    for b in range(B):
        n = len(results_tokens[b])
        out_tokens[b, :n] = torch.tensor(results_tokens[b], dtype=token_batch.dtype)
        out_mask[b, :n] = torch.tensor(results_masks[b], dtype=torch.bool)

    return out_tokens, out_mask
