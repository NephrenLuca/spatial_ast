"""
Grammar-constrained decoding mask.

At each autoregressive step, ``GrammarMask`` tracks a parse stack and
emits the set of token IDs that are structurally legal given the current
AST context (parent type, child count, expected token role).

Follows architecture.md Section 7.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set

import torch
from torch import Tensor

from core.types import NodeType, NodeRegistry, NodeSpec, DEPTH_OF
from core.tokenizer import (
    TOKEN_BOS,
    TOKEN_EOS,
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_PAD,
    Q8_OFFSET,
    ENUM_TOKEN_MAP,
    VOCAB_SIZE,
    get_node_type_token,
    get_node_type_from_token,
)


@dataclass
class ParseState:
    """Snapshot of the parser's expectation at a given decode position."""
    parent_type: Optional[NodeType]
    expecting: str  # "node_type" | "param" | "open_paren" | "close_paren_or_child"
    child_count: int = 0
    min_children: int = 0
    max_children: int = 0
    param_queue: List[str] = field(default_factory=list)
    param_idx: int = 0
    spec: Optional[NodeSpec] = None


class GrammarMask:
    """
    Stateful grammar mask that tracks a stack-based parse context.

    Call ``feed(token_id)`` for each decoded token, then
    ``get_valid_tokens()`` to get the set of legal next tokens.
    """

    def __init__(self) -> None:
        self.stack: List[ParseState] = []
        self._started = False
        self._finished = False

    def reset(self) -> None:
        self.stack.clear()
        self._started = False
        self._finished = False

    def get_valid_tokens(self) -> Set[int]:
        """Return the set of valid token IDs for the next position."""
        if self._finished:
            return {TOKEN_PAD}

        if not self._started:
            return {TOKEN_BOS}

        if not self.stack:
            return {TOKEN_EOS}

        state = self.stack[-1]

        if state.expecting == "node_type":
            return self._valid_node_tags(state.parent_type)

        if state.expecting == "param":
            return self._valid_param_tokens(state)

        if state.expecting == "open_paren":
            return {TOKEN_LPAREN}

        if state.expecting == "close_paren_or_child":
            child_tags = self._valid_node_tags(state.parent_type)
            result = set(child_tags)
            if state.child_count >= state.min_children:
                result.add(TOKEN_RPAREN)
            if state.child_count >= state.max_children:
                result = {TOKEN_RPAREN}
            return result

        return set(range(VOCAB_SIZE))

    def feed(self, token_id: int) -> None:
        """Update parse state after emitting *token_id*."""
        if self._finished:
            return

        if not self._started:
            if token_id == TOKEN_BOS:
                self._started = True
                self.stack.append(ParseState(
                    parent_type=None,
                    expecting="node_type",
                ))
            return

        if token_id == TOKEN_EOS:
            self._finished = True
            return

        if not self.stack:
            return

        state = self.stack[-1]

        if state.expecting == "node_type":
            nt = get_node_type_from_token(token_id)
            if nt is None:
                return
            spec = NodeRegistry.get(nt.name)
            param_names = [
                pname for pname, pdef in spec.param_schema.items()
                if pdef.dtype != "str"
            ]

            total_min = sum(
                slot.min_count for slot in spec.child_schema
            )
            total_max = sum(
                slot.max_count for slot in spec.child_schema
            )

            if state.parent_type is not None:
                state.child_count += 1

            if param_names:
                state.expecting = "param"
                self.stack.append(ParseState(
                    parent_type=nt,
                    expecting="param",
                    param_queue=param_names,
                    param_idx=0,
                    spec=spec,
                    min_children=total_min,
                    max_children=total_max,
                ))
            elif spec.child_schema:
                self.stack.append(ParseState(
                    parent_type=nt,
                    expecting="open_paren",
                    spec=spec,
                    min_children=total_min,
                    max_children=total_max,
                ))
            else:
                pass
            return

        if state.expecting == "param":
            state.param_idx += 1
            if state.param_idx >= len(state.param_queue):
                if state.spec and state.spec.child_schema:
                    state.expecting = "open_paren"
                else:
                    self.stack.pop()
                    if self.stack:
                        self.stack[-1].expecting = "close_paren_or_child"
            return

        if state.expecting == "open_paren":
            if token_id == TOKEN_LPAREN:
                state.expecting = "close_paren_or_child"
                state.child_count = 0
            return

        if state.expecting == "close_paren_or_child":
            if token_id == TOKEN_RPAREN:
                self.stack.pop()
                if self.stack:
                    self.stack[-1].expecting = "close_paren_or_child"
                return

            nt = get_node_type_from_token(token_id)
            if nt is not None:
                spec = NodeRegistry.get(nt.name)
                param_names = [
                    pname for pname, pdef in spec.param_schema.items()
                    if pdef.dtype != "str"
                ]
                total_min = sum(
                    slot.min_count for slot in spec.child_schema
                )
                total_max = sum(
                    slot.max_count for slot in spec.child_schema
                )

                state.child_count += 1

                if param_names:
                    self.stack.append(ParseState(
                        parent_type=nt,
                        expecting="param",
                        param_queue=param_names,
                        param_idx=0,
                        spec=spec,
                        min_children=total_min,
                        max_children=total_max,
                    ))
                elif spec.child_schema:
                    self.stack.append(ParseState(
                        parent_type=nt,
                        expecting="open_paren",
                        spec=spec,
                        min_children=total_min,
                        max_children=total_max,
                    ))

    def apply_mask(self, logits: Tensor) -> Tensor:
        """
        Mask *logits* ``[V]`` so that invalid tokens get ``-inf``.
        Returns a new tensor (does not modify in-place).
        """
        valid = self.get_valid_tokens()
        if not valid:
            return logits
        mask = torch.full_like(logits, float("-inf"))
        for t in valid:
            if 0 <= t < logits.shape[-1]:
                mask[t] = 0.0
        return logits + mask

    @staticmethod
    def _valid_node_tags(parent_type: Optional[NodeType]) -> Set[int]:
        """Token IDs of node types that are valid children of *parent_type*."""
        if parent_type is None:
            return {get_node_type_token("PROG")}
        child_tags = NodeRegistry.get_children_types(parent_type.name)
        return {get_node_type_token(tag) for tag in child_tags}

    @staticmethod
    def _valid_param_tokens(state: ParseState) -> Set[int]:
        """Token IDs valid for the current parameter being decoded."""
        if state.spec is None or state.param_idx >= len(state.param_queue):
            return set(range(VOCAB_SIZE))
        pname = state.param_queue[state.param_idx]
        pdef = state.spec.param_schema[pname]
        if pdef.dtype == "q8":
            return set(range(Q8_OFFSET, Q8_OFFSET + 256))
        if pdef.dtype == "enum" and pdef.enum_values:
            return {ENUM_TOKEN_MAP[v] for v in pdef.enum_values if v in ENUM_TOKEN_MAP}
        return set(range(VOCAB_SIZE))
