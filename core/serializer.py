"""
AST ↔ Token sequence serialiser.

Implements the DFS parenthesised format described in §4 of the architecture spec:
  serialize(node) := NODE_TAG  PARAMS  "("  children…  ")"
  serialize(leaf) := NODE_TAG  PARAMS            (no brackets)

Also produces per-token ``TokenMeta`` used as auxiliary model input.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from core.types import DEPTH_OF, NodeRegistry, NodeSpec, NodeType
from core.ast_node import ASTNode, _next_node_id
from core.tokenizer import (
    TOKEN_BOS,
    TOKEN_EOS,
    TOKEN_LPAREN,
    TOKEN_NIL,
    TOKEN_PAD,
    TOKEN_RPAREN,
    TokenMeta,
    TokenRole,
    decode_param,
    encode_param,
    get_node_type_from_token,
    get_node_type_token,
    is_q8_token,
    is_enum_token,
)

MAX_SEQ_LEN = None  # No hard limit; padding is handled by the DataLoader


class ASTSerializer:
    """
    Bidirectional serialiser: ``ASTNode ↔ List[int]``.

    ``serialize``  : ASTNode → token list (with BOS/EOS and optional padding)
    ``deserialize``: token list → ASTNode

    Both methods also track token-level metadata (depth, type, role,
    parent type, sibling index) needed by the embedding layer.

    When ``max_seq_len`` is None (default), sequences are stored at their
    natural length — padding to uniform length is deferred to the
    DataLoader via dynamic (bucket) padding.
    """

    def __init__(self, max_seq_len: int | None = MAX_SEQ_LEN) -> None:
        self.max_seq_len = max_seq_len

    # ── Serialize ───────────────────────────────────────────────────

    def serialize(
        self,
        root: ASTNode,
        *,
        pad: bool = True,
    ) -> Tuple[List[int], List[TokenMeta]]:
        """
        Convert *root* into a padded token sequence and parallel metadata.

        Returns
        -------
        tokens : list[int]
            Token IDs including BOS, EOS, and optional PAD.
        metas : list[TokenMeta]
            One ``TokenMeta`` per token (same length as *tokens*).
        """
        tokens: List[int] = []
        metas: List[TokenMeta] = []

        # BOS
        tokens.append(TOKEN_BOS)
        metas.append(TokenMeta(
            position=0,
            depth=0,
            node_type=NodeType.PROG,
            role=TokenRole.SPECIAL,
            parent_type=NodeType.NIL,
            sibling_idx=0,
        ))

        self._dfs_serialize(root, tokens, metas, parent_type=NodeType.NIL, sibling_idx=0)

        # EOS
        tokens.append(TOKEN_EOS)
        metas.append(TokenMeta(
            position=len(tokens) - 1,
            depth=0,
            node_type=NodeType.PROG,
            role=TokenRole.SPECIAL,
            parent_type=NodeType.NIL,
            sibling_idx=0,
        ))

        # Update span on root
        root = replace(root, span=(1, len(tokens) - 1))

        if pad:
            tokens, metas = self._pad(tokens, metas)

        # Fix up position indices
        for i, m in enumerate(metas):
            m.position = i

        return tokens, metas

    def _dfs_serialize(
        self,
        node: ASTNode,
        tokens: List[int],
        metas: List[TokenMeta],
        parent_type: NodeType,
        sibling_idx: int,
    ) -> None:
        start_pos = len(tokens)
        nt = node.node_type
        depth = node.depth

        # NODE_TAG token
        tag_token = get_node_type_token(nt.name)
        tokens.append(tag_token)
        metas.append(TokenMeta(
            position=len(tokens) - 1,
            depth=depth,
            node_type=nt,
            role=TokenRole.NODE_TAG,
            parent_type=parent_type,
            sibling_idx=sibling_idx,
        ))

        # PARAMS – encode in a deterministic order from the spec
        spec = self._get_spec(nt)
        for pname, pdef in spec.param_schema.items():
            if pdef.dtype == "str":
                continue  # strings are not serialised into tokens
            pval = node.params.get(pname, pdef.default)
            if pval is None:
                pval = 0 if pdef.dtype == "q8" else (
                    pdef.enum_values[0] if pdef.enum_values else 0
                )
            tok = encode_param(pval, pdef.dtype)
            tokens.append(tok)
            metas.append(TokenMeta(
                position=len(tokens) - 1,
                depth=depth,
                node_type=nt,
                role=TokenRole.PARAM_VALUE,
                parent_type=parent_type,
                sibling_idx=sibling_idx,
            ))

        # CHILDREN (with brackets)
        if node.children:
            tokens.append(TOKEN_LPAREN)
            metas.append(TokenMeta(
                position=len(tokens) - 1,
                depth=depth,
                node_type=nt,
                role=TokenRole.OPEN_PAREN,
                parent_type=parent_type,
                sibling_idx=sibling_idx,
            ))

            for i, child in enumerate(node.children):
                self._dfs_serialize(child, tokens, metas,
                                    parent_type=nt, sibling_idx=i)

            tokens.append(TOKEN_RPAREN)
            metas.append(TokenMeta(
                position=len(tokens) - 1,
                depth=depth,
                node_type=nt,
                role=TokenRole.CLOSE_PAREN,
                parent_type=parent_type,
                sibling_idx=sibling_idx,
            ))

    # ── Deserialize ─────────────────────────────────────────────────

    def deserialize(self, tokens: List[int]) -> ASTNode:
        """
        Reconstruct an AST from a token sequence.

        Strips BOS/EOS/PAD first, then performs recursive descent.
        """
        stripped = self._strip_special(tokens)
        if not stripped:
            raise ValueError("Empty token sequence after stripping specials")
        root, pos = self._dfs_deserialize(stripped, 0)
        return root

    def _dfs_deserialize(
        self, tokens: List[int], pos: int,
    ) -> Tuple[ASTNode, int]:
        if pos >= len(tokens):
            raise ValueError(f"Unexpected end of tokens at position {pos}")

        # NODE_TAG
        nt = get_node_type_from_token(tokens[pos])
        if nt is None:
            raise ValueError(
                f"Expected node-tag token at pos {pos}, got {tokens[pos]}"
            )
        pos += 1
        spec = self._get_spec(nt)

        # PARAMS
        params: Dict[str, Any] = {}
        for pname, pdef in spec.param_schema.items():
            if pdef.dtype == "str":
                params[pname] = pdef.default or ""
                continue
            if pos >= len(tokens):
                raise ValueError(
                    f"Expected param token for {nt.name}.{pname} at pos {pos}"
                )
            params[pname] = decode_param(tokens[pos], pdef.dtype)
            pos += 1

        # CHILDREN
        children: List[ASTNode] = []
        if pos < len(tokens) and tokens[pos] == TOKEN_LPAREN:
            pos += 1  # skip '('
            while pos < len(tokens) and tokens[pos] != TOKEN_RPAREN:
                child, pos = self._dfs_deserialize(tokens, pos)
                children.append(child)
            if pos < len(tokens):
                pos += 1  # skip ')'

        depth = DEPTH_OF.get(nt, 0)
        node = ASTNode(
            node_type=nt,
            depth=depth,
            children=tuple(children),
            params=params,
            node_id=_next_node_id(),
            span=(0, 0),
        )
        return node, pos

    # ── Helpers ─────────────────────────────────────────────────────

    def _pad(
        self,
        tokens: List[int],
        metas: List[TokenMeta],
    ) -> Tuple[List[int], List[TokenMeta]]:
        """Pad to ``self.max_seq_len`` if set, otherwise return as-is."""
        if self.max_seq_len is None:
            return tokens, metas
        if len(tokens) > self.max_seq_len:
            tokens = tokens[: self.max_seq_len - 1] + [TOKEN_EOS]
            metas = metas[: self.max_seq_len]
        while len(tokens) < self.max_seq_len:
            tokens.append(TOKEN_PAD)
            metas.append(TokenMeta(
                position=len(tokens) - 1,
                depth=0,
                node_type=NodeType.NIL,
                role=TokenRole.SPECIAL,
                parent_type=NodeType.NIL,
                sibling_idx=0,
            ))
        return tokens, metas

    @staticmethod
    def _strip_special(tokens: List[int]) -> List[int]:
        """Remove BOS, EOS, PAD from a token sequence."""
        return [
            t for t in tokens
            if t not in (TOKEN_BOS, TOKEN_EOS, TOKEN_PAD)
        ]

    @staticmethod
    def _get_spec(nt: NodeType) -> NodeSpec:
        return NodeRegistry.get(nt.name)
