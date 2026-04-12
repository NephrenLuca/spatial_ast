"""
Token vocabulary and encoding / decoding utilities.

Vocabulary layout (304 tokens total):
  [0 –   7]  Special tokens  (PAD, BOS, EOS, MASK, NOISE, SEP, UNK, NIL)
  [8 –  31]  Structure tokens (node-type tags + brackets + comma)
  [32 – 47]  Enum tokens      (union, intersect, subtract, new, cut, join, …)
  [48 – 303] Q8 value tokens  (0–255, token_id = value + 48)
"""

from __future__ import annotations

from enum import IntEnum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.types import NodeType

# ═══════════════════════════════════════════════════════════════════
# Special Tokens  (0-7)
# ═══════════════════════════════════════════════════════════════════
TOKEN_PAD   = 0
TOKEN_BOS   = 1
TOKEN_EOS   = 2
TOKEN_MASK  = 3
TOKEN_NOISE = 4
TOKEN_SEP   = 5
TOKEN_UNK   = 6
TOKEN_NIL   = 7

# ═══════════════════════════════════════════════════════════════════
# Structure Tokens  (8-31)
# ═══════════════════════════════════════════════════════════════════
_STRUCT_OFFSET = 8

# Node-type tag tokens map 1-to-1 with NodeType enum values.
# token_id = _STRUCT_OFFSET + NodeType.value   (for semantic types only, 0-13)
_NODE_TAG_TOKENS: Dict[str, int] = {
    nt.name: _STRUCT_OFFSET + nt.value
    for nt in NodeType
    if nt.value <= NodeType.SCL.value  # 0..13
}

# Brackets and comma occupy slots 22-24
TOKEN_LPAREN = 22
TOKEN_RPAREN = 23
TOKEN_COMMA  = 24

# ═══════════════════════════════════════════════════════════════════
# Enum Tokens  (32-47)
# ═══════════════════════════════════════════════════════════════════
_ENUM_OFFSET = 32

ENUM_VALUES: List[str] = [
    "union",       # 32
    "intersect",   # 33
    "subtract",    # 34
    "new",         # 35
    "cut",         # 36
    "join",        # 37
    "x_axis",      # 38
    "y_axis",      # 39
    "z_axis",      # 40
    "xy_plane",    # 41
    "xz_plane",    # 42
    "yz_plane",    # 43
]

ENUM_TOKEN_MAP: Dict[str, int] = {
    v: _ENUM_OFFSET + i for i, v in enumerate(ENUM_VALUES)
}
TOKEN_ENUM_MAP: Dict[int, str] = {v: k for k, v in ENUM_TOKEN_MAP.items()}

# ═══════════════════════════════════════════════════════════════════
# Q8 Value Tokens  (48-303)
# ═══════════════════════════════════════════════════════════════════
Q8_OFFSET = 48

VOCAB_SIZE = Q8_OFFSET + 256  # 304


# ═══════════════════════════════════════════════════════════════════
# TokenRole / TokenMeta
# ═══════════════════════════════════════════════════════════════════

class TokenRole(IntEnum):
    NODE_TAG    = 0
    OPEN_PAREN  = 1
    CLOSE_PAREN = 2
    PARAM_VALUE = 3
    SEPARATOR   = 4
    SPECIAL     = 5


@dataclass
class TokenMeta:
    """Per-token metadata (not part of the vocabulary; used as model input)."""
    position: int
    depth: int
    node_type: NodeType
    role: TokenRole
    parent_type: NodeType   # NIL for root
    sibling_idx: int


# ═══════════════════════════════════════════════════════════════════
# Encode / Decode helpers
# ═══════════════════════════════════════════════════════════════════

def encode_q8(value: int) -> int:
    """Encode a Q8 integer (0–255) to its token id."""
    if not (0 <= value <= 255):
        raise ValueError(f"Q8 value out of range [0, 255]: {value}")
    return Q8_OFFSET + value


def decode_q8(token_id: int) -> int:
    """Decode a Q8 token id back to its integer value (0–255)."""
    if not (Q8_OFFSET <= token_id < Q8_OFFSET + 256):
        raise ValueError(f"Token {token_id} is not a Q8 value token")
    return token_id - Q8_OFFSET


def encode_enum(value: str) -> int:
    """Encode an enum string to its token id."""
    if value not in ENUM_TOKEN_MAP:
        raise ValueError(f"Unknown enum value: {value!r}")
    return ENUM_TOKEN_MAP[value]


def decode_enum(token_id: int) -> str:
    """Decode an enum token id back to its string value."""
    if token_id not in TOKEN_ENUM_MAP:
        raise ValueError(f"Token {token_id} is not an enum token")
    return TOKEN_ENUM_MAP[token_id]


def encode_param(value: Any, dtype: str) -> int:
    """Encode a parameter value according to its dtype."""
    if dtype == "q8":
        return encode_q8(int(value))
    elif dtype == "enum":
        return encode_enum(str(value))
    elif dtype == "str":
        return TOKEN_NIL  # strings are metadata-only, not serialised
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def decode_param(token_id: int, dtype: str) -> Any:
    """Decode a token id back to a parameter value."""
    if dtype == "q8":
        return decode_q8(token_id)
    elif dtype == "enum":
        return decode_enum(token_id)
    elif dtype == "str":
        return ""
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


# ═══════════════════════════════════════════════════════════════════
# Classification helpers
# ═══════════════════════════════════════════════════════════════════

def is_q8_token(token_id: int) -> bool:
    return Q8_OFFSET <= token_id < Q8_OFFSET + 256


def is_special_token(token_id: int) -> bool:
    return 0 <= token_id <= TOKEN_NIL


def is_structure_token(token_id: int) -> bool:
    return _STRUCT_OFFSET <= token_id <= TOKEN_COMMA


def is_enum_token(token_id: int) -> bool:
    return _ENUM_OFFSET <= token_id < Q8_OFFSET


def is_node_tag_token(token_id: int) -> bool:
    return token_id in _NODE_TAG_TOKENS.values()


def get_node_type_token(tag: str) -> int:
    """Return the token id for a node-type tag string (e.g. ``'EXT'``)."""
    if tag not in _NODE_TAG_TOKENS:
        raise KeyError(f"No token for tag: {tag!r}")
    return _NODE_TAG_TOKENS[tag]


def get_node_type_from_token(token_id: int) -> Optional[NodeType]:
    """Return the NodeType for a node-tag token, or None."""
    adjusted = token_id - _STRUCT_OFFSET
    if 0 <= adjusted <= NodeType.SCL.value:
        return NodeType(adjusted)
    return None


def token_name(token_id: int) -> str:
    """Human-readable name for any token id."""
    _special_names = {
        TOKEN_PAD: "[PAD]", TOKEN_BOS: "[BOS]", TOKEN_EOS: "[EOS]",
        TOKEN_MASK: "[MASK]", TOKEN_NOISE: "[NOISE]", TOKEN_SEP: "[SEP]",
        TOKEN_UNK: "[UNK]", TOKEN_NIL: "[NIL]",
    }
    if token_id in _special_names:
        return _special_names[token_id]
    if token_id == TOKEN_LPAREN:
        return "("
    if token_id == TOKEN_RPAREN:
        return ")"
    if token_id == TOKEN_COMMA:
        return ","
    nt = get_node_type_from_token(token_id)
    if nt is not None:
        return nt.name
    if token_id in TOKEN_ENUM_MAP:
        return TOKEN_ENUM_MAP[token_id]
    if is_q8_token(token_id):
        return f"Q{decode_q8(token_id)}"
    return f"?{token_id}"
