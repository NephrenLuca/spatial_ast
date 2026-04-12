"""Unit tests for core.tokenizer – vocabulary, encode/decode."""

import pytest

from core.tokenizer import (
    ENUM_TOKEN_MAP,
    ENUM_VALUES,
    Q8_OFFSET,
    TOKEN_BOS,
    TOKEN_COMMA,
    TOKEN_EOS,
    TOKEN_LPAREN,
    TOKEN_MASK,
    TOKEN_NIL,
    TOKEN_NOISE,
    TOKEN_PAD,
    TOKEN_RPAREN,
    TOKEN_SEP,
    TOKEN_UNK,
    VOCAB_SIZE,
    TokenRole,
    decode_enum,
    decode_q8,
    encode_enum,
    encode_param,
    encode_q8,
    decode_param,
    get_node_type_from_token,
    get_node_type_token,
    is_enum_token,
    is_node_tag_token,
    is_q8_token,
    is_special_token,
    is_structure_token,
    token_name,
)
from core.types import NodeType


# ── Vocabulary Constants ────────────────────────────────────────────

class TestVocabLayout:

    def test_vocab_size(self):
        assert VOCAB_SIZE == 304

    def test_special_token_ids(self):
        assert TOKEN_PAD == 0
        assert TOKEN_BOS == 1
        assert TOKEN_EOS == 2
        assert TOKEN_MASK == 3
        assert TOKEN_NOISE == 4
        assert TOKEN_SEP == 5
        assert TOKEN_UNK == 6
        assert TOKEN_NIL == 7

    def test_bracket_ids(self):
        assert TOKEN_LPAREN == 22
        assert TOKEN_RPAREN == 23
        assert TOKEN_COMMA == 24

    def test_q8_offset(self):
        assert Q8_OFFSET == 48

    def test_no_overlapping_ranges(self):
        specials = set(range(0, 8))
        structs = set(range(8, 25))
        enums = set(range(32, 48))
        q8s = set(range(48, 304))
        assert not (specials & structs)
        assert not (structs & enums)
        assert not (enums & q8s)


# ── Q8 Encode/Decode ───────────────────────────────────────────────

class TestQ8:

    def test_encode_min(self):
        assert encode_q8(0) == Q8_OFFSET

    def test_encode_max(self):
        assert encode_q8(255) == Q8_OFFSET + 255

    def test_decode_roundtrip_full_range(self):
        for v in range(256):
            assert decode_q8(encode_q8(v)) == v

    def test_encode_out_of_range_negative(self):
        with pytest.raises(ValueError):
            encode_q8(-1)

    def test_encode_out_of_range_high(self):
        with pytest.raises(ValueError):
            encode_q8(256)

    def test_decode_invalid_token(self):
        with pytest.raises(ValueError):
            decode_q8(0)  # PAD is not a Q8 token


# ── Enum Encode/Decode ─────────────────────────────────────────────

class TestEnum:

    def test_all_enums_have_tokens(self):
        for v in ENUM_VALUES:
            assert v in ENUM_TOKEN_MAP

    def test_encode_decode_roundtrip(self):
        for v in ENUM_VALUES:
            assert decode_enum(encode_enum(v)) == v

    def test_encode_unknown_raises(self):
        with pytest.raises(ValueError):
            encode_enum("nonexistent_enum")

    def test_decode_invalid_token_raises(self):
        with pytest.raises(ValueError):
            decode_enum(999)

    def test_boolean_ops(self):
        for op in ("union", "intersect", "subtract"):
            tok = encode_enum(op)
            assert is_enum_token(tok)
            assert decode_enum(tok) == op

    def test_extrude_ops(self):
        for op in ("new", "cut", "join"):
            tok = encode_enum(op)
            assert is_enum_token(tok)


# ── encode_param / decode_param ─────────────────────────────────────

class TestParamEncoding:

    def test_q8_param(self):
        tok = encode_param(128, "q8")
        assert decode_param(tok, "q8") == 128

    def test_enum_param(self):
        tok = encode_param("new", "enum")
        assert decode_param(tok, "enum") == "new"

    def test_str_param_yields_nil(self):
        tok = encode_param("1.0", "str")
        assert tok == TOKEN_NIL

    def test_unknown_dtype_raises(self):
        with pytest.raises(ValueError):
            encode_param(42, "float64")


# ── Classification Helpers ──────────────────────────────────────────

class TestClassification:

    def test_is_q8_token(self):
        assert is_q8_token(Q8_OFFSET)
        assert is_q8_token(Q8_OFFSET + 255)
        assert not is_q8_token(Q8_OFFSET - 1)
        assert not is_q8_token(Q8_OFFSET + 256)

    def test_is_special_token(self):
        for t in range(8):
            assert is_special_token(t)
        assert not is_special_token(8)

    def test_is_structure_token(self):
        assert is_structure_token(8)   # first struct token
        assert is_structure_token(TOKEN_LPAREN)
        assert is_structure_token(TOKEN_COMMA)
        assert not is_structure_token(7)
        assert not is_structure_token(32)

    def test_is_enum_token(self):
        assert is_enum_token(32)
        assert is_enum_token(47)
        assert not is_enum_token(48)
        assert not is_enum_token(31)


# ── Node Type Token Mapping ────────────────────────────────────────

class TestNodeTypeTokens:

    def test_all_semantic_types_have_tokens(self):
        for nt in NodeType:
            if nt.value <= NodeType.SCL.value:
                tok = get_node_type_token(nt.name)
                assert tok >= 8

    def test_roundtrip(self):
        for nt in NodeType:
            if nt.value <= NodeType.SCL.value:
                tok = get_node_type_token(nt.name)
                recovered = get_node_type_from_token(tok)
                assert recovered == nt

    def test_unknown_tag_raises(self):
        with pytest.raises(KeyError):
            get_node_type_token("NONEXISTENT")

    def test_invalid_token_returns_none(self):
        assert get_node_type_from_token(TOKEN_PAD) is None
        assert get_node_type_from_token(Q8_OFFSET) is None

    def test_is_node_tag_token(self):
        tok = get_node_type_token("PROG")
        assert is_node_tag_token(tok)
        assert not is_node_tag_token(TOKEN_PAD)


# ── token_name ──────────────────────────────────────────────────────

class TestTokenName:

    def test_special_names(self):
        assert token_name(TOKEN_PAD) == "[PAD]"
        assert token_name(TOKEN_BOS) == "[BOS]"
        assert token_name(TOKEN_EOS) == "[EOS]"

    def test_bracket_names(self):
        assert token_name(TOKEN_LPAREN) == "("
        assert token_name(TOKEN_RPAREN) == ")"

    def test_node_type_name(self):
        tok = get_node_type_token("EXT")
        assert token_name(tok) == "EXT"

    def test_enum_name(self):
        tok = encode_enum("union")
        assert token_name(tok) == "union"

    def test_q8_name(self):
        tok = encode_q8(42)
        assert token_name(tok) == "Q42"


# ── TokenRole ──────────────────────────────────────────────────────

class TestTokenRole:

    def test_values(self):
        assert TokenRole.NODE_TAG == 0
        assert TokenRole.OPEN_PAREN == 1
        assert TokenRole.CLOSE_PAREN == 2
        assert TokenRole.PARAM_VALUE == 3
        assert TokenRole.SEPARATOR == 4
        assert TokenRole.SPECIAL == 5
