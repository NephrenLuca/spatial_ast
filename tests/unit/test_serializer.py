"""Unit tests for core.serializer – AST ↔ token round-trips."""

import pytest

from core.types import NodeType, NodeRegistry
from core.ast_node import ASTNode, reset_id_counter
from core.tokenizer import (
    TOKEN_BOS,
    TOKEN_EOS,
    TOKEN_LPAREN,
    TOKEN_PAD,
    TOKEN_RPAREN,
    TokenRole,
    get_node_type_token,
    encode_q8,
    encode_enum,
    is_q8_token,
)
from core.serializer import ASTSerializer

from tests.helpers import (
    build_arc_sketch_solid,
    build_circle_solid,
    build_multi_solid,
    build_rectangle_solid,
    build_triangle_solid,
    make_coord,
    make_edge,
    make_extrude,
    make_face,
    make_line,
    make_loop,
    make_program,
    make_scalar,
    make_sketch,
    make_solid,
    make_circle,
    make_arc,
    make_revolve,
)


@pytest.fixture(autouse=True)
def _reset_ids():
    reset_id_counter()
    NodeRegistry.reset()
    yield


class TestSerializeBasics:
    """Basic serialization format checks."""

    def test_bos_eos_present(self):
        ast = build_rectangle_solid()
        s = ASTSerializer(max_seq_len=512)
        tokens, metas = s.serialize(ast)
        assert tokens[0] == TOKEN_BOS
        non_pad = [t for t in tokens if t != TOKEN_PAD]
        assert non_pad[-1] == TOKEN_EOS

    def test_output_length_equals_max_seq_len(self):
        ast = build_rectangle_solid()
        s = ASTSerializer(max_seq_len=256)
        tokens, metas = s.serialize(ast, pad=True)
        assert len(tokens) == 256
        assert len(metas) == 256

    def test_no_padding_when_disabled(self):
        ast = build_rectangle_solid()
        s = ASTSerializer()
        tokens, metas = s.serialize(ast, pad=False)
        assert TOKEN_PAD not in tokens

    def test_first_real_token_is_prog(self):
        ast = build_rectangle_solid()
        s = ASTSerializer()
        tokens, _ = s.serialize(ast, pad=False)
        assert tokens[1] == get_node_type_token("PROG")

    def test_brackets_balanced(self):
        ast = build_rectangle_solid()
        s = ASTSerializer()
        tokens, _ = s.serialize(ast, pad=False)
        opens = tokens.count(TOKEN_LPAREN)
        closes = tokens.count(TOKEN_RPAREN)
        assert opens == closes


class TestRoundTrip:
    """Round-trip: deserialize(serialize(ast)) ≅ ast (structural equality)."""

    _fixtures = [
        ("rectangle", build_rectangle_solid),
        ("triangle", build_triangle_solid),
        ("circle", build_circle_solid),
        ("multi_solid", build_multi_solid),
        ("arc_sketch", build_arc_sketch_solid),
    ]

    @pytest.mark.parametrize("name,builder", _fixtures, ids=[f[0] for f in _fixtures])
    def test_roundtrip(self, name, builder):
        original = builder()
        s = ASTSerializer()
        tokens, _ = s.serialize(original, pad=False)
        recovered = s.deserialize(tokens)
        assert original.structurally_equal(recovered), (
            f"Round-trip failed for {name}:\n"
            f"Original:\n{original.pretty()}\n"
            f"Recovered:\n{recovered.pretty()}"
        )

    def test_coord_roundtrip_values(self):
        """Verify Q8 values survive the round-trip exactly."""
        for x in (0, 1, 127, 128, 255):
            for y in (0, 64, 255):
                coord = make_coord(x, y)
                line = make_line(x, y, 255 - x, 255 - y)
                edge = make_edge(line)
                loop = make_loop([edge])
                face = make_face([loop])
                sketch = make_sketch([face])
                ext = make_extrude(64)
                sol = make_solid(sketch, [ext])
                ast = make_program([sol])

                s = ASTSerializer()
                tokens, _ = s.serialize(ast, pad=False)
                recovered = s.deserialize(tokens)

                rec_line = (
                    recovered.children[0]   # SOL
                    .children[0]            # SKT
                    .children[0]            # FACE
                    .children[0]            # LOOP
                    .children[0]            # EDGE
                    .children[0]            # LN
                )
                rec_start = rec_line.children[0]
                assert rec_start.params["x"] == x
                assert rec_start.params["y"] == y

    def test_extrude_params_roundtrip(self):
        ext = make_extrude(100, 50, "cut")
        sketch = make_sketch([make_face([make_loop([
            make_edge(make_line(0, 0, 128, 0)),
            make_edge(make_line(128, 0, 0, 0)),
        ])])])
        sol = make_solid(sketch, [ext])
        ast = make_program([sol])

        s = ASTSerializer()
        tokens, _ = s.serialize(ast, pad=False)
        recovered = s.deserialize(tokens)

        rec_ext = recovered.children[0].children[1]  # SOL → EXT
        assert rec_ext.params["distance_fwd"] == 100
        assert rec_ext.params["distance_bwd"] == 50
        assert rec_ext.params["op_type"] == "cut"

    def test_revolve_roundtrip(self):
        rev = make_revolve(200, "join")
        sketch = make_sketch([make_face([make_loop([
            make_edge(make_line(0, 0, 128, 128)),
            make_edge(make_line(128, 128, 0, 0)),
        ])])])
        sol = make_solid(sketch, [rev])
        ast = make_program([sol])

        s = ASTSerializer()
        tokens, _ = s.serialize(ast, pad=False)
        recovered = s.deserialize(tokens)
        rec_rev = recovered.children[0].children[1]
        assert rec_rev.params["angle"] == 200
        assert rec_rev.params["op_type"] == "join"

    def test_circle_roundtrip(self):
        ast = build_circle_solid(cx=100, cy=200, r=50, depth=80)
        s = ASTSerializer()
        tokens, _ = s.serialize(ast, pad=False)
        recovered = s.deserialize(tokens)
        assert ast.structurally_equal(recovered)

    def test_multi_solid_roundtrip(self):
        ast = build_multi_solid()
        s = ASTSerializer()
        tokens, _ = s.serialize(ast, pad=False)
        recovered = s.deserialize(tokens)
        assert len(recovered.children) == 2
        assert ast.structurally_equal(recovered)


class TestTokenMetadata:
    """Verify per-token metadata correctness."""

    def test_meta_length_matches_tokens(self):
        ast = build_rectangle_solid()
        s = ASTSerializer()
        tokens, metas = s.serialize(ast)
        assert len(tokens) == len(metas)

    def test_bos_meta(self):
        ast = build_rectangle_solid()
        s = ASTSerializer()
        _, metas = s.serialize(ast)
        assert metas[0].role == TokenRole.SPECIAL
        assert metas[0].depth == 0

    def test_node_tag_roles(self):
        ast = build_rectangle_solid()
        s = ASTSerializer()
        tokens, metas = s.serialize(ast, pad=False)
        prog_idx = 1  # after BOS
        assert metas[prog_idx].role == TokenRole.NODE_TAG
        assert metas[prog_idx].node_type == NodeType.PROG

    def test_param_roles(self):
        ast = build_rectangle_solid()
        s = ASTSerializer()
        tokens, metas = s.serialize(ast, pad=False)
        # Find first EXT tag, then check its param tokens
        ext_indices = [
            i for i, m in enumerate(metas)
            if m.role == TokenRole.NODE_TAG and m.node_type == NodeType.EXT
        ]
        assert len(ext_indices) == 1
        ext_i = ext_indices[0]
        assert metas[ext_i + 1].role == TokenRole.PARAM_VALUE  # distance_fwd
        assert metas[ext_i + 2].role == TokenRole.PARAM_VALUE  # distance_bwd
        assert metas[ext_i + 3].role == TokenRole.PARAM_VALUE  # op_type

    def test_paren_roles(self):
        ast = build_rectangle_solid()
        s = ASTSerializer()
        tokens, metas = s.serialize(ast, pad=False)
        open_roles = [m for m in metas if m.role == TokenRole.OPEN_PAREN]
        close_roles = [m for m in metas if m.role == TokenRole.CLOSE_PAREN]
        assert len(open_roles) == len(close_roles)
        assert len(open_roles) > 0

    def test_depth_values(self):
        ast = build_rectangle_solid()
        s = ASTSerializer()
        _, metas = s.serialize(ast, pad=False)
        depths = {m.depth for m in metas if m.role == TokenRole.NODE_TAG}
        assert 0 in depths  # PROG
        assert 5 in depths  # CRD


class TestDeserializeEdgeCases:

    def test_empty_tokens_raises(self):
        s = ASTSerializer()
        with pytest.raises(ValueError):
            s.deserialize([TOKEN_BOS, TOKEN_EOS])

    def test_pad_only_raises(self):
        s = ASTSerializer()
        with pytest.raises(ValueError):
            s.deserialize([TOKEN_PAD] * 10)

    def test_truncated_sequence(self):
        """If sequence is truncated mid-tree the deserializer shouldn't crash."""
        ast = build_rectangle_solid()
        s = ASTSerializer()
        tokens, _ = s.serialize(ast, pad=False)
        partial = tokens[:10]
        # Should not raise, may produce incomplete tree
        _ = s.deserialize(partial)
