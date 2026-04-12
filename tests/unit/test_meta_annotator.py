"""Unit tests for data.meta_annotator."""

import pytest

from core.types import NodeType, NodeRegistry
from core.ast_node import reset_id_counter
from core.tokenizer import TOKEN_PAD

from data.meta_annotator import MetaAnnotator, AnnotatedSample

from tests.helpers import (
    build_rectangle_solid,
    build_circle_solid,
    build_multi_solid,
    build_triangle_solid,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_id_counter()
    NodeRegistry.reset()
    yield


class TestAnnotatedSampleShape:

    def test_all_arrays_same_length_with_padding(self):
        ast = build_rectangle_solid()
        sample = MetaAnnotator(max_seq_len=256).annotate(ast)
        L = 256
        assert len(sample.tokens) == L
        assert len(sample.depths) == L
        assert len(sample.types) == L
        assert len(sample.roles) == L
        assert len(sample.parents) == L
        assert len(sample.siblings) == L
        assert len(sample.geom_desc) == L

    def test_variable_length_no_padding(self):
        ast = build_rectangle_solid()
        sample = MetaAnnotator(max_seq_len=None).annotate(ast)
        L = sample.seq_len
        assert L > 0
        assert len(sample.tokens) == L
        assert len(sample.depths) == L
        assert len(sample.types) == L
        assert len(sample.roles) == L
        assert len(sample.parents) == L
        assert len(sample.siblings) == L
        assert len(sample.geom_desc) == L
        assert TOKEN_PAD not in sample.tokens

    def test_default_is_variable_length(self):
        ast = build_rectangle_solid()
        sample = MetaAnnotator().annotate(ast)
        assert TOKEN_PAD not in sample.tokens
        assert len(sample.tokens) == sample.seq_len

    def test_geom_desc_dim(self):
        ast = build_rectangle_solid()
        sample = MetaAnnotator().annotate(ast)
        for gd in sample.geom_desc:
            assert len(gd) == 4

    def test_seq_len_excludes_pad(self):
        ast = build_rectangle_solid()
        sample = MetaAnnotator(max_seq_len=512).annotate(ast)
        pad_count = sample.tokens.count(TOKEN_PAD)
        assert sample.seq_len == 512 - pad_count


class TestAnnotationValues:

    def test_depth_range(self):
        ast = build_rectangle_solid()
        sample = MetaAnnotator().annotate(ast)
        for d in sample.depths:
            assert 0 <= d <= 5

    def test_type_range(self):
        ast = build_rectangle_solid()
        sample = MetaAnnotator().annotate(ast)
        for t in sample.types:
            assert 0 <= t <= NodeType.NIL.value

    def test_role_range(self):
        ast = build_rectangle_solid()
        sample = MetaAnnotator().annotate(ast)
        for r in sample.roles:
            assert 0 <= r <= 5

    def test_sibling_capped(self):
        ast = build_rectangle_solid()
        sample = MetaAnnotator().annotate(ast)
        for s in sample.siblings:
            assert 0 <= s <= 63

    def test_geom_scale_range(self):
        ast = build_rectangle_solid()
        sample = MetaAnnotator().annotate(ast)
        for gd in sample.geom_desc:
            assert gd[0] >= 0.0  # scale
            assert gd[2] >= 0.0  # depth_ratio
            assert gd[2] <= 1.0


class TestDifferentFixtures:

    def test_circle(self):
        ast = build_circle_solid()
        sample = MetaAnnotator().annotate(ast)
        assert sample.seq_len > 0
        assert len(sample.tokens) == sample.seq_len

    def test_multi_solid(self):
        ast = build_multi_solid()
        sample = MetaAnnotator().annotate(ast)
        assert sample.seq_len > 0

    def test_triangle(self):
        ast = build_triangle_solid()
        sample = MetaAnnotator().annotate(ast)
        assert sample.seq_len > 0


class TestAnnotatorConsistency:

    def test_tokens_match_serializer_padded(self):
        from core.serializer import ASTSerializer
        ast = build_rectangle_solid()
        ann = MetaAnnotator(max_seq_len=256)
        sample = ann.annotate(ast)

        ser = ASTSerializer(max_seq_len=256)
        tokens, _ = ser.serialize(ast, pad=True)

        assert sample.tokens == tokens

    def test_tokens_match_serializer_variable(self):
        from core.serializer import ASTSerializer
        ast = build_rectangle_solid()
        ann = MetaAnnotator(max_seq_len=None)
        sample = ann.annotate(ast)

        ser = ASTSerializer(max_seq_len=None)
        tokens, _ = ser.serialize(ast, pad=False)

        assert sample.tokens == tokens
