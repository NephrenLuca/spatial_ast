"""Unit tests for core.types – NodeType, NodeSpec, NodeRegistry."""

import pytest

from core.types import (
    DEPTH_OF,
    MAX_DEPTH,
    MAX_EDGES,
    MAX_FACES,
    MAX_LOOPS,
    MAX_OPS,
    MAX_SOLIDS,
    ChildSlot,
    NodeRegistry,
    NodeSpec,
    NodeType,
    ParamDef,
    SEMANTIC_NODE_TYPES,
)


# ── NodeType ────────────────────────────────────────────────────────

class TestNodeType:

    def test_semantic_types_count(self):
        assert len(SEMANTIC_NODE_TYPES) == 14

    def test_special_types(self):
        assert NodeType.MASK == 14
        assert NodeType.NOISE == 15
        assert NodeType.NIL == 16

    def test_total_enum_members(self):
        assert len(NodeType) == 17

    def test_no_duplicate_values(self):
        values = [nt.value for nt in NodeType]
        assert len(values) == len(set(values))

    def test_depth_of_covers_all_semantic(self):
        for nt in SEMANTIC_NODE_TYPES:
            assert nt in DEPTH_OF, f"Missing DEPTH_OF entry for {nt.name}"

    def test_depth_ranges(self):
        for nt, depth in DEPTH_OF.items():
            assert 0 <= depth <= MAX_DEPTH

    def test_l0_is_prog(self):
        assert DEPTH_OF[NodeType.PROG] == 0

    def test_l1_types(self):
        assert DEPTH_OF[NodeType.SOL] == 1
        assert DEPTH_OF[NodeType.BOOL] == 1

    def test_l5_types(self):
        l5_types = [NodeType.LN, NodeType.ARC, NodeType.CIR,
                     NodeType.CRD, NodeType.SCL]
        for nt in l5_types:
            assert DEPTH_OF[nt] == 5


# ── ParamDef / ChildSlot ───────────────────────────────────────────

class TestParamDef:

    def test_frozen(self):
        pd = ParamDef("x", "q8")
        with pytest.raises(AttributeError):
            pd.name = "y"

    def test_enum_param(self):
        pd = ParamDef("op_type", "enum", enum_values=["new", "cut", "join"])
        assert pd.dtype == "enum"
        assert "new" in pd.enum_values

    def test_default_value(self):
        pd = ParamDef("version", "str", default="1.0")
        assert pd.default == "1.0"


class TestChildSlot:

    def test_frozen(self):
        cs = ChildSlot("solids", frozenset({"SOL"}), 1, 16)
        with pytest.raises(AttributeError):
            cs.name = "other"

    def test_allowed_types(self):
        cs = ChildSlot("curves", frozenset({"LN", "ARC", "CIR"}), 1, 1)
        assert "LN" in cs.allowed_types
        assert "ARC" in cs.allowed_types


# ── NodeRegistry ────────────────────────────────────────────────────

class TestNodeRegistry:

    @classmethod
    def setup_class(cls):
        NodeRegistry.reset()

    def test_bootstrap_registers_all_14(self):
        NodeRegistry.reset()
        NodeRegistry.bootstrap()
        assert len(NodeRegistry.all_tags()) == 14

    def test_no_duplicate_tags(self):
        tags = NodeRegistry.all_tags()
        assert len(tags) == len(set(tags))

    def test_no_duplicate_token_ids(self):
        token_ids = [s.token_id for s in NodeRegistry.all_specs()]
        assert len(token_ids) == len(set(token_ids))

    def test_get_known_tag(self):
        spec = NodeRegistry.get("EXT")
        assert spec.tag == "EXT"
        assert spec.depth == 2

    def test_get_unknown_tag_raises(self):
        with pytest.raises(KeyError):
            NodeRegistry.get("NONEXISTENT")

    def test_contains(self):
        assert NodeRegistry.contains("PROG")
        assert not NodeRegistry.contains("FOOBAR")

    def test_ext_has_params(self):
        spec = NodeRegistry.get("EXT")
        assert "distance_fwd" in spec.param_schema
        assert "distance_bwd" in spec.param_schema
        assert "op_type" in spec.param_schema
        assert spec.param_schema["distance_fwd"].dtype == "q8"
        assert spec.param_schema["op_type"].dtype == "enum"

    def test_ext_is_leaf(self):
        spec = NodeRegistry.get("EXT")
        assert len(spec.child_schema) == 0

    def test_prog_children_types(self):
        children = NodeRegistry.get_children_types("PROG")
        assert children == {"SOL", "BOOL"}

    def test_edge_children_types(self):
        children = NodeRegistry.get_children_types("EDGE")
        assert children == {"LN", "ARC", "CIR"}

    def test_crd_is_leaf(self):
        spec = NodeRegistry.get("CRD")
        assert len(spec.child_schema) == 0
        assert "x" in spec.param_schema
        assert "y" in spec.param_schema

    def test_sol_sketch_slot(self):
        spec = NodeRegistry.get("SOL")
        sketch_slot = spec.child_schema[0]
        assert sketch_slot.name == "sketch"
        assert sketch_slot.min_count == 1
        assert sketch_slot.max_count == 1

    def test_sol_ops_slot(self):
        spec = NodeRegistry.get("SOL")
        ops_slot = spec.child_schema[1]
        assert ops_slot.name == "ops"
        assert ops_slot.min_count == 1
        assert ops_slot.max_count == MAX_OPS

    def test_register_duplicate_raises(self):
        NodeRegistry.reset()
        NodeRegistry.bootstrap()
        dup = NodeSpec(tag="PROG", depth=0, token_id=999)
        with pytest.raises(ValueError, match="Duplicate tag"):
            NodeRegistry.register(dup)

    def test_ln_children(self):
        spec = NodeRegistry.get("LN")
        assert len(spec.child_schema) == 1
        assert spec.child_schema[0].min_count == 2
        assert spec.child_schema[0].max_count == 2

    def test_arc_children(self):
        spec = NodeRegistry.get("ARC")
        assert spec.child_schema[0].min_count == 3
        assert spec.child_schema[0].max_count == 3

    def test_cir_children(self):
        spec = NodeRegistry.get("CIR")
        assert len(spec.child_schema) == 2
        center_slot = spec.child_schema[0]
        radius_slot = spec.child_schema[1]
        assert "CRD" in center_slot.allowed_types
        assert "SCL" in radius_slot.allowed_types
