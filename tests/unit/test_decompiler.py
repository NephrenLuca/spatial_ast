"""Unit tests for data.deepcad_parser + data.decompiler."""

import json
import pytest

from core.types import NodeType, NodeRegistry
from core.ast_node import reset_id_counter
from core.grammar import validate_ast

from data.deepcad_parser import DeepCADParser
from data.decompiler import DeepCADDecompiler


@pytest.fixture(autouse=True)
def _reset():
    reset_id_counter()
    NodeRegistry.reset()
    yield


# ═══════════════════════════════════════════════════════════════════
# Real DeepCAD JSON fixtures (entities + sequence format)
# ═══════════════════════════════════════════════════════════════════

CIRCLE_REAL_JSON = {
    "entities": {
        "EXT1": {
            "type": "ExtrudeFeature",
            "extent_one": {"distance": {"value": 0.0254}, "type": "DistanceExtentDefinition"},
            "extent_two": {"distance": {"value": 0.0}, "type": "DistanceExtentDefinition"},
            "operation": "NewBodyFeatureOperation",
            "profiles": [{"profile": "P1", "sketch": "SKT1"}],
        },
        "SKT1": {
            "type": "Sketch",
            "profiles": {
                "P1": {
                    "loops": [{
                        "is_outer": True,
                        "profile_curves": [{
                            "type": "Circle3D",
                            "center_point": {"x": 0.0, "y": 0.0, "z": 0.0},
                            "radius": 0.09,
                            "normal": {"x": 0, "y": 0, "z": 1},
                        }]
                    }],
                    "properties": {}
                }
            },
        }
    },
    "properties": {
        "bounding_box": {
            "min_point": {"x": -0.09, "y": -0.09, "z": 0.0},
            "max_point": {"x": 0.09, "y": 0.09, "z": 0.0254},
        }
    },
    "sequence": [
        {"index": 0, "type": "Sketch", "entity": "SKT1"},
        {"index": 1, "type": "ExtrudeFeature", "entity": "EXT1"},
    ]
}

RECT_REAL_JSON = {
    "entities": {
        "EXT1": {
            "type": "ExtrudeFeature",
            "extent_one": {"distance": {"value": 0.01}, "type": "DistanceExtentDefinition"},
            "extent_two": {"distance": {"value": 0.0}, "type": "DistanceExtentDefinition"},
            "operation": "NewBodyFeatureOperation",
        },
        "SKT1": {
            "type": "Sketch",
            "profiles": {
                "P1": {
                    "loops": [{
                        "is_outer": True,
                        "profile_curves": [
                            {"type": "Line3D",
                             "start_point": {"x": 0.0, "y": 0.0, "z": 0.0},
                             "end_point": {"x": 0.1, "y": 0.0, "z": 0.0}},
                            {"type": "Line3D",
                             "start_point": {"x": 0.1, "y": 0.0, "z": 0.0},
                             "end_point": {"x": 0.1, "y": 0.05, "z": 0.0}},
                            {"type": "Line3D",
                             "start_point": {"x": 0.1, "y": 0.05, "z": 0.0},
                             "end_point": {"x": 0.0, "y": 0.05, "z": 0.0}},
                            {"type": "Line3D",
                             "start_point": {"x": 0.0, "y": 0.05, "z": 0.0},
                             "end_point": {"x": 0.0, "y": 0.0, "z": 0.0}},
                        ]
                    }]
                }
            },
        }
    },
    "properties": {
        "bounding_box": {
            "min_point": {"x": 0.0, "y": 0.0, "z": 0.0},
            "max_point": {"x": 0.1, "y": 0.05, "z": 0.01},
        }
    },
    "sequence": [
        {"index": 0, "type": "Sketch", "entity": "SKT1"},
        {"index": 1, "type": "ExtrudeFeature", "entity": "EXT1"},
    ]
}

ARC_REAL_JSON = {
    "entities": {
        "EXT1": {
            "type": "ExtrudeFeature",
            "extent_one": {"distance": {"value": 0.012}, "type": "DistanceExtentDefinition"},
            "extent_two": {"distance": {"value": 0.0}, "type": "DistanceExtentDefinition"},
            "operation": "NewBodyFeatureOperation",
        },
        "SKT1": {
            "type": "Sketch",
            "profiles": {
                "P1": {
                    "loops": [{
                        "is_outer": True,
                        "profile_curves": [
                            {"type": "Line3D",
                             "start_point": {"x": 0.0, "y": 0.0, "z": 0.0},
                             "end_point": {"x": 0.05, "y": 0.0, "z": 0.0}},
                            {"type": "Arc3D",
                             "start_point": {"x": 0.05, "y": 0.0, "z": 0.0},
                             "end_point": {"x": 0.0, "y": 0.05, "z": 0.0},
                             "center_point": {"x": 0.0, "y": 0.0, "z": 0.0},
                             "radius": 0.05,
                             "start_angle": 0.0,
                             "end_angle": 1.5707963,
                             "reference_vector": {"x": 1.0, "y": 0.0, "z": 0.0},
                             "normal": {"x": 0, "y": 0, "z": 1}},
                            {"type": "Line3D",
                             "start_point": {"x": 0.0, "y": 0.05, "z": 0.0},
                             "end_point": {"x": 0.0, "y": 0.0, "z": 0.0}},
                        ]
                    }]
                }
            },
        }
    },
    "properties": {
        "bounding_box": {
            "min_point": {"x": 0.0, "y": 0.0, "z": 0.0},
            "max_point": {"x": 0.05, "y": 0.05, "z": 0.012},
        }
    },
    "sequence": [
        {"index": 0, "type": "Sketch", "entity": "SKT1"},
        {"index": 1, "type": "ExtrudeFeature", "entity": "EXT1"},
    ]
}

MULTI_EXTRUDE_REAL_JSON = {
    "entities": {
        "SKT1": {
            "type": "Sketch",
            "profiles": {
                "P1": {
                    "loops": [{
                        "is_outer": True,
                        "profile_curves": [
                            {"type": "Line3D",
                             "start_point": {"x": 0.0, "y": 0.0, "z": 0.0},
                             "end_point": {"x": 0.1, "y": 0.0, "z": 0.0}},
                            {"type": "Line3D",
                             "start_point": {"x": 0.1, "y": 0.0, "z": 0.0},
                             "end_point": {"x": 0.0, "y": 0.0, "z": 0.0}},
                        ]
                    }]
                }
            },
        },
        "EXT1": {
            "type": "ExtrudeFeature",
            "extent_one": {"distance": {"value": 0.02}, "type": "DistanceExtentDefinition"},
            "extent_two": {"distance": {"value": 0.0}, "type": "DistanceExtentDefinition"},
            "operation": "NewBodyFeatureOperation",
        },
        "EXT2": {
            "type": "ExtrudeFeature",
            "extent_one": {"distance": {"value": 0.01}, "type": "DistanceExtentDefinition"},
            "extent_two": {"distance": {"value": 0.0}, "type": "DistanceExtentDefinition"},
            "operation": "CutFeatureOperation",
        },
    },
    "properties": {
        "bounding_box": {
            "min_point": {"x": 0.0, "y": 0.0, "z": 0.0},
            "max_point": {"x": 0.1, "y": 0.05, "z": 0.02},
        }
    },
    "sequence": [
        {"index": 0, "type": "Sketch", "entity": "SKT1"},
        {"index": 1, "type": "ExtrudeFeature", "entity": "EXT1"},
        {"index": 2, "type": "ExtrudeFeature", "entity": "EXT2"},
    ]
}

# Pre-normalised flat command list (for backward compat)
FLAT_CMD_LIST = [
    {
        "type": "sketch",
        "loops": [{"curves": [
            {"type": "line", "start_x": 0, "start_y": 0, "end_x": 128, "end_y": 128},
            {"type": "line", "start_x": 128, "start_y": 128, "end_x": 0, "end_y": 0},
        ]}]
    },
    {"type": "extrude", "distance_fwd": 64, "distance_bwd": 0, "op_type": "new"},
]


# ═══════════════════════════════════════════════════════════════════
# DeepCADParser
# ═══════════════════════════════════════════════════════════════════

class TestDeepCADParser:

    def test_parse_circle_real(self):
        cmds = DeepCADParser().parse_dict(CIRCLE_REAL_JSON)
        assert len(cmds) == 2
        assert cmds[0]["type"] == "sketch"
        assert cmds[1]["type"] == "extrude"

    def test_circle_curves(self):
        cmds = DeepCADParser().parse_dict(CIRCLE_REAL_JSON)
        curves = cmds[0]["loops"][0]["curves"]
        assert len(curves) == 1
        assert curves[0]["type"] == "circle"
        assert isinstance(curves[0]["radius"], int)
        assert 0 <= curves[0]["radius"] <= 255

    def test_parse_rectangle_real(self):
        cmds = DeepCADParser().parse_dict(RECT_REAL_JSON)
        assert len(cmds) == 2
        curves = cmds[0]["loops"][0]["curves"]
        assert len(curves) == 4
        assert all(c["type"] == "line" for c in curves)

    def test_coords_are_q8(self):
        cmds = DeepCADParser().parse_dict(RECT_REAL_JSON)
        for c in cmds[0]["loops"][0]["curves"]:
            assert 0 <= c["start_x"] <= 255
            assert 0 <= c["start_y"] <= 255

    def test_parse_arc_real(self):
        cmds = DeepCADParser().parse_dict(ARC_REAL_JSON)
        curves = cmds[0]["loops"][0]["curves"]
        arc_curves = [c for c in curves if c["type"] == "arc"]
        assert len(arc_curves) == 1
        assert "mid_x" in arc_curves[0]
        assert 0 <= arc_curves[0]["mid_x"] <= 255

    def test_multi_extrude(self):
        cmds = DeepCADParser().parse_dict(MULTI_EXTRUDE_REAL_JSON)
        assert len(cmds) == 3
        assert cmds[0]["type"] == "sketch"
        assert cmds[1]["type"] == "extrude"
        assert cmds[1]["op_type"] == "new"
        assert cmds[2]["type"] == "extrude"
        assert cmds[2]["op_type"] == "cut"

    def test_extrude_distance_quantised(self):
        cmds = DeepCADParser().parse_dict(RECT_REAL_JSON)
        ext = cmds[1]
        assert isinstance(ext["distance_fwd"], int)
        assert 0 <= ext["distance_fwd"] <= 255

    def test_flat_command_list_compat(self):
        cmds = DeepCADParser().parse_dict(FLAT_CMD_LIST)
        assert len(cmds) == 2
        assert cmds[0]["type"] == "sketch"

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            DeepCADParser().parse_dict({"not_sequence": 123})


# ═══════════════════════════════════════════════════════════════════
# DeepCADDecompiler
# ═══════════════════════════════════════════════════════════════════

class TestDeepCADDecompiler:

    def test_circle_decompile(self):
        cmds = DeepCADParser().parse_dict(CIRCLE_REAL_JSON)
        ast = DeepCADDecompiler().decompile(cmds)
        assert ast.node_type == NodeType.PROG
        r = validate_ast(ast)
        assert r.is_valid, r.errors

    def test_circle_has_cir_node(self):
        cmds = DeepCADParser().parse_dict(CIRCLE_REAL_JSON)
        ast = DeepCADDecompiler().decompile(cmds)
        circles = ast.collect(lambda n: n.node_type == NodeType.CIR)
        assert len(circles) == 1

    def test_rectangle_decompile(self):
        cmds = DeepCADParser().parse_dict(RECT_REAL_JSON)
        ast = DeepCADDecompiler().decompile(cmds)
        r = validate_ast(ast)
        assert r.is_valid, r.errors
        lines = ast.collect(lambda n: n.node_type == NodeType.LN)
        assert len(lines) == 4

    def test_arc_decompile(self):
        cmds = DeepCADParser().parse_dict(ARC_REAL_JSON)
        ast = DeepCADDecompiler().decompile(cmds)
        r = validate_ast(ast)
        assert r.is_valid, r.errors
        arcs = ast.collect(lambda n: n.node_type == NodeType.ARC)
        assert len(arcs) == 1

    def test_multi_extrude_decompile(self):
        cmds = DeepCADParser().parse_dict(MULTI_EXTRUDE_REAL_JSON)
        ast = DeepCADDecompiler().decompile(cmds)
        r = validate_ast(ast)
        assert r.is_valid, r.errors
        sol = ast.children[0]
        assert sol.node_type == NodeType.SOL
        ops = [c for c in sol.children if c.node_type in (NodeType.EXT, NodeType.REV)]
        assert len(ops) == 2

    def test_sketch_structure(self):
        cmds = DeepCADParser().parse_dict(RECT_REAL_JSON)
        ast = DeepCADDecompiler().decompile(cmds)
        sol = ast.children[0]
        skt = sol.children[0]
        assert skt.node_type == NodeType.SKT
        face = skt.children[0]
        assert face.node_type == NodeType.FACE

    def test_coord_values_valid_q8(self):
        cmds = DeepCADParser().parse_dict(RECT_REAL_JSON)
        ast = DeepCADDecompiler().decompile(cmds)
        for n in ast.dfs():
            if n.node_type == NodeType.CRD:
                assert 0 <= n.params["x"] <= 255
                assert 0 <= n.params["y"] <= 255

    def test_empty_commands_raises(self):
        with pytest.raises(ValueError):
            DeepCADDecompiler().decompile([])

    def test_flat_commands_decompile(self):
        cmds = DeepCADParser().parse_dict(FLAT_CMD_LIST)
        ast = DeepCADDecompiler().decompile(cmds)
        r = validate_ast(ast)
        assert r.is_valid, r.errors


# ═══════════════════════════════════════════════════════════════════
# Full pipeline: real JSON → parse → decompile → validate → serialize
# ═══════════════════════════════════════════════════════════════════

class TestFullPipeline:

    @pytest.mark.parametrize("fixture", [
        CIRCLE_REAL_JSON, RECT_REAL_JSON, ARC_REAL_JSON, MULTI_EXTRUDE_REAL_JSON,
    ], ids=["circle", "rectangle", "arc", "multi_extrude"])
    def test_json_to_ast_to_tokens_roundtrip(self, fixture):
        from core.serializer import ASTSerializer

        cmds = DeepCADParser().parse_dict(fixture)
        ast = DeepCADDecompiler().decompile(cmds)

        r = validate_ast(ast)
        assert r.is_valid, r.errors

        s = ASTSerializer()
        tokens, metas = s.serialize(ast, pad=False)
        recovered = s.deserialize(tokens)

        assert ast.structurally_equal(recovered)
