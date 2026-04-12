"""Unit tests for the compiler pipeline: emitter + backend + round-trip."""

import pytest

from core.types import NodeType, NodeRegistry
from core.ast_node import reset_id_counter
from core.grammar import validate_ast

from compiler.ir import IRInstruction, EXTRUDE_OP_MAP, EXTRUDE_OP_INV
from compiler.emitter import IREmitter
from compiler.backend import DeepCADBackend
from compiler.validator import CompileValidator

from data.decompiler import DeepCADDecompiler

from tests.helpers import (
    build_rectangle_solid,
    build_triangle_solid,
    build_circle_solid,
    build_multi_solid,
    build_arc_sketch_solid,
    make_coord,
    make_edge,
    make_extrude,
    make_face,
    make_line,
    make_loop,
    make_program,
    make_sketch,
    make_solid,
    make_revolve,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_id_counter()
    NodeRegistry.reset()
    yield


# ═══════════════════════════════════════════════════════════════════
# IREmitter
# ═══════════════════════════════════════════════════════════════════

class TestIREmitter:

    def test_rectangle_ir(self):
        ast = build_rectangle_solid()
        ir = IREmitter().emit(ast)
        opcodes = [i.opcode for i in ir]
        assert "sketch_start" in opcodes
        assert "sketch_end" in opcodes
        assert "loop_start" in opcodes
        assert "loop_end" in opcodes
        assert opcodes.count("line") == 4
        assert "extrude" in opcodes

    def test_triangle_ir(self):
        ast = build_triangle_solid()
        ir = IREmitter().emit(ast)
        assert sum(1 for i in ir if i.opcode == "line") == 3

    def test_circle_ir(self):
        ast = build_circle_solid()
        ir = IREmitter().emit(ast)
        assert any(i.opcode == "circle" for i in ir)

    def test_arc_ir(self):
        ast = build_arc_sketch_solid()
        ir = IREmitter().emit(ast)
        assert any(i.opcode == "arc" for i in ir)
        assert any(i.opcode == "line" for i in ir)

    def test_multi_solid_ir(self):
        ast = build_multi_solid()
        ir = IREmitter().emit(ast)
        assert sum(1 for i in ir if i.opcode == "sketch_start") == 2

    def test_extrude_params(self):
        ast = build_rectangle_solid(depth=128)
        ir = IREmitter().emit(ast)
        ext = [i for i in ir if i.opcode == "extrude"][0]
        assert len(ext.params) == 3
        assert ext.params[2] == EXTRUDE_OP_MAP["new"]

    def test_revolve_ir(self):
        sketch = make_sketch([make_face([make_loop([
            make_edge(make_line(0, 0, 128, 128)),
            make_edge(make_line(128, 128, 0, 0)),
        ])])])
        sol = make_solid(sketch, [make_revolve(200, "join")])
        ast = make_program([sol])
        ir = IREmitter().emit(ast)
        rev = [i for i in ir if i.opcode == "revolve"][0]
        assert len(rev.params) == 2

    def test_line_params_are_floats(self):
        ast = build_rectangle_solid()
        ir = IREmitter().emit(ast)
        line = [i for i in ir if i.opcode == "line"][0]
        assert all(isinstance(p, float) for p in line.params)
        assert len(line.params) == 4


# ═══════════════════════════════════════════════════════════════════
# DeepCADBackend
# ═══════════════════════════════════════════════════════════════════

class TestDeepCADBackend:

    def test_ir_to_commands_rectangle(self):
        ast = build_rectangle_solid()
        ir = IREmitter().emit(ast)
        backend = DeepCADBackend()
        cmds = backend.ir_to_commands(ir)
        assert len(cmds) == 2  # sketch + extrude
        assert cmds[0]["type"] == "sketch"
        assert cmds[1]["type"] == "extrude"

    def test_sketch_has_loops(self):
        ast = build_rectangle_solid()
        ir = IREmitter().emit(ast)
        cmds = DeepCADBackend().ir_to_commands(ir)
        sketch = cmds[0]
        assert len(sketch["loops"]) == 1
        assert len(sketch["loops"][0]["curves"]) == 4

    def test_extrude_command(self):
        ext = make_extrude(100, 50, "cut")
        sketch = make_sketch([make_face([make_loop([
            make_edge(make_line(0, 0, 128, 0)),
            make_edge(make_line(128, 0, 0, 0)),
        ])])])
        sol = make_solid(sketch, [ext])
        ast = make_program([sol])
        ir = IREmitter().emit(ast)
        cmds = DeepCADBackend().ir_to_commands(ir)
        ext_cmd = cmds[1]
        assert ext_cmd["type"] == "extrude"
        assert ext_cmd["op_type"] == "cut"

    def test_ir_roundtrip(self):
        """commands_to_ir(ir_to_commands(ir)) preserves opcode sequence."""
        ast = build_rectangle_solid()
        ir_orig = IREmitter().emit(ast)
        backend = DeepCADBackend()
        cmds = backend.ir_to_commands(ir_orig)
        ir_back = backend.commands_to_ir(cmds)

        orig_ops = [i.opcode for i in ir_orig]
        back_ops = [i.opcode for i in ir_back]
        assert orig_ops == back_ops


# ═══════════════════════════════════════════════════════════════════
# Full round-trip: AST → IR → Commands → Decompile → AST
# ═══════════════════════════════════════════════════════════════════

class TestFullRoundTrip:
    """
    End-to-end: AST → emit → backend → decompile → AST.
    The recovered AST should be structurally equivalent to the original.
    """

    _fixtures = [
        ("rectangle", build_rectangle_solid),
        ("triangle", build_triangle_solid),
        ("circle", build_circle_solid),
        ("arc_sketch", build_arc_sketch_solid),
    ]

    @pytest.mark.parametrize("name,builder", _fixtures, ids=[f[0] for f in _fixtures])
    def test_roundtrip(self, name, builder):
        original = builder()
        ir = IREmitter().emit(original)
        cmds = DeepCADBackend().ir_to_commands(ir)
        recovered = DeepCADDecompiler().decompile(cmds)

        # The recovered AST must pass grammar validation
        r = validate_ast(recovered)
        assert r.is_valid, f"Recovered AST invalid: {r.errors}"

        # Structural comparison: same node types, same geometry
        assert recovered.node_type == original.node_type
        assert len(recovered.children) == len(original.children)

    def test_multi_solid_roundtrip(self):
        original = build_multi_solid()
        ir = IREmitter().emit(original)
        cmds = DeepCADBackend().ir_to_commands(ir)
        recovered = DeepCADDecompiler().decompile(cmds)
        r = validate_ast(recovered)
        assert r.is_valid, r.errors
        assert len(recovered.children) == 2


# ═══════════════════════════════════════════════════════════════════
# CompileValidator
# ═══════════════════════════════════════════════════════════════════

class TestCompileValidator:

    def test_valid_ast_passes(self):
        ast = build_rectangle_solid()
        result = CompileValidator().validate(ast)
        assert result.is_valid

    def test_degenerate_extrude_rejected(self):
        """Extrude with both distances mapping to ~0 is degenerate."""
        sketch = make_sketch([make_face([make_loop([
            make_edge(make_line(0, 0, 128, 0)),
            make_edge(make_line(128, 0, 0, 0)),
        ])])])
        # Q8 127 → dequantize ≈ -0.0039, Q8 128 → dequantize ≈ +0.0039
        # Both are within the 0.01 threshold
        ext = make_extrude(127, 128, "new")
        sol = make_solid(sketch, [ext])
        ast = make_program([sol])
        result = CompileValidator().validate(ast)
        assert not result.is_valid
        assert any("degenerate" in e for e in result.errors)

    def test_all_fixtures_pass_validation(self):
        for builder in [build_rectangle_solid, build_triangle_solid,
                        build_circle_solid, build_arc_sketch_solid]:
            ast = builder()
            result = CompileValidator().validate(ast)
            assert result.is_valid, f"Failed for {builder.__name__}: {result.errors}"
