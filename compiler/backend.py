"""
IR -> DeepCAD command sequence backend.

Converts a flat list of ``IRInstruction`` into the dict-based DeepCAD
command format that can be serialised to JSON.  Also provides the
reverse direction (DeepCAD commands -> IR) for round-trip verification.
"""

from __future__ import annotations

from typing import Any, Dict, List

from compiler.ir import (
    BOOLEAN_OP_INV,
    EXTRUDE_OP_INV,
    IRInstruction,
)


class DeepCADBackend:
    """Bidirectional converter between IR instructions and DeepCAD commands."""

    # ── IR -> DeepCAD ───────────────────────────────────────────────

    def ir_to_commands(self, instructions: List[IRInstruction]) -> List[Dict[str, Any]]:
        """
        Convert IR instruction list to a flat list of DeepCAD command dicts.

        Sketch/loop bracket instructions are converted to composite commands
        containing their children.
        """
        commands: List[Dict[str, Any]] = []
        i = 0
        while i < len(instructions):
            instr = instructions[i]

            if instr.opcode == "sketch_start":
                sketch_cmd, i = self._collect_sketch(instructions, i)
                commands.append(sketch_cmd)

            elif instr.opcode == "extrude":
                p = instr.params
                from compiler.quantize import quantize
                commands.append({
                    "type": "extrude",
                    "distance_fwd": quantize(p[0]),
                    "distance_bwd": quantize(p[1]),
                    "op_type": EXTRUDE_OP_INV.get(p[2], "new"),
                })
                i += 1

            elif instr.opcode == "revolve":
                p = instr.params
                from compiler.quantize import quantize
                commands.append({
                    "type": "revolve",
                    "angle": quantize(p[0]),
                    "op_type": EXTRUDE_OP_INV.get(p[1], "new"),
                })
                i += 1

            elif instr.opcode == "boolean":
                commands.append({
                    "type": "boolean",
                    "op_type": BOOLEAN_OP_INV.get(instr.params[0], "union"),
                })
                i += 1

            else:
                i += 1

        return commands

    def _collect_sketch(
        self, instructions: List[IRInstruction], start: int,
    ) -> tuple:
        """Collect all instructions between sketch_start..sketch_end."""
        loops: List[Dict[str, Any]] = []
        i = start + 1  # skip sketch_start

        while i < len(instructions) and instructions[i].opcode != "sketch_end":
            if instructions[i].opcode == "loop_start":
                loop, i = self._collect_loop(instructions, i)
                loops.append(loop)
            else:
                i += 1

        if i < len(instructions):
            i += 1  # skip sketch_end

        return {"type": "sketch", "loops": loops}, i

    def _collect_loop(
        self, instructions: List[IRInstruction], start: int,
    ) -> tuple:
        """Collect curve instructions between loop_start..loop_end."""
        curves: List[Dict[str, Any]] = []
        i = start + 1  # skip loop_start

        from compiler.quantize import quantize as _q
        while i < len(instructions) and instructions[i].opcode != "loop_end":
            instr = instructions[i]
            if instr.opcode == "line":
                p = instr.params
                curves.append({
                    "type": "line",
                    "start_x": _q(p[0]), "start_y": _q(p[1]),
                    "end_x": _q(p[2]), "end_y": _q(p[3]),
                })
            elif instr.opcode == "arc":
                p = instr.params
                curves.append({
                    "type": "arc",
                    "start_x": _q(p[0]), "start_y": _q(p[1]),
                    "mid_x": _q(p[2]), "mid_y": _q(p[3]),
                    "end_x": _q(p[4]), "end_y": _q(p[5]),
                })
            elif instr.opcode == "circle":
                p = instr.params
                curves.append({
                    "type": "circle",
                    "center_x": _q(p[0]), "center_y": _q(p[1]),
                    "radius": _q(p[2]),
                })
            i += 1

        if i < len(instructions):
            i += 1  # skip loop_end

        return {"type": "loop", "curves": curves}, i

    # ── DeepCAD -> IR ───────────────────────────────────────────────

    def commands_to_ir(self, commands: List[Dict[str, Any]]) -> List[IRInstruction]:
        """Convert DeepCAD command dicts back to IR instructions."""
        instructions: List[IRInstruction] = []
        for cmd in commands:
            cmd_type = cmd["type"]

            if cmd_type == "sketch":
                instructions.append(IRInstruction("sketch_start", []))
                for loop in cmd.get("loops", []):
                    instructions.append(IRInstruction("loop_start", []))
                    for curve in loop.get("curves", []):
                        self._curve_to_ir(curve, instructions)
                    instructions.append(IRInstruction("loop_end", []))
                instructions.append(IRInstruction("sketch_end", []))

            elif cmd_type == "extrude":
                from compiler.ir import EXTRUDE_OP_MAP
                op_code = EXTRUDE_OP_MAP.get(cmd.get("op_type", "new"), 0.0)
                instructions.append(IRInstruction("extrude", [
                    cmd["distance_fwd"], cmd["distance_bwd"], op_code,
                ]))

            elif cmd_type == "revolve":
                from compiler.ir import EXTRUDE_OP_MAP
                op_code = EXTRUDE_OP_MAP.get(cmd.get("op_type", "new"), 0.0)
                instructions.append(IRInstruction("revolve", [
                    cmd["angle"], op_code,
                ]))

            elif cmd_type == "boolean":
                from compiler.ir import BOOLEAN_OP_MAP
                op_code = BOOLEAN_OP_MAP.get(cmd.get("op_type", "union"), 0.0)
                instructions.append(IRInstruction("boolean", [op_code]))

        return instructions

    @staticmethod
    def _curve_to_ir(curve: Dict[str, Any], out: List[IRInstruction]) -> None:
        ct = curve["type"]
        if ct == "line":
            out.append(IRInstruction("line", [
                curve["start_x"], curve["start_y"],
                curve["end_x"], curve["end_y"],
            ]))
        elif ct == "arc":
            out.append(IRInstruction("arc", [
                curve["start_x"], curve["start_y"],
                curve["mid_x"], curve["mid_y"],
                curve["end_x"], curve["end_y"],
            ]))
        elif ct == "circle":
            out.append(IRInstruction("circle", [
                curve["center_x"], curve["center_y"],
                curve["radius"],
            ]))
