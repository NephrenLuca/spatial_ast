"""
DeepCAD JSON → internal command list parser.

Reads the *real* DeepCAD dataset JSON format and converts it into the
normalised dict-based command list consumed by ``DeepCADDecompiler``.

DeepCAD JSON structure::

    {
        "entities": {
            "<id>": { "type": "Sketch"|"ExtrudeFeature", ... },
            ...
        },
        "properties": { "bounding_box": { "min_point": ..., "max_point": ... } },
        "sequence": [
            { "index": 0, "type": "Sketch", "entity": "<id>" },
            { "index": 1, "type": "ExtrudeFeature", "entity": "<id>" },
            ...
        ]
    }

Coordinate normalisation: all XY sketch coordinates are mapped to [0, 1]
using the model's bounding box, then quantised to Q8 [0, 255].
Extrude distances are normalised by the max bbox extent and quantised similarly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from compiler.quantize import quantize


class DeepCADParser:
    """
    Parse a single DeepCAD JSON file (or dict) into a normalised command
    list suitable for the decompiler.
    """

    def parse_file(self, path: Union[str, Path]) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return self.parse_dict(raw)

    def parse_dict(self, raw: Dict[str, Any]) -> List[Dict[str, Any]]:
        if isinstance(raw, list):
            return self._parse_flat_commands(raw)

        if "sequence" not in raw or "entities" not in raw:
            raise ValueError("Expected DeepCAD JSON with 'sequence' and 'entities' keys")

        entities = raw["entities"]
        sequence = raw["sequence"]
        bbox = raw.get("properties", {}).get("bounding_box", None)

        norm = self._build_normalizer(entities, bbox)

        commands: List[Dict[str, Any]] = []
        for step in sorted(sequence, key=lambda s: s.get("index", 0)):
            entity_id = step["entity"]
            entity = entities.get(entity_id)
            if entity is None:
                continue

            etype = entity["type"]
            if etype == "Sketch":
                parsed = self._parse_sketch(entity, norm)
                if parsed is not None:
                    commands.append(parsed)
            elif etype == "ExtrudeFeature":
                parsed = self._parse_extrude(entity, norm)
                if parsed is not None:
                    commands.append(parsed)

        return commands

    # ── normalizer ──────────────────────────────────────────────────

    def _build_normalizer(
        self, entities: Dict, bbox: Optional[Dict],
    ) -> "_CoordNormalizer":
        """Derive coordinate normalizer from bounding box or sketch coords."""
        all_xy: List[float] = []

        for ent in entities.values():
            if ent["type"] != "Sketch":
                continue
            for _pk, pv in ent.get("profiles", {}).items():
                for loop in pv.get("loops", []):
                    for c in loop.get("profile_curves", []):
                        self._collect_xy(c, all_xy)

        all_extents: List[float] = []
        for ent in entities.values():
            if ent["type"] == "ExtrudeFeature":
                e1 = _safe_extent(ent, "extent_one")
                e2 = _safe_extent(ent, "extent_two")
                all_extents.extend([abs(e1), abs(e2)])

        if bbox:
            mn = bbox.get("min_point", {})
            mx = bbox.get("max_point", {})
            xy_min = min(mn.get("x", 0), mn.get("y", 0))
            xy_max = max(mx.get("x", 0), mx.get("y", 0))
        elif all_xy:
            xy_min = min(all_xy)
            xy_max = max(all_xy)
        else:
            xy_min, xy_max = -1.0, 1.0

        extent_max = max(all_extents) if all_extents else (xy_max - xy_min)
        if extent_max < 1e-12:
            extent_max = 1.0

        return _CoordNormalizer(xy_min, xy_max, extent_max)

    @staticmethod
    def _collect_xy(curve: Dict, out: List[float]) -> None:
        for key in ("start_point", "end_point", "center_point"):
            pt = curve.get(key)
            if pt:
                out.append(pt.get("x", 0))
                out.append(pt.get("y", 0))

    # ── sketch ──────────────────────────────────────────────────────

    def _parse_sketch(self, entity: Dict, norm: "_CoordNormalizer") -> Optional[Dict]:
        profiles = entity.get("profiles", {})
        loops: List[Dict] = []

        for _pk, pv in profiles.items():
            for loop_data in pv.get("loops", []):
                curves_raw = loop_data.get("profile_curves", [])
                curves = []
                for c in curves_raw:
                    parsed = self._parse_curve(c, norm)
                    if parsed is not None:
                        curves.append(parsed)
                if curves:
                    loops.append({"type": "loop", "curves": curves})

        if not loops:
            return None
        return {"type": "sketch", "loops": loops}

    def _parse_curve(self, c: Dict, norm: "_CoordNormalizer") -> Optional[Dict]:
        ct = c.get("type", "")

        if ct == "Line3D":
            sp = c.get("start_point", {})
            ep = c.get("end_point", {})
            return {
                "type": "line",
                "start_x": norm.q_xy(sp.get("x", 0)),
                "start_y": norm.q_xy(sp.get("y", 0)),
                "end_x": norm.q_xy(ep.get("x", 0)),
                "end_y": norm.q_xy(ep.get("y", 0)),
            }

        if ct == "Arc3D":
            sp = c.get("start_point", {})
            ep = c.get("end_point", {})
            mp = self._arc_midpoint(c)
            return {
                "type": "arc",
                "start_x": norm.q_xy(sp.get("x", 0)),
                "start_y": norm.q_xy(sp.get("y", 0)),
                "mid_x": norm.q_xy(mp[0]),
                "mid_y": norm.q_xy(mp[1]),
                "end_x": norm.q_xy(ep.get("x", 0)),
                "end_y": norm.q_xy(ep.get("y", 0)),
            }

        if ct == "Circle3D":
            cp = c.get("center_point", {})
            r = c.get("radius", 0)
            return {
                "type": "circle",
                "center_x": norm.q_xy(cp.get("x", 0)),
                "center_y": norm.q_xy(cp.get("y", 0)),
                "radius": norm.q_radius(r),
            }

        return None

    @staticmethod
    def _arc_midpoint(c: Dict) -> Tuple[float, float]:
        """Compute the geometric midpoint of an Arc3D.

        Uses the center, radius, start and end angles to find the
        point at the arc's angular midpoint.
        """
        cp = c.get("center_point", {})
        cx, cy = cp.get("x", 0), cp.get("y", 0)
        r = c.get("radius", 0)
        sa = c.get("start_angle", 0)
        ea = c.get("end_angle", 0)

        if r < 1e-12:
            sp = c.get("start_point", {})
            ep = c.get("end_point", {})
            mx = (sp.get("x", 0) + ep.get("x", 0)) / 2
            my = (sp.get("y", 0) + ep.get("y", 0)) / 2
            return mx, my

        ref = c.get("reference_vector", {})
        ref_x = ref.get("x", 1.0)
        ref_y = ref.get("y", 0.0)
        base_angle = math.atan2(ref_y, ref_x)

        abs_start = base_angle + sa
        abs_end = base_angle + ea
        mid_angle = (abs_start + abs_end) / 2

        mx = cx + r * math.cos(mid_angle)
        my = cy + r * math.sin(mid_angle)
        return mx, my

    # ── extrude ─────────────────────────────────────────────────────

    def _parse_extrude(self, entity: Dict, norm: "_CoordNormalizer") -> Optional[Dict]:
        e1 = _safe_extent(entity, "extent_one")
        e2 = _safe_extent(entity, "extent_two")
        op = entity.get("operation", "NewBodyFeatureOperation")
        return {
            "type": "extrude",
            "distance_fwd": norm.q_extent(e1),
            "distance_bwd": norm.q_extent(e2),
            "op_type": self._parse_op_type(op),
        }

    @staticmethod
    def _parse_op_type(val: Any) -> str:
        mapping = {
            "NewBodyFeatureOperation": "new",
            "CutFeatureOperation": "cut",
            "JoinFeatureOperation": "join",
            "IntersectFeatureOperation": "join",
            0: "new", 1: "cut", 2: "join",
        }
        if val in mapping:
            return mapping[val]
        if isinstance(val, str) and val.lower() in ("new", "cut", "join"):
            return val.lower()
        return "new"

    # ── fallback for pre-normalised flat command lists ──────────────

    def _parse_flat_commands(self, commands: List[Dict]) -> List[Dict[str, Any]]:
        """Handle already-normalised command lists (e.g. from test fixtures)."""
        result = []
        for cmd in commands:
            ct = cmd.get("type", "")
            if ct in ("sketch", "profile"):
                loops = []
                for loop in cmd.get("loops", []):
                    curves = []
                    for c in loop.get("curves", []):
                        curves.append(c)
                    if curves:
                        loops.append({"type": "loop", "curves": curves})
                result.append({"type": "sketch", "loops": loops})
            elif ct == "extrude":
                result.append({
                    "type": "extrude",
                    "distance_fwd": cmd.get("distance_fwd", 0),
                    "distance_bwd": cmd.get("distance_bwd", 0),
                    "op_type": cmd.get("op_type", "new"),
                })
            elif ct == "revolve":
                result.append({
                    "type": "revolve",
                    "angle": cmd.get("angle", 0),
                    "op_type": cmd.get("op_type", "new"),
                })
        return result


# ── internal helpers ────────────────────────────────────────────────

class _CoordNormalizer:
    """Maps raw physical coordinates to Q8 [0, 255]."""

    def __init__(self, xy_min: float, xy_max: float, extent_max: float) -> None:
        span = xy_max - xy_min
        if span < 1e-12:
            span = 1.0
        self.xy_min = xy_min
        self.xy_span = span
        self.extent_max = extent_max

    def q_xy(self, val: float) -> int:
        normalised = (val - self.xy_min) / self.xy_span
        return quantize(normalised, q_min=0.0, q_max=1.0)

    def q_radius(self, val: float) -> int:
        normalised = abs(val) / self.xy_span
        return quantize(normalised, q_min=0.0, q_max=1.0)

    def q_extent(self, val: float) -> int:
        normalised = val / self.extent_max
        return quantize(normalised, q_min=-1.0, q_max=1.0)


def _safe_extent(entity: Dict, key: str) -> float:
    """Extract extrude distance from nested DeepCAD structure."""
    ext = entity.get(key, {})
    if isinstance(ext, dict):
        dist = ext.get("distance", {})
        if isinstance(dist, dict):
            return float(dist.get("value", 0))
        return float(dist) if dist else 0.0
    return 0.0
