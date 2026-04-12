"""
Q8 quantisation / dequantisation utilities.

Maps a continuous value in ``[q_min, q_max]`` (default ``[-1, 1]``) to an
8-bit integer in ``[0, 255]`` and back.  Round-trip error is bounded by
``(q_max - q_min) / 255 ≈ 0.00784`` for the default range.
"""

from __future__ import annotations


def quantize(
    value: float,
    q_min: float = -1.0,
    q_max: float = 1.0,
) -> int:
    """Quantise a float to an 8-bit integer in [0, 255]."""
    if q_max <= q_min:
        raise ValueError(f"q_max ({q_max}) must be > q_min ({q_min})")
    normalised = (value - q_min) / (q_max - q_min)
    return int(round(max(0.0, min(255.0, normalised * 255.0))))


def dequantize(
    q8_value: int,
    q_min: float = -1.0,
    q_max: float = 1.0,
) -> float:
    """Dequantise an 8-bit integer back to a float in [q_min, q_max]."""
    if q_max <= q_min:
        raise ValueError(f"q_max ({q_max}) must be > q_min ({q_min})")
    return q_min + (q_max - q_min) * q8_value / 255.0


# Convenience aliases used throughout the compiler / data pipeline.
Q8_MIN = 0
Q8_MAX = 255
Q8_RANGE_MIN = -1.0
Q8_RANGE_MAX = 1.0
Q8_STEP = (Q8_RANGE_MAX - Q8_RANGE_MIN) / 255.0
