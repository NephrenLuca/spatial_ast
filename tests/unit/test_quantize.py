"""Unit tests for compiler.quantize – Q8 quantize/dequantize."""

import pytest

from compiler.quantize import quantize, dequantize, Q8_STEP


class TestQuantize:

    def test_min_value(self):
        assert quantize(-1.0) == 0

    def test_max_value(self):
        assert quantize(1.0) == 255

    def test_midpoint(self):
        assert quantize(0.0) == 128

    def test_clamp_below(self):
        assert quantize(-2.0) == 0

    def test_clamp_above(self):
        assert quantize(2.0) == 255

    def test_custom_range(self):
        assert quantize(0.0, q_min=0.0, q_max=1.0) == 0
        assert quantize(1.0, q_min=0.0, q_max=1.0) == 255
        assert quantize(0.5, q_min=0.0, q_max=1.0) == 128

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError):
            quantize(0.5, q_min=1.0, q_max=0.0)


class TestDequantize:

    def test_min_value(self):
        assert dequantize(0) == pytest.approx(-1.0)

    def test_max_value(self):
        assert dequantize(255) == pytest.approx(1.0)

    def test_midpoint(self):
        val = dequantize(128)
        assert abs(val) < 0.01

    def test_custom_range(self):
        assert dequantize(0, q_min=0.0, q_max=1.0) == pytest.approx(0.0)
        assert dequantize(255, q_min=0.0, q_max=1.0) == pytest.approx(1.0)

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError):
            dequantize(128, q_min=1.0, q_max=0.0)


class TestRoundTrip:

    def test_roundtrip_full_range(self):
        """quantize(dequantize(v)) == v for all v in [0, 255]."""
        for v in range(256):
            recovered = quantize(dequantize(v))
            assert recovered == v, f"Failed for v={v}: got {recovered}"

    def test_roundtrip_error_bounded(self):
        """dequantize(quantize(x)) should be within Q8_STEP of x."""
        import random
        random.seed(42)
        for _ in range(1000):
            x = random.uniform(-1.0, 1.0)
            recovered = dequantize(quantize(x))
            assert abs(recovered - x) <= Q8_STEP + 1e-9, (
                f"Round-trip error too large for x={x}: got {recovered}"
            )

    def test_roundtrip_custom_range(self):
        for v in range(256):
            recovered = quantize(
                dequantize(v, q_min=0.0, q_max=10.0),
                q_min=0.0, q_max=10.0,
            )
            assert recovered == v
