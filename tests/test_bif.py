import math

from tools.bif import bif


def test_bif_cases():
    assert bif(3.0, 3.0)["bif"] == 1.0
    assert "inflated" in bif(4.0, 1.0)["verdict"]
    assert "SIGN FLIP" in bif(2.0, -1.0)["verdict"]
    assert math.isinf(bif(2.0, 0.0)["bif"])
    assert math.isnan(bif(float("nan"), 1.0)["bif"])
