"""平滑换因子单元测试（无行情依赖）。"""

import pandas as pd

from src.research.smooth_universe import (
    blend_progress,
    blend_weight_vectors,
    resolve_smooth_replace,
)


def test_blend_progress():
    assert blend_progress(0, 20) == 0.0
    assert blend_progress(10, 20) == 0.5
    assert blend_progress(20, 20) == 1.0


def test_blend_weight_vectors_sums_to_one():
    w_prev = pd.Series({"f1": 0.6, "f2": 0.4})
    w_new = pd.Series({"f2": 0.3, "f3": 0.7})
    mixed = blend_weight_vectors(w_new, w_prev, 0.5)
    assert abs(float(mixed.sum()) - 1.0) < 1e-9
    assert set(mixed.index) == {"f1", "f2", "f3"}


def test_resolve_smooth_replace_disabled():
    meta = {
        "smooth_replace": {
            "enabled": False,
            "prev": "factors/composite_universe_prev.json",
            "start_date": "2026-01-01",
        }
    }
    assert resolve_smooth_replace(meta, "factors/composite_universe.json") is None
