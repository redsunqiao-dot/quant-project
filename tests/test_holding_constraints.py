"""持仓约束单元测试（无行情依赖）。"""

from src.common.holding_constraints import apply_industry_cap, effective_top_n


def test_effective_top_n_respects_max_weight():
    assert effective_top_n(5, 0.15) == 7
    assert effective_top_n(10, 0.15) == 10
    assert effective_top_n(8, 0.0) == 8


def test_apply_industry_cap():
    codes = ["a", "b", "c", "d", "e"]
    ind = {"a": "银行", "b": "银行", "c": "银行", "d": "白酒", "e": "白酒"}
    out = apply_industry_cap(codes, ind, top_n=4, max_per_industry=2)
    assert out == ["a", "b", "d", "e"]


def test_apply_industry_cap_disabled():
    codes = ["a", "b", "c"]
    assert apply_industry_cap(codes, {}, top_n=2, max_per_industry=0) == ["a", "b"]
