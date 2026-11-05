"""成本/风险摘要单测。"""

from src.backtest.risk_summary import (
    build_cost_risk_block,
    format_cost_risk_lines,
    industry_concentration,
)


def test_industry_concentration():
    holdings = ["a", "b", "c", "d"]
    ind = {"a": "银行", "b": "银行", "c": "白酒", "d": "银行"}
    out = industry_concentration(holdings, ind)
    assert out["n_holdings"] == 4
    assert out["max_industry"] == "银行"
    assert out["max_industry_count"] == 3
    assert abs(float(out["max_industry_weight"]) - 0.75) < 1e-9


def test_build_cost_risk_block_gross_net():
    # 毛收益略高于净收益
    gross = [0.01, 0.01, -0.005]
    net = [0.008, 0.009, -0.006]
    block = build_cost_risk_block(
        net_returns=net,
        gross_returns=gross,
        avg_turnover=0.2,
        total_cost=100.0,
        last_holdings=["a", "b"],
        industry_by_code={"a": "银行", "b": "银行"},
    )
    assert "gross" in block and "net" in block
    assert float(block["cost_drag"]) > 0
    assert block["industry"]["max_industry_count"] == 2
    lines = format_cost_risk_lines(block)
    assert any("扣费前" in x for x in lines)
    assert any("最大行业" in x for x in lines)
