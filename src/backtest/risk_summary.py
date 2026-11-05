"""
回测风险与成本摘要：扣费前/后净值、换手、行业集中度。

不依赖完整行情；供 BacktestEngine 报告与单测复用。
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.backtest.performance_evaluator import PerformanceEvaluator


def nav_metrics(
    daily_returns: Sequence[float],
    risk_free_rate: float = 0.03,
) -> Dict[str, float]:
    """由日收益序列生成关键绩效指标。"""
    if not daily_returns:
        return {
            "total_return": float("nan"),
            "annual_return": float("nan"),
            "sharpe_ratio": float("nan"),
            "max_drawdown": float("nan"),
        }
    rets = pd.Series(list(daily_returns), dtype=float)
    cum = (1.0 + rets).cumprod()
    ev = PerformanceEvaluator(risk_free_rate=risk_free_rate)
    return ev.generate_report(cum, rets)


def industry_concentration(
    holdings: Sequence[str],
    industry_by_code: Mapping[str, str],
) -> Dict[str, float | int | str]:
    """
    等权假设下的行业集中度。

    返回：持股数、行业数、最大行业只数/占比、最大行业名。
    """
    codes = [str(c) for c in holdings]
    n = len(codes)
    if n == 0:
        return {
            "n_holdings": 0,
            "n_industries": 0,
            "max_industry_count": 0,
            "max_industry_weight": 0.0,
            "max_industry": "",
        }
    labels = []
    for c in codes:
        ind = industry_by_code.get(c)
        if ind is None or str(ind).strip() == "":
            labels.append("__NA__")
        else:
            labels.append(str(ind))
    counts = Counter(labels)
    max_ind, max_cnt = counts.most_common(1)[0]
    return {
        "n_holdings": n,
        "n_industries": len(counts),
        "max_industry_count": int(max_cnt),
        "max_industry_weight": float(max_cnt) / float(n),
        "max_industry": max_ind,
    }


def build_cost_risk_block(
    *,
    net_returns: Sequence[float],
    gross_returns: Optional[Sequence[float]] = None,
    avg_turnover: float = 0.0,
    total_cost: float = 0.0,
    last_holdings: Optional[Sequence[str]] = None,
    industry_by_code: Optional[Mapping[str, str]] = None,
    risk_free_rate: float = 0.03,
) -> Dict[str, object]:
    """组装报告用的成本与风险块。"""
    net = nav_metrics(net_returns, risk_free_rate=risk_free_rate)
    out: Dict[str, object] = {
        "net": net,
        "avg_turnover": float(avg_turnover) if avg_turnover == avg_turnover else 0.0,
        "total_cost": float(total_cost),
    }
    if gross_returns is not None and len(gross_returns) == len(net_returns):
        gross = nav_metrics(gross_returns, risk_free_rate=risk_free_rate)
        out["gross"] = gross
        if not np.isnan(gross["total_return"]) and not np.isnan(net["total_return"]):
            out["cost_drag"] = float(gross["total_return"] - net["total_return"])
        else:
            out["cost_drag"] = float("nan")
    if last_holdings is not None:
        out["industry"] = industry_concentration(
            last_holdings, industry_by_code or {}
        )
    return out


def format_cost_risk_lines(block: Mapping[str, object]) -> List[str]:
    """打印用行列表（无 emoji）。"""
    lines: List[str] = []
    net = block.get("net") or {}
    gross = block.get("gross")
    if isinstance(gross, dict) and isinstance(net, dict):
        lines.append(
            f"扣费前总收益:   {float(gross.get('total_return', float('nan')))*100:>10.2f}%"
        )
        lines.append(
            f"扣费后总收益:   {float(net.get('total_return', float('nan')))*100:>10.2f}%"
        )
        drag = block.get("cost_drag", float("nan"))
        lines.append(f"成本拖累:       {float(drag)*100:>10.2f}%")
        lines.append(
            f"扣费前夏普:     {float(gross.get('sharpe_ratio', float('nan'))):>10.2f}"
        )
        lines.append(
            f"扣费后夏普:     {float(net.get('sharpe_ratio', float('nan'))):>10.2f}"
        )
    lines.append(f"平均换手:       {float(block.get('avg_turnover', 0.0))*100:>10.2f}%")
    lines.append(f"累计交易成本:   {float(block.get('total_cost', 0.0)):>10.2f}")
    ind = block.get("industry")
    if isinstance(ind, dict) and int(ind.get("n_holdings", 0) or 0) > 0:
        lines.append(
            f"末仓持股/行业:  {int(ind['n_holdings']):>4d} / {int(ind['n_industries']):d}"
        )
        lines.append(
            f"最大行业集中:   {ind.get('max_industry', '')} "
            f"{int(ind.get('max_industry_count', 0))}只 "
            f"({float(ind.get('max_industry_weight', 0.0))*100:.1f}%)"
        )
    return lines
