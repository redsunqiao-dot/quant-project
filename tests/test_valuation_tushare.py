"""Tushare 估值字段映射（无网络）。"""

import pandas as pd

from src.data.update_valuation_data_tushare import basic_to_valuation


def test_basic_to_valuation_mapping():
    raw = pd.DataFrame(
        {
            "ts_code": ["600000.SH", "000001.SZ"],
            "trade_date": ["20260930", "20260930"],
            "pe_ttm": [5.1, 6.2],
            "pb": [0.5, 0.8],
            "ps_ttm": [1.1, 1.2],
            "pcf_ncf_ttm": [3.0, 4.0],
        }
    )
    out = basic_to_valuation(raw, "2026-09-30")
    assert list(out.columns) == [
        "date",
        "code",
        "pe_ttm",
        "pb_mrq",
        "ps_ttm",
        "pcf_ttm",
    ]
    assert set(out["code"]) == {"600000.XSHG", "000001.XSHE"}
    row = out.set_index("code").loc["600000.XSHG"]
    assert float(row["pb_mrq"]) == 0.5
    assert float(row["pcf_ttm"]) == 3.0
