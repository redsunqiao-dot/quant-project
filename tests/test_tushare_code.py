"""Tushare 代码转换单测。"""

from src.data.update_market_data_tushare import from_ts_code, to_ts_code


def test_to_from_ts_code_roundtrip():
    assert to_ts_code("600000.XSHG") == "600000.SH"
    assert to_ts_code("000001.XSHE") == "000001.SZ"
    assert from_ts_code("600000.SH") == "600000.XSHG"
    assert from_ts_code("000001.SZ") == "000001.XSHE"


def test_from_ts_code_bare_symbol():
    assert from_ts_code("600000") == "600000.XSHG"
    assert from_ts_code("000001") == "000001.XSHE"
