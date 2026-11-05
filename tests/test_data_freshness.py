"""数据新鲜度检查单测（临时目录，无真实行情）。"""

import pickle
from pathlib import Path

from src.data.data_freshness import check_data_freshness, format_report


def _write_day(folder: Path, date: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{date}.csv").write_text("code\n000001.XSHE\n", encoding="utf-8")


def test_freshness_ok(tmp_path: Path):
    cal = ["2026-01-02", "2026-01-05", "2026-01-06"]
    with open(tmp_path / "date.pkl", "wb") as f:
        pickle.dump(cal, f)
    for name in ("data_daily", "data_ud_new", "data_ret"):
        for d in cal:
            _write_day(tmp_path / name, d)

    report = check_data_freshness(tmp_path, check_optional=False)
    assert report.ok
    assert report.calendar_last == "2026-01-06"
    assert report.dir_last["data_daily"] == "2026-01-06"
    assert "ok=True" in format_report(report)


def test_freshness_missing_day(tmp_path: Path):
    cal = ["2026-01-02", "2026-01-05", "2026-01-06"]
    with open(tmp_path / "date.pkl", "wb") as f:
        pickle.dump(cal, f)
    for name in ("data_daily", "data_ud_new", "data_ret"):
        _write_day(tmp_path / name, "2026-01-02")
        _write_day(tmp_path / name, "2026-01-05")
        # 缺 2026-01-06

    report = check_data_freshness(tmp_path, check_optional=False)
    assert not report.ok
    assert any("缺" in e for e in report.errors)
