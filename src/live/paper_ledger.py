"""
模拟盘账本：记录每次 paper 跑批的持仓与买卖，不自动下单。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.common.config import get_section, ROOT


def paper_dir(path: Optional[Union[str, Path]] = None) -> Path:
    """模拟盘账本目录；可显式传入，避免短窗轨读到主配置缓存。"""
    if path:
        d = Path(path)
    else:
        paths = get_section("paths")
        d = Path(paths.get("paper_dir") or "./outputs/paper")
    if not d.is_absolute():
        d = ROOT / d
    d.mkdir(parents=True, exist_ok=True)
    return d


def append_run(
    *,
    signal_date: str,
    trade_date: str,
    codes: List[str],
    buy: List[str],
    sell: List[str],
    capital: float,
    strategy: str,
    extra: Optional[Dict[str, Any]] = None,
    paper_dir_path: Optional[Union[str, Path]] = None,
) -> Path:
    """追加一条模拟盘记录，并更新 latest.json。"""
    out = paper_dir(paper_dir_path)
    record = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "signal_date": signal_date,
        "trade_date": trade_date,
        "strategy": strategy,
        "capital": capital,
        "codes": codes,
        "buy": buy,
        "sell": sell,
    }
    if extra:
        record["extra"] = extra

    day_file = out / f"ledger_{signal_date}.json"
    history: List[Dict[str, Any]] = []
    if day_file.exists():
        history = json.loads(day_file.read_text(encoding="utf-8"))
        if not isinstance(history, list):
            history = [history]
    history.append(record)
    day_file.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    latest = out / "latest.json"
    latest.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 全量追加一行到 runs.jsonl 便于检索
    runs = out / "runs.jsonl"
    with open(runs, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return day_file
