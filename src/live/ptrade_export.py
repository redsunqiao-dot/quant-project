"""
把 live 目标持仓导出为 PTrade 可粘贴的日频调仓策略。

说明：
- PTrade 策略多在券商云端跑，一般读不到本机 outputs/live。
- 因此每次导出把目标权重写进策略源码，上传/粘贴到 PTrade 模拟盘即可。
- 代码后缀默认转成 .SS / .SZ；若券商要 .XSHG/.XSHE，导出时用 code_style=\"xshg\"。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

LIVE_DIR = Path("./outputs/live")
HOLDINGS_FILE = LIVE_DIR / "current_holdings.json"


def to_ptrade_code(code: str, style: str = "ss_sz") -> str:
    """本仓库 code → PTrade code。"""
    raw = str(code).strip()
    if style == "xshg":
        return raw
    if raw.endswith(".XSHG") or raw.endswith(".SS"):
        return raw.split(".")[0] + ".SS"
    if raw.endswith(".XSHE") or raw.endswith(".SZ"):
        return raw.split(".")[0] + ".SZ"
    num = raw.split(".")[0]
    if num.startswith(("6", "9")):
        return num + ".SS"
    return num + ".SZ"


def weights_from_codes(codes: Sequence[str], style: str = "ss_sz") -> Dict[str, float]:
    """等权目标仓。"""
    codes = [c for c in codes if c]
    if not codes:
        return {}
    w = 1.0 / len(codes)
    return {to_ptrade_code(c, style): round(w, 6) for c in codes}


def load_target_codes(holdings_path: Path = HOLDINGS_FILE) -> List[str]:
    """读取 live 保存的目标持仓。"""
    if not holdings_path.exists():
        raise FileNotFoundError(f"未找到 {holdings_path}，请先跑 python main.py --live")
    payload = json.loads(holdings_path.read_text(encoding="utf-8"))
    codes = list(payload.get("codes", []))
    if not codes:
        raise RuntimeError("current_holdings.json 中没有 codes")
    return codes


def build_strategy_source(
    targets: Dict[str, float],
    signal_date: str = "",
    note: str = "",
) -> str:
    """生成可粘贴进 PTrade 的策略文本。"""
    targets_literal = json.dumps(targets, ensure_ascii=False, indent=4)
    signal_date = signal_date or "unknown"
    note = note or "由 quant-project live 导出"
    # 不用整段 f-string，避免与策略里大量花括号冲突
    return (
        "# -*- coding: utf-8 -*-\n"
        '"""\n'
        "PTrade 日频调仓（模拟盘）模板\n"
        "信号日: " + signal_date + "\n"
        "说明: " + note + "\n"
        "\n"
        "用法：\n"
        "1. 本机先跑: python main.py --live\n"
        "2. 打开本文件，全选复制到 PTrade「新建策略」\n"
        "3. 周期选日线，先开「模拟交易」验证\n"
        "4. 以你券商 PTrade 文档为准：若 order_target_percent 不可用，\n"
        "   可改成 order_target_value / order_target\n"
        "\n"
        "注意：\n"
        "- 代码后缀当前为 .SS/.SZ；不对就改本机导出参数后重导\n"
        "- 不会自动止损止盈，只把仓位调到 TARGETS\n"
        "- 实盘前务必先模拟盘跑稳\n"
        '"""\n'
        "\n"
        "TARGETS = " + targets_literal + "\n"
        "\n"
        "\n"
        "def initialize(context):\n"
        "    g.targets = TARGETS\n"
        "    g.universe = list(TARGETS.keys())\n"
        "    set_universe(g.universe)\n"
        '    log.info("加载目标仓 %s 只，信号日 %s" % (len(g.universe), "'
        + signal_date
        + '"))\n'
        "\n"
        "\n"
        "def before_trading_start(context, data):\n"
        "    set_universe(list(g.targets.keys()))\n"
        "\n"
        "\n"
        "def handle_data(context, data):\n"
        '    """日频：调到 TARGETS 等权；不在目标内的清仓。"""\n'
        "    targets = g.targets\n"
        "    positions = context.portfolio.positions\n"
        "    for stock in list(positions.keys()):\n"
        "        pos = positions[stock]\n"
        "        amount = getattr(pos, 'amount', None)\n"
        "        if amount is None:\n"
        "            amount = getattr(pos, 'total_amount', 0)\n"
        "        if amount and stock not in targets:\n"
        "            order_target(stock, 0)\n"
        '            log.info("清仓非目标: %s" % stock)\n'
        "\n"
        "    for stock, weight in targets.items():\n"
        "        try:\n"
        "            order_target_percent(stock, weight)\n"
        '            log.info("目标权重 %s -> %.4f" % (stock, weight))\n'
        "        except Exception as e:\n"
        '            log.error("调仓失败 %s: %s" % (stock, e))\n'
    )


def export_ptrade_files(
    codes: Optional[Sequence[str]] = None,
    signal_date: str = "",
    code_style: str = "ss_sz",
    live_dir: Path = LIVE_DIR,
) -> Dict[str, Path]:
    """
    写出 ptrade_targets.json 与 ptrade_daily_rebalance.py。
    若 codes 为空，则读 current_holdings.json。
    """
    live_dir = Path(live_dir)
    live_dir.mkdir(parents=True, exist_ok=True)

    if codes is None:
        holdings_path = live_dir / "current_holdings.json"
        codes = load_target_codes(holdings_path)
        if not signal_date and holdings_path.exists():
            signal_date = str(
                json.loads(holdings_path.read_text(encoding="utf-8")).get("date", "")
            )

    targets = weights_from_codes(codes, style=code_style)
    targets_path = live_dir / "ptrade_targets.json"
    strategy_path = live_dir / "ptrade_daily_rebalance.py"

    payload = {
        "signal_date": signal_date,
        "code_style": code_style,
        "targets": targets,
        "raw_codes": list(codes),
    }
    targets_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    strategy_path.write_text(
        build_strategy_source(targets, signal_date=signal_date),
        encoding="utf-8",
    )
    return {"targets_json": targets_path, "strategy_py": strategy_path}
