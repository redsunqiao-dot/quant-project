"""
导出大 QMT 可读的本地调仓文件 + 策略模板。

本机 --live 后生成：
- outputs/live/qmt_targets.json   （QMT 策略每天读这个）
- outputs/live/qmt_read_targets.py （粘贴进大 QMT 的策略；默认只读文件不下单）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

LIVE_DIR = Path("./outputs/live")
HOLDINGS_FILE = LIVE_DIR / "current_holdings.json"


def to_qmt_code(code: str) -> str:
    """本仓库 code → 迅投常见 600000.SH / 000001.SZ。"""
    raw = str(code).strip()
    num = raw.split(".")[0]
    if raw.endswith((".XSHG", ".SS", ".SH")) or num.startswith(("6", "9")):
        return num + ".SH"
    return num + ".SZ"


def weights_from_codes(codes: Sequence[str]) -> Dict[str, float]:
    codes = [c for c in codes if c]
    if not codes:
        return {}
    w = 1.0 / len(codes)
    return {to_qmt_code(c): round(w, 6) for c in codes}


def load_target_codes(holdings_path: Path = HOLDINGS_FILE) -> List[str]:
    if not holdings_path.exists():
        raise FileNotFoundError(f"未找到 {holdings_path}，请先跑 python main.py --live")
    payload = json.loads(holdings_path.read_text(encoding="utf-8"))
    codes = list(payload.get("codes", []))
    if not codes:
        raise RuntimeError("current_holdings.json 中没有 codes")
    return codes


def build_qmt_strategy_source(targets_abs_path: str, signal_date: str = "") -> str:
    """
    生成大 QMT 策略源码：读本地目标权重，先卖非目标、再调到目标权重。
    默认 ENABLE_ORDER=False，只打印计划；改 True 后才会真正下单。
    """
    signal_date = signal_date or "unknown"
    path_literal = targets_abs_path.replace("\\", "\\\\")
    return (
        "#coding:gbk\n"
        '"""\n'
        "大 QMT：读取本地 qmt_targets.json，按目标权重日频调仓\n"
        "信号日: " + signal_date + "\n"
        "\n"
        "买卖逻辑：\n"
        "1. 持仓不在 TARGETS 内 → 清仓（卖出）\n"
        "2. TARGETS 内股票 → 调到目标权重（买入或卖出差额）\n"
        "优先 order_target_percent；若环境无该函数则回退 passorder。\n"
        "\n"
        "使用步骤：\n"
        "1. 本机: python main.py --live  （刷新 qmt_targets.json）\n"
        "2. 把本文件 TARGETS_FILE 改成 Windows 绝对路径\n"
        "3. 填写 ACCOUNT_ID，先 ENABLE_ORDER=False 看日志\n"
        "4. 确认计划无误后再 ENABLE_ORDER=True\n"
        '"""\n'
        "\n"
        "import json\n"
        "\n"
        "TARGETS_FILE = r\"" + path_literal + "\"\n"
        "ENABLE_ORDER = False  # True 才真正下单\n"
        "ACCOUNT_ID = \"\"      # 资金账号\n"
        "LOT = 100            # 整手；科创板请按券商规则自行改\n"
        "\n"
        "\n"
        "def _load_targets():\n"
        "    with open(TARGETS_FILE, \"r\", encoding=\"utf-8\") as f:\n"
        "        payload = json.load(f)\n"
        "    return payload.get(\"targets\", {}), payload.get(\"signal_date\", \"\")\n"
        "\n"
        "\n"
        "def _pos_code(pos):\n"
        "    # 兼容不同字段名\n"
        "    code = getattr(pos, \"m_strInstrumentID\", \"\") or getattr(pos, \"stockcode\", \"\")\n"
        "    market = getattr(pos, \"m_strExchangeID\", \"\") or getattr(pos, \"market\", \"\")\n"
        "    code = str(code).strip()\n"
        "    market = str(market).strip().upper()\n"
        "    if \".\" in code:\n"
        "        return code\n"
        "    if market in (\"SH\", \"SSE\"):\n"
        "        return code + \".SH\"\n"
        "    if market in (\"SZ\", \"SZSE\"):\n"
        "        return code + \".SZ\"\n"
        "    if code.startswith((\"6\", \"9\")):\n"
        "        return code + \".SH\"\n"
        "    return code + \".SZ\"\n"
        "\n"
        "\n"
        "def _pos_volume(pos):\n"
        "    for name in (\"m_nCanUseVolume\", \"m_nVolume\", \"can_use_volume\", \"volume\"):\n"
        "        v = getattr(pos, name, None)\n"
        "        if v is not None:\n"
        "            try:\n"
        "                return int(v)\n"
        "            except Exception:\n"
        "                pass\n"
        "    return 0\n"
        "\n"
        "\n"
        "def _current_positions(account_id):\n"
        "    \"\"\"返回 {code: 可用股数}\"\"\"\n"
        "    out = {}\n"
        "    try:\n"
        "        rows = get_trade_detail_data(account_id, \"stock\", \"position\")\n"
        "    except Exception as e:\n"
        "        print(\"查询持仓失败:\", e)\n"
        "        return out\n"
        "    if not rows:\n"
        "        return out\n"
        "    for pos in rows:\n"
        "        code = _pos_code(pos)\n"
        "        vol = _pos_volume(pos)\n"
        "        if code and vol > 0:\n"
        "            out[code] = out.get(code, 0) + vol\n"
        "    return out\n"
        "\n"
        "\n"
        "def _order_target_weight(ContextInfo, code, weight):\n"
        "    \"\"\"把单票调到目标权重；weight=0 表示清仓。\"\"\"\n"
        "    # 优先高级接口\n"
        "    try:\n"
        "        order_target_percent(code, float(weight), ContextInfo, ACCOUNT_ID)\n"
        "        print(\"order_target_percent\", code, weight)\n"
        "        return\n"
        "    except Exception as e1:\n"
        "        print(\"order_target_percent 不可用，回退 passorder:\", e1)\n"
        "\n"
        "    # 回退：passorder\n"
        "    # 23=买 24=卖；1113=按总资产比例；1101=按股数；5=最新价\n"
        "    try:\n"
        "        if float(weight) <= 0:\n"
        "            positions = _current_positions(ACCOUNT_ID)\n"
        "            vol = int(positions.get(code, 0))\n"
        "            vol = vol - (vol % LOT)\n"
        "            if vol > 0:\n"
        "                passorder(24, 1101, ACCOUNT_ID, code, 5, -1, vol, 1, ContextInfo)\n"
        "                print(\"passorder 卖出\", code, vol)\n"
        "            return\n"
        "        passorder(23, 1113, ACCOUNT_ID, code, 5, -1, float(weight), 1, ContextInfo)\n"
        "        print(\"passorder 买入(总资产比例)\", code, weight)\n"
        "    except Exception as e2:\n"
        "        print(\"passorder 失败\", code, e2)\n"
        "\n"
        "\n"
        "def _plan_and_trade(ContextInfo, targets):\n"
        "    positions = _current_positions(ACCOUNT_ID) if ACCOUNT_ID else {}\n"
        "    print(\"当前持仓:\", positions)\n"
        "    print(\"目标权重:\", targets)\n"
        "\n"
        "    # 1) 不在目标内的 → 卖出\n"
        "    for code, vol in positions.items():\n"
        "        if code in targets:\n"
        "            continue\n"
        "        print(\"计划卖出(非目标)\", code, \"股数\", vol)\n"
        "        if ENABLE_ORDER:\n"
        "            _order_target_weight(ContextInfo, code, 0.0)\n"
        "\n"
        "    # 2) 目标内 → 调到权重\n"
        "    for code, weight in targets.items():\n"
        "        print(\"计划调仓\", code, \"->\", weight)\n"
        "        if ENABLE_ORDER:\n"
        "            _order_target_weight(ContextInfo, code, weight)\n"
        "\n"
        "    if not ENABLE_ORDER:\n"
        "        print(\"演练模式：仅打印计划，未下单\")\n"
        "\n"
        "\n"
        "def init(ContextInfo):\n"
        "    targets, sig = _load_targets()\n"
        "    ContextInfo.targets = targets\n"
        "    ContextInfo.signal_date = sig\n"
        "    ContextInfo.last_rebalance_day = \"\"\n"
        "    print(\"QMT读到目标仓\", len(targets), \"只, 信号日\", sig)\n"
        "    for code, w in targets.items():\n"
        "        print(\"  \", code, w)\n"
        "    # 若你的客户端支持定时，可改为每日触发一次，减少重复报单：\n"
        "    # ContextInfo.run_time(\"on_rebalance\", \"1nDay\", \"9:35:00\")\n"
        "\n"
        "\n"
        "def on_rebalance(ContextInfo):\n"
        "    \"\"\"给 run_time 用的入口（若已在 init 注册）。\"\"\"\n"
        "    targets, sig = _load_targets()\n"
        "    ContextInfo.targets = targets\n"
        "    print(\"定时调仓，信号日\", sig)\n"
        "    if ENABLE_ORDER and not ACCOUNT_ID:\n"
        "        print(\"未填写 ACCOUNT_ID，拒绝下单\")\n"
        "        return\n"
        "    _plan_and_trade(ContextInfo, targets)\n"
        "\n"
        "\n"
        "def handlebar(ContextInfo):\n"
        "    # 日线主图：只在最后一根 bar 调一次，并用日期去重\n"
        "    if not ContextInfo.is_last_bar():\n"
        "        return\n"
        "    day = \"\"\n"
        "    try:\n"
        "        day = str(ContextInfo.get_bar_timetag(ContextInfo.barpos))[:8]\n"
        "    except Exception:\n"
        "        day = str(getattr(ContextInfo, \"signal_date\", \"\"))\n"
        "    if day and day == getattr(ContextInfo, \"last_rebalance_day\", \"\"):\n"
        "        return\n"
        "    ContextInfo.last_rebalance_day = day\n"
        "\n"
        "    targets, sig = _load_targets()\n"
        "    ContextInfo.targets = targets\n"
        "    print(\"handlebar 调仓检查，信号日\", sig, \"bar日\", day)\n"
        "    if ENABLE_ORDER and not ACCOUNT_ID:\n"
        "        print(\"未填写 ACCOUNT_ID，拒绝下单\")\n"
        "        return\n"
        "    _plan_and_trade(ContextInfo, targets)\n"
    )


def export_qmt_files(
    codes: Optional[Sequence[str]] = None,
    signal_date: str = "",
    live_dir: Path = LIVE_DIR,
    project_root: Optional[Path] = None,
) -> Dict[str, Path]:
    """写出 qmt_targets.json 与 qmt_read_targets.py。"""
    live_dir = Path(live_dir).resolve()
    live_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(project_root).resolve() if project_root else live_dir.parent.parent

    if codes is None:
        holdings_path = live_dir / "current_holdings.json"
        codes = load_target_codes(holdings_path)
        if not signal_date and holdings_path.exists():
            signal_date = str(
                json.loads(holdings_path.read_text(encoding="utf-8")).get("date", "")
            )

    targets = weights_from_codes(codes)
    targets_path = live_dir / "qmt_targets.json"
    strategy_path = live_dir / "qmt_read_targets.py"

    payload = {
        "signal_date": signal_date,
        "code_style": "sh_sz",
        "targets": targets,
        "raw_codes": list(codes),
    }
    targets_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    abs_targets = str(targets_path.resolve())
    strategy_path.write_text(
        build_qmt_strategy_source(abs_targets, signal_date=signal_date),
        encoding="utf-8",
    )
    return {"targets_json": targets_path, "strategy_py": strategy_path}
