"""
将研报因子登记为 candidate（不晋升 active）。

用法示例：
  .venv/bin/python register_factor.py \\
    --name cpv_ubl_wr \\
    --desc "东吴CPV：影线+威廉综合" \\
    --formula "wU*U + wB*B + wL*L + wWR*WR + wT*TREND" \\
    --source "东吴金工：上下影线，蜡烛好还是威廉好？" \\
    --hypothesis "价量形态与威廉位置刻画短期供需失衡" \\
    --windows lookback_short=5,lookback_long=20 \\
    --checklist-score 8 \\
    --impl cpv \\
    --card-path factors/research_cards/cpv_ubl_wr.md
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

from src.research.factor_library import FactorLibrary


def _parse_windows(text: str) -> Dict[str, Any]:
    """解析 lookback=20,short=5 形式。"""
    if not text.strip():
        return {}
    out: Dict[str, Any] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"windows 项缺少 = : {part}")
        key, raw = part.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        try:
            out[key] = int(raw)
        except ValueError:
            try:
                out[key] = float(raw)
            except ValueError:
                out[key] = raw
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="登记研报因子为 candidate")
    parser.add_argument("--registry", default="./factors/registry.json")
    parser.add_argument("--name", required=True, help="因子唯一名")
    parser.add_argument("--desc", required=True, help="一句话描述")
    parser.add_argument("--formula", required=True, help="公式（文本）")
    parser.add_argument("--source", default="", help="研报来源")
    parser.add_argument("--hypothesis", default="", help="经济学/行为假说")
    parser.add_argument(
        "--windows",
        default="",
        help="窗口参数，如 lookback=20,short=5,long=20",
    )
    parser.add_argument(
        "--checklist-score",
        type=int,
        default=None,
        help="模块1十项清单得分，建议 >=8 再开工",
    )
    parser.add_argument(
        "--impl",
        default="",
        help="算子键；未接线前可填占位名，计算时会报错提示先实现",
    )
    parser.add_argument("--card-path", default="", help="研报卡路径")
    parser.add_argument(
        "--data-deps",
        default="data_daily",
        help="逗号分隔数据依赖",
    )
    parser.add_argument(
        "--version",
        default="1.0",
    )
    parser.add_argument(
        "--force-status",
        default="candidate",
        choices=["candidate", "inactive", "retired"],
        help="禁止直接写成 active；晋升只能走 validate 门控",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.checklist_score is not None and args.checklist_score < 8:
        print(
            f"警告: checklist_score={args.checklist_score} < 8，"
            "按模块1标准建议暂缓实现与计算。"
        )

    lib = FactorLibrary(args.registry)
    lib.register(
        name=args.name,
        desc=args.desc,
        formula=args.formula,
        version=args.version,
        data_deps=[x.strip() for x in args.data_deps.split(",") if x.strip()],
        status=args.force_status,
        source=args.source or None,
        hypothesis=args.hypothesis or None,
        windows=_parse_windows(args.windows) or None,
        checklist_score=args.checklist_score,
        impl=args.impl or None,
        card_path=args.card_path or None,
    )
    entry = lib.get(args.name)
    print(f"已登记: {args.name} status={entry.get('status')} impl={entry.get('impl')}")
    print(f"registry: {lib.path}")


if __name__ == "__main__":
    main()
