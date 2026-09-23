"""
从 current_holdings.json 导出 PTrade 模拟盘策略。

用法：
  .venv/bin/python export_ptrade_targets.py
  .venv/bin/python export_ptrade_targets.py --code-style xshg
"""

from __future__ import annotations

import argparse

from src.live.ptrade_export import export_ptrade_files


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 PTrade 日频调仓策略")
    parser.add_argument(
        "--code-style",
        choices=["ss_sz", "xshg"],
        default="ss_sz",
        help="ss_sz=600000.SS；xshg=保持 600000.XSHG",
    )
    args = parser.parse_args()
    paths = export_ptrade_files(code_style=args.code_style)
    print(f"目标权重: {paths['targets_json']}")
    print(f"PTrade策略: {paths['strategy_py']}")
    print("请将策略文件内容粘贴到 PTrade 模拟盘（日线周期）后启动。")


if __name__ == "__main__":
    main()
