"""
导出大 QMT 本地调仓文件与演练策略。

用法：
  .venv/bin/python export_qmt_targets.py
"""

from __future__ import annotations

from src.live.qmt_export import export_qmt_files


def main() -> None:
    paths = export_qmt_files()
    print(f"目标文件: {paths['targets_json'].resolve()}")
    print(f"策略模板: {paths['strategy_py'].resolve()}")
    print("请将策略模板粘贴进大 QMT；先保持 ENABLE_ORDER=False 演练读文件。")


if __name__ == "__main__":
    main()
