#!/usr/bin/env python3
"""短窗轨模拟盘入口：lookback=60 → factors/composite_st，不覆盖主合成分。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_paper_pipeline import main as paper_main


def main() -> int:
    # 注入短窗配置；允许用户再追加 --skip-pipeline 等参数
    if "--config" not in sys.argv:
        sys.argv[1:1] = ["--config", str(ROOT / "configs" / "paper_st.json")]
    return paper_main()


if __name__ == "__main__":
    raise SystemExit(main())
