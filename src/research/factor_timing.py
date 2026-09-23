"""
基于滚动 IC 和 IR 的因子择时 (Factor Timing)。

Purpose
- Build a simple factor on/off switch based on rolling IC statistics.
目的
- 建立一个简单的因子开关信号：当因子表现不稳定时暂时失效，表现好时启用。

Knowledge points
- Timing uses past IC only (rolling window) to avoid look-ahead bias.
  择时必须仅使用**过去**的 IC 数据（例如过去 20 天），严禁使用未来数据。
- A positive rolling mean is necessary but not sufficient; IR adds stability.
  滚动均值 > 0 是基础，但不够；滚动 IR (均值/标准差) > 阈值 能确保因子不仅为正，而且足够稳定。
- Thresholds should be tuned and validated out-of-sample.
  阈值（如 IR > 0.2）是经验值，需要在样本外验证，防止过拟合。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.common.utils import ensure_dir


def build_timing_signal(
    ic_path: str,
    ic_col: str = "ic_spearman",
    window: int = 20,
    ir_threshold: float = 0.2,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    # The IC series is expected to come from 01_ic_calculation.py.
    # 读取步骤 1 生成的 IC 序列。
    ic_df = pd.read_csv(ic_path)
    if ic_col not in ic_df.columns:
        raise ValueError(f"Missing column {ic_col} in {ic_path}")

    ic_series = ic_df.set_index("date")[ic_col]
    
    # Rolling mean/std compute stability of the signal.
    # 计算滚动的均值和标准差
    roll_mean = ic_series.rolling(window).mean()
    roll_std = ic_series.rolling(window).std()
    
    # Rolling IR = Mean / Std. High IR implies consistent performance.
    roll_ir = roll_mean / roll_std
    
    # Factor is active only when signal is positive and stable.
    # 择时信号：仅当滚动 IC 为正且 IR 大于阈值时，认为因子当前有效 (factor_on = 1)。
    factor_on = (roll_mean > 0) & (roll_ir > ir_threshold)

    timing = pd.DataFrame(
        {
            "date": ic_series.index,
            "ic": ic_series.values,
            "ic_roll_mean": roll_mean.values,
            "ic_roll_std": roll_std.values,
            "ic_roll_ir": roll_ir.values,
            "factor_on": factor_on.astype(int).values,
        }
    )

    if output_dir:
        out = ensure_dir(output_dir)
        timing.to_csv(out / "ic_timing.csv", index=False)
    return timing


if __name__ == "__main__":
    path = "./outputs/day12_ic/ic_series.csv"
    if not Path(path).exists():
        print(f"Missing IC series file: {path}")
    else:
        df = build_timing_signal(
            path,
            ic_col="ic_spearman",
            window=20,
            ir_threshold=0.2,
            output_dir="./outputs/day12_ic",
        )
        print(df.tail())