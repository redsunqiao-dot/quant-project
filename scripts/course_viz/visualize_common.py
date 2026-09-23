"""Shared helpers for day13 visualization scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


from src.common.utils import ensure_dir, infer_factor_columns, load_real_panel  # type: ignore


plt.style.use("seaborn-v0_8")
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def ensure_plot_dir(base_output: str | Path, name: str) -> Path:
    """Ensure a sub directory under outputs/day13_multifactor/plots."""

    return ensure_dir(Path(base_output) / "plots" / name)


def read_csv(path: Path, **kwargs) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[Skip] Missing file: {path}")
        return None
    return pd.read_csv(path, **kwargs)


def load_panel(
    market_data_dir: str = "./data",
    ret_horizon: str = "1vwap_pct",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_dates: Optional[int] = None,
) -> Tuple[pd.DataFrame, list[str]]:
    panel = load_real_panel(
        data_dir=market_data_dir,
        ret_col=ret_horizon,
        start_date=start_date,
        end_date=end_date,
        max_dates=max_dates,
    )
    panel = panel.dropna(subset=["ret"])
    factor_cols = infer_factor_columns(panel)
    return panel, factor_cols
