"""
Utilities for Day12 factor evaluation and multi-factor modules.
Day12 因子评估和多因子模块的通用工具库。

Key assumptions
1) Factor files are cross-sectional tables keyed by `code` (one file per date; Parquet 优先，兼容 CSV).
2) Return files store forward returns aligned to the same date (e.g. 1vwap_pct).
3) Calculations are cross-sectional per date; time-series stats are built later.
关键假设
1) 因子文件是按日期存储的截面数据 (每行一只股票)，主键是 code；默认 Parquet，仍可读旧 CSV。
2) 收益率文件存储的是同日对齐的“未来收益”（即 T日文件存的是 T到T+1 的收益）。
3) 核心计算（IC、分组）都是每日独立的截面计算。

Knowledge points
- Alignment: always join by code to avoid information leakage.
  对齐：必须通过股票代码 (code) 严格对齐，防止数据错位导致结果无效。
- Standardization: z-score removes scale, enabling fair multi-factor aggregation.
  标准化：Z-score 去除量纲，使得不同因子（如价格和换手率）可以加权合并。
- Grouping: quantile buckets approximate long-short portfolio sorts.
  分组：分位数分桶近似于多空组合构建。
- Simplex projection: converts arbitrary weights into non-negative sum-to-1.
  单纯形投影：将任意权重向量转换为“非负且和为1”的向量。
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    # Create output directory if it does not exist.
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_dated_file(folder: Union[str, Path], date: str) -> Optional[Path]:
    """按日定位截面文件：优先 Parquet，其次 CSV。"""
    folder = Path(folder)
    parquet = folder / f"{date}.parquet"
    if parquet.exists():
        return parquet
    csv = folder / f"{date}.csv"
    if csv.exists():
        return csv
    return None


def list_dated_stems(folder: Union[str, Path]) -> List[str]:
    """列出目录下按日命名的文件日期（合并 parquet/csv）。"""
    folder = Path(folder)
    stems = {p.stem for p in folder.glob("????-??-??.parquet")}
    stems |= {p.stem for p in folder.glob("????-??-??.csv")}
    return sorted(stems)


def read_frame(path: Union[str, Path]) -> pd.DataFrame:
    """按后缀读取表格。"""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_factor_frame(
    df: pd.DataFrame,
    folder: Union[str, Path],
    date: str,
    index_as_code: bool = False,
) -> Path:
    """
    因子截面统一写成 Parquet。
    index_as_code=True 时把索引写成 code 列。
    同日若已有旧 CSV，删除以免读写混用到过期文件。
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if index_as_code and "code" not in out.columns:
        out = out.reset_index()
        if out.columns[0] != "code":
            out = out.rename(columns={out.columns[0]: "code"})
    path = folder / f"{date}.parquet"
    out.to_parquet(path, index=False)
    legacy = folder / f"{date}.csv"
    if legacy.exists():
        legacy.unlink()
    return path


def estimate_ln_circ_mktcap(daily: pd.DataFrame) -> pd.Series:
    """
    用成交额 / 换手率估算流通市值，再取对数。
    换手率为百分数（如 0.21 表示 0.21%），与 baostock turn 一致。
    """
    frame = daily.copy()
    if "code" in frame.columns:
        frame = frame.set_index("code")
    turn = pd.to_numeric(frame["turnover_ratio"], errors="coerce").replace(0, np.nan)
    money = pd.to_numeric(frame["money"], errors="coerce")
    mktcap = money / (turn / 100.0)
    return np.log(mktcap.replace(0, np.nan)).rename("ln_mktcap")


def list_available_dates(
    data_dir: Union[str, Path],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exposure_subdir: str = "data_barra",
    ret_subdir: str = "data_ret",
) -> List[str]:
    """Return dates that have both exposure and return files available."""

    data_path = Path(data_dir)
    with open(data_path / "date.pkl", "rb") as f:
        dates = pickle.load(f)

    filtered = []
    for date in dates:
        if start_date and date < start_date:
            continue
        if end_date and date > end_date:
            continue

        exposure_file = data_path / exposure_subdir / f"{date}.csv"
        ret_file = data_path / ret_subdir / f"{date}.csv"
        if exposure_file.exists() and ret_file.exists():
            filtered.append(date)

    return filtered


def load_real_panel(
    data_dir: Union[str, Path] = "./data",
    ret_col: str = "1vwap_pct",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_dates: Optional[int] = None,
) -> pd.DataFrame:
    """Build a panel of Barra exposures merged with forward returns."""

    data_path = Path(data_dir)
    barra_dir = data_path / "data_barra"
    ret_dir = data_path / "data_ret"

    frames: List[pd.DataFrame] = []
    for date in list_available_dates(data_dir, start_date, end_date):
        barra_file = barra_dir / f"{date}.csv"
        ret_file = ret_dir / f"{date}.csv"
        if not barra_file.exists() or not ret_file.exists():
            continue

        exposures = pd.read_csv(barra_file)
        if exposures.empty:
            continue
        exposures = exposures.rename(columns={"code": "asset"})
        exposures["date"] = date

        returns = pd.read_csv(ret_file)
        if ret_col not in returns.columns:
            continue
        returns = returns.rename(columns={"code": "asset", ret_col: "ret"})
        returns = returns[["asset", "ret"]]

        merged = pd.merge(exposures, returns, on="asset", how="inner")
        if merged.empty:
            continue
        merged["date"] = date
        frames.append(merged)

        if max_dates is not None and len(frames) >= max_dates:
            break

    if not frames:
        raise ValueError("No panel data could be loaded from the provided directory.")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["ret"])
    return panel


def infer_factor_columns(panel: pd.DataFrame, ret_col: str = "ret") -> List[str]:
    """Infer factor columns from a panel that contains date/asset/return columns."""

    ignore = {"date", "asset", ret_col}
    return [col for col in panel.columns if col not in ignore]


def load_dates(date_path: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[str]:
    # date.pkl is expected to be a list of sorted trading dates.
    with open(date_path, "rb") as f:
        dates = pickle.load(f)
    if start_date:
        dates = [d for d in dates if d >= start_date]
    if end_date:
        dates = [d for d in dates if d <= end_date]
    return dates


def infer_factor_cols(df: pd.DataFrame, exclude: Sequence[str] = ("code", "date")) -> List[str]:
    # Remove non-factor columns so downstream logic can be generic.
    return [col for col in df.columns if col not in set(exclude)]


def load_factor_frame(
    factor_dir: str,
    date: str,
    factor_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    # Load factor cross-section for a date and return selected factor columns.
    file = resolve_dated_file(factor_dir, date)
    if file is None:
        return pd.DataFrame()
    df = read_frame(file)
    if "code" in df.columns:
        df = df.set_index("code")
    cols = list(factor_cols) if factor_cols is not None else infer_factor_cols(df)
    if not cols:
        return pd.DataFrame()
    return df[cols].astype(float)


def load_factor_series(
    factor_dir: str,
    date: str,
    factor_col: Optional[str] = None,
) -> pd.Series:
    # Convenience wrapper: return one factor column as Series.
    # 便捷函数：只加载单列因子，返回 Series。
    df = load_factor_frame(factor_dir, date, None)
    if df.empty:
        return pd.Series(dtype=float)
    col = factor_col or df.columns[0]
    if col not in df.columns:
        return pd.Series(dtype=float)
    return df[col].astype(float)


def load_return_series(
    data_dir: str,
    date: str,
    ret_col: str = "1vwap_pct",
) -> pd.Series:
    # Returns are stored in data_ret/date.csv with forward return columns.
    file = Path(data_dir) / "data_ret" / f"{date}.csv"
    if not file.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(file)
    if "code" not in df.columns or ret_col not in df.columns:
        return pd.Series(dtype=float)
    df = df.set_index("code")
    return df[ret_col].astype(float)


def align_series(factor_series: pd.Series, ret_series: pd.Series) -> pd.DataFrame:
    # Align on code and drop missing values to avoid biased statistics.
    # 核心对齐逻辑：取交集并丢弃 NaN。
    aligned = pd.concat([factor_series, ret_series], axis=1).dropna()
    if aligned.empty:
        return pd.DataFrame()
    aligned.columns = ["factor", "ret"]
    return aligned


def zscore_series(series: pd.Series) -> pd.Series:
    # Cross-sectional z-score; if std is zero, return zeros to avoid infs.
    # 截面 Z-score: (x - mean) / std
    mean = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std):
        return series * 0.0
    return (series - mean) / std


def zscore_frame(frame: pd.DataFrame) -> pd.DataFrame:
    # Apply z-score to each factor column independently.
    return frame.apply(zscore_series, axis=0)


def calc_ic(factor_series: pd.Series, ret_series: pd.Series, method: str = "spearman") -> float:
    # IC is simply the cross-sectional correlation between factor and return.
    # IC 计算本质就是两个 Series 的相关系数。
    aligned = align_series(factor_series, ret_series)
    if aligned.empty:
        return np.nan
    return aligned["factor"].corr(aligned["ret"], method=method)


def calc_ic_stats(ic_series: pd.Series) -> dict:
    # Summarize IC with mean/std/IR/win-rate/t-stat.
    clean = ic_series.dropna()
    if clean.empty:
        return {
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "ic_ir": np.nan,
            "ic_win_rate": np.nan,
            "ic_t": np.nan,
            "n": 0,
        }
    mean = clean.mean()
    std = clean.std()
    ir = mean / std if std != 0 else np.nan
    win_rate = (clean > 0).mean()
    # t-value = mean / (std / sqrt(N))
    t_val = mean / (std / np.sqrt(len(clean))) if std != 0 else np.nan
    return {
        "ic_mean": mean,
        "ic_std": std,
        "ic_ir": ir,
        "ic_win_rate": win_rate,
        "ic_t": t_val,
        "n": len(clean),
    }


def assign_groups(values: pd.Series, n_groups: int) -> pd.Series:
    # Rank first to stabilize qcut when duplicates exist.
    # 技巧：先 Rank 再 qcut，可以有效处理大量相同值（如 0 值）导致的分组报错问题。
    ranks = values.rank(method="first")
    try:
        groups = pd.qcut(ranks, q=n_groups, labels=range(1, n_groups + 1))
    except ValueError:
        return pd.Series(index=values.index, dtype=float)
    return groups.astype(float)


def _group_returns_series(factor_series: pd.Series, ret_series: pd.Series, n_groups: int) -> Tuple[pd.Series, float]:
    # Compute group mean returns and monotonicity (Spearman vs group index).
    # 计算分组收益和单调性得分。
    aligned = align_series(factor_series, ret_series)
    if aligned.empty:
        return pd.Series(dtype=float), np.nan
    groups = assign_groups(aligned["factor"], n_groups=n_groups)
    if groups.empty:
        return pd.Series(dtype=float), np.nan

    # 聚合每组的平均收益
    grouped = aligned["ret"].groupby(groups).mean()
    grouped = grouped.reindex(range(1, n_groups + 1))

    # 单调性：组号 (1,2,3...) 与 组收益 的相关性
    monotonicity = grouped.corr(pd.Series(range(1, n_groups + 1)), method="spearman")
    return grouped, monotonicity


def _group_returns_panel(
    panel: pd.DataFrame,
    factor_col: str,
    ret_col: str = "ret",
    n_groups: int = 10,
) -> pd.DataFrame:
    """
    计算因子的分组单调性。
    将资产按因子值从小到大分成 n_groups 组，并计算每组的平均收益。
    """
    results = []
    for date, df in panel.groupby("date"):
        if df[factor_col].nunique() < n_groups:
            continue
        ranks = df[factor_col].rank(method="first")
        groups = pd.qcut(ranks, n_groups, labels=False) + 1
        grouped = df.assign(group=groups).groupby("group")[ret_col].mean()
        grouped.name = date
        results.append(grouped)
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


def group_returns(*args, **kwargs):
    """
    同时兼容第12课截面调用和第13课面板调用。
    - 截面: group_returns(factor_series, ret_series, n_groups) -> (Series, 单调性)
    - 面板: group_returns(panel, factor_col, ret_col=..., n_groups=...) -> DataFrame
    """
    first = args[0] if args else kwargs.get("panel", kwargs.get("factor_series"))
    if isinstance(first, pd.DataFrame):
        return _group_returns_panel(*args, **kwargs)
    return _group_returns_series(*args, **kwargs)


def project_to_simplex(weights: np.ndarray) -> np.ndarray:
    # Projection to simplex: non-negative weights that sum to 1.
    # 经典算法：将任意实数向量投影到单纯形上 (Euclidean Projection to Simplex)。
    # 参考文献: Duchi et al. (2008) "Efficient Projections onto the L1-Ball"
    if weights.ndim != 1:
        raise ValueError("weights must be 1d")
    if weights.size == 0:
        return weights
    sorted_w = np.sort(weights)[::-1]
    cumsum = np.cumsum(sorted_w)
    rho = np.nonzero(sorted_w * np.arange(1, len(weights) + 1) > (cumsum - 1))[0]
    if len(rho) == 0:
        return np.full_like(weights, 1.0 / len(weights))
    rho = rho[-1]
    theta = (cumsum[rho] - 1) / (rho + 1)
    projected = np.maximum(weights - theta, 0)
    if projected.sum() == 0:
        return np.full_like(weights, 1.0 / len(weights))
    return projected / projected.sum()


def window_dates(dates: Sequence[str], window: int) -> Iterable[Sequence[str]]:
    # Yield rolling windows of past dates (exclusive of the current date).
    # 生成滚动窗口生成器，注意是 "Exclusive" (不包含 T 日)，防止未来函数。
    if window <= 0:
        for _ in dates:
            yield []
        return
    for idx in range(len(dates)):
        start = max(0, idx - window)
        yield dates[start:idx]


def make_synthetic_panel(
    n_dates: int = 120,
    n_assets: int = 200,
    n_factors: int = 6,
    seed: int = 7,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    生成合成的模拟面板数据，用于算法测试。

    Returns:
        (panel_df, true_weights): 返回模拟面板和真实的因子权重。
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n_dates, freq="B")
    assets = [f"A{i:04d}" for i in range(n_assets)]
    factor_cols = [f"factor_{i + 1}" for i in range(n_factors)]

    true_weights = rng.normal(size=n_factors)
    true_weights = true_weights / np.sum(np.abs(true_weights))

    frames = []
    for date in dates:
        exposures = rng.normal(size=(n_assets, n_factors))
        noise = rng.normal(scale=0.5, size=n_assets)
        ret = exposures @ true_weights + noise

        df = pd.DataFrame(exposures, columns=factor_cols)
        df["ret"] = ret
        df["date"] = date.strftime("%Y-%m-%d")
        df["asset"] = assets
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    return panel, pd.Series(true_weights, index=factor_cols, name="true_weight")


def zscore_by_date(panel: pd.DataFrame, factor_cols: Sequence[str]) -> pd.DataFrame:
    """
    按日期进行截面 Z-Score 标准化。
    """
    out = panel.copy()

    def _zscore(series: pd.Series) -> pd.Series:
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            return series * 0.0
        return (series - series.mean()) / std

    out[list(factor_cols)] = out.groupby("date")[list(factor_cols)].transform(_zscore)
    return out


def industry_neutralize_by_date(
    panel: pd.DataFrame,
    factor_cols: Sequence[str],
    data_dir: Union[str, Path],
) -> pd.DataFrame:
    """
    按日期进行行业中性化处理（行业内 Z-Score 标准化）。
    """
    data_path = Path(data_dir)
    industry_dir = data_path / "data_industry"

    if not industry_dir.exists():
        print(f"[Warning] Industry data directory not found: {industry_dir}")
        print("[Warning] Falling back to simple zscore without industry neutralization")
        return zscore_by_date(panel, factor_cols)

    out = panel.copy()

    def _industry_zscore(series: pd.Series) -> pd.Series:
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            return series * 0.0
        return (series - series.mean()) / std

    for date in out["date"].unique():
        industry_file = industry_dir / f"{date}.csv"
        if not industry_file.exists():
            mask = out["date"] == date
            for col in factor_cols:
                out.loc[mask, col] = _industry_zscore(out.loc[mask, col])
            continue

        industry_df = pd.read_csv(industry_file)
        if "code" not in industry_df.columns or "industry" not in industry_df.columns:
            continue

        date_mask = out["date"] == date
        date_data = out[date_mask].copy()
        date_data = date_data.merge(
            industry_df[["code", "industry"]],
            left_on="asset",
            right_on="code",
            how="left"
        )

        for col in factor_cols:
            if col in date_data.columns:
                neutralized = date_data.groupby("industry")[col].transform(_industry_zscore)
                no_industry_mask = date_data["industry"].isna()
                if no_industry_mask.any():
                    neutralized[no_industry_mask] = _industry_zscore(
                        date_data.loc[no_industry_mask, col]
                    )
                out.loc[date_mask, col] = neutralized.values

    return out


def calc_ic_by_date(
    panel: pd.DataFrame, factor_col: str, ret_col: str = "ret"
) -> pd.Series:
    """
    按日期计算因子的信息系数，使用 Spearman 秩相关。
    """
    def _ic(df: pd.DataFrame) -> float:
        if df[factor_col].std(ddof=0) == 0 or df[ret_col].std(ddof=0) == 0:
            return np.nan
        return df[factor_col].corr(df[ret_col], method="spearman")

    return panel.groupby("date")[[factor_col, ret_col]].apply(_ic).dropna()


def calc_ic_summary(
    panel: pd.DataFrame, factor_col: str, ret_col: str = "ret"
) -> dict:
    """
    计算因子的 IC 均值、标准差和信息比率。
    """
    ic_series = calc_ic_by_date(panel, factor_col, ret_col)
    ic_mean = ic_series.mean() if not ic_series.empty else np.nan
    ic_std = ic_series.std(ddof=0) if not ic_series.empty else np.nan
    ic_ir = ic_mean / ic_std if ic_std and ic_std != 0 else np.nan
    return {"ic_mean": ic_mean, "ic_std": ic_std, "ic_ir": ic_ir}


def normalize_weights(raw: pd.Series) -> pd.Series:
    """
    将原始权重归一化，保证非负且总和为 1。
    """
    values = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    values = values.clip(lower=0)
    total = values.sum()
    if total == 0:
        return pd.Series(1.0 / len(values), index=values.index)
    return values / total


def project_simplex(values: Sequence[float]) -> np.ndarray:
    """
    将向量投影到概率单纯形。
    """
    v = np.asarray(values, dtype=float)
    n = v.size
    if n == 0:
        return v

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]

    if len(rho) == 0:
        theta = 0.0
    else:
        rho = rho[-1]
        theta = (cssv[rho] - 1) / (rho + 1)

    return np.maximum(v - theta, 0.0)


def cap_weights(weights: Sequence[float], cap: float) -> np.ndarray:
    """
    限制单一权重上限后重新归一化。
    """
    w = np.clip(np.asarray(weights, dtype=float), 0.0, cap)
    total = w.sum()
    if total == 0:
        return w
    return w / total


def ewma_smooth(weights: pd.DataFrame, alpha: float = 0.2) -> pd.DataFrame:
    """
    使用指数加权移动平均对权重时间序列平滑。
    """
    return weights.ewm(alpha=alpha, adjust=False).mean()


def weight_turnover(weights: pd.DataFrame) -> pd.Series:
    """
    计算相邻两期权重差的绝对值之和。
    """
    return weights.diff().abs().sum(axis=1).fillna(0.0)


def time_split_dates(
    dates: Sequence[str], train_size: int, test_size: int, step: int
) -> List[Tuple[List[str], List[str]]]:
    """
    生成滚动窗口的时间序列分割。
    """
    dates = list(dates)
    splits = []
    start = 0
    while start + train_size + test_size <= len(dates):
        train = dates[start : start + train_size]
        test = dates[start + train_size : start + train_size + test_size]
        splits.append((train, test))
        start += step
    return splits
