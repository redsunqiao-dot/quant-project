"""
课上多因子研究主链路：
计算 -> 股票池过滤 -> 预处理 -> 市值/行业中性化 -> IC 评价 -> 合成
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.common.utils import (
    ensure_dir,
    infer_factor_cols,
    load_return_series,
    normalize_weights,
    resolve_dated_file,
    read_frame,
    write_factor_frame,
    list_dated_stems,
)
from src.data.data_loader import DataLoader
from src.data.universe_filter import UniverseFilter
from src.research.factor_library import FactorLibrary
from src.research.factor_neutralization import FactorNeutralizer
from src.research.factor_preprocess import FactorPreprocessor
from src.research.factor_validate import validate_factors
from src.research.factor_families import (
    family_assignment_table,
    hierarchical_weights,
)
from src.research.factor_active_state import (
    apply_active_gate,
    build_ic_wide,
    rolling_active_mask,
    summarize_active_share,
)
from src.research.multifactor_weights import build_composites, compute_factor_metrics
from src.strategy.factor_calculator import FactorCalculator


FACTOR_DIRS = {
    "raw": "raw",
    "filtered": "filtered",
    "preprocessed": "preprocessed",
    "neutralized": "neutralized",
    "industry_neutralized": "industry_neutralized",
    "composite": "composite",
}

DEFAULT_COMPOSITE_UNIVERSE = "factors/composite_universe.json"


def load_composite_universe(path: str | Path) -> Tuple[List[str], dict]:
    """
    读取冻结合成因子名单。

    日更出池只用本名单成员；增删须研究+OOS 后再改 JSON，避免每日重排。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"合成冻结名单不存在: {p}")
    meta = json.loads(p.read_text(encoding="utf-8"))
    factors = [str(x) for x in meta.get("factors", []) if str(x).strip()]
    if not factors:
        raise ValueError(f"合成冻结名单为空: {p}")
    # 去重且保持顺序
    seen = set()
    ordered: List[str] = []
    for name in factors:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered, meta


def build_rolling_ic_ir_weights(
    panel: pd.DataFrame,
    factor_cols: Sequence[str],
    dates: Sequence[str],
    lookback: int = 60,
    min_history: int = 20,
) -> pd.DataFrame:
    """
    名单固定，按日滚动 IC_IR 估权。

    实现：先算全日度 IC 宽表，再 rolling mean/std → IR，并 shift(1) 避免用到当日。
    """
    from src.research.factor_active_state import build_ic_wide

    ic_wide = build_ic_wide(panel, list(factor_cols), ret_col="ret")
    if ic_wide.empty:
        eq = 1.0 / max(len(factor_cols), 1)
        return pd.DataFrame(eq, index=[str(d) for d in dates], columns=list(factor_cols))

    # 滚动 IC 均值 / 标准差 → IR；至少 min_history 个有效点
    roll_mean = ic_wide.rolling(lookback, min_periods=min_history).mean()
    roll_std = ic_wide.rolling(lookback, min_periods=min_history).std()
    roll_ir = roll_mean / roll_std.replace(0, np.nan)
    # 合成日 t 只用到 t-1 及以前
    roll_ir = roll_ir.shift(1)

    date_index = [str(d) for d in dates]
    aligned = roll_ir.reindex(date_index)
    # 早期不足窗口：用静态整段 IC_IR 填，再 ffill
    static = {}
    for col in factor_cols:
        if col in ic_wide.columns:
            s = ic_wide[col].dropna()
            if len(s) >= 2 and s.std() > 0:
                static[col] = float(s.mean() / s.std())
            elif len(s) >= 1:
                static[col] = float(s.mean())
            else:
                static[col] = 0.0
        else:
            static[col] = 0.0
    static_w = normalize_weights(pd.Series(static))
    aligned = aligned.fillna(static_w)

    rows = []
    for date, row in aligned.iterrows():
        # 负 IR 置 0 后再归一，避免做空因子方向打翻合成
        clipped = row.astype(float).clip(lower=0.0)
        w = normalize_weights(clipped)
        if float(w.abs().sum()) <= 0:
            w = static_w.reindex(aligned.columns).fillna(0.0)
        w.name = date
        rows.append(w)
    return pd.DataFrame(rows)



def _resolve_dates(loader: DataLoader, start_date: Optional[str], end_date: Optional[str], lookback_days: int) -> List[str]:
    dates = loader.get_all_dates()
    if not dates:
        raise FileNotFoundError("没有可用交易日")
    end = end_date or dates[-1]
    if start_date:
        start = start_date
    else:
        idx = dates.index(end) if end in dates else len(dates) - 1
        start = dates[max(0, idx - lookback_days + 1)]
    return [d for d in dates if start <= d <= end]


def _build_panel(factor_dir: Path, data_dir: str, dates: List[str], ret_col: str) -> pd.DataFrame:
    """把预处理后的因子和 T+1 收益拼成评价面板。"""
    rows = []
    for date in dates:
        path = resolve_dated_file(factor_dir, date)
        if path is None:
            continue
        factor = read_frame(path)
        if "code" not in factor.columns:
            continue
        factor = factor.set_index("code")
        factor_cols = infer_factor_cols(factor)
        if not factor_cols:
            continue
        ret = load_return_series(data_dir, date, ret_col=ret_col)
        if ret.empty:
            continue
        merged = factor[factor_cols].join(ret.rename("ret"), how="inner")
        merged = merged.dropna(subset=factor_cols + ["ret"])
        if merged.empty:
            continue
        merged = merged.reset_index()
        merged["date"] = date
        merged = merged.rename(columns={"code": "asset"})
        rows.append(merged)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _write_composites(
    factor_dir: Path,
    output_dir: Path,
    dates: List[str],
    factor_cols: List[str],
    weights: pd.Series,
    daily_weights: Optional[pd.DataFrame] = None,
) -> List[str]:
    """按日把合成因子写回 factors/composite；可选逐日动态权重。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    aligned = weights.reindex(factor_cols).fillna(0.0)
    for date in dates:
        path = resolve_dated_file(factor_dir, date)
        if path is None:
            continue
        df = read_frame(path)
        if "code" not in df.columns:
            continue
        use_cols = [c for c in factor_cols if c in df.columns]
        if not use_cols:
            continue
        if daily_weights is not None and date in daily_weights.index:
            w = daily_weights.loc[date].reindex(use_cols).fillna(0.0).astype(float)
            total = float(w.sum())
            if total <= 0:
                w = aligned.reindex(use_cols).fillna(0.0)
            else:
                w = w / total
        else:
            w = aligned.reindex(use_cols).fillna(0.0)
        out = df[["code"]].copy()
        if "date" in df.columns:
            out["date"] = df["date"]
        else:
            out["date"] = date
        out["composite"] = df[use_cols].fillna(0.0).values @ w.values
        for col in use_cols:
            out[col] = df[col]
        write_factor_frame(out, output_dir, date)
        written.append(date)
    return written


def run_factor_pipeline(
    data_dir: str = "./data",
    factor_root: str = "./factors",
    output_dir: str = "./outputs/factor_research",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    lookback_days: int = 240,
    ret_col: str = "1vwap_pct",
    weight_method: str = "ic_ir",
    industry_neutralize: bool = True,
    min_abs_ic: float = 0.02,
    gate_by_registry: bool = True,
    use_family_weights: bool = False,
    use_active_state: bool = False,
    active_window: int = 20,
    active_min_ir: float = 0.10,
    composite_universe_path: Optional[str] = DEFAULT_COMPOSITE_UNIVERSE,
    use_rolling_weights: bool = False,
    rolling_lookback: int = 60,
    rolling_min_history: int = 20,
) -> Dict[str, pd.DataFrame]:
    """
    跑通课上因子研究主流程，结果写入 factors/ 和 outputs/factor_research/。
    中性化采用第九课经典流程：OLS 剔除 ln(市值) + 行业哑变量，取残差。
    管线末尾做模块 5 验证：更新 registry。

    日更合成约定：
    - 成员默认来自 factors/composite_universe.json（冻结，不每日重排）；
    - 权重默认整段 IC_IR；可选 --rolling-weights 改为滚动 IC_IR；
    - 分族估权 / 活跃态门控仍默认关闭。
    """
    data_path = Path(data_dir)
    root = Path(factor_root)
    for name in FACTOR_DIRS.values():
        (root / name).mkdir(parents=True, exist_ok=True)

    library = FactorLibrary(root / "registry.json")
    library.ensure_defaults()
    library.save()

    loader = DataLoader(data_dir)
    dates = _resolve_dates(loader, start_date, end_date, lookback_days)
    print(f"研究区间: {dates[0]} ~ {dates[-1]}，共 {len(dates)} 个交易日")

    calculator = FactorCalculator(loader)
    # 按 registry 的 candidate/active 计算，避免新登记因子不被写入 raw
    calculator.calculate_from_registry(dates[0], dates[-1], registry_path=str(root / "registry.json"))

    universe = UniverseFilter()
    universe.filter_folder(
        date_path=str(data_path / "date.pkl"),
        factor_dir=str(root / "raw"),
        data_daily_dir=str(data_path / "data_daily"),
        data_ud_dir=str(data_path / "data_ud_new"),
        output_dir=str(root / "filtered"),
        start_date=dates[0],
        end_date=dates[-1],
    )

    preprocessor = FactorPreprocessor(
        fill_method="median",
        standardize="zscore",
        winsorize=True,
        n_sigma=3.0,
        winsor_method="mad",
        n_mad=5.0,
    )
    pre_dir = root / "preprocessed"
    pre_dir.mkdir(parents=True, exist_ok=True)
    for date in list_dated_stems(root / "filtered"):
        if start_date and date < start_date:
            continue
        if end_date and date > end_date:
            continue
        path = resolve_dated_file(root / "filtered", date)
        if path is None:
            continue
        df = preprocessor.load_factor_file(path)
        processed = preprocessor.preprocess_factor_df(df)
        write_factor_frame(processed, pre_dir, date, index_as_code=True)

    score_dir = pre_dir
    industry_dir = data_path / "data_industry"
    if industry_neutralize:
        missing = [d for d in dates if not (industry_dir / f"{d}.csv").exists()]
        if missing:
            print(
                f"行业文件缺失 {len(missing)} 天（例如 {missing[0]}），"
                "跳过市值/行业中性化。请先跑: .venv/bin/python update_industry_data.py"
            )
        else:
            neu_dir = root / "neutralized"
            neutralizer = FactorNeutralizer(industry_col="industry")
            neutralizer.neutralize_folder(
                factor_dir=str(pre_dir),
                barra_dir=str(data_path / "data_barra"),
                daily_dir=str(data_path / "data_daily"),
                industry_dir=str(industry_dir),
                output_dir=str(neu_dir),
                start_date=dates[0],
                end_date=dates[-1],
            )
            score_dir = neu_dir
            print(f"市值+行业中性化完成: {neu_dir}")

    panel = _build_panel(score_dir, data_dir, dates, ret_col)
    if panel.empty:
        raise RuntimeError("因子与收益无法对齐，检查 data_ret 是否覆盖同一区间")

    factor_cols = infer_factor_cols(panel.drop(columns=["ret"], errors="ignore"))
    factor_cols = [c for c in factor_cols if c not in {"date", "asset", "index"}]
    metrics = compute_factor_metrics(panel, factor_cols, "ret", n_groups=10)

    out = ensure_dir(output_dir)
    validation = validate_factors(
        panel=panel,
        factor_cols=factor_cols,
        factor_dir=score_dir,
        dates=dates,
        library=library,
        output_dir=out / "factor_validation",
        ret_col="ret",
        n_groups=10,
        min_abs_ic=min_abs_ic,
    )
    print("因子验证（模块 5）:")
    print(validation.to_string(index=False))

    active_cols = [c for c in factor_cols if c in set(library.active_names())]

    # 合成成员：优先冻结名单；否则走 registry active 门控
    universe_meta: dict = {}
    frozen_cols: Optional[List[str]] = None
    if composite_universe_path:
        uni_path = Path(composite_universe_path)
        if uni_path.exists():
            frozen_cols, universe_meta = load_composite_universe(uni_path)
            # JSON 可覆盖滚动参数
            rolling_lookback = int(universe_meta.get("rolling_lookback", rolling_lookback))
            rolling_min_history = int(
                universe_meta.get("rolling_min_history", rolling_min_history)
            )
        else:
            print(f"警告: 未找到冻结名单 {uni_path}，回退 registry active 门控")

    if frozen_cols is not None:
        use_cols = [c for c in frozen_cols if c in factor_cols]
        missing = [c for c in frozen_cols if c not in factor_cols]
        if missing:
            print(f"警告: 冻结名单中缺列（本日跳过）: {missing}")
        if not use_cols:
            raise RuntimeError(
                f"冻结名单与面板无交集，请检查 {composite_universe_path} 与 neutralized 列"
            )
        print(
            f"合成使用冻结名单 {universe_meta.get('name', '')}: "
            f"{len(use_cols)}/{len(frozen_cols)} 个因子（不每日重排成员）"
        )
    elif gate_by_registry:
        if not active_cols:
            raise RuntimeError(
                "没有因子通过准入（IC_mean 与单调性）。"
                "可查看 outputs/factor_research/factor_validation/validation_summary.csv，"
                "或临时传 gate_by_registry=False。"
            )
        use_cols = active_cols
        print(f"合成只用 active 因子: {use_cols}")
    else:
        use_cols = factor_cols
        print(f"未启用 registry 门控，合成使用全部因子: {use_cols}")

    use_metrics = metrics.loc[use_cols]
    # 基础静态权重
    equal_w = normalize_weights(pd.Series(1.0, index=use_cols))
    ic_w = normalize_weights(use_metrics["ic_mean"])
    ic_ir_w = normalize_weights(use_metrics["ic_ir"])
    ret_w = normalize_weights(use_metrics["ret_mean"])

    family_w = hierarchical_weights(use_metrics, use_cols, metric_col="ic_ir")
    fam_table = family_assignment_table(use_cols)
    print("因子分族:")
    print(fam_table.groupby("family")["factor"].count().to_string())

    weights_by_method = {
        "equal": equal_w,
        "ic": ic_w,
        "ic_ir": ic_ir_w,
        "ret": ret_w,
        "family_ic_ir": family_w,
    }

    # 默认合成：分族估权（可关）→ 再叠加活跃态门控（可关）
    if use_family_weights and weight_method in {"ic_ir", "family_ic_ir"}:
        chosen = family_w
        chosen_name = "family_ic_ir"
    else:
        if weight_method not in weights_by_method:
            raise ValueError(f"未知加权方式: {weight_method}")
        chosen = weights_by_method[weight_method]
        chosen_name = weight_method

    daily_weights = None
    active_share = None

    # 滚动 IC_IR：名单固定、权重随日更新（日更出池推荐）
    if use_rolling_weights:
        daily_weights = build_rolling_ic_ir_weights(
            panel=panel,
            factor_cols=use_cols,
            dates=dates,
            lookback=rolling_lookback,
            min_history=rolling_min_history,
        )
        chosen_name = f"rolling_ic_ir({rolling_lookback})"
        print(
            f"滚动 IC_IR 权重: lookback={rolling_lookback}, "
            f"min_history={rolling_min_history}；覆盖 {len(daily_weights)} 日"
        )

    if use_active_state:
        ic_wide = build_ic_wide(panel, use_cols, ret_col="ret")
        active_mask = rolling_active_mask(
            ic_wide,
            window=active_window,
            min_ic_mean=0.0,
            min_ir=active_min_ir,
        )
        if daily_weights is None:
            daily_weights = apply_active_gate(chosen, active_mask, dates=dates)
            chosen_name = f"{chosen_name}+活跃态"
        else:
            # 在滚动权重上乘活跃 mask 后再归一
            mask = active_mask.reindex(
                index=daily_weights.index, columns=use_cols
            ).fillna(False)
            gated = daily_weights.where(mask, 0.0)
            daily_weights = gated.div(
                gated.sum(axis=1).replace(0, pd.NA), axis=0
            ).fillna(0.0)
            chosen_name = f"{chosen_name}+活跃态"
        active_share = summarize_active_share(active_mask)
        print(
            f"活跃态门控: window={active_window}, min_ir={active_min_ir}；"
            f"平均活跃占比={float(active_share.mean()):.2%}"
        )
        active_share.to_csv(out / "factor_active_share.csv", header=["active_share"])

    if daily_weights is not None:
        daily_weights.to_csv(out / "composite_daily_weights.csv")

    composites = build_composites(panel, use_cols, weights_by_method)
    written = _write_composites(
        score_dir,
        root / "composite",
        dates,
        use_cols,
        chosen,
        daily_weights=daily_weights,
    )

    metrics.reset_index().to_csv(out / "factor_metrics.csv", index=False)
    pd.DataFrame(weights_by_method).T.to_csv(out / "weights_by_method.csv", index_label="method")
    fam_table.to_csv(out / "factor_families.csv", index=False)
    if frozen_cols is not None:
        pd.Series(use_cols, name="factor").to_csv(
            out / "composite_universe_used.csv", index=False
        )
    for name, df in composites.items():
        df.to_csv(out / f"composite_factor_{name}.csv", index=False)

    print(f"预处理完成: {pre_dir}")
    print(f"打分目录: {score_dir}")
    print(f"注册表: {library.path}")
    print(
        f"合成因子: {root / 'composite'}，共 {len(written)} 天，"
        f"加权={chosen_name}"
        + (" +活跃态" if use_active_state else "")
    )
    print("单因子评价:")
    print(metrics.to_string())
    print("权重(静态对照):")
    print(pd.DataFrame(weights_by_method).to_string())
    return {
        "metrics": metrics,
        "weights": pd.DataFrame(weights_by_method),
        "composites": composites,
        "validation": validation,
        "daily_weights": daily_weights,
        "active_share": active_share,
        "families": fam_table,
        "use_cols": use_cols,
    }
