"""
因子面板分发：把各研报/批次算子接到统一 series_map。

factor_calculator 只负责读行情与落盘；具体公式实现见 src.research.impl。
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pandas as pd

from src.common.utils import estimate_ln_circ_mktcap
from src.research.factor_ops import resolve_op
from src.research.impl.amihud_illiq import compute_illiq_panels, illiq_lookback
from src.research.impl.cutting_ctr_factors import (
    compute_cutting_ctr_panels,
    cutting_ctr_lookback,
)
from src.research.impl.cj_highvol_factors import (
    cj_highvol_lookback,
    compute_cj_highvol_panels,
)
from src.research.impl.cjx_overnight_intraday import (
    cjx_oi_lookback,
    compute_cjx_oi_panels,
)
from src.research.impl.daily_batch4_factors import (
    batch4_lookback,
    compute_dw_pct_turn_panels,
    compute_fz_bias_panels,
    compute_fz_rev_vol_flip_panels,
    compute_ky_ideal_amp_panels,
)
from src.research.impl.daily_batch5_factors import (
    batch5_lookback,
    compute_fz_rev_turn_flip_panels,
    compute_sw_dir_vol_panels,
)
from src.research.impl.daily_batch6_factors import batch6_lookback, compute_batch6_panels
from src.research.impl.daily_batch7_factors import batch7_lookback, compute_batch7_panels
from src.research.impl.daily_batch8_factors import batch8_lookback, compute_batch8_panels
from src.research.impl.daily_batch9_factors import batch9_lookback, compute_batch9_panels
from src.research.impl.daily_batch10_factors import (
    batch10_lookback,
    compute_batch10_panels,
)
from src.research.impl.daily_batch11_factors import (
    batch11_lookback,
    compute_batch11_panels,
)
from src.research.impl.daily_batch13_factors import (
    batch13_lookback,
    compute_batch13_panels,
)
from src.research.impl.gfn_factors import compute_gfn_example_panels, gfn_lookback
from src.research.impl.gs_hl_vol_surge import (
    compute_gs_hl_vol_panels,
    gs_hl_vol_lookback,
)
from src.research.impl.module6_factors import module6_lookback
from src.research.impl.ha_turn_acc_factors import (
    compute_ha_turn_acc_panels,
    ha_turn_acc_lookback,
)
from src.research.impl.overnight_gap_factors import (
    compute_overnight_gap_panels,
    overnight_gap_lookback,
)
from src.research.impl.ubl_factor import compute_ubl_by_date, ubl_lookback
from src.research.impl.xb_overnight_factors import (
    compute_xb_overnight_panels,
    xb_overnight_lookback,
)
from src.research.impl.zy_anchor_reversal import (
    compute_zy_anchor_panels,
    zy_anchor_lookback,
)
from src.research.impl.scan2026_factors import (
    compute_chip_cost_panels,
    compute_path_convex_panels,
    scan2026_lookback,
)
from src.research.impl.orgscan_factors import (
    compute_gs_near_high_panels,
    compute_gs_rsi_panels,
    compute_ky_ideal_turn_panels,
    compute_fz_moth_panels,
    compute_hx_pv_rev_panels,
    compute_xb_toi_panels,
    compute_zs_atv_panels,
    compute_zs_patv_panels,
    compute_zs_tug_panels,
    orgscan_lookback,
)


# 算子键集合（与 factor_ops.WIRED_OPS 对齐）
GFN_OPS = frozenset({"gfn_logvol_var10", "gfn_volpct_max20", "gfn_sqrt_open_low_vol"})
CJ_OPS = frozenset(
    {
        "cj_ret_skew_20",
        "cj_mom_240_skip20",
        "cj_rank_mom_220",
        "cj_corr_vol_close_20",
        "cj_turn_cv_20",
    }
)
XB_OPS = frozenset({"xb_vol_shock", "xb_amplitude", "xb_intraday"})
GAP_OPS = frozenset({"fz_jump_10", "gs_ovn_turn_corr_20"})
HA_OPS = frozenset({"ha_volup_turn_acc_20"})
ILLIQ_OPS = frozenset({"zt_illiq_20"})
CJX_OPS = frozenset(
    {
        "cjx_intraday_rev_20",
        "cjx_overnight_mom_20",
        "cjx_overnight_vol_20",
        "cjx_oi_spread_20",
    }
)
ZY_OPS = frozenset({"zy_anchor_rev", "zy_anchor_rev_vol"})
B4_OPS = frozenset(
    {"fz_bias_60", "dw_pct_turn_20", "ky_ideal_amp_20", "fz_rev_vol_flip_20"}
)
B5_OPS = frozenset({"sw_dir_vol_diff_40", "sw_mod_skew_40", "fz_rev_turn_flip_20"})
B6_OPS = frozenset({"df_maxret_20", "xn_idio_vol_20", "pk_range_vol_20"})
B7_OPS = frozenset({"ms_dastd_60", "ms_rank_vol_60", "xn_same_wd_mom_12"})
B8_OPS = frozenset({"ky_long_mom_160", "ky_long_mom2_160", "fz_panic_ret_20"})
B9_OPS = frozenset({"db_csk_xyy_120", "db_csk_down_120", "df_ext_rev_20"})
B10_OPS = frozenset({"db_csk_up_120", "db_csk_comp_120", "gk_vol_20"})
B11_OPS = frozenset(
    {
        "rs_vol_20",
        "semi_down_20",
        "xn_same_wd_rev_12",
        "ct_resid_mom_60",
        "df_mild_mom_20",
        "ovn_rev_20",
        "idt_rev_20",
        "fz_lt_mom_60",
    }
)
B13_OPS = frozenset({"yz_vol_20", "ret_kurt_60", "pv_corr_vol_20"})
GS_HL_OPS = frozenset(
    {
        "gs_low_vol_evt_5",
        "gs_high_vol_evt_5",
        "gs_hl_vol_net_5",
        "gs_low_vol_soft_120",
        "gs_high_vol_soft_120",
        "gs_hl_vol_soft_120",
    }
)
M6_OPS = frozenset(
    {"m6_ep_ttm", "m6_bp_mrq", "m6_roe", "m6_yoy_ni", "m6_peg_inv", "m6_dy"}
)
CTR_OPS = frozenset(
    {
        "ctr_turn_spread_20",
        "ctr_vol_cut_20",
        "ctr_ideal_amp_rev_20",
    }
)
SCAN26_OPS = frozenset({"ha_path_convex_20", "sw_chip_cost_60"})
SCAN26_FUND_OPS = frozenset({"yd_gp_delta"})
ORGSCAN_OPS = frozenset(
    {
        "gs_rsi_20",
        "zs_atv_20",
        "zs_patv_d20",
        "zs_tug_nr_20",
        "zs_tug_pr_20",
        "zs_yuli_20",
        "ky_ideal_turn_20",
        "gs_near_high_252",
        "fz_taylor_jump_20",
        "fz_mod_amp_20",
        "hx_pv_rev_20",
        "xb_toi_20",
    }
)
ORGSCAN_VOL_OPS = frozenset({"zs_atv_20", "zs_patv_d20", "hx_pv_rev_20", "xb_toi_20"})
ORGSCAN_OHLC_OPS = frozenset(
    {
        "zs_tug_nr_20",
        "zs_tug_pr_20",
        "zs_yuli_20",
        "gs_near_high_252",
        "fz_taylor_jump_20",
        "fz_mod_amp_20",
        "xb_toi_20",
    }
)


def _period_from_name(name: str, fallback: int) -> int:
    try:
        return int(name.rsplit("_", 1)[-1])
    except ValueError:
        return fallback


def _pick(selected: Sequence[str], ops: frozenset) -> List[str]:
    return [n for n in selected if resolve_op(n) in ops]


def _assign(
    series_map: Dict[str, pd.DataFrame],
    names: Sequence[str],
    panels: Dict[str, pd.DataFrame],
    label: str,
) -> None:
    for name in names:
        if name not in panels:
            raise RuntimeError(f"未实现的{label}因子: {name}")
        series_map[name] = panels[name]


def required_lookback(
    selected: Sequence[str],
    *,
    momentum_period: int = 20,
    reversal_period: int = 5,
    turnover_window: int = 20,
    ubl_z_window: int = 10,
    ubl_feat_window: int = 20,
    ubl_wms_n: int = 20,
) -> Tuple[int, bool]:
    """
    返回 (历史回看天数, 是否需要 OHLC/量额列)。
    """
    mom = [n for n in selected if resolve_op(n) == "momentum"]
    rev = [n for n in selected if resolve_op(n) == "reversal"]
    turn = [n for n in selected if resolve_op(n) == "turn_stable"]
    ubl = [n for n in selected if resolve_op(n) == "ubl"]

    periods = [_period_from_name(n, momentum_period) for n in mom]
    periods += [_period_from_name(n, reversal_period) for n in rev]
    periods += [_period_from_name(n, turnover_window) for n in turn]
    lookback = max(periods or [1])

    groups = [
        (ubl, ubl_lookback(ubl_z_window, ubl_feat_window, ubl_wms_n)),
        (_pick(selected, GFN_OPS), gfn_lookback()),
        (_pick(selected, CJ_OPS), cj_highvol_lookback()),
        (_pick(selected, XB_OPS), xb_overnight_lookback()),
        (_pick(selected, GAP_OPS), overnight_gap_lookback()),
        (_pick(selected, HA_OPS), ha_turn_acc_lookback()),
        (_pick(selected, ILLIQ_OPS), illiq_lookback()),
        (_pick(selected, CJX_OPS), cjx_oi_lookback()),
        (_pick(selected, ZY_OPS), zy_anchor_lookback()),
        (_pick(selected, B4_OPS), batch4_lookback()),
        (_pick(selected, B5_OPS), batch5_lookback()),
        (_pick(selected, B6_OPS), batch6_lookback()),
        (_pick(selected, B7_OPS), batch7_lookback()),
        (_pick(selected, B8_OPS), batch8_lookback()),
        (_pick(selected, B9_OPS), batch9_lookback()),
        (_pick(selected, B10_OPS), batch10_lookback()),
        (_pick(selected, B11_OPS), batch11_lookback()),
        (_pick(selected, B13_OPS), batch13_lookback()),
        (_pick(selected, GS_HL_OPS), gs_hl_vol_lookback()),
        (_pick(selected, M6_OPS), module6_lookback()),
        (_pick(selected, CTR_OPS), cutting_ctr_lookback()),
        (_pick(selected, SCAN26_OPS), scan2026_lookback()),
        (_pick(selected, SCAN26_FUND_OPS), 65),
        (_pick(selected, ORGSCAN_OPS), orgscan_lookback()),
    ]
    need_ohlc = bool(ubl)
    for names, lb in groups:
        if names:
            lookback = max(lookback, lb)
            need_ohlc = True

    # 仅 close/turn 的基础三因子不需要 OHLC
    if mom or rev or turn:
        pass
    # b4/b5/b7/b9 中部分只靠 close/turn，但仍可能要 OHLC（ky_ideal）
    # 上面 groups 已把这些算子标成 need_ohlc；与旧逻辑一致（旧逻辑对 b4/b5 也置 need_ohlc）
    return lookback, need_ohlc


def build_series_map(
    selected: Sequence[str],
    *,
    panel: pd.DataFrame,
    close: pd.DataFrame,
    turn: pd.DataFrame,
    target_dates: Sequence[str],
    momentum_period: int = 20,
    reversal_period: int = 5,
    turnover_window: int = 20,
    ubl_z_window: int = 10,
    ubl_feat_window: int = 20,
    ubl_wms_n: int = 20,
) -> Dict[str, pd.DataFrame]:
    """按 selected 计算各因子面板（index=date, columns=code）。"""
    mom_names = [n for n in selected if resolve_op(n) == "momentum"]
    rev_names = [n for n in selected if resolve_op(n) == "reversal"]
    turn_names = [n for n in selected if resolve_op(n) == "turn_stable"]
    ubl_names = [n for n in selected if resolve_op(n) == "ubl"]
    gfn_names = _pick(selected, GFN_OPS)
    cj_names = _pick(selected, CJ_OPS)
    xb_names = _pick(selected, XB_OPS)
    gap_names = _pick(selected, GAP_OPS)
    ha_names = _pick(selected, HA_OPS)
    illiq_names = _pick(selected, ILLIQ_OPS)
    cjx_names = _pick(selected, CJX_OPS)
    zy_names = _pick(selected, ZY_OPS)
    b4_names = _pick(selected, B4_OPS)
    b5_names = _pick(selected, B5_OPS)
    b6_names = _pick(selected, B6_OPS)
    b7_names = _pick(selected, B7_OPS)
    b8_names = _pick(selected, B8_OPS)
    b9_names = _pick(selected, B9_OPS)
    b10_names = _pick(selected, B10_OPS)
    b11_names = _pick(selected, B11_OPS)
    b13_names = _pick(selected, B13_OPS)
    gs_hl_names = _pick(selected, GS_HL_OPS)
    ctr_names = _pick(selected, CTR_OPS)
    scan26_names = _pick(selected, SCAN26_OPS)
    orgscan_names = _pick(selected, ORGSCAN_OPS)

    series_map: Dict[str, pd.DataFrame] = {}
    for name in mom_names:
        p = _period_from_name(name, momentum_period)
        series_map[name] = close / close.shift(p) - 1.0
    for name in rev_names:
        p = _period_from_name(name, reversal_period)
        series_map[name] = -(close / close.shift(p) - 1.0)
    for name in turn_names:
        p = _period_from_name(name, turnover_window)
        series_map[name] = -turn.rolling(p, min_periods=p).std()

    if zy_names:
        _assign(series_map, zy_names, compute_zy_anchor_panels(close), "中银锚定反转")

    if b4_names:
        if "fz_bias_60" in b4_names:
            series_map["fz_bias_60"] = compute_fz_bias_panels(close)["fz_bias_60"]
        if "dw_pct_turn_20" in b4_names:
            series_map["dw_pct_turn_20"] = compute_dw_pct_turn_panels(turn)[
                "dw_pct_turn_20"
            ]
        if "fz_rev_vol_flip_20" in b4_names:
            series_map["fz_rev_vol_flip_20"] = compute_fz_rev_vol_flip_panels(close)[
                "fz_rev_vol_flip_20"
            ]

    if b5_names:
        if "sw_dir_vol_diff_40" in b5_names or "sw_mod_skew_40" in b5_names:
            sw_panels = compute_sw_dir_vol_panels(close)
            for name in b5_names:
                if name in sw_panels:
                    series_map[name] = sw_panels[name]
        if "fz_rev_turn_flip_20" in b5_names:
            series_map["fz_rev_turn_flip_20"] = compute_fz_rev_turn_flip_panels(
                close, turn
            )["fz_rev_turn_flip_20"]

    if b7_names:
        _assign(series_map, b7_names, compute_batch7_panels(close), "日频批次7")

    if b9_names:
        _assign(series_map, b9_names, compute_batch9_panels(close), "日频批次9")

    if scan26_names and "ha_path_convex_20" in scan26_names:
        series_map["ha_path_convex_20"] = compute_path_convex_panels(close)[
            "ha_path_convex_20"
        ]
    if orgscan_names and "gs_rsi_20" in orgscan_names:
        series_map["gs_rsi_20"] = compute_gs_rsi_panels(close)["gs_rsi_20"]
    if orgscan_names and "ky_ideal_turn_20" in orgscan_names:
        series_map["ky_ideal_turn_20"] = compute_ky_ideal_turn_panels(close, turn)[
            "ky_ideal_turn_20"
        ]

    need_ohlc_block = bool(
        ubl_names
        or gfn_names
        or cj_names
        or xb_names
        or gap_names
        or ha_names
        or illiq_names
        or cjx_names
        or ("ky_ideal_amp_20" in b4_names)
        or b6_names
        or b8_names
        or b10_names
        or b11_names
        or b13_names
        or gs_hl_names
        or ctr_names
        or ("sw_chip_cost_60" in scan26_names)
        or bool(_pick(orgscan_names, ORGSCAN_VOL_OPS))
        or bool(_pick(orgscan_names, ORGSCAN_OHLC_OPS))
    )
    if not need_ohlc_block:
        return series_map

    open_ = panel.pivot_table(index="date", columns="code", values="open").sort_index()
    high = panel.pivot_table(index="date", columns="code", values="high").sort_index()
    low = panel.pivot_table(index="date", columns="code", values="low").sort_index()
    volume = panel.pivot_table(index="date", columns="code", values="volume").sort_index()
    money = panel.pivot_table(index="date", columns="code", values="money").sort_index()

    if ubl_names:
        size_rows = []
        for date, day in panel.groupby("date"):
            ln = estimate_ln_circ_mktcap(day)
            ln.name = date
            size_rows.append(ln)
        size_panel = pd.DataFrame(size_rows).sort_index()
        ubl_mat = compute_ubl_by_date(
            open_,
            high,
            low,
            close,
            size_panel,
            dates=list(target_dates),
            z_window=ubl_z_window,
            feat_window=ubl_feat_window,
            wms_n=ubl_wms_n,
            larger_better=True,
        )
        for name in ubl_names:
            series_map[name] = ubl_mat

    if gfn_names:
        _assign(
            series_map,
            gfn_names,
            compute_gfn_example_panels(open_, low, volume),
            "GFN",
        )
    if cj_names:
        _assign(
            series_map,
            cj_names,
            compute_cj_highvol_panels(close, volume, turn),
            "长江高波",
        )
    if xb_names:
        _assign(
            series_map,
            xb_names,
            compute_xb_overnight_panels(open_, high, low, close, volume),
            "西部隔夜",
        )
    if gap_names:
        _assign(
            series_map,
            gap_names,
            compute_overnight_gap_panels(open_, close, turn),
            "隔夜跳空",
        )
    if ha_names:
        _assign(
            series_map,
            ha_names,
            compute_ha_turn_acc_panels(close, volume, turn),
            "华安加速换手",
        )
    if illiq_names:
        _assign(
            series_map,
            illiq_names,
            compute_illiq_panels(close, money),
            "ILLIQ",
        )
    if cjx_names:
        _assign(
            series_map,
            cjx_names,
            compute_cjx_oi_panels(open_, close),
            "建投隔夜日内",
        )
    if "ky_ideal_amp_20" in b4_names:
        series_map["ky_ideal_amp_20"] = compute_ky_ideal_amp_panels(high, low, close)[
            "ky_ideal_amp_20"
        ]
    if b6_names:
        _assign(
            series_map,
            b6_names,
            compute_batch6_panels(open_, high, low, close),
            "日频批次6",
        )
    if b8_names:
        _assign(
            series_map,
            b8_names,
            compute_batch8_panels(high, low, close),
            "日频批次8",
        )
    if b10_names:
        _assign(
            series_map,
            b10_names,
            compute_batch10_panels(open_, high, low, close),
            "日频批次10",
        )
    if b11_names:
        _assign(
            series_map,
            b11_names,
            compute_batch11_panels(open_, high, low, close, turn),
            "日频批次11",
        )
    if b13_names:
        _assign(
            series_map,
            b13_names,
            compute_batch13_panels(open_, high, low, close, volume),
            "日频批次13",
        )
    if gs_hl_names:
        _assign(
            series_map,
            gs_hl_names,
            compute_gs_hl_vol_panels(close, volume),
            "国盛高低位放量",
        )
    if ctr_names:
        _assign(
            series_map,
            ctr_names,
            compute_cutting_ctr_panels(close, turn, volume, high=high, low=low),
            "切割轨 CTR",
        )
    if "sw_chip_cost_60" in scan26_names:
        series_map["sw_chip_cost_60"] = compute_chip_cost_panels(
            close, turn, money, volume
        )["sw_chip_cost_60"]
    if "zs_atv_20" in orgscan_names:
        series_map["zs_atv_20"] = compute_zs_atv_panels(volume)["zs_atv_20"]
    if "zs_patv_d20" in orgscan_names:
        series_map["zs_patv_d20"] = compute_zs_patv_panels(volume)["zs_patv_d20"]
    tug_names = [
        n for n in orgscan_names if n in {"zs_tug_nr_20", "zs_tug_pr_20", "zs_yuli_20"}
    ]
    if tug_names:
        tug_panels = compute_zs_tug_panels(open_, close)
        for name in tug_names:
            series_map[name] = tug_panels[name]
    if "gs_near_high_252" in orgscan_names:
        series_map["gs_near_high_252"] = compute_gs_near_high_panels(close, high)[
            "gs_near_high_252"
        ]
    moth_names = [n for n in orgscan_names if n in {"fz_taylor_jump_20", "fz_mod_amp_20"}]
    if moth_names:
        moth_panels = compute_fz_moth_panels(high, low, close)
        for name in moth_names:
            series_map[name] = moth_panels[name]
    if "hx_pv_rev_20" in orgscan_names:
        series_map["hx_pv_rev_20"] = compute_hx_pv_rev_panels(close, volume)[
            "hx_pv_rev_20"
        ]
    if "xb_toi_20" in orgscan_names:
        series_map["xb_toi_20"] = compute_xb_toi_panels(open_, close, volume)[
            "xb_toi_20"
        ]

    return series_map
