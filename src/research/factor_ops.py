"""
已实现因子算子表。

registry 里的 candidate/active 必须能在此表（或前缀规则）找到实现，
才允许进入 FactorCalculator 批量计算；否则直接报错，避免「登记了却算不出来」。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


# name -> 算子键（计算器内部分支）
IMPLEMENTED_OPS: Dict[str, str] = {
    "momentum_20": "momentum",
    "reversal_5": "reversal",
    "turn_stable_20": "turn_stable",
    "ubl": "ubl",
    "gfn_logvol_var10": "gfn_logvol_var10",
    "gfn_volpct_max20": "gfn_volpct_max20",
    "gfn_sqrt_open_low_vol": "gfn_sqrt_open_low_vol",
    "cj_ret_skew_20": "cj_ret_skew_20",
    "cj_mom_240_skip20": "cj_mom_240_skip20",
    "cj_rank_mom_220": "cj_rank_mom_220",
    "cj_corr_vol_close_20": "cj_corr_vol_close_20",
    "cj_turn_cv_20": "cj_turn_cv_20",
    "xb_vol_shock": "xb_vol_shock",
    "xb_amplitude": "xb_amplitude",
    "xb_intraday": "xb_intraday",
    "fz_jump_10": "fz_jump_10",
    "gs_ovn_turn_corr_20": "gs_ovn_turn_corr_20",
    "ha_volup_turn_acc_20": "ha_volup_turn_acc_20",
    "zt_illiq_20": "zt_illiq_20",
    "cjx_intraday_rev_20": "cjx_intraday_rev_20",
    "cjx_overnight_mom_20": "cjx_overnight_mom_20",
    "cjx_overnight_vol_20": "cjx_overnight_vol_20",
    "cjx_oi_spread_20": "cjx_oi_spread_20",
    "zy_anchor_rev": "zy_anchor_rev",
    "zy_anchor_rev_vol": "zy_anchor_rev_vol",
    "fz_bias_60": "fz_bias_60",
    "dw_pct_turn_20": "dw_pct_turn_20",
    "ky_ideal_amp_20": "ky_ideal_amp_20",
    "fz_rev_vol_flip_20": "fz_rev_vol_flip_20",
    "sw_dir_vol_diff_40": "sw_dir_vol_diff_40",
    "sw_mod_skew_40": "sw_mod_skew_40",
    "fz_rev_turn_flip_20": "fz_rev_turn_flip_20",
    "df_maxret_20": "df_maxret_20",
    "xn_idio_vol_20": "xn_idio_vol_20",
    "pk_range_vol_20": "pk_range_vol_20",
    "ms_dastd_60": "ms_dastd_60",
    "ms_rank_vol_60": "ms_rank_vol_60",
    "xn_same_wd_mom_12": "xn_same_wd_mom_12",
    "ky_long_mom_160": "ky_long_mom_160",
    "ky_long_mom2_160": "ky_long_mom2_160",
    "fz_panic_ret_20": "fz_panic_ret_20",
    "db_csk_xyy_120": "db_csk_xyy_120",
    "db_csk_down_120": "db_csk_down_120",
    "df_ext_rev_20": "df_ext_rev_20",
    "db_csk_up_120": "db_csk_up_120",
    "db_csk_comp_120": "db_csk_comp_120",
    "gk_vol_20": "gk_vol_20",
    "rs_vol_20": "rs_vol_20",
    "semi_down_20": "semi_down_20",
    "xn_same_wd_rev_12": "xn_same_wd_rev_12",
    "ct_resid_mom_60": "ct_resid_mom_60",
    "df_mild_mom_20": "df_mild_mom_20",
    "ovn_rev_20": "ovn_rev_20",
    "idt_rev_20": "idt_rev_20",
    "fz_lt_mom_60": "fz_lt_mom_60",
    "yz_vol_20": "yz_vol_20",
    "ret_kurt_60": "ret_kurt_60",
    "pv_corr_vol_20": "pv_corr_vol_20",
    "gs_low_vol_evt_5": "gs_low_vol_evt_5",
    "gs_high_vol_evt_5": "gs_high_vol_evt_5",
    "gs_hl_vol_net_5": "gs_hl_vol_net_5",
    "gs_low_vol_soft_120": "gs_low_vol_soft_120",
    "gs_high_vol_soft_120": "gs_high_vol_soft_120",
    "gs_hl_vol_soft_120": "gs_hl_vol_soft_120",
    "m6_ep_ttm": "m6_ep_ttm",
    "m6_bp_mrq": "m6_bp_mrq",
    "m6_roe": "m6_roe",
    "m6_yoy_ni": "m6_yoy_ni",
    "m6_peg_inv": "m6_peg_inv",
    "m6_dy": "m6_dy",
}

# 已接线的算子键
WIRED_OPS = frozenset(
    {
        "momentum",
        "reversal",
        "turn_stable",
        "ubl",
        "gfn_logvol_var10",
        "gfn_volpct_max20",
        "gfn_sqrt_open_low_vol",
        "cj_ret_skew_20",
        "cj_mom_240_skip20",
        "cj_rank_mom_220",
        "cj_corr_vol_close_20",
        "cj_turn_cv_20",
        "xb_vol_shock",
        "xb_amplitude",
        "xb_intraday",
        "fz_jump_10",
        "gs_ovn_turn_corr_20",
        "ha_volup_turn_acc_20",
        "zt_illiq_20",
        "cjx_intraday_rev_20",
        "cjx_overnight_mom_20",
        "cjx_overnight_vol_20",
        "cjx_oi_spread_20",
        "zy_anchor_rev",
        "zy_anchor_rev_vol",
        "fz_bias_60",
        "dw_pct_turn_20",
        "ky_ideal_amp_20",
        "fz_rev_vol_flip_20",
        "sw_dir_vol_diff_40",
        "sw_mod_skew_40",
        "fz_rev_turn_flip_20",
        "df_maxret_20",
        "xn_idio_vol_20",
        "pk_range_vol_20",
        "ms_dastd_60",
        "ms_rank_vol_60",
        "xn_same_wd_mom_12",
        "ky_long_mom_160",
        "ky_long_mom2_160",
        "fz_panic_ret_20",
        "db_csk_xyy_120",
        "db_csk_down_120",
        "df_ext_rev_20",
        "db_csk_up_120",
        "db_csk_comp_120",
        "gk_vol_20",
        "rs_vol_20",
        "semi_down_20",
        "xn_same_wd_rev_12",
        "ct_resid_mom_60",
        "df_mild_mom_20",
        "ovn_rev_20",
        "idt_rev_20",
        "fz_lt_mom_60",
        "yz_vol_20",
        "ret_kurt_60",
        "pv_corr_vol_20",
        "gs_low_vol_evt_5",
        "gs_high_vol_evt_5",
        "gs_hl_vol_net_5",
        "gs_low_vol_soft_120",
        "gs_high_vol_soft_120",
        "gs_hl_vol_soft_120",
        "m6_ep_ttm",
        "m6_bp_mrq",
        "m6_roe",
        "m6_yoy_ni",
        "m6_peg_inv",
        "m6_dy",
    }
)

# 前缀规则：未在上表显式登记时，用前缀推断
PREFIX_OPS: Tuple[Tuple[str, str], ...] = (
    ("momentum_", "momentum"),
    ("reversal_", "reversal"),
    ("turn_stable_", "turn_stable"),
    ("ubl", "ubl"),
)


def resolve_op(name: str, impl: Optional[str] = None) -> Optional[str]:
    """根据因子名或 registry.impl 解析算子键；无法解析返回 None。"""
    if impl:
        # 占位 cpv 视为未实现；正式键走接线表
        if impl == "cpv":
            return None
        if impl in WIRED_OPS:
            return impl
        return impl
    if name in IMPLEMENTED_OPS:
        return IMPLEMENTED_OPS[name]
    for prefix, op in PREFIX_OPS:
        if name == prefix or name.startswith(prefix):
            return op
    return None


def assert_ops_implemented(
    names: Sequence[str],
    impl_by_name: Optional[Dict[str, Optional[str]]] = None,
) -> List[str]:
    """
    校验一批因子均有算子实现。
    返回与 names 同序的算子键列表；任一缺失则抛错。
    """
    impl_by_name = impl_by_name or {}
    missing = []
    ops: List[str] = []
    for name in names:
        op = resolve_op(name, impl_by_name.get(name))
        if op is None or op not in WIRED_OPS:
            label = name if op is None else f"{name}(impl={op}, 尚未接线)"
            missing.append(label)
        else:
            ops.append(op)
    if missing:
        raise RuntimeError(
            "以下因子已在 registry 登记，但计算器尚未实现算子，请先写公式再计算: "
            + ", ".join(missing)
        )
    return ops
