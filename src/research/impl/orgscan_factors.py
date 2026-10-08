"""
机构库增量：日频可落地子集。

- gs_rsi_20: 国盛量价淘金（三）日频 RSI，文献负向预测，取负
- zs_atv_20: 招商异常交易量 ATV = 当日量 / 20 日均量，取负
- zs_patv_d20: 招商 PATV 的日频代理（原文用 5 分钟）；对 ATV 截面分位做
  mean/std + kurt，取负
- zs_tug_nr_20 / zs_tug_pr_20 / zs_yuli_20: 招商隔夜拉锯与渔利
- ky_ideal_turn_20: 开源振幅隐藏结构「理想换手」同切割，取负
- gs_near_high_252: 国盛创新高的日频代理，收盘相对 252 日最高
- fz_taylor_jump_20 / fz_mod_amp_20: 方正飞蛾扑火日频（修正振幅因子2）
- hx_pv_rev_20: 华西极坐标价量融合（对角马氏近似），动量形式取负
- xb_toi_20: 西部隔夜-日内拉锯 TOI 日频代理（开盘切隔夜/日内）
方向统一为越大越好。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_gs_rsi_panels(close: pd.DataFrame, window: int = 20) -> Dict[str, pd.DataFrame]:
    """
    国盛定义：窗口内上涨日平均涨幅 / (上涨日平均涨幅 + 下跌日平均跌幅绝对值) * 100。
    """
    ret = close.astype(float).pct_change()
    up = ret.clip(lower=0.0)
    down = (-ret).clip(lower=0.0)
    up_n = (ret > 0).astype(float).rolling(window, min_periods=window).sum()
    dn_n = (ret < 0).astype(float).rolling(window, min_periods=window).sum()
    avg_up = up.rolling(window, min_periods=window).sum() / up_n.replace(0, np.nan)
    avg_dn = down.rolling(window, min_periods=window).sum() / dn_n.replace(0, np.nan)
    denom = avg_up + avg_dn
    with np.errstate(invalid="ignore", divide="ignore"):
        rsi = avg_up / denom.replace(0, np.nan) * 100.0
    factor = (-rsi).replace([np.inf, -np.inf], np.nan)
    return {"gs_rsi_20": factor}


def compute_zs_atv_panels(volume: pd.DataFrame, window: int = 20) -> Dict[str, pd.DataFrame]:
    """异常交易量：短量相对 20 日均量。文献空头逻辑，取负。"""
    vol = volume.astype(float)
    ma = vol.rolling(window, min_periods=window).mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        atv = vol / ma.replace(0, np.nan)
    return {"zs_atv_20": (-atv).replace([np.inf, -np.inf], np.nan)}


def compute_zs_patv_panels(volume: pd.DataFrame, window: int = 20) -> Dict[str, pd.DataFrame]:
    """
    日频代理：ATV 截面分位的均值/标准差 + 峰度。
    持续性越强、分位越高，未来收益越差，取负。
    """
    vol = volume.astype(float)
    ma = vol.rolling(window, min_periods=window).mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        atv = vol / ma.replace(0, np.nan)
    rank = atv.rank(axis=1, pct=True)
    mean = rank.rolling(window, min_periods=window).mean()
    std = rank.rolling(window, min_periods=window).std()
    kurt = rank.rolling(window, min_periods=window).kurt()
    with np.errstate(invalid="ignore", divide="ignore"):
        patv = mean / std.replace(0, np.nan) + kurt
    return {"zs_patv_d20": (-patv).replace([np.inf, -np.inf], np.nan)}


def compute_zs_tug_panels(
    open_: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 20,
    ab_window: int = 240,
) -> Dict[str, pd.DataFrame]:
    """
    招商隔夜拉锯：负向逆转 NR（隔夜正、盘中负）频率越大越好；
    正向逆转 PR 文献 IC 为负，取负；渔利 = AB_NR - AB_PR。
    """
    px = close.astype(float)
    op = open_.reindex_like(px).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ret_oc = px / op.replace(0, np.nan) - 1.0
        ret = px / px.shift(1) - 1.0
        ret_co = (1.0 + ret) / (1.0 + ret_oc) - 1.0
    nr_d = ((ret_co > 0) & (ret_oc < 0)).astype(float)
    pr_d = ((ret_co < 0) & (ret_oc > 0)).astype(float)
    valid = ret_oc.notna() & ret_co.notna()
    nr_d = nr_d.where(valid)
    pr_d = pr_d.where(valid)
    nr = nr_d.rolling(window, min_periods=window).mean()
    pr = pr_d.rolling(window, min_periods=window).mean()
    ab_nr = nr / nr.rolling(ab_window, min_periods=max(60, ab_window // 2)).mean().replace(
        0, np.nan
    )
    ab_pr = pr / pr.rolling(ab_window, min_periods=max(60, ab_window // 2)).mean().replace(
        0, np.nan
    )
    yuli = ab_nr - ab_pr
    return {
        "zs_tug_nr_20": nr.replace([np.inf, -np.inf], np.nan),
        "zs_tug_pr_20": (-pr).replace([np.inf, -np.inf], np.nan),
        "zs_yuli_20": yuli.replace([np.inf, -np.inf], np.nan),
    }


def _high_low_slice_mean(
    value: pd.DataFrame,
    close: pd.DataFrame,
    window: int,
    lam: float,
) -> pd.DataFrame:
    """滚动窗内按收盘价切高/低价日，返回高价日均值 - 低价日均值。"""
    v = value.astype(float)
    c = close.astype(float).reindex_like(v)
    k = max(1, int(round(window * lam)))
    out = pd.DataFrame(index=v.index, columns=v.columns, dtype=float)
    v_np = v.to_numpy(dtype=float)
    c_np = c.to_numpy(dtype=float)
    n_dates, n_codes = v_np.shape
    cols_idx = np.arange(n_codes)[None, :]
    for i in range(window - 1, n_dates):
        v_win = v_np[i - window + 1 : i + 1]
        c_win = c_np[i - window + 1 : i + 1]
        order = np.argsort(c_win, axis=0)
        low_idx = order[:k, :]
        high_idx = order[-k:, :]
        v_low = np.nanmean(v_win[low_idx, cols_idx], axis=0)
        v_high = np.nanmean(v_win[high_idx, cols_idx], axis=0)
        out.iloc[i] = v_high - v_low
    return out


def compute_ky_ideal_turn_panels(
    close: pd.DataFrame,
    turn: pd.DataFrame,
    window: int = 20,
    lam: float = 0.25,
) -> Dict[str, pd.DataFrame]:
    """开源理想换手：高价日换手均值 - 低价日换手均值，文献负向，取负。"""
    diff = _high_low_slice_mean(turn, close, window=window, lam=lam)
    return {"ky_ideal_turn_20": (-diff).replace([np.inf, -np.inf], np.nan)}


def compute_gs_near_high_panels(
    close: pd.DataFrame,
    high: pd.DataFrame,
    window: int = 252,
) -> Dict[str, pd.DataFrame]:
    """接近 252 日最高：close / rolling_max(high)。创新高后反应不足，越大越好。"""
    px = close.astype(float)
    hh = high.reindex_like(px).astype(float)
    peak = hh.rolling(window, min_periods=window).max()
    with np.errstate(invalid="ignore", divide="ignore"):
        near = px / peak.replace(0, np.nan)
    return {"gs_near_high_252": near.replace([np.inf, -np.inf], np.nan)}


def compute_fz_moth_panels(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 20,
) -> Dict[str, pd.DataFrame]:
    """
    方正日频跳跃：昨低到今高的单利与对数收益差得到泰勒残项。
    残项高于截面均值则保留振幅，否则翻转；文献负向，取负。
    """
    hh = high.astype(float)
    ll = low.astype(float)
    px = close.astype(float)
    prev_low = ll.shift(1)
    prev_close = px.shift(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = hh / prev_low.replace(0, np.nan)
        simple = ratio - 1.0
        clog = np.log(ratio)
        diff = simple - clog
        taylor = 2.0 * diff - clog * clog
        amp = (hh - ll) / prev_close.replace(0, np.nan)
    cs_mean = taylor.mean(axis=1)
    torch = taylor.sub(cs_mean, axis=0) > 0
    flipped = amp.where(torch, -amp)
    jump = (-taylor.rolling(window, min_periods=window).mean()).replace(
        [np.inf, -np.inf], np.nan
    )
    mod = (-flipped.rolling(window, min_periods=window).mean()).replace(
        [np.inf, -np.inf], np.nan
    )
    return {"fz_taylor_jump_20": jump, "fz_mod_amp_20": mod}


def compute_hx_pv_rev_panels(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    window: int = 20,
) -> Dict[str, pd.DataFrame]:
    """
    华西极坐标价量融合：对角马氏距离作极径，arctan2 作极角，
    45 度调整后乘象限偏好。原文按动量取值但实证为反转，取负。
    """
    px = close.astype(float)
    vol = volume.astype(float)
    dp = px - px.shift(window)
    dv = vol - vol.shift(window)
    sp = px.rolling(window, min_periods=window).std().replace(0, np.nan)
    sv = vol.rolling(window, min_periods=window).std().replace(0, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        zp = dp / sp
        zv = dv / sv
        rho = np.sqrt(zp * zp + zv * zv)
        theta = np.arctan2(zv.to_numpy(dtype=float), zp.to_numpy(dtype=float))
    theta = np.where(theta < 0, theta + 2.0 * np.pi, theta)
    alpha = np.where(
        theta < 0.5 * np.pi,
        1.0,
        np.where(
            theta < np.pi,
            0.5,
            np.where(theta < 1.5 * np.pi, -1.0, 0.75),
        ),
    )
    adj = np.exp(-np.abs(theta - np.pi / 4.0))
    mom = pd.DataFrame(alpha * rho.to_numpy(dtype=float) * adj, index=px.index, columns=px.columns)
    return {"hx_pv_rev_20": (-mom).replace([np.inf, -np.inf], np.nan)}


def compute_xb_toi_panels(
    open_: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    window: int = 20,
) -> Dict[str, pd.DataFrame]:
    """
    西部 TOI：隔夜涨且 OID>0 的日子上，corr(OID, 量相对均量)。
    无 10:00 价，用开盘切隔夜/日内。
    """
    px = close.astype(float)
    op = open_.reindex_like(px).astype(float)
    vol = volume.reindex_like(px).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ovn = op / px.shift(1).replace(0, np.nan) - 1.0
        oc = px / op.replace(0, np.nan) - 1.0
        oid = ovn - oc
        ivr = vol / vol.rolling(window, min_periods=window).mean().replace(0, np.nan)
    mask = (ovn > 0) & (oid > 0)
    a = oid.where(mask)
    b = ivr.where(mask)
    toi = a.rolling(window, min_periods=max(8, window // 2)).corr(b)
    return {"xb_toi_20": toi.replace([np.inf, -np.inf], np.nan)}


def orgscan_lookback(window: int = 20) -> int:
    """ATV 分位矩窗、拉锯 AB 窗与 252 日新高窗取较大。"""
    return max(window * 2 + 5, window + 240 + 5, 252 + 5)
