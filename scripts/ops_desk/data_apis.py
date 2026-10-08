"""作战台数据组装：日更 / 回测缓存 / 因子体检 / 出图清单。"""

from __future__ import annotations

import csv
import json
import pickle
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.common.utils import list_dated_stems

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "ops_desk"
# 旧单文件缓存仍给出图页当主轨默认；主/短回测分存
BACKTEST_CACHE = OUT_DIR / "backtest_snapshot.json"
BACKTEST_CACHE_MAIN = OUT_DIR / "backtest_snapshot_main.json"
BACKTEST_CACHE_ST = OUT_DIR / "backtest_snapshot_st.json"
PLOTS_DIR = OUT_DIR / "plots"


def _read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _last_csv_date(dir_path: Path) -> Optional[str]:
    if not dir_path.is_dir():
        return None
    dates = sorted(p.stem for p in dir_path.glob("????-??-??.csv"))
    return dates[-1] if dates else None


def _last_dated(dir_path: Path) -> Optional[str]:
    """按日截面末日（parquet 或 csv）。"""
    if not dir_path.is_dir():
        return None
    stems = list_dated_stems(dir_path)
    return stems[-1] if stems else None


def _calendar_last() -> Optional[str]:
    pkl = ROOT / "data" / "date.pkl"
    if not pkl.exists():
        return None
    with open(pkl, "rb") as f:
        cal = [str(d) for d in pickle.load(f)]
    return cal[-1] if cal else None


def _latest_orders(live_dir: Path) -> Dict[str, Any]:
    files = sorted(live_dir.glob("????-??-??_orders.csv"))
    if not files:
        return {"file": None, "rows": [], "buy_amount": 0.0, "sell_count": 0, "buy_count": 0}
    path = files[-1]
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    buy_amount = 0.0
    buy_count = sell_count = 0
    for r in rows:
        action = r.get("action") or ""
        if action == "买入":
            buy_count += 1
            try:
                buy_amount += float(r.get("amount") or 0)
            except ValueError:
                pass
        elif action == "卖出":
            sell_count += 1
    return {
        "file": path.name,
        "rows": rows,
        "buy_amount": round(buy_amount, 2),
        "buy_count": buy_count,
        "sell_count": sell_count,
    }


def _industry_map(asof: Optional[str] = None) -> Tuple[Optional[str], Dict[str, str]]:
    ind_dir = ROOT / "data" / "data_industry"
    files = sorted(ind_dir.glob("????-??-??.csv"))
    if not files:
        return None, {}
    path = files[-1]
    if asof:
        earlier = [p for p in files if p.stem <= asof]
        if earlier:
            path = earlier[-1]
    df = pd.read_csv(path)
    col = "industry" if "industry" in df.columns else df.columns[1]
    mapping = {
        str(r["code"]): str(r[col])
        for _, r in df.iterrows()
        if pd.notna(r.get("code")) and pd.notna(r.get(col))
    }
    return path.stem, mapping


def _industry_concentration(codes: List[str], ind_map: Dict[str, str]) -> Dict[str, Any]:
    if not codes:
        return {"n": 0, "top": None, "top_share": 0.0, "hhi": 0.0, "breakdown": []}
    labels = [ind_map.get(c, "未知") for c in codes]
    cnt = Counter(labels)
    total = len(labels)
    breakdown = [
        {"industry": k, "count": v, "share": round(v / total, 4)}
        for k, v in cnt.most_common()
    ]
    shares = np.array([v / total for v in cnt.values()], dtype=float)
    hhi = float((shares ** 2).sum())
    top_ind, top_n = cnt.most_common(1)[0]
    return {
        "n": total,
        "top": top_ind,
        "top_share": round(top_n / total, 4),
        "hhi": round(hhi, 4),
        "breakdown": breakdown,
        "mapped": sum(1 for c in codes if c in ind_map),
    }


def _track_snapshot(live_rel: str, paper_rel: str, label: str) -> Dict[str, Any]:
    live_dir = ROOT / live_rel
    paper_dir = ROOT / paper_rel
    holdings = _read_json(live_dir / "current_holdings.json") or {}
    paper = _read_json(paper_dir / "latest.json") or {}
    orders = _latest_orders(live_dir)
    holdings_mtime = None
    hp = live_dir / "current_holdings.json"
    if hp.exists():
        holdings_mtime = datetime.fromtimestamp(hp.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    asof = holdings.get("date") or paper.get("signal_date")
    ind_date, ind_map = _industry_map(str(asof) if asof else None)
    conc = _industry_concentration(list(holdings.get("codes") or []), ind_map)
    conc["industry_asof"] = ind_date
    return {
        "label": label,
        "live_dir": live_rel,
        "paper_dir": paper_rel,
        "holdings": holdings,
        "holdings_mtime": holdings_mtime,
        "paper": paper,
        "orders": orders,
        "industry": conc,
    }


def _lag_ok(last: Optional[str], cal: Optional[str]) -> bool:
    return bool(last and cal and last >= cal)


def build_ops_snapshot() -> Dict[str, Any]:
    daily = _last_csv_date(ROOT / "data" / "data_daily")
    ud = _last_csv_date(ROOT / "data" / "data_ud_new")
    cal = _calendar_last()
    composite_last = _last_dated(ROOT / "factors" / "composite")
    composite_st_last = _last_dated(ROOT / "factors" / "composite_st")
    industry_last = _last_dated(ROOT / "data" / "data_industry")
    fund_last = _last_dated(ROOT / "data" / "data_fundamental")
    val_last = _last_dated(ROOT / "data" / "data_valuation")
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data": {
            "daily_last": daily,
            "ud_last": ud,
            "calendar_last": cal,
            "composite_last": composite_last,
            "composite_st_last": composite_st_last,
            "industry_last": industry_last,
            "fundamental_last": fund_last,
            "valuation_last": val_last,
            "daily_ok": _lag_ok(daily, cal),
            "ud_ok": _lag_ok(ud, cal),
            "composite_ok": _lag_ok(composite_last, daily),
            "composite_st_ok": _lag_ok(composite_st_last, daily),
            "industry_ok": _lag_ok(industry_last, daily),
            "fundamental_ok": _lag_ok(fund_last, daily),
            "valuation_ok": _lag_ok(val_last, daily),
        },
        "main": _track_snapshot("outputs/live", "outputs/paper", "主轨 composite"),
        "st": _track_snapshot(
            "outputs/live_st", "outputs/paper_st", "短线 composite_st"
        ),
    }


def _read_track_cache(path: Path) -> Optional[Dict[str, Any]]:
    data = _read_json(path)
    if not isinstance(data, dict) or not data.get("with_cost"):
        return None
    data["ok"] = True
    return data


def load_backtest_snapshot() -> Dict[str, Any]:
    """主/短分存；若只有旧单文件则当作主轨。"""
    main = _read_track_cache(BACKTEST_CACHE_MAIN)
    st = _read_track_cache(BACKTEST_CACHE_ST)
    if main is None:
        legacy = _read_track_cache(BACKTEST_CACHE)
        if legacy is not None and not legacy.get("track"):
            legacy["track"] = "main"
            main = legacy
    if main is None and st is None:
        return {
            "ok": False,
            "message": "尚无回测缓存。点「刷新主轨回测」或「刷新短线回测」。",
            "main": None,
            "st": None,
        }
    times = [x.get("generated_at") for x in (main, st) if x and x.get("generated_at")]
    ops = build_ops_snapshot()
    industry = {
        "main": ops["main"]["industry"],
        "st": ops["st"]["industry"],
    }
    return {
        "ok": True,
        "generated_at": max(times) if times else None,
        "main": main,
        "st": st,
        "industry_live": industry,
        "note": "主轨 / 短线回测分存；行业集中度为当前 live 持仓",
        # 兼容出图页旧字段：默认展示主轨（无主轨则短线）
        "with_cost": (main or st or {}).get("with_cost"),
        "no_cost": (main or st or {}).get("no_cost"),
    }


def _load_universe(path: Path) -> List[str]:
    raw = _read_json(path)
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, dict):
        return [str(x) for x in (raw.get("factors") or raw.get("names") or [])]
    return []


def _registry_by_name() -> Dict[str, Dict[str, Any]]:
    reg = _read_json(ROOT / "factors" / "registry.json") or {}
    factors = reg.get("factors") if isinstance(reg, dict) else reg
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(factors, list):
        for item in factors:
            if isinstance(item, dict) and item.get("name"):
                out[str(item["name"])] = item
    return out


def _factor_corr(universe: List[str], max_dates: int = 20) -> Dict[str, Any]:
    neu = ROOT / "factors" / "neutralized"
    files = sorted(neu.glob("????-??-??.parquet"))
    if not files:
        return {"dates": 0, "matrix": [], "labels": []}
    use = files[-max_dates:]
    labels = [n for n in universe]
    mats = []
    for path in use:
        df = pd.read_parquet(path)
        cols = [c for c in labels if c in df.columns]
        if len(cols) < 2:
            continue
        sub = df[cols].apply(pd.to_numeric, errors="coerce")
        c = sub.corr(method="spearman")
        mats.append(c.reindex(index=labels, columns=labels))
    if not mats:
        return {"dates": 0, "matrix": [], "labels": labels}
    mean = sum(mats) / len(mats)
    mean = mean.fillna(0.0)
    # 只返回上三角摘要：高相关对
    pairs = []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if j <= i:
                continue
            v = float(mean.loc[a, b]) if a in mean.index and b in mean.columns else 0.0
            pairs.append({"a": a, "b": b, "corr": round(v, 4)})
    pairs.sort(key=lambda x: abs(x["corr"]), reverse=True)
    return {
        "dates": len(mats),
        "asof": use[-1].stem,
        "labels": labels,
        "matrix": [[round(float(mean.loc[a, b]), 3) for b in labels] for a in labels],
        "top_pairs": pairs[:15],
    }


def _loo_table(path: Path, limit: int = 20) -> Dict[str, Any]:
    if not path.exists():
        return {"file": None, "rows": []}
    df = pd.read_csv(path)
    # 只看 drop_* / loo_drop_*，按收益排序
    if "dropped" in df.columns:
        sub = df[df["dropped"].notna()].copy()
    else:
        sub = df.copy()
    sort_col = "total_return" if "total_return" in sub.columns else sub.columns[-1]
    sub = sub.sort_values(sort_col, ascending=False).head(limit)
    rows = json.loads(sub.to_json(orient="records", force_ascii=False))
    return {"file": str(path.relative_to(ROOT)), "rows": rows}


def build_factors_snapshot() -> Dict[str, Any]:
    uni_main = _load_universe(ROOT / "factors" / "composite_universe.json")
    uni_st = _load_universe(ROOT / "factors" / "composite_universe_st.json")
    reg = _registry_by_name()
    rows = []
    for name in uni_main:
        item = reg.get(name) or {}
        m = item.get("last_metrics") or {}
        rows.append(
            {
                "name": name,
                "status": item.get("status"),
                "health": item.get("last_health"),
                "ic_mean": m.get("ic_mean"),
                "ic_ir": m.get("ic_ir"),
                "turnover_mean": m.get("turnover_mean"),
                "updated_at": item.get("updated_at"),
                "in_st": name in uni_st,
            }
        )
    rows.sort(key=lambda r: (r.get("ic_ir") is None, -(r.get("ic_ir") or 0)))
    loo_main = _loo_table(ROOT / "outputs" / "combo_loo_full_verify_20260923.csv")
    loo_st = _loo_table(ROOT / "outputs" / "st_track_loo_weights.csv")
    st_sum = _read_json(ROOT / "outputs" / "st_track_loo_weights_summary.json") or {}
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "universe_n": len(uni_main),
        "factors": rows,
        "corr": _factor_corr(uni_main, max_dates=20),
        "loo_main": loo_main,
        "loo_st": loo_st,
        "loo_st_best": st_sum.get("best"),
    }


def list_course_viz() -> List[Dict[str, Any]]:
    viz_dir = ROOT / "scripts" / "course_viz"
    out = []
    for path in sorted(viz_dir.glob("visualize_*.py")):
        out.append(
            {
                "id": path.stem,
                "file": str(path.relative_to(ROOT)),
                "title": path.stem.replace("visualize_", "").replace("_", " "),
            }
        )
    return out


def list_plot_images() -> List[Dict[str, str]]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    images = []
    for path in sorted(PLOTS_DIR.glob("*.png")) + sorted(PLOTS_DIR.glob("*.svg")):
        images.append(
            {
                "name": path.name,
                "url": f"/api/plots/{path.name}",
                "mtime": datetime.fromtimestamp(path.stat().st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
        )
    # 也收录 day13 已有图
    day13 = ROOT / "outputs" / "day13_multifactor" / "plots"
    if day13.is_dir():
        for path in sorted(day13.rglob("*.png"))[:40]:
            rel = path.relative_to(ROOT / "outputs")
            images.append(
                {
                    "name": str(rel),
                    "url": f"/api/outputs_file/{rel.as_posix()}",
                    "mtime": datetime.fromtimestamp(path.stat().st_mtime).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                }
            )
    return images


def build_builtin_charts() -> Dict[str, Any]:
    """页面内嵌图数据：净值 / IC 柱 / LOO 柱。"""
    bt = load_backtest_snapshot()
    fac = build_factors_snapshot()
    ic_bars = [
        {
            "name": r["name"],
            "ic_ir": r.get("ic_ir"),
            "ic_mean": r.get("ic_mean"),
        }
        for r in fac["factors"]
        if r.get("ic_ir") is not None
    ]
    loo_rows = (fac.get("loo_main") or {}).get("rows") or []
    loo_bars = [
        {
            "name": r.get("dropped") or r.get("tag"),
            "total_return": r.get("total_return"),
            "sharpe": r.get("sharpe"),
        }
        for r in loo_rows[:12]
    ]
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "equity": {
            "ok": bool(bt.get("ok")),
            "with_cost": (bt.get("with_cost") or {}).get("equity") if bt.get("ok") else [],
            "no_cost": (bt.get("no_cost") or {}).get("equity") if bt.get("ok") else [],
            "metrics_with_cost": (bt.get("with_cost") or {}).get("metrics") if bt.get("ok") else {},
            "metrics_no_cost": (bt.get("no_cost") or {}).get("metrics") if bt.get("ok") else {},
        },
        "ic_bars": ic_bars,
        "loo_bars": loo_bars,
        "corr_top": (fac.get("corr") or {}).get("top_pairs") or [],
        "viz_scripts": list_course_viz(),
        "images": list_plot_images(),
    }
