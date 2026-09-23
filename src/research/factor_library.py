"""
因子库注册表（第九课模块 5 / 第十一课研报入库）。

负责：登记因子元数据、研报出处、版本、状态（candidate/active/inactive），
以及最近一次验证指标。不负责具体计算。
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# 写回时保留的研报/验证字段
_PRESERVE_KEYS = (
    "last_metrics",
    "last_health",
    "updated_at",
    "versions",
    "source",
    "hypothesis",
    "windows",
    "checklist_score",
    "impl",
    "card_path",
)

# 当前管线里已落地的价量因子
DEFAULT_FACTORS: List[Dict[str, Any]] = [
    {
        "name": "momentum_20",
        "version": "1.0",
        "desc": "20 日动量：收盘价相对 20 个交易日前涨幅",
        "formula": "close_t / close_{t-20} - 1",
        "data_deps": ["data_daily"],
        "status": "candidate",
        "source": "课程默认价量因子",
        "hypothesis": "中期价格动量在截面上具有持续性",
        "windows": {"lookback": 20},
        "impl": "momentum",
        "checklist_score": None,
        "card_path": None,
    },
    {
        "name": "reversal_5",
        "version": "1.0",
        "desc": "5 日反转：短期跌幅取负，越大越好",
        "formula": "-(close_t / close_{t-5} - 1)",
        "data_deps": ["data_daily"],
        "status": "candidate",
        "source": "课程默认价量因子",
        "hypothesis": "短期过度反应后均值回复",
        "windows": {"lookback": 5},
        "impl": "reversal",
        "checklist_score": None,
        "card_path": None,
    },
    {
        "name": "turn_stable_20",
        "version": "1.0",
        "desc": "20 日换手稳定：换手率标准差取负",
        "formula": "-std(turnover_ratio, 20)",
        "data_deps": ["data_daily"],
        "status": "candidate",
        "source": "东方证券技术分析因子系列·量稳换手率",
        "hypothesis": "换手稳定反映持续关注，利于趋势承载",
        "windows": {"lookback": 20},
        "impl": "turn_stable",
        "checklist_score": None,
        "card_path": "factors/research_cards/turn_stable_20.md",
    },
]


class FactorLibrary:
    """按 name 索引的因子注册表，落盘为 JSON。"""

    def __init__(self, registry_path: str | Path = "./factors/registry.json"):
        self.path = Path(registry_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._factors: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            self.load()
        else:
            self.ensure_defaults(overwrite=False)
            self.save()

    def load(self) -> None:
        """从磁盘读取注册表。"""
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        items = payload.get("factors", payload if isinstance(payload, list) else [])
        self._factors = {item["name"]: item for item in items}

    def save(self) -> None:
        """写回注册表。"""
        items = [self._factors[k] for k in sorted(self._factors)]
        payload = {"factors": items}
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def ensure_defaults(self, overwrite: bool = False) -> None:
        """补齐默认价量因子；已有条目默认不覆盖。"""
        for item in DEFAULT_FACTORS:
            name = item["name"]
            if name in self._factors and not overwrite:
                continue
            self._factors[name] = deepcopy(item)

    def register(
        self,
        name: str,
        desc: str,
        formula: str,
        version: str = "1.0",
        data_deps: Optional[List[str]] = None,
        status: str = "candidate",
        source: Optional[str] = None,
        hypothesis: Optional[str] = None,
        windows: Optional[Dict[str, Any]] = None,
        checklist_score: Optional[int] = None,
        impl: Optional[str] = None,
        card_path: Optional[str] = None,
    ) -> None:
        """
        注册或更新一个因子的基础信息（保留已有验证/研报字段）。

        新因子默认 status=candidate，须过 validate 门控后才可变 active。
        """
        if status not in {"candidate", "active", "inactive", "retired"}:
            raise ValueError(f"非法 status: {status}")
        old = self._factors.get(name, {})
        entry = {
            "name": name,
            "version": version,
            "desc": desc,
            "formula": formula,
            "data_deps": data_deps or ["data_daily"],
            "status": status,
            "source": source if source is not None else old.get("source"),
            "hypothesis": hypothesis if hypothesis is not None else old.get("hypothesis"),
            "windows": windows if windows is not None else old.get("windows"),
            "checklist_score": (
                checklist_score if checklist_score is not None else old.get("checklist_score")
            ),
            "impl": impl if impl is not None else old.get("impl"),
            "card_path": card_path if card_path is not None else old.get("card_path"),
        }
        for key in _PRESERVE_KEYS:
            if key in entry and entry[key] is not None:
                continue
            if key in old:
                entry[key] = old[key]
        self._factors[name] = entry
        self.save()

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        return self._factors.get(name)

    def names(self, status: Optional[str] = None) -> List[str]:
        """列出因子名；可按状态过滤。"""
        names = sorted(self._factors)
        if status is None:
            return names
        return [n for n in names if self._factors[n].get("status") == status]

    def active_names(self) -> List[str]:
        return self.names(status="active")

    def names_for_compute(
        self,
        statuses: Sequence[str] = ("candidate", "active"),
    ) -> List[str]:
        """计算器应尝试计算的因子：默认 candidate + active。"""
        out: List[str] = []
        for status in statuses:
            out.extend(self.names(status=status))
        # 去重且保持稳定顺序
        seen = set()
        ordered = []
        for name in out:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered

    def impl_map(self, names: Optional[Sequence[str]] = None) -> Dict[str, Optional[str]]:
        """取出因子名到 impl 字段的映射。"""
        names = list(names) if names is not None else self.names()
        return {n: (self._factors.get(n) or {}).get("impl") for n in names}

    def update_metrics(
        self,
        name: str,
        metrics: Dict[str, Any],
        status: str,
        health: str,
        updated_at: str,
    ) -> None:
        """写入最近一次验证结果，并追加简版版本日志。"""
        if name not in self._factors:
            self.register(name=name, desc=name, formula="", status="candidate")

        def _to_native(value: Any) -> Any:
            if hasattr(value, "item"):
                try:
                    return value.item()
                except Exception:
                    return value
            if value is None:
                return None
            if isinstance(value, float) and value != value:  # NaN
                return None
            return value

        clean_metrics = {k: _to_native(v) for k, v in metrics.items()}
        entry = self._factors[name]
        entry["last_metrics"] = clean_metrics
        entry["status"] = status
        entry["last_health"] = health
        entry["updated_at"] = updated_at
        versions = list(entry.get("versions") or [])
        versions.append(
            {
                "v": entry.get("version", "1.0"),
                "date": updated_at,
                "status": status,
                "ic_mean": clean_metrics.get("ic_mean"),
                "ic_ir": clean_metrics.get("ic_ir"),
            }
        )
        # 只保留最近 20 条变更
        entry["versions"] = versions[-20:]
        self.save()
