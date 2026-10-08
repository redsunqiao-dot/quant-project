"""
版本化配置加载。

默认读 configs/default.json；可用环境变量 QUANT_CONFIG 或显式路径覆盖。
CLI 参数仍优先于配置文件。
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.json"

_CACHE: Optional[Dict[str, Any]] = None


def _deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """递归合并字典，override 覆盖 base。"""
    out: Dict[str, Any] = deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, Mapping)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def resolve_config_path(path: Union[str, Path, None] = None) -> Path:
    """解析配置文件路径：显式参数 > QUANT_CONFIG > 默认。"""
    if path:
        return Path(path).expanduser().resolve()
    env = os.environ.get("QUANT_CONFIG", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_CONFIG_PATH


def load_config(
    path: Union[str, Path, None] = None,
    *,
    reload: bool = False,
) -> Dict[str, Any]:
    """加载 JSON 配置；同进程默认缓存。"""
    global _CACHE
    cfg_path = resolve_config_path(path)
    if _CACHE is not None and not reload and path is None and not os.environ.get("QUANT_CONFIG"):
        return deepcopy(_CACHE)
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"配置根节点必须是对象: {cfg_path}")
    if path is None and not os.environ.get("QUANT_CONFIG"):
        _CACHE = data
    return deepcopy(data)


def get_section(name: str, path: Union[str, Path, None] = None) -> Dict[str, Any]:
    """取配置小节，不存在则返回空 dict。"""
    cfg = load_config(path)
    section = cfg.get(name, {})
    return dict(section) if isinstance(section, dict) else {}


def project_path(*parts: str, config: Optional[Mapping[str, Any]] = None) -> Path:
    """相对仓库根的路径（配置里的相对路径均相对项目根）。"""
    cfg = config or load_config()
    paths = cfg.get("paths") or {}
    if len(parts) == 1 and parts[0] in paths:
        return (ROOT / str(paths[parts[0]])).resolve()
    return (ROOT.joinpath(*parts)).resolve()
