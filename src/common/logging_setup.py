"""
统一日志规范：控制台 + outputs/logs 按日/按任务落盘。

用法:
    from src.common.logging_setup import setup_logging, get_logger
    setup_logging(task="main")
    log = get_logger(__name__)
    log.info("started")
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from src.common.config import ROOT, get_section, load_config

_INITIALIZED = False


def setup_logging(
    task: str = "app",
    *,
    level: Optional[str] = None,
    log_dir: Union[str, Path, None] = None,
    console: Optional[bool] = None,
    file: Optional[bool] = None,
) -> Path:
    """
    初始化根日志。可重复调用；首次生效。

    返回本次文件日志路径（若关闭 file 则为 log_dir）。
    """
    global _INITIALIZED
    cfg = get_section("logging")
    paths = get_section("paths")

    log_level = (level or cfg.get("level") or "INFO").upper()
    use_console = cfg.get("console", True) if console is None else console
    use_file = cfg.get("file", True) if file is None else file
    fmt = cfg.get("format") or "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    datefmt = cfg.get("datefmt") or "%Y-%m-%d %H:%M:%S"

    if log_dir is None:
        log_dir = paths.get("log_dir") or "./outputs/logs"
    log_path = Path(log_dir)
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    log_path.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d")
    file_path = log_path / f"{task}_{stamp}.log"

    root = logging.getLogger()
    if _INITIALIZED:
        return file_path

    root.handlers.clear()
    root.setLevel(getattr(logging, log_level, logging.INFO))
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    if use_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(getattr(logging, log_level, logging.INFO))
        ch.setFormatter(formatter)
        root.addHandler(ch)

    if use_file:
        fh = logging.FileHandler(file_path, encoding="utf-8")
        fh.setLevel(getattr(logging, log_level, logging.INFO))
        fh.setFormatter(formatter)
        root.addHandler(fh)

    _INITIALIZED = True
    root.info("日志已初始化 | task=%s | file=%s | level=%s", task, file_path, log_level)
    # 避免重复加载配置噪音
    _ = load_config()
    return file_path


def get_logger(name: str) -> logging.Logger:
    """获取命名 logger；若尚未 setup 则用默认配置初始化。"""
    if not _INITIALIZED:
        setup_logging(task="app")
    return logging.getLogger(name)
