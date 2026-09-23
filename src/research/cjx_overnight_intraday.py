"""兼容旧导入路径，实现已迁至 src.research.impl。"""
from src.research.impl import cjx_overnight_intraday as _impl
import sys

sys.modules[__name__] = _impl
