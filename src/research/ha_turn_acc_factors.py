"""兼容旧导入路径，实现已迁至 src.research.impl。"""
from src.research.impl import ha_turn_acc_factors as _impl
import sys

sys.modules[__name__] = _impl
