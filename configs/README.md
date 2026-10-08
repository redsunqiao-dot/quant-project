# 版本化配置

| 文件 | 说明 |
|------|------|
| `default.json` | 默认路径、管线、回测成本、live、模拟盘、日志 |

加载：`src.common.config.load_config()`。  
入口：`main.py --config ...` 或 `QUANT_CONFIG`。CLI 显式参数优先于配置。
