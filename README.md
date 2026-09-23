# quant-project

A-share multi-factor research → composite → cost-aware backtest → live target export.

面向本地研究与模拟盘的量化工程：因子注册/管线、合成分回测、收盘清单导出（PTrade / QMT）。

## 架构

```
main.py / update_*.py          # CLI 入口
src/
  data/                        # 行情、财务 PIT、估值、行业、过滤
  research/                    # 因子实现、管线、IC、合成、验证门
  strategy/                    # momentum / reversal / cpv / composite
  backtest/                    # 含成本回测引擎
  live/                        # 收盘清单与柜台导出
  common/                      # 工具与配置
factors/
  registry.json                # 因子注册表（入库）
  composite_universe.json      # 冻结合成池（入库）
  research_cards/              # 研报假说卡片（入库）
  raw|filtered|...|composite/  # parquet 中间层（本地，不入库）
data/                          # 行情与基本面缓存（本地，不入库）
outputs/                       # 日志与实验产物（本地，不入库）
scripts/                       # 运维与诊断脚本
```

## 环境

- Python >= 3.11
- 推荐 [uv](https://github.com/astral-sh/uv)

```bash
uv sync
# 或
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

依赖以 `pyproject.toml` / `uv.lock` 为准；`requirements.txt` 为兼容导出。

## 数据准备

`data/` 与因子 parquet **不进 Git**（体积约数 GB）。本地需具备：

| 路径 | 说明 |
|------|------|
| `data/date.pkl` | 交易日历 |
| `data/data_daily/` | 日线 |
| `data/data_ret/` 等 | 收益与衍生 |
| `data/data_industry/` | 行业 |
| `data/data_fundamental/` / `data/data_valuation/` | 财务 PIT / 估值（可选，模块 6） |

增量更新（baostock 若被限流请 `--workers 1`，勿频繁重试）：

```bash
python update_market_data.py
python update_industry_data.py
python update_fundamental_data.py
python update_valuation_data.py
```

详见 `data/README.md`。

## 常用命令

```bash
# 因子主链路：计算 → 过滤 → 预处理 → IC → 合成
python main.py --pipeline

# 合成分回测（默认策略 composite，含成本）
python main.py --backtest --strategy composite

# 第13课滚动权重 + raw/neu 验证
python main.py --backtest --strategy multifactor

# 收盘清单（默认 composite，实盘资金/持仓数见 CLI）
python main.py --live
```

可选：`--weight ic_ir`、`--rolling-weights`、`--start` / `--end`、`--top-n`、`--rebalance`。

## 入库约定

| 入库 | 不入库 |
|------|--------|
| `src/`、根目录入口脚本、`scripts/` | `data/**` 行情与缓存 |
| `factors/registry.json`、`composite_universe.json`、`research_cards/` | `factors/{raw,filtered,...}/` parquet |
| `pyproject.toml`、`uv.lock`、`requirements.txt` | `outputs/**`、`.venv/`、`.env` |
| `.cursor/rules/` | 实盘持仓/订单 JSON |

克隆后按「数据准备」恢复本地数据，再跑 `--pipeline` 生成因子层。

## 许可

仅供学习与研究；实盘自负风险。行情与第三方数据源请遵守其服务条款。
