# Tushare 拉数方案（对齐本仓库落盘）

目标：在 baostock 限流/封禁时，用 **Tushare Pro** 补齐与现有目录一致的数据，不改因子/回测主链路读法。

代码现状：行情增量已实现 → `update_market_data_tushare.py`（实现见 `src/data/update_market_data_tushare.py`）。  
下文把「已落地」与「待扩展」分开写清。

---

## 1. 前置条件

| 项 | 说明 |
|----|------|
| 注册 | https://tushare.pro/register |
| Token | 环境变量 `TUSHARE_TOKEN`（或 `TS_TOKEN`），勿写入仓库；`.env.example` 已留位 |
| 依赖 | `pyproject.toml` optional：`pip install tushare` 或 `uv sync --extra data-alt` |
| 积分 | 日线/每日指标约 120+；涨跌停、财务、申万行业通常要更高档（以官网权限表为准） |
| 频控 | 单接口建议间隔 ≥0.3s；按日面板拉取，禁止全市场逐票狂打 |

原则：

1. **只增量**：以本地 `data_daily` 最后一日为锚，补缺失交易日。  
2. **不覆盖课包历史**（除非显式 `--overwrite`）。  
3. **代码统一**为 `XXXXXX.XSHG` / `XXXXXX.XSHE`（Tushare `ts_code` ↔ 项目码已有映射函数）。  
4. 财务只按 **公告日 PIT**，禁止用报告期偷未来。

---

## 2. 目录与接口映射

| 本地路径 | 用途 | Tushare 接口（建议） | 状态 |
|----------|------|---------------------|------|
| `data/date.pkl` | 交易日历 | `trade_cal`（SSE/SZSE）或沿用本地已有日 | 行情脚本末尾已 `update_date_pkl` |
| `data/data_daily/{date}.csv` | 后复权 OHLCV + 换手 | `daily` + 锚定链式后复权；换手 `daily_basic.turnover_rate` | **已落地** |
| `data/data_ud_new/{date}.csv` | 不复权开盘/昨收、涨跌停、停牌/ST | `daily`（不复权）+ `stk_limit` + `stock_basic`（ST） | **已落地** |
| `data/data_ret/` | 前向收益 | 本地由日线重算 `update_forward_returns` | **已落地**（随行情） |
| `data/data_valuation/{date}.csv` | pe_ttm / pb_mrq / ps_ttm / pcf_ttm | `daily_basic`（`pe_ttm,pb,ps_ttm,pcf_ncf_ttm`）按日截面 | **已落地** |
| `data/data_industry/{date}.csv` | 行业（申万一级优先） | `index_classify` + `index_member`（或 `index_member_all`）落 code→industry | **待写** |
| `data/_cache/fundamental/` | 季报缓存 | `income` / `balancesheet` / `fina_indicator` / `disclosure_date` | **待写** |
| `data/data_fundamental/{date}.csv` | PIT 日截面 | 缓存按 `ann_date`/`actual_date` ≤ 交易日展开（复用现有 PIT 逻辑） | **待写** |

字段对齐（不可随意改列名）：

- 日线：`date,code,open,close,low,high,volume,money,turnover_ratio`  
- 交易状态：`date,code,open,pre_close,high_limit,low_limit,paused,zt,dt,st`  
- 估值：`date,code,pe_ttm,pb_mrq,ps_ttm,pcf_ttm`  
- 财务 PIT：`date,code,roe,np_margin,gp_margin,net_profit,eps_ttm,yoy_ni,yoy_eps,pub_date,stat_date`

---

## 3. 行情（已落地）操作手册

```bash
export TUSHARE_TOKEN=你的token
# 增量：从本地最后一日之后补到今天
.venv/bin/python update_market_data_tushare.py

# 指定区间 / 纠错覆盖
.venv/bin/python update_market_data_tushare.py --start 2026-09-28 --end 2026-09-30 --overwrite
```

口径要点：

- `data_daily`：用「锚定日本地后复权收盘 × (当日不复权价 / 昨收)」接到历史课包序列，避免与 baostock 后复权绝对值硬拼。  
- `money`：Tushare `amount` 为千元 → ×1000 成元。  
- `turnover_ratio`：用 `turnover_rate`（百分数，与 baostock turn 一致）。  
- 涨跌停：优先 `stk_limit`；无权限时按规则推算（脚本内已处理）。

与 akshare 备源关系：二者都是 **baostock 备源**；同一日不要混源乱序覆盖。纠错日用 `--overwrite` 且单一数据源。

---

## 4. 估值（已落地）操作手册

```bash
export TUSHARE_TOKEN=你的token
.venv/bin/python update_valuation_data_tushare.py
.venv/bin/python update_valuation_data_tushare.py --start 2026-09-01 --end 2026-09-30 --overwrite
```

字段映射：`pe_ttm←pe_ttm`，`pb_mrq←pb`，`ps_ttm←ps_ttm`，`pcf_ttm←pcf_ncf_ttm`。  
默认相对估值目录增量；无估值文件时只补最近 240 个有日线的交易日。失败重试 3 次仍空则退出（不写半日）。

---

## 5. 行业（待扩展）方案

**脚本名（建议）**：`update_industry_data_tushare.py`

1. `index_classify(level='L1', src='SW2021')` 取申万一级行业列表。  
2. 对每个行业指数拉成分 → `code → industry_name`。  
3. 按 `date.pkl` 交易日写截面（成分变更日才变；可日更复制最新映射，与现有 industry 用法一致）。  
4. 缓存映射表到 `data/_cache/sw_l1_industry_map_ts.csv`，避免每次全量打接口。

若积分不足：继续用现有 akshare/本地 industry，不强行切源。

---

## 6. 财务 PIT（待扩展）方案

**目标**：产出与 `update_fundamental_data.py` 相同的 cache + PIT，供 m6 因子。

**建议两段式**（与现网一致）：

### 6.1 拉缓存 `data/_cache/fundamental/{code}.parquet`

按股票（或按报告期批量）拉取：

| 本地字段 | 来源建议 |
|----------|----------|
| roe, np_margin, gp_margin, eps_ttm 等 | `fina_indicator` |
| net_profit | `income` 或 fina_indicator |
| yoy_ni, yoy_eps | fina_indicator 同比字段或自算 |
| stat_date | 报告期 `end_date` |
| pub_date | **优先** `fina_indicator.ann_date` / `disclosure_date.actual_date`；无则退回 `ann_date` |

注意：

- 单位与 baostock 对齐（净利润等量级），写入前做一次抽样对照（同股同报告期）。  
- 全市场首次建缓存：按 ts_code 列表串行或极低并发（workers=1），写进度到日志。  
- 增量：只补「本地 cache 缺失或 end_date 更新」的票。

### 6.2 重写 PIT

优先复用现有 `--skip-fetch --overwrite` 的 PIT 展开逻辑：  
只要 cache 列名与 `FUND_COLS` 一致，即可：

```bash
# 设想：Tushare 只负责填 cache，PIT 仍走现脚本
.venv/bin/python update_fundamental_data.py --skip-fetch --overwrite --start ... --end ...
```

若 cache 路径/列略有差异，抽公共 `rewrite_fundamental_pit(cache_dir, ...)`，baostock / tushare 共用（结课后再抽，避免中途大改）。

---

## 7. 推荐落地顺序

| 序 | 事项 | 理由 |
|----|------|------|
| 1 | 行情日更用 Tushare（已有） | 解开 baostock 行情停更 |
| 2 | 估值按日 `daily_basic`（已有） | 与 m6/BP 强相关 |
| 3 | 行业申万映射 | 中性化/持仓行业约束依赖 |
| 4 | 财务 cache → 复用 PIT | 积分与字段对齐成本最高，放最后 |

日更（命令行）：

```bash
export TUSHARE_TOKEN=...
.venv/bin/python update_market_data_tushare.py
.venv/bin/python update_valuation_data_tushare.py
# 行业/财务按周或按需，不必每日
```

或作战台按钮（同一 token 环境启动 server）：

```bash
export TUSHARE_TOKEN=...
.venv/bin/python scripts/ops_desk/server.py
# 浏览器打开日更页 →「Tushare 补行情」/「Tushare 补估值」
```

---

## 8. 风险与验收

- **复权断裂**：新上市、停牌久、锚定日无该票 → 跳过该票并计数；验收看补日覆盖率与随机 5 票 vs 旧源收盘比。  
- **混源**：同一历史区间不要 baostock/akshare/tushare 交叉 overwrite。  
- **权限报错**：空表或「积分不足」直接失败退出，不写半日文件。  
- **验收清单**：  
  - 补日后 `date.pkl` 含新日；  
  - `data_daily`/`data_ud_new` 同行数接近；  
  - `python main.py --live --skip-pipeline`（或 paper）能出非空清单；  
  - 估值近窗非空率与财务 PIT 末日股票数合理（约 5k 级）。

---

## 9. 与主链路关系

- 不改冻结名单 `b_pool_pv_32_oos`。  
- 数据源切换只影响 `data/`；因子 parquet 仍由 `--pipeline` 重算。  
- baostock 解封后可切回主源；Tushare 保持为备源脚本，不删。
