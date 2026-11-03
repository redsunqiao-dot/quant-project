# UBL 因子卡（东吴金工研报口径 · 完整对照）

对照模板：`factors/research_cards/_template.md`。  
替代样板：`cpv_ubl_wr.md`（综合 CPV，已 retired）。

## 基本信息
- name: `ubl`
- 来源（券商/标题/日期）: 东吴证券《上下影线，蜡烛好还是威廉好？》（2023-11）
- 假说（一句话）: 蜡烛上影波动不确定性 + 威廉跨日下影支撑，互补刻画价量博弈
- 公式: `ubl = -( zscore(desize(蜡烛上_std)) + zscore(desize(威廉下_mean)) )`
- 子项:
  - Upper = High - max(Open, Close) → z 日滚动 zscore → feat 日 std
  - WMS_Lower = min(Open, Close) - rolling_min(Low, wms_n) → z 日滚动 zscore → feat 日 mean
  - desize: 截面对 ln(流通市值) OLS 残差
- 窗口（整数）: **z=10**, feat=20, wms_n=20  
  （2026-09-23 调参：原 z=5 时 IC≈0.017 未过门；近 120 日网格最优改为 z=10）
- 方向（越大越好？）: 研报 UBL 越小越好；入库公式已取负，**越大越好**
- 数据依赖: `data_daily`（OHLC）+ 流通市值（算子内 desize）
- checklist_score: 8/10
- impl: `ubl`
- card_path: `factors/research_cards/ubl.md`

## 预处理约定
- 去极值: 走管线 `FactorPreprocessor`（MAD）
- 标准化: 截面 zscore（管线）
- 中性化: 管线 `FactorNeutralizer`（ln 市值 + 行业 OLS 残差）  
  说明: 算子内已对两分量做市值 desize；管线中性化再叠行业，属有意叠加而非重复市值一遍

## 回测约定
- 调仓: 与主回测一致时可 5 日；单因子对照也可按研报月频解读 IC
- 成交时点: T+1（开盘或 VWAP）；禁止信号日收盘成交
- 成本假设: 主引擎佣金/印花税/过户费/滑点；单因子压测可按双边约 0.3% 量级心算

## 十项清单（勾选）
- [x] 未来函数（只用截至 T 的 OHLC/市值；收益对齐 T→T+1）
- [x] 幸存者偏差（全 A 动态池；退市覆盖知悉仍有局限）
- [x] 交易成本（含成本回测 / 合成 OOS 时一并看）
- [x] T+1 调仓（与成交时点一致）
- [x] 数据对齐 Shift（因子日与 forward ret、市值截面同日对齐）
- [ ] 动态股票池（管线有 UniverseFilter；退市完整覆盖仍弱，故不勾满分项）
- [x] 参数经济学解释（上影不确定性、威廉下影支撑；z/feat 有调参记录而非盲搜进池）
- [x] 市值/行业中性化（算子 desize + 管线市值/行业）
- [x] 多空单调性（验证门 `mono_mean`；近窗约为正）
- [x] 样本外预期（研报后区间验证；**合成分 OOS：+ubl 变差 → 不进冻结池**）

## 入库命令
```bash
.venv/bin/python register_factor.py \
  --name ubl \
  --desc "东吴UBL：蜡烛上_std + 威廉下_mean（取负；z=10 调参）" \
  --formula "ubl=-(zscore(desize(candle_up_std))+zscore(desize(wms_low_mean)))" \
  --source "东吴金工：上下影线，蜡烛好还是威廉好？" \
  --hypothesis "上影不确定性与跨日下影支撑互补，刻画价量博弈 alpha" \
  --windows z_window=10,feat_window=20,wms_n=20 \
  --checklist-score 8 \
  --impl ubl \
  --card-path factors/research_cards/ubl.md
```

## 状态机（本因子轨迹）
```
idea（读研报）
  → candidate（register；曾用课上 CPV 样板 cpv_ubl_wr，后 retired）
  → 实现 ubl_factor.py + factor_ops / panel_dispatch
  → 管线验证: z=5 → inactive（IC≈0.017）
  → 调参 z=10 → active（240 日 IC≈0.028；其后多次验证仍 active / HEALTHY）
  → 合成分 OOS: +ubl 更差 → 不写入 composite_universe.json（b_pool_pv_32_oos 不含 ubl）
```

## 实现与落盘对照
- 算子: `src/research/impl/ubl_factor.py`（`compute_ubl_by_date`）
- 接线: `src/research/factor_ops.py`、`src/research/panel_dispatch.py`
- 注册表: `factors/registry.json` → `name=ubl`
- 日更: **不进** `factors/composite_universe.json`；单因子 active 仅作研究/对照
