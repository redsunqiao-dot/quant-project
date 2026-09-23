# CPV 综合因子卡（第十一课样板 · candidate）

## 基本信息
- name: `cpv_ubl_wr`
- 来源: 东吴金工《上下影线，蜡烛好还是威廉好？》；课程 day11 CPV 复现
- 假说: 蜡烛上下影线/实体刻画短线供需与博弈，威廉位置刻画超买超卖，合成后捕捉价量形态 alpha
- 公式: `CPV = wU*U + wB*B + wL*L + wWR*WR + wT*TREND`（默认等权）
- 子项:
  - `U=(U5+U20)/2`, `U_n = mean(UpperShadow,n)/mean(High-Low,n)`
  - `B=(B5+B20)/2`, `B_n = mean(|Close-Open|,n)/mean(High-Low,n)`
  - `L=(L5+L20)/2`, `L_n = mean(LowerShadow,n)/mean(High-Low,n)`
  - `WR_n = mean((Close-Low)/(High-Low)*100, n)`, `TREND = WR5 - WR20`
- 窗口: short=5, long=20
- 方向: 子项符号与研报一致后，再合成「越大越好」
- 数据依赖: `data_daily`（OHLC）
- checklist_score: 8/10

## 预处理约定
- 去极值: 3σ（可与管线一致；厚尾可改 MAD）
- 标准化: 截面 zscore
- 中性化: ln(市值) + 行业 OLS 残差

## 回测约定
- 调仓: 月频（月初/月末按复现设定）
- 成交时点: T+1（开盘或 VWAP，禁止信号日收盘成交）
- 成本假设: 双边约 0.3% 量级压测

## 十项清单
- [x] 未来函数（坚持 T+1）
- [x] 幸存者偏差（全 A 动态池，知悉局限）
- [x] 交易成本
- [x] T+1 调仓
- [x] 数据对齐
- [ ] 动态成分更严（退市完整覆盖仍弱）
- [x] 参数经济学解释（影线/威廉）
- [x] 市值/行业中性化
- [x] 多空单调性（验证门要求）
- [x] 样本外预期（发布日后另验）

## 入库状态
- 当前: **retired**（已由研报口径 `ubl` 替代）
- 请改用: `factors/research_cards/ubl.md` 与 registry 名 `ubl`
