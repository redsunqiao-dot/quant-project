# UBL 因子卡（东吴金工研报口径）

## 基本信息
- name: `ubl`
- 来源: 东吴证券《上下影线，蜡烛好还是威廉好？》（2023-11）
- 假说: 蜡烛上影波动不确定性 + 威廉跨日下影支撑，互补刻画价量博弈
- 公式: `ubl = -( zscore(desize(蜡烛上_std)) + zscore(desize(威廉下_mean)) )`
- 子项:
  - Upper = High - max(Open, Close) → z 日 zscore → feat 日 std
  - WMS_Lower = min(Open, Close) - rolling_min(Low, wms_n) → z 日 zscore → feat 日 mean
  - desize: 截面对 ln(流通市值) OLS 残差
- 窗口（2026-09-23 调参）: **z=10**, feat=20, wms_n=20（原 z=5 IC≈0.017 未过门；网格近 120 日最优）
- 方向: 研报 UBL 越小越好；入库已取负，越大越好
- 数据依赖: data_daily
- checklist_score: 8
- impl: ubl

## 预处理约定（管线）
- 去极值 / 标准化 / 市值+行业中性化：走现有 FactorPreprocessor + FactorNeutralizer
- 算子内已含分量级市值 desize；管线再中性化可叠加行业

## 状态
inactive(z=5) → 调参 z=10 → 240 日验证 IC≈0.028 → **active**
