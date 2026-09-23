# 量稳换手率因子卡

## 基本信息
- name: `turn_stable_20`
- 来源: 东方证券技术分析因子系列 · 量稳换手率
- 假说: 换手率稳定反映持续关注与成交承载，利于趋势；越稳越好故取负标准差
- 公式: `-std(turnover_ratio, 20)`
- 窗口: lookback=20
- 方向: 越大越好
- 数据依赖: data_daily
- checklist_score: 8

## 状态
已实现算子 `turn_stable`，可由验证门晋升为 active。
