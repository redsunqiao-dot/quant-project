# 国金·GFlowNet 低相关性量价因子挖掘

## 基本信息
- source: 20260410-国金证券-Alpha掘金系列之二十二：基于GFlowNet的低相关性量价因子挖掘策略.pdf
- 假说: GFlowNet 可生成低相关量价公式族；图表18给出可复现日频表达式
- 已落地因子（越大越好）:
  - gfn_logvol_var10 = -ts_var(cs_demean(log(volume)), 10)
  - gfn_volpct_max20 = -cs_zscore(ts_max(pct_change(volume,10), 20))
  - gfn_sqrt_open_low_vol = cs_winsorize(sqrt(open)/(low*volume))
- checklist_score: 8
- 状态: candidate → validate
