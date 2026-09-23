# 日频批次：ILLIQ / 隔夜-日内 / 锚定反转

## ILLIQ（中投 / Amihud）
- source: `20120907-中投证券-ILLIQ非流动性因子的测算.pdf`
- `zt_illiq_20` = `mean(|ret|/money, 20)`，越大越不流动

## 隔夜-日内（中信建投逐鹿二十九）
- source: `20251125-中信建投-隔夜-日内异象因子及领先滞后分析.pdf`
- `cjx_intraday_rev_20` = `-mean(close/open-1, 20)`
- `cjx_overnight_mom_20` = `mean(open/close_lag-1, 20)`
- `cjx_overnight_vol_20` = `-std(overnight, 20)`
- `cjx_oi_spread_20` = `mean(overnight-intraday, 20)`
- 未落地：集合竞价量、量价相关等需开盘量字段

## 锚定反转（中银）
- source: `20220902-中银国际-锚定反转因子构建与增强.pdf`
- i≈10 日、j≈65 日
- `zy_anchor_rev` / `zy_anchor_rev_vol`（×窗内波动）

## 暂缓
方正「飞蛾扑火」：日跳跃度依赖分钟频。
