# 日频批次：乖离 / PctTurn / 理想振幅 / 波动翻转反转

## 方正乖离率
- `20180806-方正…基于均线的乖离率选股因子.pdf`
- `fz_bias_60` = `-(close/MA60-1)`

## 东吴换手变化率
- `20210515-东吴…量稳换手率…pdf` 前情 PctTurn20
- `dw_pct_turn_20` = `-mean(turn/MA40_lag(turn)-1, 20)`
- STR≈现有 `turn_stable_20`，不重复登记

## 开源理想振幅
- `20200516-开源…振幅因子的隐藏结构.pdf`
- `ky_ideal_amp_20` = `-(mean(amp|高价25%)-mean(amp|低价25%))`，amp=`high/low-1`

## 方正日间反转-波动翻转（球队硬币子集）
- `20220611-方正…球队硬币因子构建.pdf`
- `fz_rev_vol_flip_20`：低波截面翻转 mean(ret,20) 后再取负
