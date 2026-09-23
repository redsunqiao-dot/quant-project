# 日频批次：MAX / 异质波动 / Parkinson

## MAX 彩票（东方波动率系列相关）
- 参考《波动率因子的逻辑与非对称使用》中 MAXRet 对波动异象的解释
- `df_maxret_20` = `-max(ret, 20)`

## 异质波动代理（西南异质波动率简化）
- 原文需逐日 Barra 截面回归；此处用「日收益减截面均值」的滚动标准差近似
- `xn_idio_vol_20` = `-std(ret - cs_mean(ret), 20)`

## Parkinson 极差波动
- `pk_range_vol_20` = `-sqrt(sum(ln(H/L)^2)/(4 N ln2))`

## 本批暂缓
安信实现偏度（需 5 分钟）、方正水中行舟（需分钟成交额相关）。
