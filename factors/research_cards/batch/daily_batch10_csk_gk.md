# 日频批次10：上行/复合协偏度 + GK 低波

## 东北协偏度续
- `db_csk_up_120`：仅市场上行日子样本的 CSK_XYY（研报 RankIC 为正）
- `db_csk_comp_120`：截面 `z(上行) - z(下行原始)`（复合简化）

## Garman-Klass 极差波动
- 日度 `GK = 0.5*ln(H/L)^2 - (2ln2-1)*ln(C/O)^2`
- `gk_vol_20` = `-mean(GK, 20)`

## 本批暂缓
海通动态反转（需指数波段划分）、开源 ERR（需分钟 bar）。
