# 切割轨 · CTR 日频子集（第 8–9 课）

来源：换手率切割刀 CTR、因子切割论「理想反转」、成交量切割动量（聪明版日频代理）。

## 假说

- 信息在「安静」交易日（低换手/低量）更易形成持续定价；「拥挤」日（高换手/高量）更易反转。
- 高振幅日过度反应更强，切割后取反转 leg。

## 公式（窗口 20，窗内按该股中位数切割）

| 因子 | 公式要点 |
|------|----------|
| `ctr_turn_spread_20` | mean(r \| turn≤med) − mean(r \| turn>med) |
| `ctr_vol_cut_20` | mean(r \| vol≤med) − mean(r \| vol>med) |
| `ctr_ideal_amp_rev_20` | −mean(r \| amp>med)，amp=(high/low−1) |

## 验证

`python scripts/cutting_track_run.py` → `outputs/cutting_track/`；**不过验证门不进合成/ live**。
