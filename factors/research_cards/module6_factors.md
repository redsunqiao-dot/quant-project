# 模块6·估值/质量/成长/PEG/股息率

## 来源
课件模块 6 样例策略（股息率&PEG、估值/质量/成长）；本仓库用 baostock 估值 + 财务 PIT + 分红缓存落地。

## 假说
低估值、高质量、高成长、高股息的股票截面上预期收益更高；PEG 综合估值与成长。

## 已落地
| name | 公式 | 数据 |
|---|---|---|
| m6_ep_ttm | `1/pe_ttm`（pe>0） | valuation |
| m6_bp_mrq | `1/pb_mrq`（pb>0） | valuation |
| m6_roe | `roe` | fundamental PIT |
| m6_yoy_ni | `clip(yoy_ni, -2, 5)` | fundamental PIT |
| m6_peg_inv | `-pe / max(yoy_eps,yoy_ni)`（增速>5%） | 两者 |
| m6_dy | 近一年现金分红 / 收盘价（PIT） | dividend cache + close |

## 组合层
PEG 与股息率组合策略仍可在组合/回测层叠加，非单因子字段。
