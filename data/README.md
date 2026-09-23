# 本地数据目录

本目录**不提交到 Git**（体量大、可重拉）。仓库内仅保留本说明与 `.gitkeep`。

## 建议布局

```
data/
  date.pkl                 # 交易日列表
  data_daily/              # 日线 OHLCV
  data_ret/                # 收益相关
  data_ud_new/             # 涨跌停等
  data_industry/           # 行业分类
  data_fundamental/        # 财务 PIT
  data_valuation/          # 估值
  data_barra/              # 可选
  _cache/                  # 拉取缓存
```

## 如何生成

在项目根目录、已安装依赖后：

```bash
python update_market_data.py
python update_industry_data.py
python update_fundamental_data.py   # baostock 限流时 workers=1
python update_valuation_data.py
```

具体参数以各脚本内说明为准。财务/估值若遇数据源黑名单，应暂停续拉，勿加大并发。
