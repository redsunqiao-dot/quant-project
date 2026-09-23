# 研报因子卡模板（模块 1 十项清单 ≥8 再 register candidate）

## 基本信息
- name:
- 来源（券商/标题/日期）:
- 假说（一句话）:
- 公式:
- 窗口（整数）:
- 方向（越大越好？）:
- 数据依赖:
- checklist_score: /10

## 预处理约定
- 去极值:
- 标准化:
- 中性化:

## 回测约定
- 调仓:
- 成交时点:
- 成本假设:

## 十项清单（勾选）
- [ ] 未来函数
- [ ] 幸存者偏差
- [ ] 交易成本
- [ ] T+1 调仓
- [ ] 数据对齐 Shift
- [ ] 动态股票池
- [ ] 参数经济学解释
- [ ] 市值/行业中性化
- [ ] 多空单调性
- [ ] 样本外预期

## 入库命令
```bash
.venv/bin/python register_factor.py \
  --name <name> \
  --desc "..." \
  --formula "..." \
  --source "..." \
  --hypothesis "..." \
  --windows lookback=20 \
  --checklist-score 8 \
  --impl <op_or_placeholder> \
  --card-path factors/research_cards/<name>.md
```

## 状态机
idea → candidate →（实现算子+管线验证）→ active / inactive
