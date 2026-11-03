# 研报因子卡目录说明

本目录是「读研报」的文档产出，**不是**日更信号源。  
真正进日更的是冻结名单 `factors/composite_universe.json`；卡片只服务研究决策与 registry 溯源（`card_path`）。

---

## 目录怎么放

| 位置 | 用途 | 例子 |
|------|------|------|
| `_template.md` | 空白模板 | 新单因子深挖时复制 |
| `README.md` | 本说明 | — |
| **根目录** | 单因子 / 主题卡（一对一或一对少） | `ubl.md`、`turn_stable_20.md`、`module6_factors.md` |
| `batch/` | 批次扫库、多因子摘要、机构/课包汇总 | `daily_batch*.md`、`orgscan_daily.md`、`scan2026_daily.md` |
| `skip/` | 读完即跳过 / retired 备忘（短） | `xy_west_learn_186.md`、`htsc_gold_outlook.md` |

`register_factor.py --card-path` 与 `registry.json` 的 `card_path` 须写**实际相对路径**，例如：

- `factors/research_cards/ubl.md`
- `factors/research_cards/batch/orgscan_daily.md`
- `factors/research_cards/skip/xy_west_learn_186.md`

---

## 五种卡（厚薄不必统一）

1. **完整单因子对照**（推荐深挖时用）  
   对齐 `_template.md`：基本信息、预处理、回测、十项清单、入库命令、状态机。  
   **样板：`ubl.md`**。

2. **短单因子 / 主题卡**（历史够用即可）  
   有假说/公式/checklist，未铺满十项。不必为整齐全部扩成 UBL 厚度。

3. **批次 / 扫库卡**（`batch/`）  
   一张卡挂多个 registry 名，或整批跳过表 + 验证摘要。

4. **跳过备忘**（`skip/`）  
   来源 + retired/跳过原因即可（约 5～10 行）。

5. **模板** `_template.md`  
   只作空白骨架，不挂因子。

一张卡 **不必** 对应一个因子；一对多、一对零都正常。一对一的权威是 `registry.json` 的 `name`。

---

## 新研报建议流程

1. 读报 → 判断能否日频落地。  
2. **跳过** → 在 `skip/` 写短备忘（或写入当次 `batch/` 汇总的跳过表）。  
3. **要做** → 优先按 `ubl.md` 写完整卡到根目录 → `register` → impl 接线 → `--pipeline`。  
4. 批量扫日期/机构 → 用 `batch/` 汇总卡，可落地的再拆单卡或直接 register 并指向该汇总卡。

过门（active）≠ 进冻结池；进池须另做合成分 OOS 并手改 `composite_universe.json`。
