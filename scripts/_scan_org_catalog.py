# 按机构分类全库归档：标题分级，不打开 PDF
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/Users/qiaojh/Downloads/阿波量化资料/0.量化阿波全套量化研报/按机构分类")
REG = Path("/Users/qiaojh/Documents/quant-project/factors/registry.json")
OUT = Path("/Users/qiaojh/Documents/quant-project/outputs/org_scan_catalog.txt")

EN_ORGS = {
    "Barclays",
    "Bernstein",
    "BofA",
    "CITI",
    "Deutsche Bank",
    "Goldman Sachs",
    "HSBC",
    "JPMorgan",
    "Jefferies",
    "Morgan Stanley",
    "Nomura",
    "UBS",
}

SKIP_NON_EQ = re.compile(
    r"固收|债券|信用|转债|可转债|利率债|宏观|汇率|黄金|商品|期货|期权|"
    r"策略周报|周报|月报|点评|快评|晨会|行情回顾|投资策略(?!.*因子)|"
    r"FOF|REITs|基金研究|资产配置|债市",
    re.I,
)
SKIP_FREQ = re.compile(
    r"L2|逐笔|委托|撤单|挂单|高频|分钟|tick|订单簿|集合竞价|"
    r"Agent|大模型|LLM|GPT|OpenClaw|GFlowNet|遗传规划|神经网络|"
    r"深度学习|图网络|Transformer|GRU|LSTM|XGBoost|FactorVAE|"
    r"指增|指数增强|行业轮动|风格轮动",
    re.I,
)
KEEP = re.compile(
    r"因子|选股|量价|换手|动量|反转|波动|成交量|筹码|"
    r"Alpha|alpha|截面|价量|影线|威廉|隔夜|振幅|跳跃"
)

# 标题已覆盖（仓库已有实现或本轮已落地/明确代理）
COVERED = [
    (re.compile(r"上下影线|威廉|UBL|蜡烛"), "ubl"),
    (re.compile(r"量稳换手"), "turn_stable_20"),
    (re.compile(r"优加换手|UTR|PctTurn"), "dw_pct_turn_20"),
    (re.compile(r"切割刀|CTR"), "ctr_*"),
    (re.compile(r"协偏度"), "db_csk_*"),
    (re.compile(r"高低位放量|高／低位放量|高/低位放量"), "gs_*vol*"),
    (re.compile(r"理想振幅|振幅因子的隐藏结构"), "ky_ideal_amp/turn"),
    (re.compile(r"长端动量"), "ky_long_mom*"),
    (re.compile(r"成交量冲击"), "xb_vol_shock"),
    (re.compile(r"路径凸性"), "ha_path_convex_20"),
    (re.compile(r"筹码(成本|收益|结构|分布)"), "sw_chip_cost_60"),
    (re.compile(r"留存筹码"), "minute-skip"),
    (re.compile(r"毛利率"), "yd_gp_delta"),
    (re.compile(r"RSI"), "gs_rsi_20"),
    (re.compile(r"PATV|持续异常交易量|异常交易量"), "zs_atv/patv"),
    (re.compile(r"拉锯|渔利"), "zs_tug/yuli"),
    (re.compile(r"飞蛾扑火|跳跃.*振幅|振幅.*跳跃"), "fz_mod_amp/taylor"),
    (re.compile(r"创新高股票"), "gs_near_high_252"),
    (re.compile(r"加速换手"), "ha_volup_turn_acc_20"),
    (re.compile(r"隔夜涨跌变为有效|知情交易"), "gs_ovn_turn_corr_20"),
    (re.compile(r"隔夜跳空|跳一跳"), "fz_jump_10"),
    (re.compile(r"ILLIQ|非流动性"), "zt_illiq_20"),
    (re.compile(r"特质波动"), "xn_idio_vol_20"),
    (re.compile(r"重拾自信"), "minute-skip"),
    (re.compile(r"一视同仁"), "minute-skip"),
    (re.compile(r"羊群"), "minute/l2-skip"),
    (re.compile(r"净换手率"), "l2-skip"),
    (re.compile(r"CPV|分时版"), "minute/cpv"),
    (re.compile(r"RPV聪明"), "minute-skip"),
    (re.compile(r"GFlowNet"), "gfn_*"),
    (re.compile(r"高波环境下的有效因子"), "cj_*"),
    (re.compile(r"乖离"), "fz_bias_60"),
]


def bucket(org: str, name: str) -> str:
    if org in EN_ORGS:
        return "overseas"
    if SKIP_NON_EQ.search(name) and not KEEP.search(name):
        return "skip_macro_credit"
    if SKIP_FREQ.search(name):
        return "skip_minute_l2_ml"
    if not KEEP.search(name):
        return "skip_other"
    for pat, _tag in COVERED:
        if pat.search(name):
            return "covered"
    return "uncovered_keep"


def main() -> None:
    factors = json.loads(REG.read_text(encoding="utf-8"))["factors"]
    n_fac = len(factors)
    counts: Counter = Counter()
    uncovered = []
    n_pdf = 0
    for p in ROOT.rglob("*.pdf"):
        n_pdf += 1
        org = p.parent.name
        b = bucket(org, p.name)
        counts[b] += 1
        if b == "uncovered_keep":
            uncovered.append((org, p.name))

    uncovered.sort(key=lambda x: (x[0], x[1]))
    lines = [
        f"n_pdf={n_pdf} registry_factors={n_fac}",
        f"overseas={counts['overseas']}",
        f"skip_macro_credit={counts['skip_macro_credit']}",
        f"skip_minute_l2_ml={counts['skip_minute_l2_ml']}",
        f"skip_other={counts['skip_other']}",
        f"covered_by_title={counts['covered']}",
        f"uncovered_keep={counts['uncovered_keep']}",
        "",
        "=== uncovered 日频选股标题（未与已实现关键词撞上）===",
    ]
    by_org = defaultdict(list)
    for org, name in uncovered:
        by_org[org].append(name)
    for org in sorted(by_org, key=lambda o: -len(by_org[o])):
        lines.append(f"\n[{org}] n={len(by_org[org])}")
        for name in by_org[org][:80]:
            lines.append(f"  {name}")
        if len(by_org[org]) > 80:
            lines.append(f"  ... +{len(by_org[org]) - 80}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))
    print("wrote", OUT, "uncovered", len(uncovered))


if __name__ == "__main__":
    main()
