# 按机构分类全库归类：已覆盖 / 跳过原因 / 剩余日频可落地
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/Users/qiaojh/Downloads/阿波量化资料/0.量化阿波全套量化研报/按机构分类")
REG = Path("/Users/qiaojh/Documents/quant-project/factors/registry.json")
OUT = Path("/Users/qiaojh/Documents/quant-project/outputs/org_scan_full.txt")

EN_ORGS = {
    "Barclays", "Bernstein", "BofA", "CITI", "Deutsche Bank",
    "Goldman Sachs", "HSBC", "JPMorgan", "Jefferies",
    "Morgan Stanley", "Nomura", "UBS",
}

# 标题关键词 -> 已有因子或本轮已落地
COVERED = re.compile(
    r"上下影线|蜡烛好还是威廉|量稳换手|优加换手|UTR|STR|"
    r"换手率切割刀|CTR|聪明换手|RPV|"
    r"协偏度|特质波动|ILLIQ|非流动性|"
    r"高低位放量|高／低位放量|高/低位放量|"
    r"理想振幅|振幅因子的隐藏结构|理想换手|"
    r"长端动量|路径凸性|筹码成本|留存筹码|"
    r"毛利率变动|异常交易量|PATV|"
    r"隔夜.?拉锯|渔利因子|"
    r"飞蛾扑火|股价跳跃及其对振幅|"
    r"加速换手|隔夜涨跌变为有效|"
    r"成交量冲击与隔夜|"
    r"乖离|惊恐|草木皆兵|"
    r"GFlowNet|gfn_|"
    r"模块.?6|EP.?TTM|账面市值",
    re.I,
)

SKIP_RULES = [
    ("en_house", lambda org, n: org in EN_ORGS),
    ("固收宏观", lambda org, n: bool(re.search(
        r"固收|债券|信用|转债|可转债|利率|宏观|汇率|黄金|商品|期货|期权|"
        r"FOF|REITs|信用债|利率债", n))),
    ("周报点评", lambda org, n: bool(re.search(r"策略周报|周报|月报|点评|快评|晨会", n))),
    ("分钟L2", lambda org, n: bool(re.search(
        r"L2|逐笔|委托|撤单|挂单|高频|分钟|tick|订单簿|集合竞价|分时", n, re.I))),
    ("ML_DL", lambda org, n: bool(re.search(
        r"神经网络|深度学习|遗传规划|DFQ|GRU|TCN|VAE|图网络|"
        r"Agent|大模型|LLM|GPT|OpenClaw|XGBoost|树模型|知识蒸馏", n, re.I))),
    ("指增轮动择时", lambda org, n: bool(re.search(r"指增|指数增强|行业轮动|风格轮动|择时|ETF轮动", n))),
    ("分析师另类", lambda org, n: bool(re.search(
        r"分析师|一致预期|研报情感|新闻情绪|舆情|公募重仓|黑白马|"
        r"财务附注|主营业务拆分|短贷长投|真实盈余|异常现金流", n))),
    ("covered", lambda org, n: bool(COVERED.search(n))),
]

KEEP = re.compile(
    r"因子|选股|量价|换手|动量|反转|波动|成交量|筹码|"
    r"Alpha|alpha|截面|价量|影线|威廉|"
    r"基本面|估值|盈利|质量|成长|红利"
)


def main() -> None:
    factors = json.loads(REG.read_text(encoding="utf-8"))["factors"]
    src_bits = []
    for f in factors:
        s = (f.get("source") or "") + (f.get("name") or "") + (f.get("desc") or "")
        src_bits.append(s)
    src_blob = " ".join(src_bits)

    n_pdf = 0
    buckets = defaultdict(list)
    leftover = []
    for p in ROOT.rglob("*.pdf"):
        n_pdf += 1
        org = p.parent.name
        name = p.name
        tagged = False
        for key, fn in SKIP_RULES:
            if fn(org, name):
                buckets[key].append(f"{org}\t{name}")
                tagged = True
                break
        if tagged:
            continue
        if not KEEP.search(name):
            buckets["非选股标题"].append(f"{org}\t{name}")
            continue
        leftover.append((org, name))

    leftover.sort(key=lambda x: (x[0], x[1]))
    lines = [
        f"n_pdf={n_pdf}",
        "bucket_counts:",
    ]
    for k, v in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"  {k}\t{len(v)}")
    lines.append(f"  leftover_daily_like\t{len(leftover)}")
    lines.append("")
    lines.append("=== leftover_daily_like ===")
    for org, name in leftover:
        lines.append(f"{org}\t{name}")
    lines.append("")
    lines.append("=== leftover_by_org ===")
    for org, cnt in Counter(o for o, _ in leftover).most_common():
        lines.append(f"{cnt}\t{org}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("done", OUT)
    print("\n".join(lines[:20]))
    print("leftover", len(leftover))
    for org, name in leftover[:40]:
        print(f"  {org} | {name}")


if __name__ == "__main__":
    main()
