# 一次性扫描：按机构分类 PDF vs registry，筛日频选股候选
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path("/Users/qiaojh/Downloads/阿波量化资料/0.量化阿波全套量化研报/按机构分类")
REG = Path("/Users/qiaojh/Documents/quant-project/factors/registry.json")
OUT = Path("/Users/qiaojh/Documents/quant-project/outputs/org_scan_unregistered.txt")

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
PREFER = {
    "开源证券",
    "东吴证券",
    "华泰证券",
    "申万宏源",
    "申万宏源证券",
    "长江证券",
    "广发证券",
    "中信建投证券",
    "中信证券",
    "国泰君安",
    "海通证券",
    "兴业证券",
    "华创证券",
    "方正证券",
    "国盛证券",
    "西部证券",
    "中泰证券",
    "东方证券",
    "招商证券",
    "天风证券",
    "光大证券",
    "平安证券",
    "国金证券",
    "浙商证券",
    "华安证券",
    "源达信息",
    "东北证券",
    "东兴证券",
    "国联民生证券",
    "国泰海通证券",
    "华西证券",
    "山西证券",
    "中银国际",
    "中邮证券",
}

SKIP = re.compile(
    r"固收|债券|信用|转债|可转债|利率|宏观|汇率|黄金|商品|期货|期权|"
    r"策略周报|周报|月报|点评|快评|晨会|"
    r"L2|逐笔|委托|撤单|挂单|高频|分钟|tick|"
    r"Agent|大模型|LLM|GPT|OpenClaw|"
    r"指增|指数增强|行业轮动|风格轮动|择时",
    re.I,
)
KEEP = re.compile(
    r"因子|选股|量价|换手|动量|反转|波动|成交量|筹码|"
    r"Alpha|alpha|截面|价量|影线|威廉|"
    r"基本面|估值|盈利|质量|成长|红利|PEG|ROE|毛利率"
)


def norm(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("～", "~").replace("—", "-").replace("–", "-")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\.pdf$", "", s)
    return s


def score(org: str, name: str) -> int:
    s = 0
    if re.search(r"选股因子|量价因子|多因子|因子选股", name):
        s += 5
    if "因子" in name:
        s += 2
    if "选股" in name:
        s += 2
    if re.match(r"^202[3-6]", name):
        s += 3
    if org in PREFER:
        s += 2
    if re.search(r"高频|分钟|L2|日内|tick", name, re.I):
        s -= 8
    if re.search(r"指增|组合|轮动|择时", name):
        s -= 3
    return s


def main() -> None:
    factors = json.loads(REG.read_text(encoding="utf-8"))["factors"]
    src_keys = set()
    for f in factors:
        n = norm(f.get("source", ""))
        if n:
            src_keys.add(n)
            src_keys.add(n[:48])
            stem = re.sub(r"^\d{8}-[^-]+-", "", n)
            if stem:
                src_keys.add(stem[:40])

    cands = []
    already = 0
    n_pdf = 0
    for p in ROOT.rglob("*.pdf"):
        n_pdf += 1
        org = p.parent.name
        if org in EN_ORGS:
            continue
        name = p.name
        if not KEEP.search(name):
            continue
        if SKIP.search(name):
            continue
        n = norm(name)
        stem = re.sub(r"^\d{8}-[^-]+-", "", n)
        if n in src_keys or n[:48] in src_keys or (stem[:40] and stem[:40] in src_keys):
            already += 1
            continue
        cands.append((score(org, name), org, name))

    cands.sort(key=lambda x: (-x[0], x[1], x[2]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"n_pdf={n_pdf} unregistered={len(cands)} already_matched={already}",
        "",
        "=== TOP 80 ===",
    ]
    for s, org, name in cands[:80]:
        lines.append(f"{s:2d}\t{org}\t{name}")
    lines.append("")
    lines.append("=== org counts ===")
    for org, cnt in Counter(o for _, o, _ in cands).most_common(25):
        lines.append(f"{cnt}\t{org}")
    lines.append("")
    lines.append("=== ALL (score>=4) ===")
    for s, org, name in cands:
        if s < 4:
            break
        lines.append(f"{s:2d}\t{org}\t{name}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("done", OUT)
    print(lines[0])
    print("\n".join(lines[3:23]))


if __name__ == "__main__":
    main()
