"""Recompute the headline numbers of the RuleSight paper from the stored answers.

    python analysis/reproduce_tables.py      # prints the tables and checks them against the paper

Inputs: data/records.csv (one scored answer per model, setting and instance) and data/answers.jsonl.gz
(raw answers with token usage, cost and latency). Setting keys: s0 zero-shot, bcot chain-of-thought,
s1 few-shot, s2 rule cards and exemplars, s3nomeas RuleSight (routed), s3 RuleSight unrouted,
s3norule without rule cards, s3noex without exemplars, s3raw unrouted without the verifier,
uilens UI-Lens prompting (the per-category prompts published with UI-Lens)."""
import csv, gzip, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
T = lambda v: str(v).strip().lower() in ("1", "true", "1.0")
UL = defaultdict(dict)
for r in csv.DictReader(open(ROOT / "data" / "records.csv", encoding="utf-8-sig")):
    if r["dataset"] == "ui_lens": UL[(r["model"], r["setting"])][r["sid"]] = r
FULL = sorted(UL[("haiku", "s0")]); SUB = sorted(UL[("sonnet55", "s0")])
SPLIT = {"haiku": FULL, "qwen3vl8b": FULL, "sonnet55": SUB, "gpt54mini": SUB, "gemini38flash": SUB, "internvl35": SUB}
CODES = ("TO", "AE", "UC", "CC", "CO")

def confusion(m, s, sids):
    d = UL[(m, s)]; ids = [x for x in sids if x in d]
    return np.array([T(d[x]["truth"]) for x in ids]), np.array([T(d[x]["pred"]) for x in ids]), ids
def mcc(t, p):
    tp = np.sum(t & p, -1); tn = np.sum(~t & ~p, -1); fp = np.sum(~t & p, -1); fn = np.sum(t & ~p, -1)
    den = np.sqrt(((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).astype(float))
    return (tp * tn - fp * fn) / np.where(den == 0, 1, den)
def rates(t, p):
    tp = (t & p).sum(); fp = (~t & p).sum(); fn = (t & ~p).sum(); tn = (~t & ~p).sum()
    return dict(MCC=float(mcc(t, p)), F1=float(2 * tp / max(1, 2 * tp + fp + fn)), FA=100 * fp / max(1, fp + tn), MISS=100 * fn / max(1, fn + tp))
def boot(m, a, b, sids, n=10000, seed=20261101):
    A, B = UL[(m, a)], UL[(m, b)]; ids = [x for x in sids if x in A and x in B]
    t = np.array([T(A[x]["truth"]) for x in ids]); pa = np.array([T(A[x]["pred"]) for x in ids]); pb = np.array([T(B[x]["pred"]) for x in ids])
    idx = np.random.default_rng(seed).integers(0, len(ids), size=(n, len(ids))); d = np.sort(mcc(t[idx], pb[idx]) - mcc(t[idx], pa[idx]))
    return float(mcc(t, pb) - mcc(t, pa)), float(d[int(0.025 * n)]), float(d[int(0.975 * n) - 1])

usage = defaultdict(list)
with gzip.open(ROOT / "data" / "answers.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        x = json.loads(line); u = x.get("usage") or {}
        if (u.get("prompt_tokens") or 0) > 0:
            usage[(x["tag"], x["cond"])].append((x["sid"], float(x.get("cost") or 0), float(x.get("latency") or 0), u.get("prompt_tokens", 0), u.get("completion_tokens", 0)))
def cost(m, s, sids):   # averaged over the calls with recorded token usage, as in the paper
    keep = set(sids); v = [c for c in usage[(m, s)] if c[0] in keep]
    return dict(usd=1000 * np.mean([c[1] for c in v]), lat=np.mean([c[2] for c in v]), inp=np.mean([c[3] for c in v]), out=np.mean([c[4] for c in v]))

checks = []
def check(name, got, want, tol): checks.append((name, got, want, abs(got - want) <= tol))
print("Table 4  zero-shot -> RuleSight, paired instances")
want4 = {"haiku": (0.280, 0.433), "sonnet55": (0.448, 0.555), "gpt54mini": (0.210, 0.185), "gemini38flash": (0.432, 0.432), "qwen3vl8b": (0.020, 0.120)}
for m, sids in SPLIT.items():
    common = [x for x in sids if x in UL[(m, "s0")] and x in UL[(m, "s3nomeas")]]
    a = rates(*confusion(m, "s0", common)[:2]); b = rates(*confusion(m, "s3nomeas", common)[:2]); d, lo, hi = boot(m, "s0", "s3nomeas", common)
    print(f"  {m:14s} n={len(common):4d}  MCC {a['MCC']:.3f} -> {b['MCC']:.3f} ({d:+.3f}, 95% CI {lo:.3f} to {hi:.3f})  F1 {a['F1']:.3f} -> {b['F1']:.3f}  FA {a['FA']:.1f} -> {b['FA']:.1f}  miss {a['MISS']:.1f} -> {b['MISS']:.1f}")
    if m in want4: check(f"{m} zero-shot MCC", a["MCC"], want4[m][0], 0.0011); check(f"{m} RuleSight MCC", b["MCC"], want4[m][1], 0.0011)
print("Table 5  Claude Haiku 5.5 configurations on the 400-instance subset")
for s, w in {"s0": 0.284, "bcot": 0.355, "uilens": 0.477, "s1": 0.194, "s2": 0.460, "s3nomeas": 0.488, "s3": 0.478, "s3raw": 0.472, "s3norule": 0.243, "s3noex": 0.452}.items():
    r = rates(*confusion("haiku", s, SUB)[:2]); print(f"  {s:9s} MCC {r['MCC']:.3f}  F1 {r['F1']:.3f}"); check(f"haiku {s} MCC (subset)", r["MCC"], w, 0.0011)
check("haiku uilens F1 (subset)", rates(*confusion("haiku", "uilens", SUB)[:2])["F1"], 0.671, 0.0011)
print("Non-inferiority  Claude Haiku 5.5 with RuleSight against zero-shot Claude Sonnet 5.5, 400-instance subset")
A_, B_ = UL[("sonnet55", "s0")], UL[("haiku", "s3nomeas")]; ni = [x for x in SUB if x in A_ and x in B_]
t_ = np.array([T(A_[x]["truth"]) for x in ni]); pa_ = np.array([T(A_[x]["pred"]) for x in ni]); pb_ = np.array([T(B_[x]["pred"]) for x in ni])
ix_ = np.random.default_rng(20261101).integers(0, len(ni), size=(10000, len(ni))); dd_ = np.sort(mcc(t_[ix_], pb_[ix_]) - mcc(t_[ix_], pa_[ix_]))
diff_, low_ = float(mcc(t_, pb_) - mcc(t_, pa_)), float(dd_[int(0.05 * 10000)])
print(f"  difference {diff_:+.3f}, one-sided 95% lower bound {low_:+.3f}, margin -0.050: " + ("non-inferior" if low_ > -0.05 else "not shown"))
check("haiku RuleSight minus sonnet zero-shot", diff_, 0.040, 0.0011); check("non-inferiority lower bound", low_, -0.044, 0.003)
print("Routing on the full test split, unrouted s3 -> routed s3nomeas")
for m, w in {"haiku": (0.438, 0.433, 1.27, 1.08), "qwen3vl8b": (0.003, 0.120, None, None)}.items():
    a = rates(*confusion(m, "s3", FULL)[:2]); b = rates(*confusion(m, "s3nomeas", FULL)[:2]); c0, c1 = cost(m, "s3", FULL), cost(m, "s3nomeas", FULL)
    print(f"  {m:10s} MCC {a['MCC']:.3f} -> {b['MCC']:.3f}  USD per 1,000 screenshots {c0['usd']:.2f} -> {c1['usd']:.2f}")
    check(f"{m} unrouted MCC", a["MCC"], w[0], 0.0011); check(f"{m} routed MCC", b["MCC"], w[1], 0.0011)
    if w[2]: check(f"{m} unrouted cost", c0["usd"], w[2], 0.0051); check(f"{m} routed cost", c1["usd"], w[3], 0.0051)
print("Table 6  MCC per category, zero-shot -> RuleSight")
for m, sids in SPLIT.items():
    cells = []
    for code in CODES:
        cs = [x for x in sids if x.split("-")[1] == code]
        cells.append(f"{code} {rates(*confusion(m, 's0', cs)[:2])['MCC']:.3f}->{rates(*confusion(m, 's3nomeas', cs)[:2])['MCC']:.3f}")
    print(f"  {m:14s} " + "  ".join(cells))
check("haiku text overflow RuleSight MCC", rates(*confusion("haiku", "s3nomeas", [x for x in FULL if x.split("-")[1] == "TO"])[:2])["MCC"], 0.522, 0.0011)
print("Table 9  diagnostic quality and cost, zero-shot -> RuleSight")
num = lambda v: float(v) if v not in ("", "None") else None
for m, sids in SPLIT.items():
    vals = []
    for s in ("s0", "s3nomeas"):
        d = UL[(m, s)]; pos = [d[x] for x in sids if x in d and T(d[x]["truth"])]; flag = [d[x] for x in sids if x in d and T(d[x]["pred"])]
        cf = np.mean([num(r["type_f1"]) for r in pos if num(r["type_f1"]) is not None]); lr = np.mean([num(r["loc_ap"]) for r in pos if num(r["loc_ap"]) is not None])
        rc = 100 * np.mean([T(r["rule_ok"]) for r in flag if r["rule_ok"] not in ("", "None")]); vals.append((cf, lr, rc, cost(m, s, sids)))
    a, b = vals
    print(f"  {m:14s} category F1 {a[0]:.3f} -> {b[0]:.3f}  localization {a[1]:.3f} -> {b[1]:.3f}  rule cited {a[2]:.1f} -> {b[2]:.1f}%  "
          f"latency {a[3]['lat']:.1f} -> {b[3]['lat']:.1f} s  output tokens {a[3]['out']:.0f} -> {b[3]['out']:.0f}  USD per 1,000 {a[3]['usd']:.2f} -> {b[3]['usd']:.2f}")
    if m == "haiku":
        check("haiku category F1", b[0], 0.442, 0.0011); check("haiku localization recall", b[1], 0.213, 0.0011)
        check("haiku rule citation rate", b[2], 96.6, 0.051); check("haiku zero-shot cost", a[3]["usd"], 0.81, 0.0051); check("haiku RuleSight cost", b[3]["usd"], 1.08, 0.0051)
    if m == "sonnet55": check("sonnet zero-shot cost", a[3]["usd"], 14.66, 0.0051)
bad = [c for c in checks if not c[3]]
print(f"\n{len(checks) - len(bad)} of {len(checks)} checks against the paper passed")
for name, got, want, _ in bad: print(f"  MISMATCH {name}: {got:.4f} vs {want}")
sys.exit(1 if bad else 0)
