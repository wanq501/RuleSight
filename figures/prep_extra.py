import os
# -*- coding: utf-8 -*-
"""Per-instance data for the richer panels, from the exported records of the final runs (records.csv, results.json, screens.csv)."""
import csv, json, re, collections
import numpy as np
import sys
EXP = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RULESIGHT_EXPORT", "/tmp/exp4")   # the unpacked paper-data export
rows = list(csv.DictReader(open(f"{EXP}/records.csv", encoding="utf-8-sig")))
res = json.load(open(f"{EXP}/results.json", encoding="utf-8"))
TASK = {"TO": "Text overflow", "AE": "Abnormal ellipsis", "UC": "Undisplayed content", "CC": "Cropped content", "CO": "Container overlap"}
RULES = {"Text overflow": {"D6"}, "Abnormal ellipsis": {"D7"}, "Undisplayed content": {"D3", "D4"}, "Cropped content": {"D8"}, "Container overlap": {"D1", "D2"}}
CATS = list(RULES); MODELS = ["haiku", "sonnet55", "gpt54mini", "gemini38flash", "qwen3vl8b", "internvl35"]
cat_of = lambda sid: TASK[re.match(r"UL-([A-Z]{2})-", sid).group(1)]
R = collections.defaultdict(dict)
for r in rows:
    if r["dataset"] == "ui_lens" and r["pred"] != "": R[(r["model"], r["setting"])][r["sid"]] = r
raw = {(x["tag"], x["cond"], x["sid"]): x for x in res["records"]}
sub400 = sorted(R[("haiku", "bcot")]); full = sorted(R[("haiku", "s0")])
def mcc(t, p):
    t = np.asarray(t, bool); p = np.asarray(p, bool); tp = (t & p).sum(); tn = (~t & ~p).sum(); fp = (~t & p).sum(); fn = (t & ~p).sum()
    d = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))); return float((tp * tn - fp * fn) / d) if d else 0.0
def vec(m, s, sids): return np.array([int(R[(m, s)][x]["truth"]) for x in sids]), np.array([int(R[(m, s)][x]["pred"]) for x in sids])
rng = np.random.default_rng(20261101); OUT = {}
# --- RQ2: nine settings on the 400 subset, MCC with 95% bootstrap CI, false-alarm and miss rates
SET9 = ["s0", "bcot", "s1", "s2", "s3nomeas", "s3", "s3norule", "s3noex", "s3nover"]; idx = rng.integers(0, len(sub400), (2000, len(sub400)))
OUT["set9"] = {}
for s in SET9:
    t, p = vec("haiku", s, sub400); boots = [mcc(t[i], p[i]) for i in idx]
    OUT["set9"][s] = dict(mcc=mcc(t, p), lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)), fa=float((p[t == 0] == 1).mean()), miss=float((p[t == 1] == 0).mean()),
                          nd=[int(R[("haiku", s)][x]["n_defects"] or 0) for x in sub400])
# --- RQ2: per-category MCC along the component steps, with 95% bootstrap bands
STEPS = ["s0", "s1", "s2", "s3", "s3nomeas"]; OUT["steps"] = {}
for c in CATS:
    sids = [x for x in sub400 if cat_of(x) == c]; ii = rng.integers(0, len(sids), (600, len(sids)))
    for s in STEPS:
        t, p = vec("haiku", s, sids); b = [mcc(t[i], p[i]) for i in ii]; OUT["steps"][f"{c}|{s}"] = dict(mcc=mcc(t, p), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)))
# --- RQ3: instance transitions of Claude Haiku 5.5 on the full test split, per category
OUT["trans"] = {}
for c in CATS + ["All"]:
    sids = [x for x in full if c == "All" or cat_of(x) == c]; cls = []
    for x in sids:
        a = R[("haiku", "s0")][x]["correct"] == "true"; b = R[("haiku", "s3nomeas")][x]["correct"] == "true"; cls.append("fixed" if (b and not a) else "lost" if (a and not b) else "both" if a else "neither")
    OUT["trans"][c] = cls
# --- RQ3: rule citation under RuleSight per model (share of reported defects citing each rule), categories per defective report, confidence of the labeled category
OUT["cite"] = {}; OUT["ncat"] = {}; OUT["conf"] = {}
for m in MODELS:
    cnt = collections.Counter(); tot = 0
    for x in R[(m, "s3nomeas")]:
        rr = raw.get((m, "s3nomeas", x))
        try: cl = json.loads(rr["raw"]) if rr and rr.get("raw") else []
        except Exception: cl = []
        for k in cl if isinstance(cl, list) else []:
            if isinstance(k, dict): tot += 1; cnt[str(k.get("rule") or "none")] += 1
    OUT["cite"][m] = {k: v / max(1, tot) for k, v in cnt.items()}
for s in ("s0", "s3nomeas"):
    for c in CATS:
        vals, confs = [], []
        for x in full:
            if cat_of(x) != c or R[("haiku", s)][x]["truth"] != "1": continue
            rr = raw.get(("haiku", s, x))
            try: cl = json.loads(rr["raw"]) if rr and rr.get("raw") else []
            except Exception: cl = []
            cl = [k for k in cl if isinstance(k, dict)] if isinstance(cl, list) else []
            vals.append(len({str(k.get("type")) for k in cl})); cf = [float(k.get("confidence") or 0) for k in cl if str(k.get("type")) in RULES[c]]
            if cf: confs.append(max(cf))
        OUT["ncat"][f"{s}|{c}"] = vals; OUT["conf"][f"{s}|{c}"] = confs
# --- RQ3 and RQ4: category F1 and localization per category, Claude Haiku 5.5, full split (means over defective instances)
OUT["catf1"] = {}
for s in ("s0", "s3nomeas"):
    allf, alll = [], []
    for c in CATS:
        d = [R[("haiku", s)][x] for x in full if cat_of(x) == c and R[("haiku", s)][x]["truth"] == "1"]
        f = [float(r["type_f1"] or 0) for r in d]; l = [float(r["loc_ap"] or 0) for r in d]; allf += f; alll += l
        OUT["catf1"][f"{s}|{c}"] = dict(f1=float(np.mean(f)), loc=float(np.mean(l)))
    OUT["catf1"][f"{s}|All"] = dict(f1=float(np.mean(allf)), loc=float(np.mean(alll)))
# --- RQ4: latency, output tokens, evidence and rule completeness per model
OUT["rq4x"] = {}
for m in MODELS:
    for s in ("s0", "s3nomeas"):
        d = R[(m, s)]; lat = [float(r["latency_s"] or 0) for r in d.values()]; tok = []
        for x in d:
            u = (raw.get((m, s, x)) or {}).get("usage") or {}; tok.append(int(u.get("completion_tokens") or 0))
        flag = [r for r in d.values() if r["pred"] == "1"]
        OUT["rq4x"][f"{m}|{s}"] = dict(lat=lat, tok=tok, ev=float(np.mean([r["evidence_ok"] == "true" for r in flag])) if flag else 0.0, rule=float(np.mean([r["rule_ok"] == "true" for r in flag])) if flag else 0.0)
json.dump(OUT, open("extra.json", "w"))
print("set9", {k: round(v["mcc"], 3) for k, v in OUT["set9"].items()})
print("catf1 All", {k: (round(v["f1"], 3), round(v["loc"], 3)) for k, v in OUT["catf1"].items() if k.endswith("All")})
print("trans TO", collections.Counter(OUT["trans"]["Text overflow"]), "| All", collections.Counter(OUT["trans"]["All"]))
print("cite haiku", sorted(((k, round(v, 3)) for k, v in OUT["cite"]["haiku"].items()), key=lambda t: -t[1])[:9])
print("ev/rule haiku", {s: (round(OUT["rq4x"][f"haiku|{s}"]["ev"], 3), round(OUT["rq4x"][f"haiku|{s}"]["rule"], 3)) for s in ("s0", "s3nomeas")})
print("tokens median", {m: (int(np.median(OUT["rq4x"][f"{m}|s0"]["tok"])), int(np.median(OUT["rq4x"][f"{m}|s3nomeas"]["tok"]))) for m in MODELS})
print("ndef mean", {s: round(float(np.mean(OUT["set9"][s]["nd"])), 2) for s in SET9})
# --- RQ2: per-category MCC of the seven settings of Table 6; RQ4: mean prompt and output tokens per call
SET7 = ["s0", "bcot", "s1", "s2", "s3nomeas", "s3", "s3norule"]; OUT["cat7"] = {}
for s in SET7:
    for c in CATS + ["All"]:
        sids = [x for x in sub400 if c == "All" or cat_of(x) == c]; t, p = vec("haiku", s, sids); OUT["cat7"][f"{s}|{c}"] = mcc(t, p)
OUT["tokens"] = {}
for m in MODELS:
    for s in ("s0", "s3nomeas"):
        us = [((raw.get((m, s, x)) or {}).get("usage") or {}) for x in R[(m, s)]]
        OUT["tokens"][f"{m}|{s}"] = dict(prompt=float(np.mean([int(u.get("prompt_tokens") or 0) for u in us])), completion=float(np.mean([int(u.get("completion_tokens") or 0) for u in us])))
json.dump(OUT, open("extra.json", "w"))
print("cat7 All", {s: round(OUT["cat7"][f"{s}|All"], 3) for s in SET7})
print("tokens", {k: (round(v["prompt"]), round(v["completion"])) for k, v in OUT["tokens"].items()})
print("trans", {c: dict(collections.Counter(OUT["trans"][c])) for c in ("Abnormal ellipsis", "Undisplayed content")})
print("catf1", {k: (round(v["f1"], 3), round(v["loc"], 3)) for k, v in OUT["catf1"].items()})
# --- wide heatmaps: per-category MCC, F1, false-alarm and miss rates of the seven settings; category F1 and localization of every model
def f1(t, p):
    t = np.asarray(t, bool); p = np.asarray(p, bool); tp = (t & p).sum(); fp = (~t & p).sum(); fn = (t & ~p).sum(); return float(2 * tp / max(1, 2 * tp + fp + fn))
OUT["set7cat"] = {}
for s in SET7:
    for c in CATS + ["All"]:
        sids = [x for x in sub400 if c == "All" or cat_of(x) == c]; t, p = vec("haiku", s, sids)
        OUT["set7cat"][f"{s}|{c}"] = dict(mcc=mcc(t, p), f1=f1(t, p), fa=float((p[t == 0] == 1).mean()), miss=float((p[t == 1] == 0).mean()))
OUT["diag"] = {}
for m in MODELS:
    for s in ("s0", "s3nomeas"):
        d = R[(m, s)]
        for c in CATS + ["All"]:
            rr = [r for x, r in d.items() if (c == "All" or cat_of(x) == c) and r["truth"] == "1"]
            OUT["diag"][f"{m}|{s}|{c}"] = dict(f1=float(np.mean([float(r["type_f1"] or 0) for r in rr])), loc=float(np.mean([float(r["loc_ap"] or 0) for r in rr])), n=len(rr))
        us = [((raw.get((m, s, x)) or {}).get("usage") or {}) for x in d]
        OUT["rq4x"][f"{m}|{s}"]["inp"] = [int(u.get("prompt_tokens") or 0) for u in us]
        OUT["rq4x"][f"{m}|{s}"]["cost"] = [1000 * float(r["cost_usd"] or 0) for r in d.values()]
json.dump(OUT, open("extra.json", "w"))
print("set7cat All", {s: tuple(round(v, 3) for v in OUT["set7cat"][f"{s}|All"].values()) for s in SET7})
print("diag All", {k: (round(v["f1"], 3), round(v["loc"], 3)) for k, v in OUT["diag"].items() if k.endswith("|All")})
print("sonnet per cat", {k: (round(v["f1"], 3), round(v["loc"], 3)) for k, v in OUT["diag"].items() if k.startswith("sonnet55")})
