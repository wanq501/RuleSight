"""RuleSight scorer (design §13.6–13.7).

    python score.py --manifest data/RuleSight_data.json --runs runs/*.jsonl --out results/results.json [--boot 10000]

Per (model, setting, dataset): T1 P/R/F1 and AUROC, T2 type macro-F1, T3 localization AP@0.5, T4 rule-citation and
evidence consistency, cost and latency. Paired bootstrap CIs for the pre-registered contrasts (H1 S3 vs S0; H3 each
ablation vs S3; H4 S3 vs S3-meas on E-Defect) with Holm correction; H2 non-inferiority vs the detectors (margin 2 pt).
results.json carries one row per record for the console and a "stats" block for the paper.
"""
from __future__ import annotations
import argparse, glob, json, random
from collections import defaultdict
from pathlib import Path


def iou(a, b):
    ax, ay, aw, ah = a[:4]; bx, by, bw, bh = b[:4]
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx)); iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy; union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def iter_records(path, bad=None):
    """the JSON records of a run file, tolerant of damage: a line that carries two records glued together yields both,
    a line that is not JSON at all is skipped (counted in bad[0] when given)"""
    dec = json.JSONDecoder()
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            pos = 0
            while pos < len(line):
                try:
                    obj, end = dec.raw_decode(line, pos)
                except ValueError:
                    nxt = line.find("{", pos + 1)
                    if bad is not None: bad[0] += 1
                    if nxt < 0: break
                    pos = nxt; continue
                if isinstance(obj, dict): yield obj
                pos = end
                while pos < len(line) and line[pos] in " \t\r,": pos += 1


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


KB = json.loads((Path(__file__).resolve().parent / "rulekb" / "rules.json").read_text(encoding="utf-8"))
UL_MAP = KB.get("category_map", {}).get("UI-Lens", {})


TAU = KB.get("confidence_threshold", {})   # per setting, chosen on the UI-Lens Dev split only; a claim below it does not count


def rec_types(rec):
    """the defect types that count for a record: every verified claim at or above the setting's confidence threshold"""
    tau = float(TAU.get(rec.get("setting"), 0) or 0)
    ver = rec.get("verified")
    if ver is None or tau <= 0: return set(rec.get("types") or [])
    out = set()
    for d in ver:
        try: c = float(d.get("confidence") if d.get("confidence") is not None else 1.0)
        except Exception: c = 1.0
        if c >= tau: out.add(str(d.get("type", "")).upper())
    return out


def pred_of(rec, item):
    """the prediction for this item: for a UI-Lens (page, task) row, whether the counted defects include a rule of that task; else any defect"""
    task = item.get("ui_lens_task"); ts = rec_types(rec)
    if task and task in UL_MAP:
        return int(bool(ts & set(UL_MAP[task])))
    if rec.get("verified") is not None and float(TAU.get(rec.get("setting"), 0) or 0) > 0: return int(bool(ts))
    p = rec.get("pred")
    return None if p is None else int(p)


def type_ok(pred_type, truth_tag):
    """a truth box tagged with a UI-Lens task matches any rule the task maps to; otherwise the rule ids must be equal"""
    return str(pred_type) in UL_MAP.get(str(truth_tag), [str(truth_tag)])


def groups_of(rec, item):
    """the tables a record counts in: its item's dataset and the subsets the item belongs to (a UI-Lens-400 row also counts in UI-Lens)"""
    gs = [item.get("dataset")] + list(item.get("subsets") or [])
    if rec.get("dataset") and rec["dataset"] not in gs: gs.append(rec["dataset"])
    return [g for g in gs if g]


def per_record(rec, item):
    """T1 correctness, T2 type F1, T3 AP@0.5 (fraction of truth boxes matched by a predicted box of a matching type), T4 flags"""
    truth = item.get("label"); pred = pred_of(rec, item)
    m = {"truth": truth, "pred": pred, "correct": (truth is not None and pred is not None and int(truth) == int(pred))}
    t_types, p_types = set(item.get("types") or []), set(rec.get("types") or [])
    if t_types or p_types:
        tp = len(t_types & p_types); m["type_f1"] = prf(tp, len(p_types - t_types), len(t_types - p_types))[2]
    boxes = item.get("boxes") or []
    if boxes:
        hits = 0
        for tb in boxes:
            if any(iou(tb, d.get("box") or [0, 0, 0, 0]) >= 0.5 and (len(tb) < 5 or type_ok(d.get("type"), tb[4])) for d in rec.get("verified", [])): hits += 1
        m["loc_ap"] = hits / len(boxes)
    ver = rec.get("verified") or []
    if ver:
        m["rule_ok"] = all(str(d.get("rule", d.get("type"))).upper() == str(d.get("type")).upper() for d in ver) and (not t_types or bool({d["type"] for d in ver} & t_types))
        m["evidence_ok"] = all(bool(d.get("verifier")) and (d.get("verifier") or {}).get("basis") != "unverified" for d in ver)   # T4: every kept finding backed by a measurement
        m["verified_share"] = (sum(1 for d in ver if (d.get("verifier") or {}).get("basis") not in (None, "unverified")) / len(ver)) if ver else None
    return m


def aggregate(records, items):
    groups = defaultdict(list)
    for r in records:
        if "error" in r or r["sid"] not in items: continue
        for g in groups_of(r, items[r["sid"]]): groups[(r["model"], r["setting"], g)].append(r)
    table = {}
    for key, rs in groups.items():
        tp = fp = fn = tn = 0; tf1, lap, rok, eok, cost, lat = [], [], [], [], 0.0, []
        for r in rs:
            m = per_record(r, items[r["sid"]]); t, p = m["truth"], m["pred"]
            if t is not None and p is not None:
                tp += int(t == 1 and p == 1); fp += int(t == 0 and p == 1); fn += int(t == 1 and p == 0); tn += int(t == 0 and p == 0)
            if "type_f1" in m: tf1.append(m["type_f1"])
            if "loc_ap" in m: lap.append(m["loc_ap"])
            if "rule_ok" in m: rok.append(m["rule_ok"]); eok.append(m["evidence_ok"])
            cost += r.get("cost_usd", 0)
            if not r.get("shared_call"): lat.append(r.get("latency_s", 0))
        p, rc, f1 = prf(tp, fp, fn)
        lat_s = sorted(lat)
        table["|".join(key)] = {"model": key[0], "setting": key[1], "dataset": key[2], "n": len(rs), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(p, 4), "recall": round(rc, 4), "f1": round(f1, 4),
                                "type_macro_f1": round(sum(tf1) / len(tf1), 4) if tf1 else None, "loc_ap50": round(sum(lap) / len(lap), 4) if lap else None,
                                "rule_ok": round(sum(rok) / len(rok), 4) if rok else None, "evidence_ok": round(sum(eok) / len(eok), 4) if eok else None,
                                "cost_usd": round(cost, 4), "cost_per_shot": round(cost / len(rs), 6) if rs else None, "calls": sum(1 for r in rs if not r.get("shared_call")),
                                "cost_per_call": round(cost / max(1, sum(1 for r in rs if not r.get("shared_call"))), 6), "latency_median": lat_s[len(lat_s) // 2] if lat_s else None, "latency_p95": lat_s[int(0.95 * (len(lat_s) - 1))] if lat_s else None}
    return table


def paired_bootstrap(a_recs, b_recs, items, n_boot=10000, seed=20261101):
    """F1(A) - F1(B) over the screenshots both settings cover; returns (diff, lo, hi). Vectorised: 10,000 draws over 20,000 rows take seconds."""
    import numpy as np
    A = {r["sid"]: r for r in a_recs}; B = {r["sid"]: r for r in b_recs}
    sids = [s for s in A if s in B and items[s].get("label") is not None]
    if len(sids) < 10: return None
    t = np.array([int(items[s]["label"]) for s in sids], dtype=np.int8)
    pa = np.array([int(pred_of(A[s], items[s]) or 0) for s in sids], dtype=np.int8)
    pb = np.array([int(pred_of(B[s], items[s]) or 0) for s in sids], dtype=np.int8)
    def f1_vec(idx, p):
        tt, pp = t[idx], p[idx]
        tp = np.sum((tt == 1) & (pp == 1), axis=-1).astype(float); fp = np.sum((tt == 0) & (pp == 1), axis=-1).astype(float); fn = np.sum((tt == 1) & (pp == 0), axis=-1).astype(float)
        pr = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0); rc = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
        return np.where(pr + rc > 0, 2 * pr * rc / np.maximum(pr + rc, 1e-12), 0.0)
    n = len(sids); all_idx = np.arange(n)
    d0 = float(f1_vec(all_idx, pa) - f1_vec(all_idx, pb))
    rng = np.random.default_rng(seed); diffs = np.empty(n_boot)
    chunk = max(1, min(n_boot, 200_000_000 // max(1, n)))   # memory guard: draws × rows per block
    for k in range(0, n_boot, chunk):
        m = min(chunk, n_boot - k); idx = rng.integers(0, n, size=(m, n))
        diffs[k:k + m] = f1_vec(idx, pa) - f1_vec(idx, pb)
    diffs.sort()
    p_two = float(min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))))
    return {"n": n, "diff": round(d0, 4), "ci95": [round(float(diffs[int(0.025 * n_boot)]), 4), round(float(diffs[max(0, int(0.975 * n_boot) - 1)]), 4)], "p_two_sided": round(p_two, 4)}


def holm(pvals):
    """Holm step-down; returns adjusted p in the original order"""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i]); adj = [0] * len(pvals); run = 0
    for rank, i in enumerate(idx):
        run = max(run, (len(pvals) - rank) * pvals[i]); adj[i] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    man = json.loads(Path(a.manifest).read_text(encoding="utf-8")); items = {s["sid"]: s for s in man["screens"]}
    try:
        import run_api as _RA; cur_pv = _RA.PROMPT_VERSION
    except Exception:
        cur_pv = None
    stale = 0; bad = [0]
    latest = {}   # one record per (model, screen, setting): the last success, else the last attempt (a retried row must not count twice)
    for pat in a.runs:
        for f in sorted(glob.glob(pat)):
            if ".old-" in Path(f).name: continue   # answers made with earlier prompts, kept aside
            for r in iter_records(f, bad):
                k = (r.get("model"), r.get("sid"), r.get("setting"))
                if cur_pv and r.get("pv") != cur_pv: stale += 1; continue   # made with other prompts or rule cards
                if "error" not in r or k not in latest or "error" in latest[k]: latest[k] = r
    records = list(latest.values())
    if stale: print(f"{stale} records made under another prompt version left out")
    if bad[0]: print(f"{bad[0]} damaged lines skipped (two writers on one file, or a cut-off line)")
    import re as _re
    refused = defaultdict(set)   # model -> screens its provider refused on content grounds in any setting: left out of every setting of that model
    for r in records:
        if "error" in r and _re.search(r"prohibited content|content policy|moderation|flagged", str(r["error"]), _re.I): refused[r["model"]].add(r["sid"])
    if refused:
        records = [r for r in records if r.get("sid") not in refused.get(r.get("model"), set())]
        print("provider refusals (left out of that model's comparisons): " + ", ".join(f"{m} {len(v)}" for m, v in refused.items()))
    table = aggregate(records, items)
    by = defaultdict(list)
    for r in records:
        if "error" not in r and r["sid"] in items:
            for g in groups_of(r, items[r["sid"]]): by[(r["model"], r["setting"], g)].append(r)
    tests = []
    for (model, setting, ds), rs in list(by.items()):
        if setting == "s3" and ds in ("ui_lens", "ui_lens_400", "test_real", "test_real_400"):
            for other, name in (("s0", "H1 S3 vs S0"), ("s3norule", "H3 -rule"), ("s3nomeas", "H3 -meas"), ("s3noex", "H3 -exemplars"), ("s3nover", "H3 -verifier")):
                if (model, other, ds) in by:
                    bs = paired_bootstrap(rs, by[(model, other, ds)], items, a.boot)
                    if bs: tests.append({"test": name, "model": model, "dataset": ds, **bs})
        if setting == "s3" and ds in ("e_defect", "uicrit") and (model, "s3nomeas", ds) in by:
            bs = paired_bootstrap(rs, by[(model, "s3nomeas", ds)], items, a.boot)
            if bs: tests.append({"test": f"H4 measurement vs none ({ds})", "model": model, "dataset": ds, **bs})
    if tests:
        adj = holm([t["p_two_sided"] for t in tests])
        for t, p in zip(tests, adj): t["p_holm"] = round(p, 4)
    # H2 non-inferiority: S3 F1 >= detector F1 - 0.02 (point estimate; CI from the bootstrap when both cover the same screenshots)
    noninf = []
    for (model, setting, ds), row in table.items() if False else []: pass
    for k, row in table.items():
        if row["setting"] == "s3" and row["dataset"] in ("ui_lens", "ui_lens_400", "test_real", "test_real_400"):
            for det in ("det_owleye", "det_nighthawk"):
                dk = next((kk for kk, rr in table.items() if rr["setting"] == det and rr["dataset"] == row["dataset"]), None)
                if dk: noninf.append({"test": f"H2 S3 vs {det}", "model": row["model"], "dataset": row["dataset"], "f1_s3": row["f1"], "f1_det": table[dk]["f1"], "margin": 0.02, "non_inferior": row["f1"] >= table[dk]["f1"] - 0.02})
    # UI-Lens: the benchmark's own five tasks, so the numbers line up with the UI-Lens paper (a page is positive for a task when our verified
    # defects include one of the rules that task maps to); UICrit: per-rule P/R/F1 against the designers' critiques
    ul_map = UL_MAP
    ul_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    uc_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in records:
        if "error" in r or r["sid"] not in items: continue
        it = items[r["sid"]]
        if it.get("dataset") in ("ui_lens", "ui_lens_dev") and it.get("ui_lens_task"):   # per task, the test split and the Dev split kept apart
            task = it["ui_lens_task"]
            pred = bool(pred_of(r, it)); truth = bool(it.get("label"))
            k = (r["model"], r["setting"], task if it.get("dataset") == "ui_lens" else "dev:" + task); st = ul_stats[k]
            st["tp" if pred and truth else "fp" if pred else "fn" if truth else "tn"] += 1
        if it.get("dataset") == "uicrit":
            tt, pp = set(it.get("types") or []), set(r.get("types") or [])
            for rid in ("E1", "E2", "E3", "E4", "E6", "D1", "D2", "D6", "D8"):
                k = (r["model"], r["setting"], rid)
                if rid in pp and rid in tt: uc_stats[k]["tp"] += 1
                elif rid in pp: uc_stats[k]["fp"] += 1
                elif rid in tt: uc_stats[k]["fn"] += 1
    # the confidence threshold, chosen on Dev only: claims below tau do not count; F1 for each tau, per model and setting
    TAUS = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    def types_at(r, tau):
        ver = r.get("verified")
        if ver is None: return set(r.get("types") or [])
        out = set()
        for d in ver:
            try: c = float(d.get("confidence") if d.get("confidence") is not None else 1.0)
            except Exception: c = 1.0
            if c >= tau: out.add(str(d.get("type", "")).upper())
        return out
    tau_acc = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    for r in records:
        if "error" in r or r["sid"] not in items: continue
        it = items[r["sid"]]
        if it.get("dataset") != "ui_lens_dev": continue
        truth = bool(it.get("label")); task = it.get("ui_lens_task")
        for tau in TAUS:
            ts = types_at(r, tau)
            pred = bool(ts & set(UL_MAP[task])) if task in UL_MAP else bool(ts)
            tau_acc[(r["model"], r["setting"], tau)]["tp" if pred and truth else "fp" if pred else "fn" if truth else "tn"] += 1
    tau_curve = [{"model": m, "setting": s0, "tau": tau, **v, "f1": round(prf(v["tp"], v["fp"], v["fn"])[2], 4), "precision": round(prf(v["tp"], v["fp"], v["fn"])[0], 4), "recall": round(prf(v["tp"], v["fp"], v["fn"])[1], 4)}
                 for (m, s0, tau), v in sorted(tau_acc.items())]
    ui_lens_tasks = [{"model": m, "setting": s, "task": t, **v, "precision": round(prf(v["tp"], v["fp"], v["fn"])[0], 4), "recall": round(prf(v["tp"], v["fp"], v["fn"])[1], 4), "f1": round(prf(v["tp"], v["fp"], v["fn"])[2], 4)} for (m, s, t), v in sorted(ul_stats.items())]
    uicrit_rules = [{"model": m, "setting": s, "rule": t, **v, "precision": round(prf(v["tp"], v["fp"], v["fn"])[0], 4), "recall": round(prf(v["tp"], v["fp"], v["fn"])[1], 4), "f1": round(prf(v["tp"], v["fp"], v["fn"])[2], 4)} for (m, s, t), v in sorted(uc_stats.items()) if v["tp"] + v["fp"] + v["fn"]]
    runs = []
    for r in records:
        if "error" in r or r["sid"] not in items: continue
        m = per_record(r, items[r["sid"]])
        runs.append({"sid": r["sid"], "setting": r["setting"], "model": r["model"], "model_id": r.get("model_id"), "pred": pred_of(r, items[r["sid"]]), "types": r.get("types", []), "verified": r.get("verified", []),
                     "type_f1": m.get("type_f1"), "loc_ap": m.get("loc_ap"), "rule_ok": m.get("rule_ok"), "evidence_ok": m.get("evidence_ok"), "tokens_in": r.get("tokens_in", 0), "tokens_out": r.get("tokens_out", 0),
                     "cost_usd": r.get("cost_usd", 0), "latency_s": r.get("latency_s", 0)})
    out = {"executor": {"name": "RuleSight executor", "manifest": man.get("name"), "manifest_md5": man.get("md5"), "rulekb": man.get("rulekb"), "scored": Path(a.out).name, "prompt_version": cur_pv},
           "table": list(table.values()), "stats": {"refused_by_provider": {m: sorted(v) for m, v in refused.items()}, "tau_curve_dev": tau_curve, "paired_bootstrap": tests, "non_inferiority": noninf, "n_boot": a.boot, "alpha": 0.05, "family": "H1-H5 Holm", "ui_lens_tasks": ui_lens_tasks, "uicrit_rules": uicrit_rules}, "runs": runs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(runs)} records → {len(table)} rows; {len(tests)} paired tests; {len(noninf)} non-inferiority checks → {a.out}")
    for row in table.values():
        print(f"  {row['model']:10s} {row['setting']:10s} {row['dataset']:14s} n={row['n']:4d} P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f} cost/shot={row['cost_per_shot']}")


if __name__ == "__main__":
    main()
