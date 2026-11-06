# -*- coding: utf-8 -*-
"""The tables behind the figures of the results section, from the same data as the figures (catprm.json, figdata.json, extra.json)."""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, "catprm.json"))); D = json.load(open(os.path.join(HERE, "figdata.json"))); G = D["G"]; X = json.load(open(os.path.join(HERE, "extra.json")))
MODS = [("haiku", "Claude Haiku 5.5"), ("sonnet55", "Claude Sonnet 5.5"), ("gpt54mini", "GPT-5.4 mini"), ("gemini38flash", "Gemini 3.8 Flash"), ("qwen3vl8b", "Qwen3-VL-8B"), ("internvl35", "InternVL3.5-8B")]
CATS = ["Text overflow", "Abnormal ellipsis", "Undisplayed content", "Cropped content", "Container overlap"]; CS = ["Overflow", "Ellipsis", "Undisplayed", "Cropped", "Overlap"]
b = lambda s, on: f"\\textbf{{{s}}}" if on else s
T = {}
# A: MCC per model and category, both oracles (behind Figs. 7 and 3D bars)
r = ["\\begin{table}[!htbp]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{2.2pt}\n\\caption{MCC of every model per display-defect category under zero-shot prompting (ZS) and RuleSight (RS), the values behind Fig.~\\ref{fig:bars3d}. Bold marks the higher MCC of the two oracles.}\n\\label{tab:catmcc}\n",
     "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l" + "rr" * 6 + "@{}}\n\\toprule\n & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{c}}}" for c in CS + ["All"]) + " \\\\\n" + "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(6)) + "\nModel & " + " & ".join(["ZS & RS"] * 6) + " \\\\\n\\midrule\n"]
for m, nm in MODS:
    cells = []
    for c in CATS + ["All"]:
        a_, b_ = P[f"{m}|s0|{c}"]["MCC"], P[f"{m}|s3nomeas|{c}"]["MCC"]; cells += [f"{a_:.3f}", b(f"{b_:.3f}", b_ > a_ + 1e-9)]
    r.append(f"{nm} & " + " & ".join(cells) + " \\\\\n")
r.append("\\bottomrule\n\\end{tabular*}\n\\end{table}\n"); T["catmcc"] = "".join(r)
# B: false-alarm and miss rates over all categories (behind Fig. 7)
r = ["\\begin{table}[!htbp]\n\\centering\n\\footnotesize\n\\caption{False-alarm rate on the defect-free instances and miss rate on the defective instances over all categories (\\%), the values behind Fig.~\\ref{fig:errors}(f). Bold marks the lower rate.}\n\\label{tab:errors}\n\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrrrrr@{}}\n\\toprule\n & \\multicolumn{3}{c}{False alarm} & \\multicolumn{3}{c}{Miss} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nModel & ZS & RS & Change & ZS & RS & Change \\\\\n\\midrule\n"]
for m, nm in MODS:
    za, ra = 100 * P[f"{m}|s0|All"]["FA"], 100 * P[f"{m}|s3nomeas|All"]["FA"]; zm, rm = 100 * P[f"{m}|s0|All"]["MISS"], 100 * P[f"{m}|s3nomeas|All"]["MISS"]
    r.append(f"{nm} & {za:.1f} & {b(f'{ra:.1f}', ra < za - 1e-9)} & {ra - za:+.1f} & {zm:.1f} & {b(f'{rm:.1f}', rm < zm - 1e-9)} & {rm - zm:+.1f} \\\\\n")
r.append("\\bottomrule\n\\end{tabular*}\n\\end{table}\n"); T["errors"] = "".join(r)
# C: change in MCC of each component step per category with 95% bootstrap intervals (behind Fig. 8(a) to (d))
S = G["strips"]; steps = [("cards", "Adding the rule cards"), ("nocards", "Removing the rule cards"), ("route_haiku", "Routing, Claude Haiku 5.5"), ("route_qwen", "Routing, Qwen3-VL-8B")]
r = ["\\begin{table}[!htbp]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{2.5pt}\n\\caption{Change in MCC from each component step per display-defect category with its 95\\% paired bootstrap interval, the values behind Fig.~\\ref{fig:rq2}(a) to (d).}\n\\label{tab:steps}\n\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l" + "c" * 6 + "@{}}\n\\toprule\nStep & " + " & ".join(CS + ["All"]) + " \\\\\n\\midrule\n"]
for key, nm in steps:
    cells = []
    for c in CATS + ["All"]:
        d = np.array(S[key][c]["d"]); o = S[key][c]["obs"]; lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5); cells.append(f"{o:+.3f} \\newline ({lo:+.2f}, {hi:+.2f})")
    r.append(f"{nm} & " + " & ".join(f"\\makecell{{{x.split(' \\newline ')[0]}\\\\{x.split(' \\newline ')[1]}}}" for x in cells) + " \\\\\n")
r.append("\\bottomrule\n\\end{tabular*}\n\\end{table}\n"); T["steps"] = "".join(r)
# D: corrected and lost instances per model (behind Fig. 9)
F = G["flip"]
r = ["\\begin{table}[!htbp]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\caption{Instances that RuleSight corrected and lost against zero-shot prompting, as counts and as the net change in percentage points, the values behind Fig.~\\ref{fig:rq3}.}\n\\label{tab:flip}\n\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrrrrrrrr@{}}\n\\toprule\n & \\multicolumn{3}{c}{All instances} & \\multicolumn{3}{c}{Defective} & \\multicolumn{3}{c}{Defect-free} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\nModel & Corrected & Lost & Net & Corrected & Lost & Net & Corrected & Lost & Net \\\\\n\\midrule\n"]
for m, nm in MODS:
    cells = []
    for key in ("All", "pos", "neg"):
        f = F[f"{m}|{key}"]; net = f["fixed_pct"] - f["broken_pct"]; cells += [f"{f['fixed']:,}", f"{f['broken']:,}", b(f"{net:+.1f}", net > 0)]
    r.append(f"{nm} & " + " & ".join(cells) + " \\\\\n")
r.append("\\bottomrule\n\\end{tabular*}\n\\end{table}\n"); T["flip"] = "".join(r)
# E: share of the labeled defects reported under the labeled category, and category F1 and localization of Claude Haiku 5.5 (behind Fig. 10)
RULE = {"Abnormal ellipsis": ["D7"], "Text overflow": ["D6"], "Undisplayed content": ["D3", "D4"], "Cropped content": ["D8"], "Container overlap": ["D1", "D2"]}
NL = {"Abnormal ellipsis": 431, "Text overflow": 310, "Undisplayed content": 365, "Cropped content": 342, "Container overlap": 389}
def diag(cnt, nlab, c): return 100 * max(cnt.get(f"{c}|{t}", 0) for t in RULE[c]) / nlab[c]
r = ["\\begin{table}[!htbp]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\caption{Share of the labeled defects reported under their labeled category (\\%) for Claude Haiku 5.5 and Claude Sonnet 5.5, and category F1 and localization recall of Claude Haiku 5.5, per display-defect category, the values behind Fig.~\\ref{fig:rq3b}. Bold marks the higher value.}\n\\label{tab:diag}\n\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrrrrrrr@{}}\n\\toprule\n & \\multicolumn{2}{c}{Haiku, labeled} & \\multicolumn{2}{c}{Sonnet, labeled} & \\multicolumn{2}{c}{Haiku, category F1} & \\multicolumn{2}{c}{Haiku, localization} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\nCategory & ZS & RS & ZS & RS & ZS & RS & ZS & RS \\\\\n\\midrule\n"]
CF = D["confusion"]; CSN = G["conf_sonnet"]
for c, cs in zip(CATS, CS):
    hz, hr = diag(CF["s0"], NL, c), diag(CF["s3nomeas"], NL, c); sz, sr = diag(CSN["s0"]["counts"], CSN["s0"]["nlab"], c), diag(CSN["s3nomeas"]["counts"], CSN["s3nomeas"]["nlab"], c)
    fz, fr = X["catf1"][f"s0|{c}"]["f1"], X["catf1"][f"s3nomeas|{c}"]["f1"]; lz, lr = X["catf1"][f"s0|{c}"]["loc"], X["catf1"][f"s3nomeas|{c}"]["loc"]
    r.append(f"{c} & {hz:.1f} & {b(f'{hr:.1f}', hr > hz)} & {sz:.1f} & {b(f'{sr:.1f}', sr > sz)} & {fz:.3f} & {b(f'{fr:.3f}', fr > fz)} & {lz:.3f} & {b(f'{lr:.3f}', lr > lz)} \\\\\n")
r.append("\\bottomrule\n\\end{tabular*}\n\\end{table}\n"); T["diag"] = "".join(r)
# F: latency, tokens, reported defects and cost per call (behind Fig. 13(d) to (h))
R1 = D["rq1"]
r = ["\\begin{table}[!htbp]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{2.4pt}\n\\caption{Mean latency per screenshot, mean input and output tokens per call, mean reported defects per screenshot and API cost per 1,000 screenshots under zero-shot prompting (ZS) and RuleSight (RS), the values behind Fig.~\\ref{fig:rq4}(d) to (h).}\n\\label{tab:costx}\n\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l" + "rr" * 5 + "@{}}\n\\toprule\n & \\multicolumn{2}{c}{Latency (s)} & \\multicolumn{2}{c}{Input tokens} & \\multicolumn{2}{c}{Output tokens} & \\multicolumn{2}{c}{Defects} & \\multicolumn{2}{c}{USD per 1,000} \\\\\n" + "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(5)) + "\nModel & " + " & ".join(["ZS & RS"] * 5) + " \\\\\n\\midrule\n"]
for m, nm in MODS:
    z, s_ = X["rq4x"][f"{m}|s0"], X["rq4x"][f"{m}|s3nomeas"]; c0, c1 = R1[m]["cost"]
    cost = lambda v: "local" if v == 0 else f"{v:.2f}"
    r.append(f"{nm} & {np.mean(z['lat']):.1f} & {np.mean(s_['lat']):.1f} & {np.mean(z['inp']):,.0f} & {np.mean(s_['inp']):,.0f} & {np.mean(z['tok']):.0f} & {np.mean(s_['tok']):.0f} & {np.mean(z['nd']):.2f} & {np.mean(s_['nd']):.2f} & {cost(c0)} & {cost(c1)} \\\\\n")
r.append("\\bottomrule\n\\end{tabular*}\n\\end{table}\n"); T["costx"] = "".join(r)
# G: MCC at chosen thresholds per category for Claude Haiku 5.5 (behind Fig. 15)
TH = G["thr_cat"]; taus = np.array(G["taus"]); pick = [0.0, 0.3, 0.5, 0.7]; idx = [int(np.argmin(np.abs(taus - p))) for p in pick]
r = ["\\begin{table}[!htbp]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\caption{MCC of Claude Haiku 5.5 at four confidence thresholds per display-defect category under zero-shot prompting (ZS) and RuleSight (RS), the values behind Fig.~\\ref{fig:tau}. Bold marks the higher MCC.}\n\\label{tab:tau}\n\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l" + "rr" * 4 + "@{}}\n\\toprule\n & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$\\tau={p}$}}" for p in pick) + " \\\\\n" + "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(4)) + "\nCategory & " + " & ".join(["ZS & RS"] * 4) + " \\\\\n\\midrule\n"]
for c in CATS + ["All"]:
    cells = []
    for i in idx:
        a_, b_ = TH[f"s0|{c}"]["mcc"][i], TH[f"s3nomeas|{c}"]["mcc"][i]; cells += [f"{a_:.3f}", b(f"{b_:.3f}", b_ > a_ + 1e-9)]
    r.append(("All categories" if c == "All" else c) + " & " + " & ".join(cells) + " \\\\\n")
r.append("\\bottomrule\n\\end{tabular*}\n\\end{table}\n"); T["tau"] = "".join(r)
json.dump(T, open(os.path.join(HERE, "tables_extra.json"), "w")); print({k: len(v) for k, v in T.items()})
print(T["errors"][-900:])
