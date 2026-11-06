# -*- coding: utf-8 -*-
"""Experiment figures in the style of the user's own ErgoSight and FEGAR figures: flat colours from the Nature reference
palette, white panels with a dashed grid, bold labels, a white legend or note box inside the panel, bold letters below.
Every row is four panels of one kind with the same axes geometry, or one full-width panel."""
import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib import colors as mcolors, cm
from house import *
import figs_rich2 as R2

X = json.load(open(os.path.join(HERE, "extra.json")))
L8 = "abcdefgh"; CT = D["categories"]; SHADE = "#FBEDED"
BLUES = mcolors.LinearSegmentedColormap.from_list("blues", ["#FFFFFF", "#DCE9F5", "#B9D3EC", "#8FB6E0", "#5B8BC9", "#3B5FA0"])
PINKS = mcolors.LinearSegmentedColormap.from_list("pinks", ["#FFFFFF", "#FBEBF2", "#F2C5DA", "#E286AF", "#C25B8E"])
CODE = {"Text overflow": "TO", "Abnormal ellipsis": "AE", "Undisplayed content": "UC", "Cropped content": "CC", "Container overlap": "CO"}
import figs_rich as _RICH
SET7 = _RICH.SET7

def frame(ax, title, xl, yl, grid="both"):
    style(ax, title=title, xlabel=xl, ylabel=yl, grid=grid)

def inner_legend(ax, handles, loc="lower right", ncol=2):
    lg = ax.legend(handles=handles, loc=loc, ncol=ncol, fontsize=FS, frameon=True, fancybox=False, framealpha=0.95, edgecolor="#C9C9C9", handletextpad=0.3, columnspacing=0.8, borderpad=0.35, labelspacing=0.25, handlelength=1.0)
    lg.get_frame().set_linewidth(0.5)

def fig6():   # per category of each model, MCC under the baseline and under RuleSight (ErgoSight Fig. 9)
    specs = [(m, MNAME[m], "Zero-shot", lambda m, c: CT[f"{m}|{c}"]["zs"], lambda m, c: CT[f"{m}|{c}"]["rs"], D["rq1"][m]["rs"]["mcc"] - D["rq1"][m]["zs"]["mcc"], D["rq1"][m]["n"] if "n" in D["rq1"][m] else None) for m in MODELS]
    specs += [("haiku", "Haiku 5.5, chain-of-thought", "Chain-of-thought", lambda m, c: X["cat7"][f"bcot|{c}"], lambda m, c: X["cat7"][f"s3nomeas|{c}"], X["set9"]["s3nomeas"]["mcc"] - X["set9"]["bcot"]["mcc"], 400),
              ("haiku", "Haiku 5.5, few-shot", "Few-shot", lambda m, c: X["cat7"][f"s1|{c}"], lambda m, c: X["cat7"][f"s3nomeas|{c}"], X["set9"]["s3nomeas"]["mcc"] - X["set9"]["s1"]["mcc"], 400)]
    NI = {"haiku": 3635, "sonnet55": 400, "gpt54mini": 400, "gemini38flash": 400, "qwen3vl8b": 3635, "internvl35": 399}
    lo, hi = -0.25, 0.8
    for k, (m, title, base, fz, fr, dm, n) in enumerate(specs):
        fig, ax = panel(left=0.30, bottom=0.37)
        ax.fill_between([lo, hi], [lo, hi], [hi, hi], color=SHADE, lw=0, zorder=0); ax.plot([lo, hi], [lo, hi], ls=(0, (3, 2)), color="#444444", lw=0.6, zorder=2)
        hs = []
        for c in CATS:
            ax.scatter(fz(m, c), fr(m, c), s=16, color=CATC[c], edgecolor="white", linewidth=0.4, zorder=4); hs.append(dot(CATC[c], CODE[c], ms=3.6, mec="white"))
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box"); ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8]); ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])
        ax.set_xticklabels(["-0.2", "0", "0.2", "0.4", "0.6", "0.8"]); ax.set_yticklabels(["-0.2", "0", "0.2", "0.4", "0.6", "0.8"])
        note(ax, f"{('0.000' if abs(dm) < 5e-4 else f'{dm:+.3f}')} MCC\n{n or NI[m]:,} instances", loc="upper left"); inner_legend(ax, hs, loc="lower right", ncol=3)
        frame(ax, title, f"{base} MCC", "RuleSight MCC"); letter(fig, L8[k]); save(fig, f"Fig6{L8[k]}")
    legend_strip("Fig6_legend", None, rows=[[Patch(fc=SHADE, ec="#C9C9C9", lw=0.4, label="RuleSight higher"), Line2D([], [], ls=(0, (3, 2)), color="#444444", lw=0.6, label="Equal MCC")], [dot(CATC[c], f"{CODE[c]}  {c}", ms=3.6, mec="white") for c in CATS]])

def diverge(name, labels, rows, title, l, bold=None):
    fig, ax = panel(left=0.46, bottom=0.40, right=0.10); n = len(labels)
    for i, bars in enumerate(rows):
        k = len(bars); hgt = 0.78 / k
        for j, (fa, mi, col) in enumerate(bars):
            yy = i - 0.39 + hgt * (j + 0.5); ax.barh(yy, -100 * fa, hgt * 0.92, color=col, ec="white", lw=0.3, zorder=3); ax.barh(yy, 100 * mi, hgt * 0.92, color=col, ec="white", lw=0.3, zorder=3)
    ax.axvline(0, color=SPINE, lw=0.6, zorder=4); ax.set_xlim(-100, 100); ax.set_xticks([-100, -50, 0, 50, 100]); ax.set_xticklabels(["100", "50", "0", "50", "100"])
    ax.set_yticks(range(n)); ax.set_yticklabels(labels); ax.set_ylim(n - 0.5, -0.5); ax.tick_params(axis="y", length=0)
    if bold is not None: ax.get_yticklabels()[bold].set_fontweight("bold")
    frame(ax, title, None, None, grid="x"); ax.text(0.25, -0.15, "False alarm", transform=ax.transAxes, ha="center", va="top", fontweight="bold", color=INK); ax.text(0.75, -0.15, "Miss", transform=ax.transAxes, ha="center", va="top", fontweight="bold", color=INK)
    letter(fig, l); save(fig, name)

def fig7():
    C = G["cell"]; A = G["conf400"]; R = G["route"]
    for k, cat in enumerate(CATS + ["All"]):
        rows = [[(C[f"{m}|{cat}|zs"]["fa"], C[f"{m}|{cat}|zs"]["miss"], ZS), (C[f"{m}|{cat}|rs"]["fa"], C[f"{m}|{cat}|rs"]["miss"], RS)] for m in MODELS]
        diverge(f"Fig7{L8[k]}", [MSHORT[m] for m in MODELS], rows, CSHORT[cat] if cat != "All" else "All categories", L8[k])
    cfg = [("s0", "Zero-shot", ZS), ("bcot", "CoT", COT), ("s1", "Few-shot", FEW), ("s2", "Cards + ex.", "#E286AF"), ("s3nomeas", "RuleSight", RS), ("s3", "Unrouted", RS_L), ("s3norule", "w/o cards", GREY)]
    diverge("Fig7g", [c[1] for c in cfg], [[(A[f"{c[0]}|All"]["fa"], A[f"{c[0]}|All"]["miss"], c[2])] for c in cfg], "Haiku 5.5, settings", "g", bold=4)
    cl = CATS + ["All"]
    diverge("Fig7h", [CSHORT[c] for c in cl], [[(R[f"qwen3vl8b|s3|{c}"]["fa"], R[f"qwen3vl8b|s3|{c}"]["miss"], RS_L), (R[f"qwen3vl8b|s3nomeas|{c}"]["fa"], R[f"qwen3vl8b|s3nomeas|{c}"]["miss"], RS)] for c in cl], "Qwen3-VL-8B routing", "h")
    legend_strip("Fig7_legend", None, rows=[[box(ZS, "Zero-shot"), box(RS, "RuleSight (Ours)"), box(RS_L, "RuleSight, unrouted")], [box(COT, "Chain-of-thought (g)"), box(FEW, "Few-shot (g)"), box("#E286AF", "Cards + exemplars (g)"), box(GREY, "Without rule cards (g)")]])

def fig8():   # strips (FEGAR Fig. 12) and the full-width heatmap
    S = G["strips"]; spec = [("cards", "Adding rule cards", 1, RS), ("nocards", "Without rule cards", -1, PURPLE), ("route_haiku", "Routing, Haiku 5.5", 1, RS), ("route_qwen", "Routing, Qwen3-VL-8B", 1, RS)]
    cols = CATS + ["All"]; rng = np.random.default_rng(0)
    for k, (key, title, sign, col) in enumerate(spec):
        fig, ax = panel(left=0.34, bottom=0.48, h=1.44); allv = []
        for j, cat in enumerate(cols):
            d = np.array(S[key][cat]["d"]); obs = S[key][cat]["obs"]; allv += list(d); jx = rng.uniform(-0.3, 0.3, d.size); helps = sign * d > 1e-9; hurts = ~helps
            ax.scatter((j + jx)[hurts], d[hurts], s=3.0, color="#E6E6E6", edgecolor="#8A8A8A", linewidth=0.2, zorder=4)
            ax.scatter((j + jx)[helps], d[helps], s=3.0, color=col, edgecolor="white", linewidth=0.2, alpha=0.9, zorder=5)
            ax.plot([j - 0.36, j + 0.36], [obs, obs], color=INK, lw=1.3, zorder=6, solid_capstyle="butt")
        top, bot = max(allv), min(allv); span = top - bot; ax.set_ylim(min(bot - 0.06 * span, -0.03), max(top + 0.08 * span, 0.03))
        ax.axhline(0, color=SPINE, lw=0.5, ls=(0, (3, 2)), zorder=2); ax.set_xticks(range(len(cols))); ax.set_xticklabels([CSHORT[c] for c in cols], rotation=40, ha="right", rotation_mode="anchor"); ax.set_xlim(-0.6, len(cols) - 0.4)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4)); ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f")); frame(ax, title, None, "Change in MCC", grid="y"); letter(fig, L8[k]); save(fig, f"Fig8{L8[k]}")
    legend_strip("Fig8_legend", [dot(RS, "Step helps", mec="white"), dot(PURPLE, "Removal hurts (b)", mec="white"), dot("#E6E6E6", "Step hurts", mec="#8A8A8A"), Line2D([], [], color=INK, lw=1.3, label="Observed change")])
    R2.wide_heat.__globals__["SEQ"] = BLUES
    S7 = X["set7cat"]; rows = [t[1] for t in SET7]; b = [i for i, t in enumerate(SET7) if t[0] == "s3nomeas"]; lab = [CSHORT[c] for c in cols]
    groups = [("MCC (×100)", lab, [[100 * S7[f"{s}|{c}"]["mcc"] for c in cols] for s, _, _ in SET7], BLUES, -20, 80, "{:.0f}"), ("F1 (×100)", lab, [[100 * S7[f"{s}|{c}"]["f1"] for c in cols] for s, _, _ in SET7], BLUES, -20, 80, "{:.0f}"),
              ("False alarm (%)", lab, [[100 * S7[f"{s}|{c}"]["fa"] for c in cols] for s, _, _ in SET7], PINKS, 0, 80, "{:.0f}"), ("Miss (%)", lab, [[100 * S7[f"{s}|{c}"]["miss"] for c in cols] for s, _, _ in SET7], PINKS, 0, 80, "{:.0f}")]
    R2.wide_heat("Fig8e", "Configurations of Claude Haiku 5.5 per display-defect category, 400-instance subset", rows, groups, "e", bold_rows=b)

def fig9():
    import figs_rich as RICH
    for k, (c, t) in enumerate((("Text overflow", "Text overflow"), ("Abnormal ellipsis", "Abnormal ellipsis"), ("Undisplayed content", "Undisplayed content"), ("All", "All five categories"))): RICH.dots(f"Fig9{'abcd'[k]}", c, t, "abcd"[k])
    F = G["flip"]
    for k, (key, title) in enumerate((("Text overflow", "Text overflow"), ("Abnormal ellipsis", "Abnormal ellipsis"), ("Undisplayed content", "Undisplayed content"), ("All", "All instances"))):
        fig, ax = panel(left=0.40, bottom=0.36)
        for i, m in enumerate(MODELS):
            f = F[f"{m}|{key}"]; fx, br = f["fixed_pct"], f["broken_pct"]; net = fx - br
            ax.barh(i, -br, 0.64, color=ZS, ec="white", lw=0.3, zorder=3); ax.barh(i, fx, 0.64, color=RS, ec="white", lw=0.3, zorder=3)
            ax.scatter(net, i, s=8, marker="D", color="white", edgecolor=INK, linewidth=0.5, zorder=5); ax.text(55, i, f"{net:+.1f}", ha="right", va="center", fontsize=FS, color=RS_D if net > 0 else "#6A6A6A")
        ax.axvline(0, color=SPINE, lw=0.6, zorder=4); ax.set_xlim(-40, 56); ax.set_xticks([-30, 0, 30]); ax.set_xticklabels(["30", "0", "30"])
        ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([MSHORT[m] for m in MODELS]); ax.set_ylim(len(MODELS) - 0.5, -0.5); ax.tick_params(axis="y", length=0)
        frame(ax, title, "Instances (%)", None, grid="x"); letter(fig, "efgh"[k]); save(fig, f"Fig9{'efgh'[k]}")
    legend_strip("Fig9_legend", [dot(RS, "Corrected by RuleSight", marker="s"), dot(ZS, "Lost by RuleSight", marker="s"), dot(RICH.TCOL["both"], "Correct under both", marker="s", mec="none"), dot(RICH.TCOL["neither"], "Wrong under both", marker="s", mec="none"), dot("white", "Net change", marker="D", ms=3.6, mec=INK)])

TYPES = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "none"]; TLAB = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "No"]
CORRECT = {"Abnormal ellipsis": {"D7"}, "Text overflow": {"D6"}, "Undisplayed content": {"D3", "D4"}, "Cropped content": {"D8"}, "Container overlap": {"D1", "D2"}}
def fig10():
    N_LAB = {"Abnormal ellipsis": 431, "Text overflow": 310, "Undisplayed content": 365, "Cropped content": 342, "Container overlap": 389}
    mats = [("Haiku 5.5, zero-shot", D["confusion"]["s0"], N_LAB), ("Haiku 5.5, RuleSight", D["confusion"]["s3nomeas"], N_LAB), ("Sonnet 5.5, zero-shot", G["conf_sonnet"]["s0"]["counts"], G["conf_sonnet"]["s0"]["nlab"]), ("Sonnet 5.5, RuleSight", G["conf_sonnet"]["s3nomeas"]["counts"], G["conf_sonnet"]["s3nomeas"]["nlab"])]
    for k, (title, counts, nlab) in enumerate(mats):
        fig, ax = panel(left=0.47, bottom=0.40)
        for i, c in enumerate(CATS):
            for j, t in enumerate(TYPES):
                v = counts.get(f"{c}|{t}", 0) / nlab[c]; ok = t in CORRECT[c]
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc=BLUES(min(v / 0.9, 1)), ec="white", lw=0.6, zorder=2))
                if ok: ax.add_patch(Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fc="none", ec=GT, lw=0.9, zorder=3))
                if v >= 0.10 or (ok and v >= 0.005): ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=FS - 0.5, color="white" if v > 0.62 else INK, fontweight="bold" if ok else "normal", zorder=4)
        ax.set_xlim(-0.5, len(TYPES) - 0.5); ax.set_ylim(len(CATS) - 0.5, -0.5); ax.set_xticks(range(len(TYPES))); ax.set_xticklabels(TLAB, rotation=90); ax.set_yticks(range(len(CATS))); ax.set_yticklabels([CSHORT[c] for c in CATS])
        frame(ax, title, "Reported category", None, grid=None); ax.tick_params(length=0)
        for sp in ax.spines.values(): sp.set_color("#9A9A9A"); sp.set_linewidth(0.5)
        letter(fig, "abcd"[k]); save(fig, f"Fig10{'abcd'[k]}")
    fig = plt.figure(figsize=(TEXTW, 0.30)); cax = fig.add_axes([0.38, 0.50, 0.28, 0.22]); cb = fig.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(0, 90), cmap=BLUES), cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.4); cb.set_ticks([0, 30, 60, 90]); cb.ax.tick_params(labelsize=FS, length=1.5, width=0.4, pad=1)
    fig.text(0.37, 0.61, "Share of the labeled defects (%)", ha="right", va="center", fontsize=FS); fig.patches.append(Rectangle((0.70, 0.46), 0.012, 0.30, transform=fig.transFigure, fc="white", ec=GT, lw=0.9)); fig.text(0.717, 0.61, "Labeled category", ha="left", va="center", fontsize=FS); save(fig, "Fig10_legend")
    rows, M1, M2, rc, bold = [], [], [], [], []
    for m in MODELS:
        for s, tag, col in (("s0", "zero-shot", ZS_D), ("s3nomeas", "RuleSight", RS)):
            rows.append(f"{MSHORT[m]}, {tag}"); rc.append(col); M1.append([100 * X["diag"][f"{m}|{s}|{c}"]["f1"] for c in CATS + ["All"]]); M2.append([100 * X["diag"][f"{m}|{s}|{c}"]["loc"] for c in CATS + ["All"]])
            if s == "s3nomeas": bold.append(len(rows) - 1)
    lab = [CSHORT[c] for c in CATS + ["All"]]
    R2.wide_heat("Fig10e", "Category F1 and localization recall of every model on the defective instances, per display-defect category", rows, [("Category F1 (×100)", lab, M1, BLUES, 0, 80, "{:.0f}"), ("Localization recall at IoU 0.5 (×100)", lab, M2, BLUES, 0, 80, "{:.0f}")], "e", bold_rows=bold, row_colors=rc, cell_h=0.118, left=0.98, bold_values=False)

def gbars(ax, vals0, vals1, err0=None, err1=None, w=0.38):
    x = np.arange(len(vals0))
    ax.bar(x - w / 2, vals0, w, color=ZS, ec="white", lw=0.3, zorder=3); ax.bar(x + w / 2, vals1, w, color=RS, ec="white", lw=0.3, zorder=3)
    if err0 is not None: ax.errorbar(x - w / 2, vals0, yerr=err0, fmt="none", ecolor=INK, elinewidth=0.5, capsize=1.2, capthick=0.5, zorder=4); ax.errorbar(x + w / 2, vals1, yerr=err1, fmt="none", ecolor=INK, elinewidth=0.5, capsize=1.2, capthick=0.5, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels([MSHORT[m] for m in MODELS], rotation=40, ha="right", rotation_mode="anchor"); ax.set_xlim(-0.6, len(vals0) - 0.4)

def fig11():   # diagnostic quality and cost (paper Fig. 13)
    Q = D["rq4"]; R1 = D["rq1"]
    specs = [("Rule citation rate", "Reports citing their rule (%)", lambda m, s: Q[m][s]["rule_cited"], (0, 105)), ("Category F1", "Category F1", lambda m, s: Q[m][s]["category_f1"], (0, 0.62)), ("Localization recall", "Recall at IoU 0.5", lambda m, s: Q[m][s]["loc_recall"], (0, 0.32)), ("API cost", "USD per 1,000 screenshots", lambda m, s: R1[m]["cost"][0 if s == "s0" else 1], None)]
    for k, (title, yl, f, ylim) in enumerate(specs):
        fig, ax = panel(left=0.34, bottom=0.42)
        v0 = [f(m, "s0") for m in MODELS]; v1 = [f(m, "s3nomeas") for m in MODELS]; gbars(ax, v0, v1)
        if ylim: ax.set_ylim(*ylim); ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        else:
            ax.set_yscale("symlog", linthresh=0.5, linscale=0.4); ax.set_ylim(0, 40); ax.set_yticks([0, 1, 10]); ax.set_yticklabels(["0", "1", "10"])
            for i, m in enumerate(MODELS):
                if f(m, "s3nomeas") == 0: ax.text(i, 0.05, "local", ha="center", va="bottom", rotation=90, fontsize=FS, color="#6A6A6A")
        frame(ax, title, None, yl, grid="y"); letter(fig, "abcd"[k]); save(fig, f"Fig11{'abcd'[k]}")
    def stat(key, sc=1.0):
        out = []
        for s in ("s0", "s3nomeas"):
            mv, ev = [], []
            for m in MODELS: v = np.asarray(X["rq4x"][f"{m}|{s}"][key], float) * sc; mv.append(float(v.mean())); ev.append(float(1.96 * v.std(ddof=1) / np.sqrt(v.size)))
            out.append((mv, ev))
        return out
    for k, (key, title, yl, sc) in enumerate((("lat", "Latency per screenshot", "Seconds", 1.0), ("tok", "Output tokens per call", "Tokens", 1.0), ("inp", "Input tokens per call", "Thousand tokens", 1e-3), ("nd", "Defects per screenshot", "Reported defects", 1.0))):
        (m0, e0), (m1, e1) = stat(key, sc); fig, ax = panel(left=0.34, bottom=0.42); gbars(ax, m0, m1, e0, e1)
        ax.set_ylim(0, 1.18 * max(max(m0), max(m1))); ax.yaxis.set_major_locator(ticker.MaxNLocator(4)); frame(ax, title, None, yl, grid="y"); letter(fig, "efgh"[k]); save(fig, f"Fig11{'efgh'[k]}")
    legend_strip("Fig11_legend", [box(ZS, "Zero-shot"), box(RS, "RuleSight (Ours)"), Line2D([], [], color=INK, lw=0.5, marker="_", label="95% confidence interval of the mean, (e) to (h)")])

def fig13():
    import figs_rich as RICH
    RICH.fig13()

if __name__ == "__main__":
    fig6(); fig7(); fig8(); fig9(); fig10(); fig11(); fig13(); print("clean figures written")
