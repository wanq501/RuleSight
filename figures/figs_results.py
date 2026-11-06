# -*- coding: utf-8 -*-
"""Result figures at printed size (house.py): every panel 1.32 x 1.40 in, four per row, one legend strip per figure."""
import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch, Rectangle
from matplotlib import colors as mcolors
from house import *

L8 = "abcdefgh"

def fig6():   # every defective instance at the confidence with which each oracle reported its labeled category (FEGAR Fig. 10, ErgoSight Fig. 9)
    CS = json.load(open(os.path.join(HERE, "conf_scatter.json"))); rng = np.random.default_rng(7); stats = {}
    spec = [(f"{m}|s0", MNAME[m], "Zero-shot", ZS) for m in MODELS] + [("haiku|bcot", "Claude Haiku 5.5", "Chain-of-thought", COT), ("haiku|s1", "Claude Haiku 5.5", "Few-shot", FEW)]
    for k, (key, title, base, col) in enumerate(spec):
        d = CS[key]; x, y, lab = np.array(d["x"]), np.array(d["y"]), np.array(d["label"]); m_ = lab == 1; x, y = x[m_], y[m_]
        jx, jy = rng.uniform(-0.035, 0.035, x.size), rng.uniform(-0.035, 0.035, x.size); X, Y = np.clip(x + jx, -0.06, 1.06), np.clip(y + jy, -0.06, 1.06)
        up, down = y > x, y < x; tie = ~(up | down); big = x.size > 1000; s = 1.3 if big else 3.2; a = 0.55 if big else 0.8
        fig, ax = panel(left=0.30, bottom=0.37)
        ax.fill_between([-0.08, 1.08], [-0.08, 1.08], [1.08, 1.08], color=WIN, alpha=0.10, lw=0, zorder=0); ax.plot([-0.08, 1.08], [-0.08, 1.08], ls=(0, (3, 2)), color="#555555", lw=0.6, zorder=2)
        ax.scatter(X[tie], Y[tie], s=s, color=GREY_L, edgecolor="#9A9A9A", linewidth=0.15, alpha=a, zorder=3)
        ax.scatter(X[down], Y[down], s=s, color=col, edgecolor="white", linewidth=0.15, alpha=a + 0.1, zorder=4)
        ax.scatter(X[up], Y[up], s=s, color=RS, edgecolor="white", linewidth=0.15, alpha=a + 0.1, zorder=5)
        ax.set_xlim(-0.08, 1.08); ax.set_ylim(-0.08, 1.08); ax.set_aspect("equal", adjustable="box"); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"]); ax.set_yticklabels(["0", "0.5", "1"])
        sh = 100 * up.sum() / max(1, up.sum() + down.sum()); stats[key] = dict(n=int(x.size), up=int(up.sum()), down=int(down.sum()), share=round(sh, 1))
        note(ax, f"{sh:.0f}% of changes\nare increases", loc="lower right")
        style(ax, title=title, xlabel=f"{base} confidence", ylabel="RuleSight confidence"); letter(fig, L8[k]); save(fig, f"Fig6{L8[k]}")
    json.dump(stats, open(os.path.join(HERE, "fig6_stats.json"), "w"), indent=1); print("fig6", {k: v["share"] for k, v in stats.items()})
    legend_strip("Fig6_legend", [Line2D([], [], ls=(0, (3, 2)), color="#555555", lw=0.6, label="Equal confidence"), dot(RS, "RuleSight (Ours) higher", mec="white"), dot(ZS, "Zero-shot higher", mec="white"),
                                 dot(COT, "Chain-of-thought higher", mec="white"), dot(FEW, "Few-shot higher", mec="white"), dot(GREY_L, "Unchanged", mec="#9A9A9A")])

def diverge(name, labels, rows, title, l, left=0.46, bold=None):
    """rows: per label a list of (false_alarm, miss, color) drawn top to bottom inside the label's band"""
    fig, ax = panel(left=left, bottom=0.40, right=0.10); n = len(labels)
    for i, bars in enumerate(rows):
        k = len(bars); hgt = 0.78 / k
        for j, (fa, mi, col) in enumerate(bars):
            yy = i - 0.39 + hgt * (j + 0.5); hb = hgt * 0.46
            gbar(ax, 0, -100 * fa, yy - hb, yy + hb, col, along="x"); gbar(ax, 0, 100 * mi, yy - hb, yy + hb, col, along="x")
    ax.axvline(0, color=SPINE, lw=0.6, zorder=4); ax.set_xlim(-100, 100); ax.set_xticks([-100, -50, 0, 50, 100]); ax.set_xticklabels(["100", "50", "0", "50", "100"])
    ax.set_yticks(range(n)); ax.set_yticklabels(labels); ax.set_ylim(n - 0.5, -0.5); ax.tick_params(axis="y", length=0)
    if bold is not None: ax.get_yticklabels()[bold].set_fontweight("bold")
    style(ax, title=title, grid="x")
    ax.text(0.25, -0.15, "False alarm", transform=ax.transAxes, ha="center", va="top", fontweight="bold", color=INK)
    ax.text(0.75, -0.15, "Miss", transform=ax.transAxes, ha="center", va="top", fontweight="bold", color=INK)
    letter(fig, l); save(fig, name)

def fig7():
    C = G["cell"]; A = G["conf400"]; R = G["route"]; assert abs(C["haiku|All|zs"]["fa"] - 0.2531) < 0.001
    for k, cat in enumerate(CATS + ["All"]):
        rows = [[(C[f"{m}|{cat}|zs"]["fa"], C[f"{m}|{cat}|zs"]["miss"], ZS), (C[f"{m}|{cat}|rs"]["fa"], C[f"{m}|{cat}|rs"]["miss"], RS)] for m in MODELS]
        diverge(f"Fig7{L8[k]}", [MSHORT[m] for m in MODELS], rows, CSHORT[cat] if cat != "All" else "All categories", L8[k])
    cfg = [("s0", "Zero-shot", ZS), ("bcot", "CoT", COT), ("s1", "Few-shot", FEW), ("s2", "Cards + ex.", "#E286AF"), ("s3nomeas", "RuleSight", RS), ("s3", "Unrouted", RS_L), ("s3norule", "w/o cards", GREY)]
    diverge("Fig7g", [c[1] for c in cfg], [[(A[f"{c[0]}|All"]["fa"], A[f"{c[0]}|All"]["miss"], c[2])] for c in cfg], "Haiku 5.5, settings", "g", bold=4)
    cl = CATS + ["All"]
    diverge("Fig7h", [CSHORT[c] for c in cl], [[(R[f"qwen3vl8b|s3|{c}"]["fa"], R[f"qwen3vl8b|s3|{c}"]["miss"], RS_L), (R[f"qwen3vl8b|s3nomeas|{c}"]["fa"], R[f"qwen3vl8b|s3nomeas|{c}"]["miss"], RS)] for c in cl], "Qwen3-VL-8B routing", "h")
    legend_strip("Fig7_legend", None, rows=[[box(ZS, "Zero-shot"), box(RS, "RuleSight (Ours)"), box(RS_L, "RuleSight, unrouted")], [box(COT, "Chain-of-thought (g)"), box(FEW, "Few-shot (g)"), box("#E286AF", "Cards + exemplars (g)"), box(GREY, "Without rule cards (g)")]])

def fig8():   # change from each component step per category (FEGAR Fig. 12)
    S = G["strips"]; assert abs(S["cards"]["All"]["obs"] - 0.2659) < 0.001
    spec = [("cards", "+ Rule cards", 1, RS), ("nocards", "Without rule cards", -1, PURPLE), ("route_haiku", "Routing, Haiku 5.5", 1, RS), ("route_qwen", "Routing, Qwen3-VL-8B", 1, RS)]
    cols = CATS + ["All"]; rng = np.random.default_rng(0)
    for k, (key, title, sign, col) in enumerate(spec):
        fig, ax = panel(left=0.34, bottom=0.48, h=1.44); allv = []
        for j, cat in enumerate(cols):
            d = np.array(S[key][cat]["d"]); obs = S[key][cat]["obs"]; allv += list(d); jx = rng.uniform(-0.3, 0.3, d.size)
            helps = sign * d > 1e-9; hurts = sign * d < -1e-9
            g = np.linspace(d.min(), d.max(), 80); kd = kde(d, g); kd = 0.36 * kd / kd.max()
            ax.fill_betweenx(g, j - kd, j, color=tint(col, 0.6), lw=0, zorder=2); ax.plot(j - kd, g, color=tint(col, 0.2), lw=0.4, zorder=2)
            jx = rng.uniform(0.04, 0.34, d.size)
            ax.scatter((j + jx)[hurts], d[hurts], s=2.2, color="#E6E6E6", edgecolor="#777777", linewidth=0.15, zorder=4)
            ax.scatter((j + jx)[helps], d[helps], s=2.2, color=col, edgecolor="#555555", linewidth=0.15, alpha=0.9, zorder=5)
            ax.plot([j - 0.36, j + 0.36], [obs, obs], color=INK, lw=1.2, zorder=6, solid_capstyle="butt")
        top, bot = max(allv), min(allv); span = top - bot; ax.set_ylim(min(bot - 0.06 * span, -0.03), max(top + 0.08 * span, 0.03))
        ax.axhline(0, color=SPINE, lw=0.5, ls=(0, (3, 2)), zorder=2); ax.set_xticks(range(len(cols))); ax.set_xticklabels([CSHORT[c] for c in cols], rotation=40, ha="right", rotation_mode="anchor"); ax.set_xlim(-0.6, len(cols) - 0.4)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4)); style(ax, title=title, ylabel="Change in MCC", grid="y"); letter(fig, L8[k]); save(fig, f"Fig8{L8[k]}")
    legend_strip("Fig8_legend", [dot(RS, "Step helps"), dot(PURPLE, "Removal hurts (b)"), dot("#E6E6E6", "Step hurts", mec="#777777"), Line2D([], [], color=INK, lw=1.3, label="Observed change")])

def fig9():   # instances corrected and lost (ErgoSight Fig. 11)
    F = G["flip"]; assert F["haiku|Text overflow"]["fixed"] == 181
    subs = [(c, CSHORT[c]) for c in CATS] + [("All", "All instances"), ("pos", "Defective instances"), ("neg", "Defect-free instances")]
    for k, (key, title) in enumerate(subs):
        fig, ax = panel(left=0.40, bottom=0.36)
        for i, m in enumerate(MODELS):
            f = F[f"{m}|{key}"]; fx, br = f["fixed_pct"], f["broken_pct"]; net = fx - br
            gbar(ax, 0, -br, i - 0.32, i + 0.32, ZS, along="x"); gbar(ax, 0, fx, i - 0.32, i + 0.32, RS, along="x")
            ax.scatter(net, i, s=7, marker="D", color="white", edgecolor=INK, linewidth=0.5, zorder=5)
            ax.text(55, i, f"{net:+.1f}", ha="right", va="center", fontsize=FS, color=RS_D if net > 0 else "#6A6A6A")
        ax.axvline(0, color=SPINE, lw=0.6, zorder=4); ax.set_xlim(-40, 56); ax.set_xticks([-30, 0, 30]); ax.set_xticklabels(["30", "0", "30"])
        ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([MSHORT[m] for m in MODELS]); ax.set_ylim(len(MODELS) - 0.5, -0.5); ax.tick_params(axis="y", length=0)
        style(ax, title=title, xlabel="Instances (%)", grid="x"); letter(fig, L8[k]); save(fig, f"Fig9{L8[k]}")
    legend_strip("Fig9_legend", [box(RS, "Corrected by RuleSight"), box(ZS, "Lost by RuleSight"), dot("white", "Net change, value at right", marker="D", ms=3.6, mec=INK)])

TYPES = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "none"]; TLAB = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "No"]
CORRECT = {"Abnormal ellipsis": {"D7"}, "Text overflow": {"D6"}, "Undisplayed content": {"D3", "D4"}, "Cropped content": {"D8"}, "Container overlap": {"D1", "D2"}}
HEAT = mcolors.LinearSegmentedColormap.from_list("house_blue", ["#F4F7FB", "#DCE9F2", "#B9D3EC", "#90B7E2", "#5B8FCB", "#3F5F99"])
def fig10():
    N_LAB = {"Abnormal ellipsis": 431, "Text overflow": 310, "Undisplayed content": 365, "Cropped content": 342, "Container overlap": 389}
    mats = [("Haiku 5.5, zero-shot", D["confusion"]["s0"], N_LAB), ("Haiku 5.5, RuleSight", D["confusion"]["s3nomeas"], N_LAB),
            ("Sonnet 5.5, zero-shot", G["conf_sonnet"]["s0"]["counts"], G["conf_sonnet"]["s0"]["nlab"]), ("Sonnet 5.5, RuleSight", G["conf_sonnet"]["s3nomeas"]["counts"], G["conf_sonnet"]["s3nomeas"]["nlab"])]
    for k, (title, counts, nlab) in enumerate(mats):
        fig, ax = panel(left=0.47, bottom=0.40)
        M = np.array([[counts.get(f"{c}|{t}", 0) / nlab[c] for t in TYPES] for c in CATS])
        for i, c in enumerate(CATS):
            for j, t in enumerate(TYPES):
                v = M[i, j]; ok = t in CORRECT[c]
                ax.add_patch(FancyBboxPatch((j - 0.45, i - 0.45), 0.9, 0.9, boxstyle="round,pad=0,rounding_size=0.18", fc=HEAT(min(v / 0.9, 1)), ec=GT if ok else "none", lw=0.9 if ok else 0, zorder=2))
                if v >= 0.10 or (ok and v >= 0.005): ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=FS - 0.5, color="white" if v > 0.5 else INK, fontweight="bold" if ok else "normal", zorder=3)
        ax.set_xlim(-0.55, len(TYPES) - 0.45); ax.set_ylim(len(CATS) - 0.45, -0.55)
        ax.set_xticks(range(len(TYPES))); ax.set_xticklabels(TLAB, rotation=90); ax.set_yticks(range(len(CATS))); ax.set_yticklabels([CSHORT[c] for c in CATS])
        style(ax, title=title, xlabel="Reported category", grid=None); ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
        letter(fig, "abcd"[k]); save(fig, f"Fig10{'abcd'[k]}")
    fig = plt.figure(figsize=(TEXTW, 0.30)); cax = fig.add_axes([0.38, 0.50, 0.28, 0.22]); cb = fig.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(0, 90), cmap=HEAT), cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.4); cb.set_ticks([0, 30, 60, 90]); cb.ax.tick_params(labelsize=FS, length=1.5, width=0.4, pad=1)
    fig.text(0.37, 0.61, "Share of the labeled defects (%)", ha="right", va="center", fontsize=FS); fig.patches.append(FancyBboxPatch((0.70, 0.46), 0.012, 0.30, boxstyle="round,pad=0,rounding_size=0.002", transform=fig.transFigure, fc="white", ec=GT, lw=0.9))
    fig.text(0.717, 0.61, "Labeled category", ha="left", va="center", fontsize=FS); save(fig, "Fig10_legend")

def fig11():
    Q = D["rq4"]; R1 = D["rq1"]; assert (Q["haiku"]["s0"]["rule_cited"], Q["haiku"]["s3nomeas"]["rule_cited"]) == (0.1, 96.6)
    specs = [("Rule citation rate", "Reports citing their rule (%)", lambda m, s: Q[m][s]["rule_cited"], (0, 105)), ("Category F1", "Category F1", lambda m, s: Q[m][s]["category_f1"], (0, 0.62)),
             ("Localization recall", "Recall at IoU 0.5", lambda m, s: Q[m][s]["loc_recall"], (0, 0.32)), ("API cost", "USD per 1,000 screenshots", lambda m, s: R1[m]["cost"][0 if s == "s0" else 1], None)]
    x = np.arange(len(MODELS)); w = 0.38
    for k, (title, yl, f, ylim) in enumerate(specs):
        fig, ax = panel(left=0.30, bottom=0.42)
        if ylim:
            ax.set_ylim(*ylim); ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
            for i, m in enumerate(MODELS): gbar(ax, i - w, i, 0, f(m, "s0"), ZS); gbar(ax, i, i + w, 0, f(m, "s3nomeas"), RS)
            ax.set_xlim(-0.6, len(MODELS) - 0.4)
        else:
            ax.set_yscale("symlog", linthresh=0.5, linscale=0.4); ax.set_ylim(0, 40); ax.set_yticks([0, 1, 10]); ax.set_yticklabels(["0", "1", "10"])
            ax.bar(x - w / 2, [f(m, "s0") for m in MODELS], w, color=ZS, lw=0.4, ec=ZS, zorder=3); ax.bar(x + w / 2, [f(m, "s3nomeas") for m in MODELS], w, color=RS, lw=0.4, ec=RS, zorder=3)
            for i, m in enumerate(MODELS):
                if f(m, "s3nomeas") == 0: ax.text(i, 0.05, "local", ha="center", va="bottom", rotation=90, fontsize=FS, color="#6A6A6A")
        ax.set_xticks(x); ax.set_xticklabels([MSHORT[m] for m in MODELS], rotation=40, ha="right", rotation_mode="anchor")
        style(ax, title=title, ylabel=yl, grid="y"); letter(fig, "abcd"[k]); save(fig, f"Fig11{'abcd'[k]}")
    legend_strip("Fig11_legend", [box(ZS, "Zero-shot"), box(RS, "RuleSight (Ours)")])

def fig13():
    T = G["thr_cat"]; taus = np.array(G["taus"]); assert abs(max(T["s3nomeas|Text overflow"]["mcc"]) - 0.605) < 0.001
    for k, (cat, title) in enumerate((("All", "All categories"), ("Text overflow", "Overflow"), ("Abnormal ellipsis", "Ellipsis"), ("Undisplayed content", "Undisplayed"))):
        fig, ax = panel(left=0.30, bottom=0.37)
        for st, col, mfc in (("s0", ZS_D, "white"), ("s3nomeas", RS, RS)):
            d = T[f"{st}|{cat}"]; ax.fill_between(taus, d["lo"], d["hi"], color=col, alpha=0.16, lw=0, zorder=2); ax.plot(taus, d["mcc"], color=col, lw=0.9, marker="o", ms=2.0, mfc=mfc, mec=col, mew=0.5, zorder=3)
        d = T[f"s3nomeas|{cat}"]; j = int(np.argmax(d["mcc"])); ax.scatter(taus[j], d["mcc"][j], s=22, marker="*", color=RS, edgecolor=INK, linewidth=0.3, zorder=5)
        ax.set_xlim(-0.03, 0.93); ax.set_ylim(-0.08, 0.75); ax.set_xticks([0, 0.3, 0.6, 0.9]); ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        style(ax, title=title, xlabel="Confidence threshold", ylabel="MCC"); letter(fig, "abcd"[k]); save(fig, f"Fig13{'abcd'[k]}")
    legend_strip("Fig13_legend", [Line2D([], [], color=ZS_D, lw=0.9, marker="o", ms=2.5, mfc="white", label="Zero-shot"), Line2D([], [], color=RS, lw=0.9, marker="o", ms=2.5, label="RuleSight (Ours)"),
                                  Patch(fc="#BBBBBB", alpha=0.5, label="95% bootstrap band"), dot(RS, "Best threshold", marker="*", ms=6, mec=INK)])

if __name__ == "__main__":
    fig6(); fig7(); fig8(); fig9(); fig10(); fig11(); fig13(); print("result panels written")
