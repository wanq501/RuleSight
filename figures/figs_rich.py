# -*- coding: utf-8 -*-
"""Richer experiment figures (second rows and redesigned panels) in the house style, from extra.json and figdata.json."""
import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch, Rectangle
from matplotlib import colors as mcolors, cm
from house import *
import figs_results as FR

X = json.load(open(os.path.join(HERE, "extra.json")))
SEQ = mcolors.LinearSegmentedColormap.from_list("seq", ["#FFFFFF", "#CAE5F8", "#9DCBED", "#70ACDE", "#5271AE", "#4F587D"])
PINK = mcolors.LinearSegmentedColormap.from_list("pink", ["#FFFFFF", "#FAEFF5", "#F2C5DA", "#E286AF", "#C68DC0", "#776B97"])
DIV = mcolors.LinearSegmentedColormap.from_list("div", ["#5271AE", "#9DCBED", "#F7F7F7", "#F2C5DA", "#D85B59"])
FR.HEAT = SEQ
SET7 = [("s0", "Zero-shot", ZS), ("bcot", "CoT", COT), ("s1", "Few-shot", FEW), ("s2", "Cards + ex.", RS_L), ("s3", "Unrouted", RS_D), ("s3nomeas", "RuleSight", RS), ("s3norule", "w/o cards", GREY)]

def tiles(ax, M, rows, cols, cmap, vmin, vmax, fmt="{:.2f}", bold=None, fs=FS - 0.6):
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i][j]; c = cmap((v - vmin) / (vmax - vmin)); ax.add_patch(FancyBboxPatch((j - 0.45, i - 0.45), 0.9, 0.9, boxstyle="round,pad=0,rounding_size=0.16", fc=c, ec="none", zorder=2))
            lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fs, color="white" if lum < 0.55 else INK, fontweight="bold" if (bold and bold(i, j)) else "normal", zorder=3)
    ax.set_xlim(-0.55, len(cols) - 0.45); ax.set_ylim(len(rows) - 0.45, -0.55); ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(rows))); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)

# ------------------------------------------------------------------ Fig. 6: per-instance scatter with marginal distributions
def fig6():
    """one scatter per comparison, square plot area as large as the panel allows, the share of increases under the title"""
    CS = json.load(open(os.path.join(HERE, "conf_scatter.json"))); rng = np.random.default_rng(7); stats = {}
    spec = [(f"{m}|s0", MNAME[m], "Zero-shot", ZS) for m in MODELS] + [("haiku|bcot", "Haiku 5.5, chain-of-thought", "Chain-of-thought", COT), ("haiku|s1", "Haiku 5.5, few-shot", "Few-shot", FEW)]
    W, Hh = PW, 1.58; L, B, S = 0.29, 0.37, 0.93
    for k, (key, title, base, col) in enumerate(spec):
        d = CS[key]; x, y, lab = np.array(d["x"]), np.array(d["y"]), np.array(d["label"]); m_ = lab == 1; x, y = x[m_], y[m_]
        X_, Y_ = np.clip(x + rng.uniform(-0.03, 0.03, x.size), -0.05, 1.05), np.clip(y + rng.uniform(-0.03, 0.03, x.size), -0.05, 1.05)
        up, down = y > x, y < x; tie = ~(up | down); big = x.size > 1000; s = 1.5 if big else 3.6; a = 0.6 if big else 0.85
        fig = plt.figure(figsize=(W, Hh)); ax = fig.add_axes([L / W, B / Hh, S / W, S / Hh])
        ax.fill_between([-0.07, 1.07], [-0.07, 1.07], [1.07, 1.07], color=WIN, alpha=0.30, lw=0, zorder=0); ax.plot([-0.07, 1.07], [-0.07, 1.07], ls=(0, (3, 2)), color="#666666", lw=0.6, zorder=2)
        ax.scatter(X_[tie], Y_[tie], s=s, color=GREY_L, edgecolor="#A6A6A6", linewidth=0.15, alpha=a, zorder=3)
        ax.scatter(X_[down], Y_[down], s=s, color=col, edgecolor="white", linewidth=0.15, alpha=a + 0.1, zorder=4)
        ax.scatter(X_[up], Y_[up], s=s, color=RS, edgecolor="white", linewidth=0.15, alpha=a + 0.1, zorder=5)
        ax.set_xlim(-0.07, 1.07); ax.set_ylim(-0.07, 1.07); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"]); ax.set_yticklabels(["0", "0.5", "1"])
        sh = 100 * up.sum() / max(1, up.sum() + down.sum()); stats[key] = dict(n=int(x.size), up=int(up.sum()), down=int(down.sum()), share=round(sh, 1))
        style(ax, xlabel=base, ylabel="RuleSight")
        fig.text((L + S / 2) / W, 1 - 0.012, title, ha="center", va="top", fontsize=FS_T, fontweight="bold", color=INK)
        fig.text((L + S / 2) / W, 1 - 0.15 / Hh, f"{sh:.1f}% of changes are increases", ha="center", va="top", fontsize=FS - 0.3, color=RS_D if sh >= 50 else "#6A6A6A")
        letter(fig, "abcdefgh"[k], y=0.008); save(fig, f"Fig6{'abcdefgh'[k]}")
    legend_strip("Fig6_legend", [dot(RS, "RuleSight (Ours) higher", mec="white"), dot(ZS, "Zero-shot higher", mec="white"), dot(COT, "Chain-of-thought higher", mec="white"), dot(FEW, "Few-shot higher", mec="white"),
                                 dot(GREY_L, "Unchanged", mec="#A6A6A6"), Line2D([], [], ls=(0, (3, 2)), color="#666666", lw=0.6, label="Equal confidence")])
    return stats

# ------------------------------------------------------------------ Fig. 8, second row
def fig8_row2():
    S9 = X["set9"]; fig, ax = panel(left=0.30, bottom=0.44)
    for i, (s, lab, col) in enumerate(SET7):
        v = S9[s]; gbar(ax, i - 0.34, i + 0.34, 0, v["mcc"], col); ax.errorbar(i, v["mcc"], yerr=[[v["mcc"] - v["lo"]], [v["hi"] - v["mcc"]]], fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.6, capthick=0.6, zorder=5)
    ax.set_xticks(range(len(SET7))); ax.set_xticklabels([t[1] for t in SET7], rotation=40, ha="right", rotation_mode="anchor"); ax.get_xticklabels()[5].set_fontweight("bold")
    ax.set_xlim(-0.6, len(SET7) - 0.4); ax.set_ylim(0, 0.7); ax.yaxis.set_major_locator(ticker.MaxNLocator(4)); style(ax, title="Settings, all categories", ylabel="MCC", grid="y"); letter(fig, "e"); save(fig, "Fig8e")
    steps = ["s0", "s1", "s2", "s3", "s3nomeas"]; slab = ["Zero-shot", "Few-shot", "+ Cards", "Unrouted", "RuleSight"]; fig, ax = panel(left=0.33, bottom=0.44); xs = np.arange(len(steps))
    for c in CATS:
        v = np.array([X["steps"][f"{c}|{s}"]["mcc"] for s in steps]); lo = np.array([X["steps"][f"{c}|{s}"]["lo"] for s in steps]); hi = np.array([X["steps"][f"{c}|{s}"]["hi"] for s in steps])
        ax.fill_between(xs, lo, hi, color=CATC[c], alpha=0.14, lw=0, zorder=2); ax.plot(xs, v, color=CATC[c], lw=1.1, marker="o", ms=2.2, mec="white", mew=0.3, zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels(slab, rotation=40, ha="right", rotation_mode="anchor"); ax.get_xticklabels()[4].set_fontweight("bold"); ax.set_xlim(-0.3, len(steps) - 0.7); ax.set_ylim(-0.3, 0.95)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5)); style(ax, title="Steps per category", ylabel="MCC"); letter(fig, "f"); save(fig, "Fig8f")
    fig, ax = panel(left=0.30, bottom=0.37)
    for s, lab, col in SET7:
        v = S9[s]; ax.scatter(100 * v["fa"], 100 * v["miss"], s=8 + 40 * max(v["mcc"], 0), color=col, edgecolor=INK if s == "s3nomeas" else "white", linewidth=0.6 if s == "s3nomeas" else 0.4, zorder=4, alpha=0.95)
        dx, dy, ha = {"s0": (0.9, 1.6, "left"), "bcot": (0.9, -1.8, "left"), "s1": (-0.9, 1.8, "right"), "s2": (0.9, -1.8, "left"), "s3": (-0.9, -1.6, "right"), "s3nomeas": (0.9, -1.9, "left"), "s3norule": (0.9, 1.8, "left")}[s]
        ax.text(100 * v["fa"] + dx, 100 * v["miss"] + dy, lab, ha=ha, va="center", fontsize=FS - 0.3, color=INK, fontweight="bold" if s == "s3nomeas" else "normal", zorder=6)
    ax.set_xlim(5, 32); ax.set_ylim(28, 70); ax.set_xticks([10, 20, 30]); ax.set_yticks([30, 40, 50, 60, 70]); style(ax, title="False alarms and misses", xlabel="False alarm (%)", ylabel="Miss (%)"); letter(fig, "g"); save(fig, "Fig8g")
    cols = CATS + ["All"]; M = [[100 * X["cat7"][f"{s}|{c}"] for c in cols] for s, _, _ in SET7]
    fig, ax = panel(left=0.46, bottom=0.44); tiles(ax, M, SET7, cols, DIV, -75, 75, fmt="{:.0f}", bold=lambda i, j: SET7[i][0] == "s3nomeas")
    ax.set_xticklabels([CSHORT[c] for c in cols], rotation=40, ha="right", rotation_mode="anchor"); ax.set_yticklabels([t[1] for t in SET7]); ax.get_yticklabels()[5].set_fontweight("bold")
    style(ax, title="Per-category MCC", grid=None); [sp.set_visible(False) for sp in ax.spines.values()]; letter(fig, "h"); save(fig, "Fig8h")
    legend_strip("Fig8_legend", None, rows=[[dot(RS, "Step helps"), dot(PURPLE, "Removal hurts (b)"), dot("#E6E6E6", "Step hurts", mec="#777777"), Line2D([], [], color=INK, lw=1.2, label="Observed change (a) to (d)"), Patch(fc=tint(GREY, 0.6), label="Density of the resamples")],
                                            [Line2D([], [], color=CATC[c], lw=1.1, marker="o", ms=2.5, label=c) for c in CATS] + [Patch(fc=RS, label="RuleSight (Ours), (e) (g)")]])

# ------------------------------------------------------------------ Fig. 9: instances corrected and lost
TCOL = {"fixed": RS, "lost": ZS, "both": "#CFE3F1", "neither": "#8C96A3"}
def dots(name, cat, title, l):
    cls = X["trans"][cat]; order = ["fixed", "lost", "both", "neither"]; seq = [c for o in order for c in cls if c == o]; n = len(seq); nc = int(np.ceil(np.sqrt(n * 1.0))); nr = int(np.ceil(n / nc))
    fig, ax = panel(left=0.08, bottom=0.30, right=0.08, top=0.15); ii = np.arange(n); xx = ii % nc; yy = ii // nc
    s = max(0.25, (0.95 / nc * 72) ** 2 * 0.55)
    for o in order:
        mk = np.array([c == o for c in seq]); ax.scatter(xx[mk], yy[mk], s=s, color=TCOL[o], edgecolor="none", marker="s" if nc > 40 else "o", zorder=3)
    ax.set_xlim(-1, nc); ax.set_ylim(nr, -1); ax.set_aspect("equal"); ax.axis("off")
    cnt = {o: sum(1 for c in cls if c == o) for o in order}
    fig.text(0.5, 1 - 0.02 / PH, title, ha="center", va="top", fontsize=FS_T, fontweight="bold", color=INK)
    fig.text(0.5, 0.20 / PH, f"+{cnt['fixed']:,} corrected, {cnt['lost']:,} lost of {n:,}", ha="center", va="center", fontsize=FS, color=INK)
    letter(fig, l); save(fig, name); return cnt

def fig9():
    F = G["flip"]; out = {}
    for k, (c, t) in enumerate((("Text overflow", "Text overflow"), ("Abnormal ellipsis", "Abnormal ellipsis"), ("Undisplayed content", "Undisplayed content"), ("All", "All five categories"))):
        out[c] = dots(f"Fig9{'abcd'[k]}", c, t, "abcd"[k])
    for k, (key, title) in enumerate((("All", "All instances"), ("pos", "Defective instances"), ("neg", "Defect-free instances"))):
        fig, ax = panel(left=0.40, bottom=0.36)
        for i, m in enumerate(MODELS):
            f = F[f"{m}|{key}"]; fx, br = f["fixed_pct"], f["broken_pct"]; net = fx - br
            gbar(ax, 0, -br, i - 0.32, i + 0.32, ZS, along="x"); gbar(ax, 0, fx, i - 0.32, i + 0.32, RS, along="x")
            ax.scatter(net, i, s=7, marker="D", color="white", edgecolor=INK, linewidth=0.5, zorder=5); ax.text(55, i, f"{net:+.1f}", ha="right", va="center", fontsize=FS, color=RS_D if net > 0 else "#6A6A6A")
        ax.axvline(0, color=SPINE, lw=0.6, zorder=4); ax.set_xlim(-40, 56); ax.set_xticks([-30, 0, 30]); ax.set_xticklabels(["30", "0", "30"])
        ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([MSHORT[m] for m in MODELS]); ax.set_ylim(len(MODELS) - 0.5, -0.5); ax.tick_params(axis="y", length=0)
        style(ax, title=title, xlabel="Instances (%)", grid="x"); letter(fig, "efg"[k]); save(fig, f"Fig9{'efg'[k]}")
    M = [[F[f"{m}|{c}"]["fixed_pct"] - F[f"{m}|{c}"]["broken_pct"] for c in CATS] for m in MODELS]
    fig, ax = panel(left=0.40, bottom=0.44); tiles(ax, M, MODELS, CATS, DIV, -16, 16, fmt="{:+.0f}")
    ax.set_xticklabels([CSHORT[c] for c in CATS], rotation=40, ha="right", rotation_mode="anchor"); ax.set_yticklabels([MSHORT[m] for m in MODELS]); style(ax, title="Net change (points)", grid=None)
    [sp.set_visible(False) for sp in ax.spines.values()]; letter(fig, "h"); save(fig, "Fig9h")
    legend_strip("Fig9_legend", [dot(RS, "Corrected by RuleSight", marker="s"), dot(ZS, "Lost by RuleSight", marker="s"), dot(TCOL["both"], "Correct under both", marker="s", mec="none"), dot(TCOL["neither"], "Wrong under both", marker="s", mec="none"),
                                 dot("white", "Net change", marker="D", ms=3.6, mec=INK)])
    return out

# ------------------------------------------------------------------ Fig. 10, second row
def fig10_row2():
    rules = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]; M = [[100 * X["cite"][m].get(r, 0) for m in MODELS] for r in rules]
    fig, ax = panel(left=0.24, bottom=0.44); tiles(ax, M, rules, MODELS, PINK, 0, 45, fmt="{:.0f}")
    ax.set_xticklabels([MSHORT[m] for m in MODELS], rotation=40, ha="right", rotation_mode="anchor"); ax.set_yticklabels(rules); style(ax, title="Rules cited (%)", grid=None); [sp.set_visible(False) for sp in ax.spines.values()]
    letter(fig, "e"); save(fig, "Fig10e")
    fig, ax = panel(left=0.46, bottom=0.37); g = np.linspace(0, 1, 120)
    for i, c in enumerate(CATS):
        base = len(CATS) - 1 - i
        for s, col in (("s0", ZS), ("s3nomeas", RS)):
            v = np.array(X["conf"][f"{s}|{c}"])
            if v.size < 3: continue
            k = kde(v, g, bw=0.05); k = 0.85 * k / k.max() * (0.35 + 0.65 * min(1, v.size / 400)); ax.fill_between(g, base, base + k, color=col, alpha=0.42, lw=0, zorder=3 + (s == "s3nomeas")); ax.plot(g, base + k, color=col, lw=0.6, zorder=4)
        ax.axhline(base, color="#C9C9C9", lw=0.4, zorder=1)
    ax.set_yticks(range(len(CATS))); ax.set_yticklabels([CSHORT[c] for c in CATS][::-1]); ax.set_ylim(-0.1, len(CATS) + 0.2); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
    style(ax, title="Confidence of reports", xlabel="Reported confidence", grid="x"); ax.tick_params(axis="y", length=0); letter(fig, "f"); save(fig, "Fig10f")
    cols = CATS + ["All"]
    for k, (key, title, yl, ylim) in enumerate((("f1", "Category F1 per category", "Category F1", (0, 0.72)), ("loc", "Localization per category", "Recall at IoU 0.5", (0, 0.5)))):
        fig, ax = panel(left=0.30, bottom=0.44)
        for i, c in enumerate(cols): gbar(ax, i - 0.38, i, 0, X["catf1"][f"s0|{c}"][key], ZS); gbar(ax, i, i + 0.38, 0, X["catf1"][f"s3nomeas|{c}"][key], RS)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([CSHORT[c] for c in cols], rotation=40, ha="right", rotation_mode="anchor"); ax.set_xlim(-0.6, len(cols) - 0.4); ax.set_ylim(*ylim); ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        style(ax, title=title, ylabel=yl, grid="y"); letter(fig, "gh"[k]); save(fig, f"Fig10{'gh'[k]}")
    fig = plt.figure(figsize=(TEXTW, 0.40))
    for x0, cmap, lab, lo, hi in ((0.25, SEQ, "Labeled defects (%), (a) to (d)", 0, 90), (0.72, PINK, "Rules cited (%), (e)", 0, 45)):
        cax = fig.add_axes([x0, 0.66, 0.18, 0.14]); cb = fig.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(lo, hi), cmap=cmap), cax=cax, orientation="horizontal"); cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=FS, length=1.5, width=0.4, pad=1)
        fig.text(x0 - 0.012, 0.73, lab, ha="right", va="center", fontsize=FS)
    fig.legend(handles=[Patch(fc="white", ec=GT, lw=0.9, label="Labeled category"), box(ZS, "Zero-shot"), box(RS, "RuleSight (Ours)")], loc="center", bbox_to_anchor=(0.5, 0.2), ncol=3, frameon=False, fontsize=FS, columnspacing=1.6)
    save(fig, "Fig10_legend")

# ------------------------------------------------------------------ Fig. 11, second row
def split_violin(ax, i, a, b, ca, cb, w=0.36):
    for v, col, side in ((a, ca, -1), (b, cb, 1)):
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        if v.size < 3: continue
        hi = np.percentile(v, 95); v = v[v <= hi]; g = np.linspace(v.min(), v.max(), 80); k = kde(v, g); k = w * k / k.max()
        ax.fill_betweenx(g, i, i + side * k, color=col, alpha=0.75, lw=0, zorder=3); ax.plot([i + side * 0.04, i + side * w * 0.7], [np.median(v)] * 2, color=INK, lw=0.7, zorder=4)

def fig11_row2():
    R1 = D["rq1"]; fig, ax = panel(left=0.30, bottom=0.37); lx = lambda c: np.log10(max(c, 0.06))
    for m in MODELS:
        (c0, c1) = R1[m]["cost"]; z0, z1 = R1[m]["zs"]["mcc"], R1[m]["rs"]["mcc"]; f0, f1 = R1[m]["zs"]["F1"], R1[m]["rs"]["F1"]; col = MCOL[m]
        ax.annotate("", xy=(lx(c1), z1), xytext=(lx(c0), z0), arrowprops=dict(arrowstyle="-|>", color=col, lw=0.7, mutation_scale=5, shrinkA=3, shrinkB=3.5), zorder=3)
        ax.scatter(lx(c0), z0, s=10 + 50 * f0, facecolor="white", edgecolor=col, linewidth=0.8, zorder=4); ax.scatter(lx(c1), z1, s=10 + 50 * f1, facecolor=col, edgecolor=INK, linewidth=0.3, zorder=5)
    ax.set_xticks([lx(0.06), 0, 1]); ax.set_xticklabels(["local", "1", "10"]); ax.set_xlim(lx(0.06) - 0.25, 1.4); ax.set_ylim(-0.15, 0.7); ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    style(ax, title="Cost and MCC", xlabel="USD per 1,000 screenshots", ylabel="MCC"); letter(fig, "e"); save(fig, "Fig11e")
    for k, (key, title, yl, ylim) in enumerate((("lat", "Latency per screenshot", "Seconds", None), ("tok", "Output tokens per call", "Tokens", None))):
        top = 1.10 * max(np.percentile(X["rq4x"][f"{m}|{st}"][key], 95) for m in MODELS for st in ("s0", "s3nomeas")); ylim = (0, top)
        fig, ax = panel(left=0.34, bottom=0.42)
        for i, m in enumerate(MODELS): split_violin(ax, i, X["rq4x"][f"{m}|s0"][key], X["rq4x"][f"{m}|s3nomeas"][key], ZS, RS)
        ax.set_xticks(range(len(MODELS))); ax.set_xticklabels([MSHORT[m] for m in MODELS], rotation=40, ha="right", rotation_mode="anchor"); ax.set_xlim(-0.6, len(MODELS) - 0.4); ax.set_ylim(*ylim); ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        style(ax, title=title, ylabel=yl, grid="y"); letter(fig, "fg"[k]); save(fig, f"Fig11{'fg'[k]}")
    fig, ax = panel(left=0.30, bottom=0.42)
    for i, m in enumerate(MODELS): gbar(ax, i - 0.38, i, 0, X["tokens"][f"{m}|s0"]["prompt"] / 1000, ZS); gbar(ax, i, i + 0.38, 0, X["tokens"][f"{m}|s3nomeas"]["prompt"] / 1000, RS)
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels([MSHORT[m] for m in MODELS], rotation=40, ha="right", rotation_mode="anchor"); ax.set_xlim(-0.6, len(MODELS) - 0.4); ax.set_ylim(0, 13); ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    style(ax, title="Input tokens per call", ylabel="Thousand tokens", grid="y"); letter(fig, "h"); save(fig, "Fig11h")
    legend_strip("Fig11_legend", None, rows=[[box(ZS, "Zero-shot"), box(RS, "RuleSight (Ours)"), Line2D([], [], color=INK, lw=0.7, label="Median (f) (g)")],
                                             [dot(MCOL[m], MSHORT[m], mec="none") for m in MODELS] + [dot("white", "Zero-shot (e)", mec=INK), dot("#888888", "RuleSight (e), size F1", mec=INK)]])

# ------------------------------------------------------------------ Fig. 13: threshold, with the per-category curves of reference 7
def fig13():
    T = G["thr_cat"]; taus = np.array(G["taus"]); tf = np.linspace(taus[0], taus[-1], 120)
    sm = lambda v: np.interp(tf, taus, v)
    for k, (cat, title) in enumerate((("All", "All categories"), ("Text overflow", "Overflow"), ("Abnormal ellipsis", "Ellipsis"), ("Undisplayed content", "Undisplayed"))):
        fig, ax = panel(left=0.30, bottom=0.37)
        for st, col in (("s0", ZS_D), ("s3nomeas", RS)):
            d = T[f"{st}|{cat}"]; ax.fill_between(tf, sm(d["lo"]), sm(d["hi"]), color=col, alpha=0.18, lw=0, zorder=2); ax.plot(tf, sm(d["mcc"]), color=col, lw=1.2, zorder=3)
        d = T[f"s3nomeas|{cat}"]; j = int(np.argmax(d["mcc"])); ax.scatter(taus[j], d["mcc"][j], s=22, marker="*", color=RS, edgecolor=INK, linewidth=0.3, zorder=5)
        ax.set_xlim(0, 0.9); ax.set_ylim(-0.05, 0.72); ax.set_xticks([0, 0.3, 0.6, 0.9]); ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        style(ax, title=title, xlabel="Confidence threshold", ylabel="MCC"); letter(fig, "abcd"[k]); save(fig, f"Fig13{'abcd'[k]}")
    for k, (mode, title) in enumerate((("s0", "Zero-shot, per category"), ("s3nomeas", "RuleSight, per category"), ("d", "Change, per category"))):
        fig, ax = panel(left=0.30, bottom=0.37)
        for c in CATS:
            if mode == "d": v = np.array(T[f"s3nomeas|{c}"]["mcc"]) - np.array(T[f"s0|{c}"]["mcc"]); ax.plot(tf, sm(v), color=CATC[c], lw=1.1, zorder=3)
            else:
                d = T[f"{mode}|{c}"]; ax.fill_between(tf, sm(d["lo"]), sm(d["hi"]), color=CATC[c], alpha=0.15, lw=0, zorder=2); ax.plot(tf, sm(d["mcc"]), color=CATC[c], lw=1.1, zorder=3)
        if mode == "d": ax.axhline(0, color="#777777", lw=0.5, ls=(0, (3, 2))); ax.set_ylim(-0.25, 0.5); ax.set_yticks([-0.2, 0, 0.2, 0.4])
        else: ax.set_ylim(-0.05, 0.72); ax.set_yticks([0, 0.2, 0.4, 0.6])
        ax.set_xlim(0, 0.9); ax.set_xticks([0, 0.3, 0.6, 0.9]); ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        style(ax, title=title, xlabel="Confidence threshold", ylabel="Change in MCC" if mode == "d" else "MCC"); letter(fig, "efg"[k]); save(fig, f"Fig13{'efg'[k]}")
    fig, ax = panel(left=0.30, bottom=0.37)
    for c in CATS:
        v = np.array(X["conf"][f"s3nomeas|{c}"]); n = sum(1 for x in X["trans"][c] if True)
        npos = len(X["ncat"][f"s3nomeas|{c}"]); r = [100 * np.mean(v >= t) * len(v) / max(1, npos) for t in tf]; ax.plot(tf, r, color=CATC[c], lw=1.1, zorder=3)
    ax.set_xlim(0, 0.9); ax.set_xticks([0, 0.3, 0.6, 0.9]); ax.set_ylim(0, 100); ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    style(ax, title="RuleSight, kept reports", xlabel="Confidence threshold", ylabel="Defects reported (%)"); letter(fig, "h"); save(fig, "Fig13h")
    legend_strip("Fig13_legend", None, rows=[[Line2D([], [], color=ZS_D, lw=1.2, label="Zero-shot"), Line2D([], [], color=RS, lw=1.2, label="RuleSight (Ours)"), Patch(fc="#BBBBBB", alpha=0.5, label="95% bootstrap band"), dot(RS, "Best threshold", marker="*", ms=6, mec=INK)],
                                             [Line2D([], [], color=CATC[c], lw=1.2, label=c) for c in CATS]])

if __name__ == "__main__":
    st = fig6(); print("fig6", {k: v["share"] for k, v in st.items()})
    FR.fig7(); FR.fig8(); fig8_row2(); print("fig9", fig9()); FR.fig10(); fig10_row2(); FR.fig11(); fig11_row2(); fig13(); print("rich panels written")
