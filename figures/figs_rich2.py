# -*- coding: utf-8 -*-
"""Row consistency: every row is four panels of one kind with the same axes geometry, or one full-width panel (the width of
four, as in the user's MT-GUI heatmap). The 3D figure uses one coordinate system and one viewing angle for all eight panels."""
import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib import colors as mcolors, cm
from house import *
import figs_rich as RICH, figs_results as FR
X = json.load(open(os.path.join(HERE, "extra.json")))
SEQ, PINK, DIV = RICH.SEQ, RICH.PINK, RICH.DIV
COLS6 = CATS + ["All"]

def wide_heat(name, title, rows, groups, letter_, bold_rows=(), row_colors=None, cell_h=0.135, left=0.74, bold_values=True):
    """one full-width heatmap: groups = [(name, col_labels, matrix, cmap, vmin, vmax, fmt)], rounded tiles, group headers, colorbars at right"""
    W = TEXTW; nr = len(rows); right, gap = 0.62, 0.07; ncol = sum(len(g[1]) for g in groups)
    cw = (W - left - right - gap * (len(groups) - 1)) / ncol; H = 0.20 + 0.20 + nr * cell_h + 0.42 + 0.14
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    top = H - 0.20 - 0.20; x0 = left; cmaps = []
    fig.text(0.5, 1 - 0.04 / H, title, ha="center", va="top", fontsize=FS_T, fontweight="bold", color=INK)
    for gname, cols, M, cmap, vmin, vmax, fmt in groups:
        gw = cw * len(cols); ax.text(x0 + gw / 2, top + 0.13, gname, ha="center", va="bottom", fontsize=FS, fontweight="bold", color=INK)
        ax.plot([x0 + 0.02, x0 + gw - 0.02], [top + 0.10, top + 0.10], color=cmap(0.75), lw=1.2, solid_capstyle="round")
        for i in range(nr):
            for j in range(len(cols)):
                v = M[i][j]; c = cmap(np.clip((v - vmin) / (vmax - vmin), 0, 1)); xx = x0 + j * cw; yy = top - (i + 1) * cell_h
                ax.add_patch(FancyBboxPatch((xx + 0.012, yy + 0.010), cw - 0.024, cell_h - 0.020, boxstyle="round,pad=0,rounding_size=0.022", fc=c, ec="none"))
                lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
                ax.text(xx + cw / 2, yy + cell_h / 2, fmt.format(v), ha="center", va="center", fontsize=FS - 0.3, color="white" if lum < 0.55 else INK, fontweight="bold" if (bold_values and i in bold_rows) else "normal")
        for j, cl in enumerate(cols): ax.text(x0 + j * cw + cw / 2, top - nr * cell_h - 0.04, cl, ha="right", va="top", rotation=40, rotation_mode="anchor", fontsize=FS, color=INK)
        if (cmap, vmin, vmax) not in [(c_[0], c_[1], c_[2]) for c_ in cmaps]: cmaps.append((cmap, vmin, vmax, gname))
        x0 += gw + gap
    for i, rl in enumerate(rows):
        ax.text(left - 0.05, top - (i + 0.5) * cell_h, rl, ha="right", va="center", fontsize=FS, color=(row_colors[i] if row_colors else INK), fontweight="bold" if i in bold_rows else "normal")
    hb = nr * cell_h / max(1, len(cmaps)) - 0.06
    for k, (cmap, vmin, vmax, gname) in enumerate(cmaps):
        y1 = top - k * (hb + 0.06) - hb; cax = fig.add_axes([(W - right + 0.18) / W, y1 / H, 0.07 / W, hb / H]); cb = fig.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(vmin, vmax), cmap=cmap), cax=cax)
        cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=FS, length=1.5, width=0.4, pad=1); cb.ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
    fig.text(0.5, 0.012, f"({letter_})", ha="center", va="bottom", fontsize=FS_L, color=INK); save(fig, name)

def fig8_wide():
    S = X["set7cat"]; set7 = RICH.SET7; rows = [t[1] for t in set7]; b = [i for i, t in enumerate(set7) if t[0] == "s3nomeas"]
    lab = [CSHORT[c] for c in COLS6]
    groups = [("MCC (×100)", lab, [[100 * S[f"{s}|{c}"]["mcc"] for c in COLS6] for s, _, _ in set7], SEQ, -20, 80, "{:.0f}"),
              ("F1 (×100)", lab, [[100 * S[f"{s}|{c}"]["f1"] for c in COLS6] for s, _, _ in set7], SEQ, -20, 80, "{:.0f}"),
              ("False alarm (%)", lab, [[100 * S[f"{s}|{c}"]["fa"] for c in COLS6] for s, _, _ in set7], PINK, 0, 80, "{:.0f}"),
              ("Miss (%)", lab, [[100 * S[f"{s}|{c}"]["miss"] for c in COLS6] for s, _, _ in set7], PINK, 0, 80, "{:.0f}")]
    wide_heat("Fig8e", "Configurations of Claude Haiku 5.5 per display-defect category, 400-instance subset", rows, groups, "e", bold_rows=b)

def fig10_wide():
    rows, M1, M2, rc, bold = [], [], [], [], []
    for m in MODELS:
        for s, tag, col in (("s0", "zero-shot", ZS_D), ("s3nomeas", "RuleSight", RS)):
            rows.append(f"{MSHORT[m]}, {tag}"); rc.append(col)
            M1.append([100 * X["diag"][f"{m}|{s}|{c}"]["f1"] for c in COLS6]); M2.append([100 * X["diag"][f"{m}|{s}|{c}"]["loc"] for c in COLS6])
            if s == "s3nomeas": bold.append(len(rows) - 1)
    lab = [CSHORT[c] for c in COLS6]
    wide_heat("Fig10e", "Category F1 and localization recall of every model on the defective instances, per display-defect category", rows,
              [("Category F1 (×100)", lab, M1, SEQ, 0, 80, "{:.0f}"), ("Localization recall at IoU 0.5 (×100)", lab, M2, SEQ, 0, 80, "{:.0f}")], "e", bold_rows=bold, row_colors=rc, cell_h=0.118, left=0.98, bold_values=False)

def fig9_row2():
    F = G["flip"]
    for k, (key, title) in enumerate((("Text overflow", "Text overflow"), ("Abnormal ellipsis", "Abnormal ellipsis"), ("Undisplayed content", "Undisplayed content"), ("All", "All instances"))):
        fig, ax = panel(left=0.40, bottom=0.36)
        for i, m in enumerate(MODELS):
            f = F[f"{m}|{key}"]; fx, br = f["fixed_pct"], f["broken_pct"]; net = fx - br
            gbar(ax, 0, -br, i - 0.32, i + 0.32, ZS, along="x"); gbar(ax, 0, fx, i - 0.32, i + 0.32, RS, along="x")
            ax.scatter(net, i, s=7, marker="D", color="white", edgecolor=INK, linewidth=0.5, zorder=5); ax.text(55, i, f"{net:+.1f}", ha="right", va="center", fontsize=FS, color=RS_D if net > 0 else "#6A6A6A")
        ax.axvline(0, color=SPINE, lw=0.6, zorder=4); ax.set_xlim(-40, 56); ax.set_xticks([-30, 0, 30]); ax.set_xticklabels(["30", "0", "30"])
        ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([MSHORT[m] for m in MODELS]); ax.set_ylim(len(MODELS) - 0.5, -0.5); ax.tick_params(axis="y", length=0)
        style(ax, title=title, xlabel="Instances (%)", grid="x"); letter(fig, "efgh"[k]); save(fig, f"Fig9{'efgh'[k]}")

def fig11_row2():
    for k, (key, title, yl, sc) in enumerate((("lat", "Latency per screenshot", "Seconds", 1), ("tok", "Output tokens per call", "Tokens", 1), ("inp", "Input tokens per call", "Thousand tokens", 1e-3), ("nd", "Defects per screenshot", "Reported defects", 1))):
        fig, ax = panel(left=0.34, bottom=0.42); top = 0
        for i, m in enumerate(MODELS):
            a = np.array(X["rq4x"][f"{m}|s0"][key], float) * sc; b = np.array(X["rq4x"][f"{m}|s3nomeas"][key], float) * sc
            RICH.split_violin(ax, i, a, b, ZS, RS); top = max(top, np.percentile(a, 95), np.percentile(b, 95))
        ax.set_ylim(0, 1.10 * top); ax.yaxis.set_major_locator(ticker.MaxNLocator(4, integer=(key == "nd")))
        ax.set_xticks(range(len(MODELS))); ax.set_xticklabels([MSHORT[m] for m in MODELS], rotation=40, ha="right", rotation_mode="anchor"); ax.set_xlim(-0.6, len(MODELS) - 0.4)
        style(ax, title=title, ylabel=yl, grid="y"); letter(fig, "efgh"[k]); save(fig, f"Fig11{'efgh'[k]}")
    legend_strip("Fig11_legend", [box(ZS, "Zero-shot (left half in (e) to (h))"), box(RS, "RuleSight (Ours) (right half)"), Line2D([], [], color=INK, lw=0.7, label="Median")])

# ------------------------------------------------------------------ Fig. 12: one coordinate system and one viewing angle for all eight panels
import figs_3d as F3
ZL, ZT, VIEW, ASPECT, RECT = (-0.4, 1.0), [-0.4, 0, 0.4, 0.8], dict(elev=24, azim=-128), (1.1, 1.6, 0.8), (0.08, -0.02, 0.88, 1.02)
def bars3(name, val, title, l, cmap, norm):
    fig, ax = F3.canvas(title, l, rect=RECT, h=1.42, ty=0.90); NSEG = 10
    for i, m in enumerate(F3.ROWS):
        for j, c in enumerate(F3.COLS):
            z = val(m, c); lo, hi = (0.0, z) if z >= 0 else (z, 0.0); e = np.linspace(lo, hi, NSEG + 1)
            for k in range(NSEG): ax.bar3d(j - 0.3, i - 0.3, e[k], 0.6, 0.6, max(e[k + 1] - e[k], 1e-6), color=cmap(norm((e[k] + e[k + 1]) / 2)), edgecolor="none", shade=True, alpha=0.97)
    ax.set_xticks(range(len(F3.COLS))); ax.set_xticklabels([F3.CODE[c] for c in F3.COLS]); ax.set_yticks(range(len(F3.ROWS))); ax.set_yticklabels([F3.MS3[m] for m in F3.ROWS], ha="left", va="center")
    ax.set_xlim(-0.6, len(F3.COLS) - 0.4); ax.set_ylim(-0.6, len(F3.ROWS) - 0.4); ax.set_zlim(*ZL); ax.set_zticks(ZT); ax.view_init(**VIEW); ax.set_box_aspect(ASPECT); save(fig, name)

def fig12():
    CT = D["categories"]; C = G["cell"]; key = lambda c: {"Text overflow": "Text overflow", "Abnormal ellipsis": "Abnormal ellipsis", "Undisplayed content": "Undisplayed content", "Cropped content": "Cropped content", "Container overlap": "Container overlap"}[c]
    vn = mcolors.Normalize(ZL[0], ZL[1]); dn = mcolors.Normalize(-0.4, 0.4)
    spec = [("a", "MCC, zero-shot", lambda m, c: CT[f"{m}|{c}"]["zs"], F3.MCC_CM, vn), ("b", "MCC, RuleSight", lambda m, c: CT[f"{m}|{c}"]["rs"], F3.MCC_CM, vn),
            ("c", "Change in MCC", lambda m, c: CT[f"{m}|{c}"]["rs"] - CT[f"{m}|{c}"]["zs"], F3.DIV_CM, dn), ("d", "Change in F1", lambda m, c: CT[f"{m}|{c}"]["rs_f1"] - CT[f"{m}|{c}"]["zs_f1"], F3.DIV_CM, dn),
            ("e", "F1, zero-shot", lambda m, c: CT[f"{m}|{c}"]["zs_f1"], F3.MCC_CM, vn), ("f", "F1, RuleSight", lambda m, c: CT[f"{m}|{c}"]["rs_f1"], F3.MCC_CM, vn),
            ("g", "False alarm, zero-shot", lambda m, c: C[f"{m}|{key(c)}|zs"]["fa"], F3.MCC_CM, vn), ("h", "False alarm, RuleSight", lambda m, c: C[f"{m}|{key(c)}|rs"]["fa"], F3.MCC_CM, vn)]
    for l, title, fn, cmap, norm in spec: bars3(f"Fig12{l}", fn, title, l, cmap, norm)
    fig = plt.figure(figsize=(TEXTW, 0.24))
    for x0, cmap, lo, hi, lab, ticks in ((0.24, F3.MCC_CM, ZL[0], ZL[1], "Value, (a) (b) (e) to (h)", [-0.4, 0, 0.4, 0.8]), (0.70, F3.DIV_CM, -0.4, 0.4, "Change, (c) (d)", [-0.4, -0.2, 0, 0.2, 0.4])):
        cax = fig.add_axes([x0, 0.45, 0.20, 0.25]); cb = fig.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(lo, hi), cmap=cmap), cax=cax, orientation="horizontal"); cb.outline.set_linewidth(0.4); cb.set_ticks(ticks)
        cb.ax.tick_params(labelsize=FS, length=1.5, width=0.4, pad=1); fig.text(x0 - 0.012, 0.575, lab, ha="right", va="center", fontsize=FS)
    save(fig, "Fig12_legend")

if __name__ == "__main__":
    fig8_wide(); fig9_row2(); fig10_wide(); fig11_row2(); fig12(); print("row-consistent panels written")
