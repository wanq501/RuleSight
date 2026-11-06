# -*- coding: utf-8 -*-
"""Fig. 3D, eight three-dimensional panels at printed size, four per row, house palette and 6 pt text."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.proj3d import proj_transform
from house import *

MCC_CM = mcolors.LinearSegmentedColormap.from_list("house_seq", ["#F6EFC7", "#C9E3A0", "#7DC4A6", "#4A9DB8", "#4F6FB0", "#5B4D97"])
DIV_CM = mcolors.LinearSegmentedColormap.from_list("house_div", [(0, "#3F6FB6"), (0.25, "#90B7E2"), (0.5, "#F5F5F5"), (0.75, "#F09A9B"), (1, "#C9353F")])
MCOL = {"haiku": "#4F587D", "sonnet55": "#776B97", "gpt54mini": "#E8B54A", "gemini38flash": "#F08A4B", "qwen3vl8b": "#4E9E7A", "internvl35": "#7FB3DD"}
ROWS = ["internvl35", "qwen3vl8b", "gpt54mini", "haiku", "gemini38flash", "sonnet55"]
MS3 = {"haiku": "Haiku", "sonnet55": "Sonnet", "gpt54mini": "GPT", "gemini38flash": "Gemini", "qwen3vl8b": "Qwen", "internvl35": "Intern"}
COLS = ["Undisplayed content", "Container overlap", "Text overflow", "Abnormal ellipsis", "Cropped content"]
CODE = {"Undisplayed content": "D3", "Container overlap": "D2", "Text overflow": "D6", "Abnormal ellipsis": "D7", "Cropped content": "D8"}
CT = D["categories"]

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *a, **kw): super().__init__((0, 0), (0, 0), *a, **kw); self._v = (xs, ys, zs)
    def do_3d_projection(self, renderer=None):
        xs, ys, zs = self._v; x, y, z = proj_transform(xs, ys, zs, self.axes.M); self.set_positions((x[0], y[0]), (x[1], y[1])); return min(z)

def canvas(title, l, rect=(-0.04, 0.09, 1.0, 0.82), h=1.62, ty=0.985):
    fig = plt.figure(figsize=(PW, h)); ax = fig.add_axes(list(rect), projection="3d")
    fig.text(0.5, ty, title, ha="center", va="top", fontsize=FS_T, fontweight="bold", color=INK); fig.text(0.5, 0.005, f"({l})", ha="center", va="bottom", fontsize=FS_L, color=INK)
    for p in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane): p.set_facecolor("white"); p.set_edgecolor("#CFCFCF"); p.set_linewidth(0.4); p.set_alpha(1.0)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis): a._axinfo["grid"].update(color="#D6D6D6", linestyle=(0, (2, 1.5)), linewidth=0.3); a._axinfo["tick"]["inward_factor"] = 0.0; a._axinfo["tick"]["outward_factor"] = 0.2; a.line.set_linewidth(0.5)
    ax.tick_params(labelsize=FS, pad=-3.5)
    return fig, ax

def bars(name, val, title, l, cmap, vmin, vmax, zt):
    fig, ax = canvas(title, l, rect=(-0.01, 0.09, 0.90, 0.82)); norm = mcolors.Normalize(vmin, vmax); NSEG = 10
    for i, m in enumerate(ROWS):
        for j, c in enumerate(COLS):
            z = val(m, c); lo, hi = (0.0, z) if z >= 0 else (z, 0.0); e = np.linspace(lo, hi, NSEG + 1)
            for k in range(NSEG): ax.bar3d(j - 0.3, i - 0.3, e[k], 0.6, 0.6, max(e[k + 1] - e[k], 1e-6), color=cmap(norm((e[k] + e[k + 1]) / 2)), edgecolor="none", shade=True, alpha=0.97)
    ax.set_xticks(range(len(COLS))); ax.set_xticklabels([CODE[c] for c in COLS]); ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([MS3[m] for m in ROWS], ha="left", va="center")
    ax.set_zlim(vmin, vmax); ax.set_zticks(zt); ax.view_init(elev=22, azim=-60); ax.set_box_aspect((1.1, 1.6, 0.8)); save(fig, name)

def ribbons(name, st, title, l):
    T = G["thr_cat"]; taus = np.array(G["taus"]); norm = mcolors.Normalize(-0.25, 0.7); fig, ax = canvas(title, l, rect=(0.0, 0.09, 0.86, 0.82))
    order = ["Container overlap", "Cropped content", "Undisplayed content", "Abnormal ellipsis", "Text overflow"]
    for i, cat in enumerate(order):
        z = np.array(T[f"{st}|{cat}"]["mcc"]); tf = np.linspace(taus[0], taus[-1], 50); zf = np.interp(tf, taus, z); X, Y = np.meshgrid(tf, [i - 0.32, i + 0.32]); Z = np.vstack([zf, zf])
        ax.plot_surface(X, Y, Z, facecolors=MCC_CM(norm(Z)), rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False, alpha=0.97)
        for yy in (i - 0.32, i + 0.32): ax.plot(tf, np.full_like(tf, yy), zf, color="#555555", lw=0.3)
    ax.set_xticks([0, 0.45, 0.9]); ax.set_xticklabels(["0", "0.45", "0.9"]); ax.set_yticks(range(len(order))); ax.set_yticklabels([CODE[c] for c in order], ha="left", va="center")
    ax.set_zlim(-0.05, 0.65); ax.set_zticks([0, 0.3, 0.6]); ax.view_init(elev=24, azim=-60); ax.set_box_aspect((1.15, 1.25, 0.8)); save(fig, name)

def scatter_prm(name, l):
    fig, ax = canvas("Precision, recall, MCC", l, rect=(0.07, 0.09, 0.86, 0.82)); Z0 = -0.15
    for key, v in D["configs"].items():
        m, st = key.split("|"); col = MCOL[m]
        ax.scatter(v["P"], v["R"], Z0, s=2.5, color=col, alpha=0.35, edgecolor="none", depthshade=False)
        if st == "s0": ax.scatter(v["P"], v["R"], v["mcc"], s=9, facecolor="white", edgecolor=col, linewidth=0.7, depthshade=False)
        elif st == "s3nomeas": ax.scatter(v["P"], v["R"], v["mcc"], s=22, marker="*", facecolor=col, edgecolor=INK, linewidth=0.25, depthshade=False)
        else: ax.scatter(v["P"], v["R"], v["mcc"], s=3, facecolor=col, edgecolor="none", alpha=0.8, depthshade=False)
    for m in MODELS:
        a, b = D["configs"][f"{m}|s0"], D["configs"][f"{m}|s3nomeas"]; ax.add_artist(Arrow3D([a["P"], b["P"]], [a["R"], b["R"]], [a["mcc"], b["mcc"]], mutation_scale=5, lw=0.6, arrowstyle="-|>", color=MCOL[m], shrinkA=2, shrinkB=3))
    ax.set_xlim(0.4, 0.9); ax.set_ylim(0, 0.8); ax.set_zlim(Z0, 0.65); ax.set_xticks([0.5, 0.7, 0.9]); ax.set_yticks([0, 0.4, 0.8]); ax.set_zticks([0, 0.3, 0.6])
    ax.set_xlabel("P", labelpad=-9, fontsize=FS); ax.set_ylabel("R", labelpad=-9, fontsize=FS); ax.view_init(elev=22, azim=-128); ax.set_box_aspect((1.1, 1.1, 0.85)); save(fig, name)

def bubbles(name, l):
    R1 = D["rq1"]; fig, ax = canvas("Cost, latency, MCC", l, rect=(0.07, 0.09, 0.84, 0.82)); lx = lambda c: np.log10(max(c, 0.05))
    for m in MODELS:
        (c0, c1), (l0, l1) = R1[m]["cost"], R1[m]["latency"]; z0, z1 = R1[m]["zs"]["mcc"], R1[m]["rs"]["mcc"]; f0, f1 = R1[m]["zs"]["F1"], R1[m]["rs"]["F1"]; col = MCOL[m]
        for cx, lt, z, f, fill in ((c0, l0, z0, f0, False), (c1, l1, z1, f1, True)):
            ax.plot([lx(cx)] * 2, [lt] * 2, [-0.15, z], color=col, lw=0.4, ls=(0, (1.5, 1)), alpha=0.8)
            ax.scatter(lx(cx), lt, z, s=6 + 40 * f, facecolor=col if fill else "white", edgecolor=col if not fill else INK, linewidth=0.6 if not fill else 0.25, alpha=0.95, depthshade=False)
        ax.add_artist(Arrow3D([lx(c0), lx(c1)], [l0, l1], [z0, z1], mutation_scale=5, lw=0.6, arrowstyle="-|>", color=col, shrinkA=3, shrinkB=4))
    ax.set_xticks([lx(0.05), 0, 1]); ax.set_xticklabels(["local", "1", "10"]); ax.set_xlim(lx(0.05) - 0.2, 1.35); ax.set_ylim(0, 11); ax.set_zlim(-0.15, 0.65); ax.set_yticks([0, 5, 10]); ax.set_zticks([0, 0.3, 0.6])
    ax.set_xlabel("USD", labelpad=-9, fontsize=FS); ax.set_ylabel("s", labelpad=-9, fontsize=FS); ax.view_init(elev=20, azim=-55); ax.set_box_aspect((1.1, 1.1, 0.85)); save(fig, name)

def legend():
    fig = plt.figure(figsize=(TEXTW, 0.40))
    for x0, cmap, lo, hi, lab, ticks in ((0.27, MCC_CM, -0.25, 0.7, "MCC, (a) (b) (e) (f)", [-0.2, 0, 0.2, 0.4, 0.6]), (0.70, DIV_CM, -0.4, 0.4, "Change, (c) (d)", [-0.4, -0.2, 0, 0.2, 0.4])):
        cax = fig.add_axes([x0, 0.62, 0.22, 0.15]); cb = fig.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(lo, hi), cmap=cmap), cax=cax, orientation="horizontal"); cb.outline.set_linewidth(0.4); cb.set_ticks(ticks)
        cb.ax.tick_params(labelsize=FS, length=1.5, width=0.4, pad=1); fig.text(x0 - 0.012, 0.695, lab, ha="right", va="center", fontsize=FS)
    hs = [Line2D([], [], ls="none", marker="o", ms=3.5, mfc=MCOL[m], mec="none", label=MSHORT[m]) for m in MODELS] + [Line2D([], [], ls="none", marker="o", ms=3.5, mfc="white", mec=INK, mew=0.6, label="Zero-shot"), Line2D([], [], ls="none", marker="*", ms=5, mfc="#777777", mec=INK, mew=0.3, label="RuleSight (g)"), Line2D([], [], ls="none", marker="o", ms=3.5, mfc="#777777", mec=INK, mew=0.3, label="RuleSight (h)")]
    fig.legend(handles=hs, loc="center", bbox_to_anchor=(0.5, 0.18), ncol=9, frameon=False, fontsize=FS, handletextpad=0.3, columnspacing=0.8)
    save(fig, "Fig12_legend")

if __name__ == "__main__":
    bars("Fig12a", lambda m, c: CT[f"{m}|{c}"]["zs"], "MCC, zero-shot", "a", MCC_CM, -0.25, 0.7, [0, 0.3, 0.6])
    bars("Fig12b", lambda m, c: CT[f"{m}|{c}"]["rs"], "MCC, RuleSight", "b", MCC_CM, -0.25, 0.7, [0, 0.3, 0.6])
    bars("Fig12c", lambda m, c: CT[f"{m}|{c}"]["rs"] - CT[f"{m}|{c}"]["zs"], "Change in MCC", "c", DIV_CM, -0.4, 0.4, [-0.3, 0, 0.3])
    bars("Fig12d", lambda m, c: CT[f"{m}|{c}"]["rs_f1"] - CT[f"{m}|{c}"]["zs_f1"], "Change in F1", "d", DIV_CM, -0.4, 0.4, [-0.3, 0, 0.3])
    ribbons("Fig12e", "s0", "Threshold, zero-shot", "e"); ribbons("Fig12f", "s3nomeas", "Threshold, RuleSight", "f")
    scatter_prm("Fig12g", "g"); bubbles("Fig12h", "h"); legend(); print("3D panels written")
