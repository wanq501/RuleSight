# -*- coding: utf-8 -*-
"""The three 3D figures as rows of four panels, like every other figure of the paper.
Fig3DBars (reference 5): per model, MCC per category under zero-shot prompting and RuleSight as gradient bars with values.
Fig14 (reference 8): MCC ribbons over the confidence threshold, Claude Haiku 5.5 and Claude Sonnet 5.5 under both oracles.
Fig3DBubbles (references 6 and 7): per model, every category moving from zero-shot prompting to RuleSight in the space of
precision, recall and MCC, with shadows on the floor. Every row shares one coordinate system and one viewing angle."""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors, cm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import PchipInterpolator
from house import *

P = json.load(open(os.path.join(HERE, "catprm.json"))); RB = json.load(open(os.path.join(HERE, "ribbons.json"))); TAUS = np.array(RB["taus"])
CATS5 = ["Text overflow", "Abnormal ellipsis", "Undisplayed content", "Cropped content", "Container overlap"]
SHORT = {"Text overflow": "Overflow", "Abnormal ellipsis": "Ellipsis", "Undisplayed content": "Undispl.", "Cropped content": "Cropped", "Container overlap": "Overlap"}
CODE = {"Text overflow": "TO", "Abnormal ellipsis": "AE", "Undisplayed content": "UC", "Cropped content": "CC", "Container overlap": "CO"}
MOD4 = [("haiku", "Claude Haiku 5.5"), ("sonnet55", "Claude Sonnet 5.5"), ("gpt54mini", "GPT-5.4 mini"), ("qwen3vl8b", "Qwen3-VL-8B")]
REF5 = mcolors.LinearSegmentedColormap.from_list("ref5", ["#9E0142", "#D53E4F", "#F46D43", "#FDAE61", "#FEE08B", "#E6F598", "#ABDDA4", "#66C2A5", "#3288BD", "#5E4FA2"])
RIB = mcolors.LinearSegmentedColormap.from_list("ribbon", ["#3B2A6B", "#3E5C9C", "#3D8DB4", "#4DB6A7", "#8BD17C", "#E7E36A"])
CATCOL = {"Text overflow": "#776B97", "Abnormal ellipsis": "#3BB5B5", "Undisplayed content": "#79B95E", "Cropped content": "#E2C640", "Container overlap": "#F08A4B"}
W, H = PW, 1.38

def ax3(fig, view, aspect, pane="white"):
    ax = fig.add_axes([0.05, 0.03, 0.86, 1.07], projection="3d"); ax.computed_zorder = False
    for p in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane): p.set_facecolor(pane); p.set_edgecolor("#C3C8CF"); p.set_linewidth(0.4)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a._axinfo["grid"].update(color="#D3D6DB", linestyle=(0, (3, 2)), linewidth=0.3); a._axinfo["tick"]["inward_factor"] = 0.0; a._axinfo["tick"]["outward_factor"] = 0.2; a.line.set_linewidth(0.6); a.line.set_color(INK)
    ax.view_init(**view); ax.set_box_aspect(aspect, zoom=0.93); ax.tick_params(labelsize=FS - 0.6, pad=-3.2); return ax

def head(fig, title, l):
    fig.text(0.5, 0.99, title, ha="center", va="top", fontsize=FS_T, fontweight="bold", color=INK); fig.text(0.5, 0.004, f"({l})", ha="center", va="bottom", fontsize=FS_L, color=INK)

def gbar(ax, x, y, z, color, dx, dy, order, n=20):
    if z < 0: n = 1
    lo, hi = (0.0, z) if z >= 0 else (z, 0.0); hi = max(hi, lo + 0.004); zs = np.linspace(lo, hi, n + 1); polys, cols = [], []
    for k in range(n):
        t = (k + 0.5) / n; c = tint(color, 0.85 * (1 - t) ** 1.3) if z >= 0 else color; r, g, b = mcolors.to_rgb(c)
        polys += [[(x, y, zs[k]), (x + dx, y, zs[k]), (x + dx, y, zs[k + 1]), (x, y, zs[k + 1])], [(x + dx, y, zs[k]), (x + dx, y + dy, zs[k]), (x + dx, y + dy, zs[k + 1]), (x + dx, y, zs[k + 1])]]
        cols += [c, (r * 0.86, g * 0.86, b * 0.86)]
    top = hi if z >= 0 else lo; polys.append([(x, y, top), (x + dx, y, top), (x + dx, y + dy, top), (x, y + dy, top)]); cols.append(tint(color, 0.15))
    pc = Poly3DCollection(polys, facecolors=cols, edgecolors=cols, linewidths=0.2); pc.set_zorder(order); ax.add_collection3d(pc)

NB = mcolors.Normalize(-0.1, 0.75)
def repel(fig, labs, dz=0.03, rounds=40):
    """lift a value label until it overlaps no other: the RuleSight label of a pair moves, or the upper one within a row"""
    for _ in range(rounds):
        fig.canvas.draw(); r = fig.canvas.get_renderer(); bb = [t.get_window_extent(r).expanded(1.04, 1.08) for t, _ in labs]; moved = False
        for a in range(len(labs)):
            for b in range(a + 1, len(labs)):
                if not bb[a].overlaps(bb[b]): continue
                (ta, ja), (tb, jb) = labs[a], labs[b]
                t = ta if ja > jb else tb if jb > ja else (ta if bb[a].y0 >= bb[b].y0 else tb)
                x, y, z = t.get_position_3d() if hasattr(t, "get_position_3d") else t._position3d
                t.set_position_3d((x, y, z + dz)); moved = True
        if not moved: break
def bars(name, m, title, l):
    fig = plt.figure(figsize=(W, H)); view = dict(elev=36, azim=-62); ax = ax3(fig, view, (1.45, 0.95, 0.85))
    az = np.deg2rad(view["azim"]); eye = np.array([np.cos(az), np.sin(az)]); D = 0.5; YP = (0.0, 2.2)
    cells = sorted(((i, j) for i in range(5) for j in range(2)), key=lambda ij: (ij[0] + D / 2) * eye[0] + (YP[ij[1]] + D / 2) * eye[1])
    for k, (i, j) in enumerate(cells):
        z = P[f"{m}|{('s0', 's3nomeas')[j]}|{CATS5[i]}"]["MCC"]; gbar(ax, i - D / 2, YP[j] - D / 2, z, REF5(NB(z)), D, D, k + 1)
    labs = []
    for i in range(5):
        for j in range(2):
            z = P[f"{m}|{('s0', 's3nomeas')[j]}|{CATS5[i]}"]["MCC"]
            labs.append((ax.text(i, YP[j], max(z, 0) + 0.04, f"{z:.2f}".replace("0.", "."), ha="center", va="bottom", fontsize=FS - 1.9, fontweight="bold", color=INK, zorder=1000,
                                 bbox=dict(boxstyle="square,pad=0.06", fc="white", ec="none", alpha=0.8)), j))
    ax.set_xticks(range(5)); ax.set_xticklabels([CODE[c] for c in CATS5], ha="right", va="center", rotation=0); ax.set_yticks(list(YP)); ax.set_yticklabels(["ZS", "RS"], ha="left", va="center")
    ax.set_xlim(-0.6, 4.6); ax.set_ylim(-0.6, 2.8); ax.set_zlim(-0.25, 0.75); ax.set_zticks([0, 0.3, 0.6]); ax.set_zticklabels(["0", ".3", ".6"]); head(fig, title, l)
    repel(fig, labs); save(fig, name)

NR = mcolors.Normalize(0.0, 0.70); RIBCATS = ["Container overlap", "Cropped content", "Undisplayed content", "Abnormal ellipsis", "Text overflow"]
def ribbons(name, m, st, title, l):
    fig = plt.figure(figsize=(W, H)); ax = ax3(fig, dict(elev=24, azim=-60), (1.3, 1.45, 0.8), pane="#FBFBFC")
    tf = np.linspace(0, 0.9, 91); grid = np.round(np.arange(0, 0.91, 0.1), 2); idx = [int(np.argmin(np.abs(TAUS - g))) for g in grid]
    for i, cat in enumerate(RIBCATS):
        z = PchipInterpolator(grid, np.array(RB["mcc"][f"{m}|{st}|{cat}"])[idx])(tf); y0, y1 = i - 0.34, i + 0.34; X, Y = np.meshgrid(tf, [y0, y1]); Z = np.vstack([z, z])
        ax.plot_surface(X, Y, Z, facecolors=RIB(NR(Z)), rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False, alpha=0.98)
        ax.plot(tf, np.full_like(tf, y0), z, color="#2E2E2E", lw=0.6); ax.plot(tf, np.full_like(tf, y1), z, color="#2E2E2E", lw=0.25, alpha=0.6)
    ax.set_xlim(0, 0.9); ax.set_ylim(-0.6, 4.6); ax.set_zlim(-0.05, 0.75); ax.set_xticks([0, 0.4, 0.8]); ax.set_xticklabels(["0", ".4", ".8"]); ax.set_yticks(range(5)); ax.set_yticklabels([CODE[c] for c in RIBCATS], ha="left", va="center"); ax.set_zticks([0, 0.3, 0.6]); ax.set_zticklabels(["0", ".3", ".6"]); ax.set_zticklabels(["0", ".3", ".6"])
    ax.set_xlabel("Threshold", fontsize=FS - 0.6, fontweight="bold", labelpad=-9); head(fig, title, l); save(fig, name)

def bubbles(name, m, title, l):
    fig = plt.figure(figsize=(W, H)); ax = ax3(fig, dict(elev=22, azim=-58), (1.1, 1.1, 0.95)); Z0 = -0.2
    for c in CATS5:
        a, b = P[f"{m}|s0|{c}"], P[f"{m}|s3nomeas|{c}"]; s0, s1 = 6 + 0.05 * a["pos"], 6 + 0.05 * b["pos"]
        for v, s in ((a, s0), (b, s1)): ax.scatter(v["P"], v["R"], Z0, s=s * 0.7, color="#C3C8CF", alpha=0.6, edgecolor="none", depthshade=False, zorder=1)
        ax.plot([a["P"], b["P"]], [a["R"], b["R"]], [a["MCC"], b["MCC"]], color=CATCOL[c], lw=0.9, alpha=0.9, zorder=2)
        ax.plot([b["P"]] * 2, [b["R"]] * 2, [Z0, b["MCC"]], color=CATCOL[c], lw=0.3, alpha=0.5, zorder=2)
        ax.scatter(a["P"], a["R"], a["MCC"], s=s0, color="white", edgecolor=CATCOL[c], linewidth=0.8, depthshade=False, zorder=3)
        ax.scatter(b["P"], b["R"], b["MCC"], s=s1, color=CATCOL[c], edgecolor="white", linewidth=0.4, depthshade=False, zorder=4)
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0, 1.0); ax.set_zlim(Z0, 0.75); ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", ".5", "1"]); ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["0", ".5", "1"]); ax.set_zticks([0, 0.3, 0.6]); ax.set_zticklabels(["0", ".3", ".6"])
    ax.set_xlabel("Precision", fontsize=FS - 0.6, fontweight="bold", labelpad=-9); ax.set_ylabel("Recall", fontsize=FS - 0.6, fontweight="bold", labelpad=-9); fig.text(0.975, 0.64, "MCC", rotation=90, ha="center", va="center", fontsize=FS - 0.6, fontweight="bold", color=INK); head(fig, title, l); save(fig, name)

def colorbar_strip(name, cmap, norm, label, ticks):
    fig = plt.figure(figsize=(TEXTW, 0.24)); cax = fig.add_axes([0.40, 0.46, 0.30, 0.24])
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal"); cb.outline.set_linewidth(0.4); cb.set_ticks(ticks); cb.ax.tick_params(labelsize=FS, length=1.5, width=0.4, pad=1)
    fig.text(0.39, 0.58, label, ha="right", va="center", fontsize=FS); save(fig, name)

if __name__ == "__main__":
    for k, (m, nm) in enumerate(MOD4): bars(f"Fig3DBars{'abcd'[k]}", m, nm, "abcd"[k])
    colorbar_strip("Fig3DBars_legend", REF5, NB, "MCC of the bar, ZS zero-shot and RS RuleSight", [0, 0.2, 0.4, 0.6])
    for k, (m, st, t) in enumerate((("haiku", "s0", "Haiku 5.5, zero-shot"), ("haiku", "s3nomeas", "Haiku 5.5, RuleSight"), ("sonnet55", "s0", "Sonnet 5.5, zero-shot"), ("sonnet55", "s3nomeas", "Sonnet 5.5, RuleSight"))):
        ribbons(f"Fig14{'abcd'[k]}", m, st, t, "abcd"[k])
    colorbar_strip("Fig14_legend", RIB, NR, "MCC, the height and colour of each ribbon", [0, 0.2, 0.4, 0.6])
    for k, (m, nm) in enumerate(MOD4): bubbles(f"Fig3DBubbles{'abcd'[k]}", m, nm, "abcd"[k])
    hs = [Line2D([], [], ls="none", marker="o", ms=4.2, mfc=CATCOL[c], mec="white", label=SHORT[c].replace("Undispl.", "Undisplayed")) for c in CATS5] + [Line2D([], [], ls="none", marker="o", ms=4.2, mfc="white", mec="#666666", label="Zero-shot (open)"), Line2D([], [], ls="none", marker="o", ms=4.2, mfc="#8A8A8A", mec="white", label="RuleSight (filled)"), Line2D([], [], ls="none", marker="o", ms=4.2, mfc="#C3C8CF", mec="none", label="Shadow")]
    legend_strip("Fig3DBubbles_legend", hs); print("3D rows written")
