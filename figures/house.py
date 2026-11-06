# -*- coding: utf-8 -*-
"""House style of every RuleSight figure. Each canvas is drawn at its printed size (the paper's text width is 390 pt = 5.40 in,
a panel four per row is 0.245 of it), so the point sizes below are the printed sizes and every figure prints its text at the
same size: 6 pt for all text, 6.5 pt bold for panel titles, 7 pt for panel letters. One palette for all figures:
RuleSight red, zero-shot blue, labeled green, neutral grey; FEGAR's purple for removal steps."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
D = json.load(open(os.path.join(HERE, "figdata.json"), encoding="utf-8")); G = D["G"]
TEXTW = 390.0 / 72.27                      # 5.397 in
PW = 0.245 * TEXTW                         # 1.322 in, one panel four per row
PH = 1.40                                  # panel height
FS, FS_T, FS_L = 6.0, 6.5, 7.0             # body, title, letter (printed points)

def _font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    return next((c for c in ("Times New Roman", "Liberation Serif", "TeX Gyre Termes", "DejaVu Serif") if c in names), "serif")
plt.rcParams.update({
    "font.family": _font(), "font.size": FS, "axes.titlesize": FS_T, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
    "legend.fontsize": FS, "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 1.5, "lines.linewidth": 0.9, "grid.linewidth": 0.4, "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "stix", "axes.unicode_minus": False, "savefig.dpi": 600})

INK, SPINE, GRIDC = "#262626", "#5A5A5A", "#E3E3E3"
# palettes given by the user (Nature reference): #4F587D #C68DC0 #C2E0EE #776B97 #DBC8ED | #5271AE #70ACDE #F5CC7D #FFA660 #D85B59 | #CAE5F8 #9DCBED #FAEFF5 #F2C5DA #E286AF
RS, RS_D, RS_L = "#D85B59", "#B8423F", "#EBA9A8"            # RuleSight
ZS, ZS_D, ZS_L = "#70ACDE", "#5271AE", "#CAE5F8"            # zero-shot
GT = "#2E9E5B"                                              # labeled / correct
GREY, GREY_L = "#9AA3B0", "#DDE2E8"
PURPLE, COT, FEW = "#776B97", "#8CC5A3", "#C9B6E4"          # removal step, chain-of-thought, few-shot
WIN = "#F2C5DA"                                             # region where RuleSight is higher
CATC = {"Text overflow": "#776B97", "Abnormal ellipsis": "#3BB5B5", "Undisplayed content": "#79B95E", "Cropped content": "#E2C640", "Container overlap": "#F08A4B"}
MCOL = {"haiku": "#4F587D", "sonnet55": "#776B97", "gpt54mini": "#C68DC0", "gemini38flash": "#F5CC7D", "qwen3vl8b": "#70ACDE", "internvl35": "#FFA660"}
MODELS = ["haiku", "sonnet55", "gpt54mini", "gemini38flash", "qwen3vl8b", "internvl35"]
MNAME = {"haiku": "Claude Haiku 5.5", "sonnet55": "Claude Sonnet 5.5", "gpt54mini": "GPT-5.4 mini", "gemini38flash": "Gemini 3.8 Flash", "qwen3vl8b": "Qwen3-VL-8B", "internvl35": "InternVL3.5-8B"}
MSHORT = {"haiku": "Haiku", "sonnet55": "Sonnet", "gpt54mini": "GPT mini", "gemini38flash": "Gemini", "qwen3vl8b": "Qwen", "internvl35": "InternVL"}
CATS = ["Text overflow", "Abnormal ellipsis", "Undisplayed content", "Cropped content", "Container overlap"]
CSHORT = {"Text overflow": "Overflow", "Abnormal ellipsis": "Ellipsis", "Undisplayed content": "Undisplayed", "Cropped content": "Cropped", "Container overlap": "Overlap", "All": "All"}

def panel(left=0.30, bottom=0.36, right=0.05, top=0.15, w=PW, h=PH, projection=None):
    """a canvas of the printed panel size with fixed margins in inches, so every panel of a figure lines up"""
    fig = plt.figure(figsize=(w, h))
    rect = [left / w, bottom / h, 1 - (left + right) / w, 1 - (bottom + top) / h]
    ax = fig.add_axes(rect, projection=projection) if projection else fig.add_axes(rect)
    return fig, ax

def style(ax, title=None, xlabel=None, ylabel=None, grid="both"):
    for s in ax.spines.values(): s.set_color(SPINE); s.set_linewidth(0.6)
    ax.tick_params(colors=INK, length=2.0, width=0.5)
    if grid: ax.grid(True, axis=grid, ls="-", color=GRIDC, lw=0.45); ax.set_axisbelow(True)
    if title: ax.set_title(title, fontsize=FS_T, fontweight="bold", pad=2.5, color=INK)
    if xlabel: ax.set_xlabel(xlabel, fontweight="bold", color=INK)
    if ylabel: ax.set_ylabel(ylabel, fontweight="bold", color=INK)

def letter(fig, l, y=0.006):
    fig.text(0.5, y, f"({l})", ha="center", va="bottom", fontsize=FS_L, color=INK)

def note(ax, text, loc="upper left"):
    x, y, ha, va = {"upper left": (0.04, 0.96, "left", "top"), "lower right": (0.96, 0.04, "right", "bottom"), "upper right": (0.96, 0.96, "right", "top"), "lower left": (0.04, 0.04, "left", "bottom")}[loc]
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=FS, color=INK, zorder=9, linespacing=1.15, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#C9C9C9", lw=0.5, alpha=0.92))

def legend_strip(name, handles, ncol=None, h=0.17, w=TEXTW, rows=None):
    """one legend row, or several rows given as lists of handles"""
    rows = rows or [handles]; fig = plt.figure(figsize=(w, h * len(rows)))
    for k, hs in enumerate(rows):
        fig.legend(handles=hs, loc="center", bbox_to_anchor=(0.5, 1 - (k + 0.5) / len(rows)), ncol=len(hs), frameon=False, fontsize=FS, handlelength=1.4, handletextpad=0.4, columnspacing=1.1, borderaxespad=0)
    save(fig, name)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf")); fig.savefig(os.path.join(OUT, name + ".png"), dpi=600, facecolor="white"); plt.close(fig)

def dot(color, label, marker="o", ms=4.2, mec="#555555", mfc=None, mew=0.4):
    return Line2D([], [], ls="none", marker=marker, ms=ms, mfc=mfc or color, mec=mec, mew=mew, label=label)

def box(color, label, ec="none"):
    return Patch(fc=color, ec=ec, lw=0.4, label=label)


def tint(color, t):
    """mixes a color with white; t = 0 gives the color, t = 1 white"""
    from matplotlib import colors as mc
    r, g, b = mc.to_rgb(color); return (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)

def gbar(ax, x0, x1, y0, y1, color, along="y", light=0.55, edge=True, z=3):
    """a bar filled with a gradient from a light tint at its base to the full color at its end (along the value axis)"""
    import numpy as np
    from matplotlib.patches import Rectangle
    from matplotlib.colors import LinearSegmentedColormap
    if abs(x1 - x0) < 1e-12 or abs(y1 - y0) < 1e-12: return
    cm = LinearSegmentedColormap.from_list("g", [tint(color, light), color]); n = 64
    grad = np.linspace(0, 1, n)
    if along == "y":
        img = grad.reshape(-1, 1) if y1 >= y0 else grad[::-1].reshape(-1, 1); ext = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
    else:
        img = grad.reshape(1, -1) if x1 >= x0 else grad[::-1].reshape(1, -1); ext = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
    im = ax.imshow(img, extent=ext, origin="lower", cmap=cm, aspect="auto", interpolation="bicubic", zorder=z)
    rect = Rectangle((ext[0], ext[2]), ext[1] - ext[0], ext[3] - ext[2], fc="none", ec=tint(color, -0.0) if edge else "none", lw=0.4 if edge else 0, zorder=z + 0.1)
    ax.add_patch(rect); im.set_clip_path(rect)

def kde(v, grid, bw=None):
    import numpy as np
    v = np.asarray(v, float); sd = v.std() if v.std() > 0 else 1e-3; bw = bw or 1.06 * sd * len(v) ** (-0.2)
    return np.exp(-0.5 * ((grid[:, None] - v[None, :]) / bw) ** 2).sum(1) / (len(v) * bw * np.sqrt(2 * np.pi))
