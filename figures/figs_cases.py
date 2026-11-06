# -*- coding: utf-8 -*-
"""Interface comparison figures at printed size (house.py): one cell per UI-Lens page, four per row, 6 pt text. The defect
region is cropped from the screenshot without stretching; green dashed is the labeled box, blue and red are the boxes Claude
Haiku 5.5 reported under zero-shot prompting and RuleSight; one square per model and oracle, green when correct."""
import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
from PIL import Image
from house import *

TASK_RULES = {"Text Overflow": {"D6"}, "Abnormal Text Ellipsis": {"D7"}, "Undisplayed Content": {"D3", "D4"}, "Cropped Content": {"D8"}, "Container Overlap": {"D1", "D2"}}
TASK_NAME = {"Text Overflow": "Text overflow", "Abnormal Text Ellipsis": "Abnormal ellipsis", "Undisplayed Content": "Undisplayed content", "Cropped Content": "Cropped content", "Container Overlap": "Container overlap"}
CH = 2.02                                   # cell height (in)
IMG = (0.02, 0.90 / CH, 0.96, 0.94 / CH)    # image area: 0.94 in high between the text block (below 0.90 in) and the title

def overlap(a, b):
    return max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])) * max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))

def target(c):
    rules = TASK_RULES[c["task"]]; rs = [x["box_px"] for x in c["models"]["haiku"]["rulesight"]["claims"] if x["box_px"] and x["type"] in rules]
    gts = [g["box_px"] for g in c["ground_truth"] if g.get("box_px")]
    if gts: return max(gts, key=lambda g: (sum(overlap(g, r) for r in rs), -g[1]))
    zs = [(x["box_px"], x["type"]) for x in c["models"]["haiku"]["zero_shot"]["claims"] if x["box_px"]]
    return ([b for b, t in zs if t in rules] or [b for b, t in zs] or [None])[0]

def blur_faces(img, c):
    """blur the faces of real people in a screenshot (privacy), leaving every labeled or reported box untouched"""
    import cv2
    from PIL import ImageFilter
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    g = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY); faces = det.detectMultiScale(g, scaleFactor=1.1, minNeighbors=7, minSize=(28, 28))
    keep = [x["box_px"] for x in c["ground_truth"] if x.get("box_px")] + [x["box_px"] for m in ("zero_shot", "rulesight") for x in c["models"]["haiku"][m]["claims"] if x["box_px"]]
    for (x, y, w, h) in [tuple(int(v) for v in f) for f in faces]:
        if any(overlap((x, y, w, h), b) > 0.3 * w * h for b in keep): continue
        pad = int(0.15 * w); box = (max(0, x - pad), max(0, y - pad), min(img.width, x + w + pad), min(img.height, y + h + pad))
        img.paste(img.crop(box).filter(ImageFilter.GaussianBlur(radius=int(max(4, w // 6)))), box[:2])
    return img

def window(W, H, box, aspect, img=None):
    """the crop around the target box; its edges are then moved, within limits that keep the box well inside, to run through
    background rather than through a line of text"""
    if box is None: w = min(W, H * aspect); return (0, 0, w, w / aspect)
    bw, bh = box[2], box[3]; w = max(2.6 * bw, 2.6 * bh * aspect, 0.55 * W); w = min(w, W, H * aspect); h = w / aspect
    cx, cy = box[0] + bw / 2, box[1] + bh / 2; x0, y0 = min(max(0, cx - w / 2), W - w), min(max(0, cy - h / 2), H - h)
    if img is not None:
        g = np.asarray(img.convert("L"), dtype=float); m = 0.08
        def cost(xx, yy, axis):
            a, b_, c_, d = int(xx), int(yy), int(xx + w) - 1, int(yy + h) - 1
            return float(g[b_, a:c_].std() + g[d, a:c_].std()) if axis == 0 else float(g[b_:d, a].std() + g[b_:d, c_].std())
        lo, hi = max(0, box[1] + bh + m * h - h), min(H - h, box[1] - m * h)
        if lo < hi: y0 = min(np.linspace(max(lo, y0 - 0.2 * h), min(hi, y0 + 0.2 * h), 41), key=lambda yy: cost(x0, yy, 0) + 0.02 * abs(yy - y0))
        lo, hi = max(0, box[0] + bw + m * w - w), min(W - w, box[0] - m * w)
        if lo < hi: x0 = min(np.linspace(max(lo, x0 - 0.2 * w), min(hi, x0 + 0.2 * w), 41), key=lambda xx: cost(xx, y0, 1) + 0.02 * abs(xx - x0))
    return (x0, y0, w, h)

PLACED = []
def tag(ax, b, text, color, win):
    """the category tag at the top-left corner of the box, kept wholly inside the image: moved left at the right edge and
    inside the box when the box touches the top"""
    x0, y0, w, h = win
    if b[0] + b[2] < x0 or b[0] > x0 + w or b[1] + b[3] < y0 or b[1] > y0 + h: return
    tw, th = 0.13 * w, 0.075 * h
    x = min(max(b[0], x0), x0 + w - tw); y = b[1] if b[1] - th >= y0 else max(b[1], y0) + th
    # the first free spot among the corner, one tag to the right, one tag lower, or one tag to the left
    hit = lambda xx, yy: any(xx < px + pw and px < xx + tw and yy - th < py and py - ph < yy for px, py, pw, ph in PLACED)
    for dx, dy in ((0, 0), (tw, 0), (0, th), (tw, th), (-tw, 0), (0, 2 * th)):
        xx = min(max(x + dx, x0), x0 + w - tw); yy = min(max(y + dy, y0 + th), y0 + h)
        if not hit(xx, yy): x, y = xx, yy; break
    PLACED.append((x, y, tw, th))
    ax.text(x, y, text, ha="left", va="bottom", fontsize=FS - 0.5, color="white", fontweight="bold", zorder=7, clip_on=True, bbox=dict(boxstyle="square,pad=0.1", fc=color, ec="none"))

def cell(c, root, name, l):
    img = blur_faces(Image.open(os.path.join(root, c["group"], c["sub"], c["sid"].replace("/", "_") + ".jpg")).convert("RGB"), c); W, H = img.size
    fig = plt.figure(figsize=(PW, CH)); ax = fig.add_axes(list(IMG)); aspect = IMG[2] * PW / (IMG[3] * CH)
    win = window(W, H, target(c), aspect, img); x0, y0, w, h = win; PLACED.clear()
    ax.imshow(img, extent=(0, W, H, 0), interpolation="lanczos", zorder=1); ax.set_xlim(x0, x0 + w); ax.set_ylim(y0 + h, y0)
    for g in c["ground_truth"]:
        b = g["box_px"]; ax.add_patch(Rectangle((b[0], b[1]), b[2], b[3], fill=False, ec=GT, lw=0.9, ls=(0, (3, 1.5)), zorder=4))
    for x in c["models"]["haiku"]["zero_shot"]["claims"]:
        if x["box_px"]: b = x["box_px"]; ax.add_patch(Rectangle((b[0], b[1]), b[2], b[3], fill=False, ec=ZS_D, lw=0.7, zorder=5)); tag(ax, b, x["type"], ZS_D, win)
    for x in c["models"]["haiku"]["rulesight"]["claims"]:
        if x["box_px"]: b = x["box_px"]; ax.add_patch(Rectangle((b[0], b[1]), b[2], b[3], fill=False, ec=RS, lw=0.9, zorder=6)); tag(ax, b, x["type"], RS, win)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color("#B9BEC6"); s.set_linewidth(0.5)
    rules = TASK_RULES[c["task"]]; rule_txt = "/".join(sorted(rules)); defective = c["label"] == 1
    fig.text(0.5, 1 - 0.025 / CH, f"{TASK_NAME[c['task']]}, {'defective' if defective else 'defect-free'}", ha="center", va="top", fontsize=FS_T, fontweight="bold", color=INK)
    def said(side):
        ts = sorted({x["type"] for x in c["models"]["haiku"][side]["claims"] if x.get("type")})
        if not defective: hit = sorted(set(ts) & rules); return ", ".join(hit) if hit else f"no {rule_txt}"
        return ", ".join(ts) if ts else "no defect"
    yy = IMG[1] - 0.05 / CH
    lines = [(f"Labeled  {rule_txt if defective else 'no ' + rule_txt}", GT), (f"Zero-shot  {said('zero_shot')}", ZS_D), (f"RuleSight  {said('rulesight')}", RS)]
    for k, (t, col) in enumerate(lines): fig.text(0.5, yy - k * 0.105 / CH, t, ha="center", va="top", fontsize=FS, color=col, fontweight="bold" if k == 0 else "normal")
    sq, gap = 0.092, 0.024   # model squares (in): six squares and five gaps end 0.1 in inside the cell
    for r, (side, lab) in enumerate((("zero_shot", "Zero-shot"), ("rulesight", "RuleSight"))):
        ycen = (yy - 3 * 0.105 / CH) - (0.07 + r * 0.13) / CH; x_start = 0.50 / PW
        fig.text(x_start - 0.04 / PW, ycen, lab, ha="right", va="center", fontsize=FS, color=INK)
        for k, m in enumerate(MODELS):
            v = c["models"].get(m, {}).get(side); col = GT if (v and v["correct"]) else GREY_L
            fig.patches.append(FancyBboxPatch((x_start + k * (sq + gap) / PW, ycen - sq / 2 / CH), sq / PW, sq / CH, boxstyle="round,pad=0,rounding_size=0.004", transform=fig.transFigure, fc=col, ec="none"))
    n0 = sum(1 for m in MODELS if c["models"].get(m) and c["models"][m]["zero_shot"]["correct"]); n1 = sum(1 for m in MODELS if c["models"].get(m) and c["models"][m]["rulesight"]["correct"])
    fig.text(0.5, 0.20 / CH, f"{n0} to {n1} of 6 models correct", ha="center", va="center", fontsize=FS, color=INK, fontweight="bold")
    fig.text(0.5, 0.012, f"({l})", ha="center", va="bottom", fontsize=FS_L, color=INK); save(fig, name)
    return n0, n1

def choose(cases):
    six = [c for c in cases if c["models_compared"] == 6]; n = lambda c, s: sum(1 for m in MODELS if c["models"].get(m) and c["models"][m][s] and c["models"][m][s]["correct"])
    def best(task, sub, k):
        pool = [c for c in six if (task is None or c["task"] == task) and c["sub"] == sub and (c["ground_truth"] or sub == "defect_free") and n(c, "rulesight") > n(c, "zero_shot")]
        return sorted(pool, key=lambda c: (-(n(c, "rulesight") - n(c, "zero_shot")), -n(c, "rulesight"), c["sid"]))[:k]
    return (best("Text Overflow", "defective", 4) + best("Abnormal Text Ellipsis", "defective", 4) + best("Undisplayed Content", "defective", 4),
            best("Cropped Content", "defective", 4) + best(None, "defect_free", 8))

def legend():
    sq = lambda col, lab: Line2D([], [], ls="none", marker="s", ms=4.2, mfc=col, mec="none", label=lab)
    hs = [Line2D([], [], color=GT, lw=0.9, ls=(0, (3, 1.5)), label="Labeled defect"), Line2D([], [], color=ZS_D, lw=0.9, label="Haiku zero-shot"), Line2D([], [], color=RS, lw=0.9, label="Haiku RuleSight"),
          sq(GT, "Model correct"), sq(GREY_L, "Model wrong")]
    fig = plt.figure(figsize=(TEXTW, 0.30)); fig.legend(handles=hs, loc="center", bbox_to_anchor=(0.5, 0.68), ncol=5, frameon=False, fontsize=FS, handlelength=1.6, handletextpad=0.4, columnspacing=1.6)
    fig.text(0.5, 0.18, "Squares from left to right  Claude Haiku 5.5, Claude Sonnet 5.5, GPT-5.4 mini, Gemini 3.8 Flash, Qwen3-VL-8B, InternVL3.5-8B", ha="center", va="center", fontsize=FS, color=INK)
    for nm in ("FigCaseA_legend", "FigCaseB_legend"): fig.savefig(os.path.join(OUT, nm + ".pdf")); fig.savefig(os.path.join(OUT, nm + ".png"), dpi=600, facecolor="white")
    plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--curated", required=True); a = ap.parse_args()
    cases = json.load(open(os.path.join(a.curated, "cases.json"), encoding="utf-8")); A, B = choose(cases); sel = {}
    for nm, lst in (("FigCaseA", A), ("FigCaseB", B)):
        sel[nm] = [(c["sid"], c["task"], c["sub"], *cell(c, a.curated, f"{nm}{'abcdefghijkl'[k]}", "abcdefghijkl"[k])) for k, c in enumerate(lst)]
    legend(); json.dump(sel, open(os.path.join(OUT, "case_selection.json"), "w"), indent=1); print({k: [f"{r[3]}->{r[4]}" for r in v] for k, v in sel.items()})
