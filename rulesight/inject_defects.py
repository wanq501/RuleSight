"""Inject display defects into real app screenshots (design §13.1, Rico-synth).

A Rico screen is a screenshot plus a view hierarchy whose nodes carry pixel bounds [x1, y1, x2, y2] in the 1440x2560 frame,
a class name and (for text nodes) the text. The eight display rules are injected the way Nighthawk and later work did,
so every injected defect comes with its exact box and type. Nothing here uses model output.

    python inject_defects.py --rico data/rico --out data/rico_synth --n 2000 [--seed 20261101]
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont

RICO_W, RICO_H = 1440, 2560


def flatten(node, acc, depth=0):
    if not isinstance(node, dict): return acc
    b = node.get("bounds")
    if isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
        acc.append({"bounds": [int(v) for v in b], "class": str(node.get("class") or ""), "text": node.get("text") or "", "label": node.get("componentLabel") or "",
                    "clickable": bool(node.get("clickable")), "depth": depth, "visible": node.get("visible-to-user", True), "n_children": len(node.get("children") or [])})
    for c in node.get("children") or []: flatten(c, acc, depth + 1)
    return acc


def load_hierarchy(path):
    try: j = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    except Exception: return []
    root = j.get("activity", {}).get("root", j) if isinstance(j, dict) else j
    return [n for n in flatten(root, []) if n["visible"]]


def pseudo_nodes(img):
    """screens whose view hierarchy is missing or empty: text runs and element boxes from the measurement tools, in Rico's 1440x2560 frame"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import measure as M
    W, H = img.size; fx, fy = RICO_W / W, RICO_H / H
    nodes = []
    for (x, y, w, h, *_rest) in M.text_like_boxes(img):
        nodes.append({"bounds": [int(x * fx), int(y * fy), int((x + w) * fx), int((y + h) * fy)], "class": "TextView", "text": "text", "label": "Text", "clickable": False, "depth": 1, "visible": True, "n_children": 0})
    for (x, y, w, h) in M.element_boxes(img):
        if w < W * 0.95:   # full-width bands are rows, not components
            nodes.append({"bounds": [int(x * fx), int(y * fy), int((x + w) * fx), int((y + h) * fy)], "class": "ImageView", "text": "", "label": "Image", "clickable": True, "depth": 1, "visible": True, "n_children": 0})
    return nodes


def scale_box(b, sx, sy):
    x1, y1, x2, y2 = b
    return [int(x1 * sx), int(y1 * sy), max(1, int((x2 - x1) * sx)), max(1, int((y2 - y1) * sy))]


def pick(nodes, pred, rng):
    c = [n for n in nodes if pred(n)]
    return rng.choice(c) if c else None


def is_text(n): return ((n["text"] or "").strip() != "" and "Text" in n["class"]) or n["label"] in ("Text", "Text Button")
def is_image(n): return "Image" in n["class"] or n["label"] in ("Image", "Icon", "Background Image")
def is_component(n): return n["n_children"] == 0 and (n["clickable"] or n["label"] in ("Text Button", "Icon", "Image", "Input", "Checkbox", "Toggle"))
def sane(box, W, H, min_w=20, min_h=12):
    x, y, w, h = box
    return 0 <= x < W and 0 <= y < H and w >= min_w and h >= min_h and x + w <= W + 2 and y + h <= H + 2


def bg_color(img, box):
    x, y, w, h = box
    px = img.crop((x, y, x + w, y + h)).resize((8, 8)).getdata()
    px = sorted(px, key=lambda p: sum(p[:3]))
    return px[-1][:3]   # the brightest sample: the usual background


def inject(img, nodes, kind, rng, font):
    """returns (new image, box [x,y,w,h]) or None when the screen offers no suitable element"""
    W, H = img.size; sx, sy = W / RICO_W, H / RICO_H
    boxes = [(n, scale_box(n["bounds"], sx, sy)) for n in nodes]
    texts = [(n, b) for n, b in boxes if is_text(n) and sane(b, W, H)]
    images = [(n, b) for n, b in boxes if is_image(n) and sane(b, W, H, 40, 40)]
    comps = [(n, b) for n, b in boxes if is_component(n) and sane(b, W, H, 40, 24)]
    out = img.copy(); d = ImageDraw.Draw(out)
    if kind == "D1" and len(texts) >= 2:   # text overlap: a text run pasted over another, shifted
        (n1, b1), (n2, b2) = rng.sample(texts, 2)
        crop = img.crop((b1[0], b1[1], b1[0] + b1[2], b1[1] + b1[3]))
        dx, dy = rng.randint(-b2[2] // 4, b2[2] // 4), rng.randint(-b2[3] // 2, b2[3] // 2)
        px, py = max(0, b2[0] + dx), max(0, b2[1] + dy)
        out.paste(crop, (px, py), _alpha(crop))
        ux, uy = min(px, b2[0]), min(py, b2[1]); ux2, uy2 = max(px + b1[2], b2[0] + b2[2]), max(py + b1[3], b2[1] + b2[3])
        return out, [ux, uy, min(W, ux2) - ux, min(H, uy2) - uy]
    if kind == "D2" and len(comps) >= 2:   # component occlusion: one component moved onto part of another
        (n1, b1), (n2, b2) = rng.sample(comps, 2)
        crop = img.crop((b1[0], b1[1], b1[0] + b1[2], b1[1] + b1[3]))
        px = b2[0] + rng.randint(0, max(0, b2[2] // 3)); py = b2[1] + rng.randint(0, max(0, b2[3] // 3))
        out.paste(crop, (px, py))
        return out, [px, py, min(W - px, b1[2]), min(H - py, b1[3])]
    if kind == "D3" and images:   # missing image: the image slot becomes a flat placeholder, sometimes with a broken-image mark
        n, b = rng.choice(images)
        col = rng.choice([(238, 238, 238), (245, 245, 245), (224, 224, 224), (255, 255, 255)])
        d.rectangle([b[0], b[1], b[0] + b[2], b[1] + b[3]], fill=col)
        if rng.random() < 0.5:
            cx, cy = b[0] + b[2] // 2, b[1] + b[3] // 2; s = max(6, min(b[2], b[3]) // 6)
            d.rectangle([cx - s, cy - s, cx + s, cy + s], outline=(160, 160, 160), width=2); d.line([cx - s, cy - s, cx + s, cy + s], fill=(160, 160, 160), width=2)
        return out, b
    if kind == "D4" and texts:   # null value: the text is replaced by a programming placeholder
        n, b = rng.choice(texts)
        col = bg_color(img, b); d.rectangle([b[0], b[1], b[0] + b[2], b[1] + b[3]], fill=col)
        word = rng.choice(["null", "undefined", "NaN", "{{title}}", "%s", "None", "[object Object]", "null null"])
        fnt = _fit_font(font, word, b[2], b[3])
        d.text((b[0] + 2, b[1] + max(0, (b[3] - fnt.size) // 2)), word, fill=_contrast_color(col), font=fnt)
        return out, b
    if kind == "D5":   # blurred or torn band
        lo = max(4, min(24, H // 20)); band = rng.randint(lo, max(lo, H // 6))   # small screenshots must not make an empty range
        y0 = rng.randint(H // 8, max(H // 8, H - H // 4 - band))
        region = out.crop((0, y0, W, y0 + band))
        if rng.random() < 0.5: region = region.filter(ImageFilter.GaussianBlur(radius=rng.uniform(4, 9)))
        else:
            shift = rng.randint(W // 12, W // 4); region = Image.fromarray(__import__("numpy").roll(__import__("numpy").asarray(region), shift, axis=1))
        out.paste(region, (0, y0))
        return out, [0, y0, W, band]
    if kind == "D6" and texts:   # text overflow: the text run pasted so that it crosses its container's right edge
        n, b = rng.choice(texts)
        crop = img.crop((b[0], b[1], b[0] + b[2], b[1] + b[3]))
        extra = rng.randint(b[2] // 3, b[2]); px = min(W - 10, b[0] + extra)
        out.paste(crop, (px, b[1]), _alpha(crop))
        return out, [px, b[1], min(W - px, b[2]), b[3]]
    if kind == "D7" and texts:   # abnormal ellipsis: a short label gets "..." though space is free
        n, b = rng.choice([t for t in texts if t[1][2] >= 60] or texts)
        cut = max(10, int(b[2] * rng.uniform(0.45, 0.7)))
        col = bg_color(img, b)
        d.rectangle([b[0] + cut, b[1], b[0] + b[2], b[1] + b[3]], fill=col)
        fnt = _fit_font(font, "...", 40, b[3]); d.text((b[0] + cut + 1, b[1] + max(0, (b[3] - fnt.size) // 2)), "...", fill=_contrast_color(col), font=fnt)
        return out, b
    if kind == "D8" and (images or comps):   # cropped content: part of an image or component slides past the screen edge
        n, b = rng.choice(images or comps)
        crop = img.crop((b[0], b[1], b[0] + b[2], b[1] + b[3]))
        col = bg_color(img, b); d.rectangle([b[0], b[1], b[0] + b[2], b[1] + b[3]], fill=col)
        px = W - b[2] // 2 if rng.random() < 0.5 else -(b[2] // 2)
        out.paste(crop, (px, b[1]))
        return out, [max(0, px), b[1], min(W, px + b[2]) - max(0, px), b[3]]
    return None


def _alpha(crop):
    """paste a text crop with its light background made transparent, so only the glyphs land"""
    g = crop.convert("L"); px = g.load(); w, h = g.size
    a = Image.new("L", (w, h), 0); ap = a.load()
    for yy in range(h):
        for xx in range(w):
            ap[xx, yy] = 255 if px[xx, yy] < 170 else 0
    return a


def _fit_font(font, word, w, h):
    size = max(10, min(int(h * 0.8), 60))
    try:
        f = ImageFont.truetype(font, size) if font else ImageFont.load_default()
        if font:
            while size > 10 and f.getlength(word) > w: size -= 2; f = ImageFont.truetype(font, size)
        return f
    except Exception:
        return ImageFont.load_default()


def _contrast_color(bg):
    return (30, 30, 30) if sum(bg) > 380 else (235, 235, 235)


def synthesize(rico_dir, out_dir, n, seed=20261101, kinds=("D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"), clean_ratio=0.35, font=None, exclude=(), on_progress=None):
    """Rico screens from rico_dir (<id>.jpg|png + <id>.json), n output screens: clean_ratio of them untouched (negatives), the rest with one injected defect each."""
    rico_dir, out_dir = Path(rico_dir), Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    exclude = {str(x) for x in exclude}   # test screens (UICrit) never become synthetic data: no leakage into exemplars or detector training
    screens = sorted(p for p in rico_dir.glob("*.json") if p.stem not in exclude and (p.with_suffix(".jpg").exists() or p.with_suffix(".png").exists()))
    rng.shuffle(screens)
    items = []
    for p in screens:
        if len(items) >= n: break
        if on_progress and len(items) % 25 == 0: on_progress(len(items), n)
        img_p = p.with_suffix(".jpg") if p.with_suffix(".jpg").exists() else p.with_suffix(".png")
        try:
            img = Image.open(img_p).convert("RGB"); nodes = load_hierarchy(p)
            if len(nodes) < 3: nodes = pseudo_nodes(img)
        except Exception:
            continue
        sid = p.stem
        if rng.random() < clean_ratio:
            out_p = out_dir / f"{sid}_clean.jpg"; img.save(out_p, quality=92)
            items.append({"sid": f"RS-{sid}-clean", "file": out_p.name, "label": 0, "types": [], "boxes": [], "rico_id": sid}); continue
        kind = rng.choice(kinds)
        def _try(kd):
            try: return inject(img, nodes, kd, rng, font)
            except Exception: return None   # one awkward screen never stops the whole set
        res = _try(kind)
        if res is None:   # the screen has nothing to inject this kind into: try the other kinds once
            for k2 in rng.sample(kinds, len(kinds)):
                res = _try(k2)
                if res: kind = k2; break
        if res is None: continue
        new, box = res
        out_p = out_dir / f"{sid}_{kind}.jpg"; new.save(out_p, quality=92)
        items.append({"sid": f"RS-{sid}-{kind}", "file": out_p.name, "label": 1, "types": [kind], "boxes": [box + [kind]], "rico_id": sid})
    (out_dir / "rico_synth.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return items


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rico", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=2000); ap.add_argument("--seed", type=int, default=20261101); ap.add_argument("--font", default=None)
    a = ap.parse_args()
    items = synthesize(a.rico, a.out, a.n, a.seed, font=a.font)
    from collections import Counter
    print(len(items), "screens;", dict(Counter(t for it in items for t in (it["types"] or ["clean"]))))
