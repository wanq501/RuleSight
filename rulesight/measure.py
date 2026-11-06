"""RuleSight measurement tools (design §13.3).

Deterministic measurements for one screenshot: OCR text boxes, element boxes, WCAG contrast of text against its
background, physical conversions (px -> arcmin / mm from ppi and viewing distance), colour-vision and glare
simulation, and the geometric checks the display rules need. The output is the "measurement table" that goes into
the S3 prompt and that the verifier re-reads. Nothing here looks at labels.

    python measure.py --image shot.png --device phone [--ocr auto|off] --out shot.measure.json
"""
from __future__ import annotations
import argparse, json, math, os, re, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
KB = json.loads((HERE / "rulekb" / "rules.json").read_text(encoding="utf-8"))
DEVICES = KB["device_defaults"]
NULL_RE = re.compile("|".join(next(r for r in KB["rules"] if r["id"] == "D4")["threshold"]["patterns"]), re.I)


# ----------------------------------------------------------------------------- colour and physics
def srgb_to_lin(c):
    c = np.asarray(c, dtype=float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def luminance(rgb):
    r, g, b = srgb_to_lin(rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def contrast_ratio(fg, bg):
    l1, l2 = luminance(fg), luminance(bg)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)

def px_to_mm(px, ppi): return px * 25.4 / ppi

def px_to_arcmin(px, ppi, distance_mm):
    return math.degrees(2 * math.atan(px_to_mm(px, ppi) / (2 * distance_mm))) * 60

def simulate_cvd(img, kind="deuteranopia"):
    """Brettel-style projection (Machado 2009 matrices for severity 1.0)."""
    M = {"deuteranopia": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
         "protanopia": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]}.get(kind)
    if M is None: return img
    a = np.asarray(img.convert("RGB"), dtype=float)
    out = np.clip(a @ np.array(M).T, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

def simulate_glare(img, lux=10000, screen_nits=500, reflectance=0.05):
    """Veiling luminance added by ambient light: L_veil = E * rho / pi (cd/m2); the screen's black lifts, contrast shrinks."""
    if lux <= 1000: return img
    veil = lux * reflectance / math.pi                      # cd/m2 added uniformly
    a = srgb_to_lin(np.asarray(img.convert("RGB"), dtype=float))
    lin = (a * screen_nits + veil) / (screen_nits + veil)   # renormalise to the new white
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return Image.fromarray(np.clip(srgb * 255, 0, 255).astype(np.uint8))


# ----------------------------------------------------------------------------- boxes
_OCR = {"engine": None, "api": None, "tried": False, "said": False}
OCR_LANG = os.environ.get("RULESIGHT_OCR_LANG", "ch")   # PaddleOCR's Chinese model reads English too; UI-Lens is Chinese

def _ocr_engine():
    """one PaddleOCR instance for the whole run (loading it costs seconds); works with the 2.x API (ocr.ocr) and the 3.x API (ocr.predict)"""
    if _OCR["tried"]: return _OCR["engine"], _OCR["api"]
    _OCR["tried"] = True
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception as e:
        print(f"[measure] PaddleOCR 不可用（{str(e)[:80]}）：改用边缘估计文字框，没有文字内容", file=sys.stderr); return None, None
    v3 = {"lang": OCR_LANG, "use_doc_orientation_classify": False, "use_doc_unwarping": False, "use_textline_orientation": False}
    for kwargs, api in (({"use_angle_cls": False, "lang": OCR_LANG, "show_log": False}, "v2"),
                        (dict(v3, text_detection_model_name="PP-OCRv5_mobile_det", text_recognition_model_name="PP-OCRv5_mobile_rec"), "v3"),   # light models: fast on CPU, enough for rendered UI text
                        (v3, "v3"), ({"lang": OCR_LANG}, "auto")):
        try:
            eng = PaddleOCR(**kwargs)
            _OCR["engine"], _OCR["api"] = eng, api
            print(f"[measure] PaddleOCR 已加载（{api} 接口，lang={OCR_LANG}）", file=sys.stderr); return eng, api
        except Exception:
            continue
    print("[measure] PaddleOCR 加载失败：改用边缘估计文字框", file=sys.stderr); return None, None


def _ocr_parse(res):
    """(x, y, w, h, text, conf) from either API's answer"""
    out = []
    if not res: return out
    first = res[0] if isinstance(res, list) else res
    if isinstance(first, dict) or hasattr(first, "get") or hasattr(first, "json"):   # 3.x: a result object / dict with rec_texts, rec_scores, rec_polys or rec_boxes
        d = first.json.get("res", first.json) if hasattr(first, "json") and isinstance(getattr(first, "json"), dict) else first
        texts = list(d.get("rec_texts") or []); scores = list(d.get("rec_scores") or [])
        polys = d.get("rec_polys"); boxes = d.get("rec_boxes")
        geo = list(polys) if polys is not None and len(polys) else (list(boxes) if boxes is not None else [])
        for k, t in enumerate(texts):
            if k >= len(geo): break
            g = np.asarray(geo[k], dtype=float).reshape(-1)
            if g.size == 4: x1, y1, x2, y2 = g
            else: xs, ys = g[0::2], g[1::2]; x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            out.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), str(t), float(scores[k]) if k < len(scores) else 0.0))
        return out
    for line in (first or []):   # 2.x: [[pts, (text, conf)], ...]
        try:
            pts, (txt, conf) = line
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            out.append((int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys)), str(txt), float(conf)))
        except Exception:
            continue
    return out


def ocr_boxes(img, mode="auto"):
    """[(x, y, w, h, text, conf)] from PaddleOCR when installed, else a crude edge-based estimate (no text)."""
    if mode != "off":
        eng, api = _ocr_engine()
        if eng is not None:
            arr = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()   # PaddleOCR expects BGR like cv2
            try:
                res = eng.ocr(arr, cls=False) if api == "v2" else (eng.predict(arr) if hasattr(eng, "predict") else eng.ocr(arr))
                return _ocr_parse(res), "paddleocr"
            except Exception as e:
                if not _OCR["said"]:
                    _OCR["said"] = True; print(f"[measure] PaddleOCR 识别出错（{str(e)[:80]}），这张改用边缘估计", file=sys.stderr)
    return text_like_boxes(img), "fallback"

def text_like_boxes(img):
    """Rows of dark-on-light (or light-on-dark) strokes grouped into line boxes; a stand-in when no OCR is available."""
    g = np.asarray(img.convert("L"), dtype=float)
    edges = np.abs(np.gradient(g, axis=1)) + np.abs(np.gradient(g, axis=0))
    H, W = g.shape
    boxes = []
    for thr in (40, 12):   # strong strokes first, then faint text (low contrast is itself a defect to measure)
        rows = (edges > thr).sum(axis=1)
        y = 0
        while y < H:
            if rows[y] > W * 0.02:
                y0 = y
                while y < H and rows[y] > W * 0.01: y += 1
                if 6 <= y - y0 <= 60:
                    cols = (edges[y0:y, :] > thr).sum(axis=0)
                    xs = np.where(cols > 0)[0]
                    if len(xs):
                        cand = (int(xs.min()), y0, int(xs.max() - xs.min()), y - y0, "", 0.0)
                        if not any(box_ios(cand, b) > 0.5 for b in boxes): boxes.append(cand)
            y += 1
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes

def element_boxes(img):
    """Rectangular regions bounded by edges (cards, buttons, images): a cheap stand-in for DOM boxes."""
    e = img.convert("L").filter(ImageFilter.FIND_EDGES)
    a = np.asarray(e, dtype=float) > 60
    H, W = a.shape
    boxes = []
    # horizontal and vertical edge runs
    for y in range(H):
        row = a[y]
        if row.sum() > W * 0.3:
            boxes.append(("h", y))
    ys = [b[1] for b in boxes]
    rects = []
    for i in range(len(ys) - 1):
        if 24 <= ys[i + 1] - ys[i] <= H // 2:
            rects.append((0, ys[i], W, ys[i + 1] - ys[i]))
    return rects[:40]

def box_ios(a, b):
    """intersection over the smaller box"""
    ax, ay, aw, ah = a[:4]; bx, by, bw, bh = b[:4]
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx)); iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy; small = min(aw * ah, bw * bh) or 1
    return inter / small

def region_contrast(img, box, pad=2):
    """text colour vs background inside a box: the two dominant luminance clusters"""
    x, y, w, h = [int(v) for v in box[:4]]
    crop = np.asarray(img.convert("RGB").crop((max(0, x - pad), max(0, y - pad), x + w + pad, y + h + pad)), dtype=float).reshape(-1, 3)
    if len(crop) < 8: return None
    lum = luminance(crop.T)
    lo, hi = np.percentile(lum, 10), np.percentile(lum, 90)
    fg = crop[lum <= lo + 1e-6].mean(axis=0) if (lum <= lo + 1e-6).any() else crop[np.argmin(lum)]
    bg = crop[lum >= hi - 1e-6].mean(axis=0) if (lum >= hi - 1e-6).any() else crop[np.argmax(lum)]
    return round(float(contrast_ratio(fg, bg)), 2), [int(v) for v in fg], [int(v) for v in bg]

def row_correlation_dips(img, band_px=8, corr_min=0.3):
    g = np.asarray(img.convert("L"), dtype=float)
    dips, run, start = [], 0, 0
    for y in range(1, g.shape[0]):
        a, b = g[y - 1], g[y]
        if a.std() < 1 or b.std() < 1: c = 1.0
        else: c = float(np.corrcoef(a, b)[0, 1])
        if c < corr_min:
            if run == 0: start = y
            run += 1
        else:
            if run >= band_px: dips.append((start, run))
            run = 0
    return dips


# ----------------------------------------------------------------------------- the table
def measure(image_path, device="phone", ocr="auto", dom_boxes=None):
    img = Image.open(image_path)
    W, H = img.size
    dev = dict(DEVICES.get(device, DEVICES["phone"]))
    texts, ocr_src = ocr_boxes(img, ocr)
    elems = dom_boxes or element_boxes(img)
    rows = []
    # text rows: contrast, size, visual angle
    for i, t in enumerate(texts):
        x, y, w, h, txt, conf = t
        cr = region_contrast(img, t)
        xh = h * 0.5                                   # x-height ≈ half the line box
        rows.append({"kind": "text", "id": f"t{i}", "box": [x, y, w, h], "text": txt[:80], "ocr_conf": round(conf, 2),
                     "contrast": cr[0] if cr else None, "fg": cr[1] if cr else None, "bg": cr[2] if cr else None,
                     "height_mm": round(px_to_mm(h, dev["ppi"]), 2), "xheight_arcmin": round(px_to_arcmin(xh, dev["ppi"], dev["distance_mm"]), 1),
                     "null_value": bool(txt and NULL_RE.search(txt))})
    # element rows: physical size, spacing to the nearest neighbour
    for i, e in enumerate(elems):
        x, y, w, h = e[:4]
        near = None
        for j, f in enumerate(elems):
            if j == i: continue
            gap = max(f[0] - (x + w), x - (f[0] + f[2]), f[1] - (y + h), y - (f[1] + f[3]))
            if near is None or gap < near: near = gap
        rows.append({"kind": "element", "id": f"e{i}", "box": [int(x), int(y), int(w), int(h)], "width_mm": round(px_to_mm(w, dev["ppi"]), 2), "height_mm": round(px_to_mm(h, dev["ppi"]), 2),
                     "nearest_gap_mm": round(px_to_mm(near, dev["ppi"]), 2) if near is not None else None})
    # geometric facts for the display rules
    facts = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            ios = box_ios(texts[i], texts[j])
            if ios >= 0.15: facts.append({"rule": "D1", "ios": round(ios, 2), "boxes": [list(texts[i][:4]), list(texts[j][:4])]})
    for i in range(len(elems)):
        for j in range(i + 1, len(elems)):
            a, b = elems[i], elems[j]
            if box_ios(a, b) >= 0.2 and not (a[0] <= b[0] and a[1] <= b[1] and a[0] + a[2] >= b[0] + b[2] and a[1] + a[3] >= b[1] + b[3]) and not (b[0] <= a[0] and b[1] <= a[1] and b[0] + b[2] >= a[0] + a[2] and b[1] + b[3] >= a[1] + a[3]):
                facts.append({"rule": "D2", "ios": round(box_ios(a, b), 2), "boxes": [list(a[:4]), list(b[:4])]})
    for t in texts:
        x, y, w, h = t[:4]
        if x + w > W + 2 or x < -2: facts.append({"rule": "D6", "overflow_px": int(max(x + w - W, -x)), "box": list(t[:4])})
        if t[4] and t[4].rstrip().endswith(("…", "...")): facts.append({"rule": "D7", "ellipsis": True, "text": t[4][-40:], "box": list(t[:4])})
    for e in elems:   # D8 cropped content: an element box that runs past the screen
        ex, ey, ew, eh = [int(v) for v in e[:4]]
        over = max(ex + ew - W, ey + eh - H, -ex, -ey)
        if over >= 4: facts.append({"rule": "D8", "overflow_px": int(over), "box": [ex, ey, ew, eh]})
    for t in texts:
        if t[4] and NULL_RE.search(t[4]): facts.append({"rule": "D4", "text": t[4][:60], "box": list(t[:4])})
    for (start, run) in row_correlation_dips(img):
        facts.append({"rule": "D5", "band": [0, start, W, run]})
    # simulated views for E7 / E8 when the device asks for them
    sims = {}
    if dev.get("cvd") and dev["cvd"] != "none":
        sims["cvd"] = dev["cvd"]
    if dev.get("lux", 0) >= 10000:
        g = simulate_glare(img, dev["lux"])
        worst = None
        for t in texts:
            cr = region_contrast(g, t)
            if cr and (worst is None or cr[0] < worst): worst = cr[0]
        sims["glare_min_text_contrast"] = worst
    return {"image": str(image_path), "size": [W, H], "device": dev, "ocr": ocr_src, "rows": rows, "facts": facts, "simulations": sims,
            "thresholds": {r["id"]: r["threshold"] for r in KB["rules"]}}


def _sbox(b, k):
    return [int(round(v * k)) for v in b[:4]] if isinstance(b, (list, tuple)) and len(b) >= 4 else b


def _sfact(f, k):
    if k == 1: return f
    g = dict(f)
    for key in ("box", "band"):
        if key in g: g[key] = _sbox(g[key], k)
    if "boxes" in g: g["boxes"] = [_sbox(b, k) for b in g["boxes"]]
    return g


def table_text(m, max_rows=60, scale=1.0, sent=None):
    """compact text of the measurement table for the prompt; scale maps the original pixels to the pixels of the image the model is shown (sent = its size)"""
    k = float(scale or 1.0)
    W, H = sent if sent else (max(1, int(m['size'][0] * k)), max(1, int(m['size'][1] * k)))
    lines = [f"image {W}x{H} px (coordinates below are pixels of the image you see); device ppi {m['device']['ppi']}, distance {m['device']['distance_mm']} mm, lux {m['device'].get('lux')}, cvd {m['device'].get('cvd')}"]
    for r in m["rows"][:max_rows]:
        r = dict(r, box=_sbox(r["box"], k))
        if r["kind"] == "text":
            lines.append(f"{r['id']} text box {r['box']} contrast {r['contrast']} xheight {r['xheight_arcmin']} arcmin height {r['height_mm']} mm" + (f" text '{r['text']}'" if r['text'] else "") + (" NULL-PATTERN" if r["null_value"] else ""))
        else:
            lines.append(f"{r['id']} element box {r['box']} size {r['width_mm']}x{r['height_mm']} mm nearest gap {r['nearest_gap_mm']} mm")
    for f in m["facts"][:40]:
        lines.append("fact " + json.dumps(_sfact(f, k), ensure_ascii=False))
    if m["simulations"]:
        lines.append("simulations " + json.dumps(m["simulations"]))
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True); ap.add_argument("--device", default="phone"); ap.add_argument("--ocr", default="auto"); ap.add_argument("--out")
    a = ap.parse_args()
    m = measure(a.image, a.device, a.ocr)
    out = a.out or (str(Path(a.image).with_suffix("")) + ".measure.json")
    Path(out).write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    print(table_text(m))
