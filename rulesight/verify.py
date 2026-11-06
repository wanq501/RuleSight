"""RuleSight verifier (design §3.3, §13.2).

A claimed defect is checked against the measurements:
  * ergonomic rules E1..E8 are measurable by construction: kept only when a measurement crosses the rule's threshold for this device;
  * display rules D1..D8: kept when a measured fact supports them (text boxes overlapping, a null-like or ellipsised OCR text, an element past
    the screen edge, ...), removed when the measurement contradicts them (OCR read the region and found no null value / no ellipsis; a missing
    image claimed on a detailed picture), and kept as "unverified" when the tools cannot see that kind of defect there — weak coverage must not
    delete true findings. D5 (tearing) rests on the model.
Removed claims go to "suspected" and leave the final report, which is what the metrics score.
"""
from __future__ import annotations
import json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
KB = json.loads((HERE / "rulekb" / "rules.json").read_text(encoding="utf-8"))
RULES = {r["id"]: r for r in KB["rules"]}
TYPES = list(RULES)


def _iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx)); iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy; union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _near(box, rows, kind=None, min_iou=0.1):
    best = None
    for r in rows:
        if kind and r["kind"] != kind: continue
        v = _iou(box, r["box"])
        if v >= min_iou and (best is None or v > best[0]): best = (v, r)
    return best[1] if best else None


def check_ergonomic(d, table):
    """returns (ok, evidence_dict) for E1..E8"""
    t, box, th, dev = d["type"], d.get("box"), RULES[d["type"]]["threshold"], table["device"]
    rows = table["rows"]
    if t == "E1":
        r = _near(box, rows, "text") if box else None
        if r and r.get("contrast") is not None:
            large = r["height_mm"] and (r["height_mm"] * 72 / 25.4) >= th["large_pt"]
            need = th["large_min"] if large else th["normal_min"]
            return (r["contrast"] < need, {"row": r["id"], "contrast": r["contrast"], "threshold": need})
        if box and table.get("image"):   # no row near the claimed box: measure the box itself
            try:
                import measure as M
                from PIL import Image
                cr = M.region_contrast(Image.open(table["image"]), box)
                if cr: return (cr[0] < th["normal_min"], {"row": "box", "contrast": cr[0], "threshold": th["normal_min"]})
            except Exception:
                pass
        return (False, {"reason": "no text row with contrast near the box"})
    if t == "E2":
        r = _near(box, rows, "element") if box else None
        return (False, {"reason": "boundary contrast not measured in v1"}) if r is None else (False, {"row": r["id"], "reason": "boundary contrast not measured in v1"})
    if t == "E3":
        r = _near(box, rows, "text") if box else None
        if r and r.get("xheight_arcmin") is not None:
            return (r["xheight_arcmin"] < th["min_arcmin"], {"row": r["id"], "xheight_arcmin": r["xheight_arcmin"], "threshold": th["min_arcmin"]})
        return (False, {"reason": "no text row near the box"})
    if t == "E4":
        r = _near(box, rows, "element") if box else None
        if r:
            mm_min = min(r["width_mm"], r["height_mm"])
            need_mm = th["material_dp"] * 25.4 / 160          # dp -> mm at 160 dpi reference
            touch = dev.get("ppi", 92) >= 200
            return (touch and mm_min < need_mm, {"row": r["id"], "min_mm": mm_min, "threshold_mm": round(need_mm, 2), "touch_device": touch})
        return (False, {"reason": "no element row near the box"})
    if t == "E5":
        ov = table.get("simulations", {}).get("horizontal_overflow_320")
        return (bool(ov), {"horizontal_overflow_320": ov})
    if t == "E6":
        r = _near(box, rows, "element") if box else None
        if r and r.get("nearest_gap_mm") is not None:
            need_mm = th["min_dp"] * 25.4 / 160
            return (r["nearest_gap_mm"] < need_mm and min(r["width_mm"], r["height_mm"]) < th["small_target_dp"] * 25.4 / 160, {"row": r["id"], "gap_mm": r["nearest_gap_mm"], "threshold_mm": round(need_mm, 2)})
        return (False, {"reason": "no element row near the box"})
    if t == "E7":
        return (table["device"].get("cvd", "none") != "none" and bool(table.get("simulations", {}).get("cvd_lost_pair")), {"cvd": table["device"].get("cvd")})
    if t == "E8":
        g = table.get("simulations", {}).get("glare_min_text_contrast")
        return (g is not None and g < th["min"], {"glare_min_text_contrast": g, "threshold": th["min"]})
    return (False, {"reason": "unknown rule"})


def _frac_inside(a, b):
    """share of box a that lies inside box b"""
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx)); iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return (ix * iy) / max(1, aw * ah)


def _region_texts(box, rows):
    """OCR text rows that belong to the claimed region: most of the text box lies in it, or the claim lies mostly in the text box
    (only rows that carry text, i.e. PaddleOCR ran); a neighbour touched at the edge does not count"""
    return [r for r in rows if r["kind"] == "text" and r.get("text") and (_frac_inside(r["box"], box) >= 0.5 or _frac_inside(box, r["box"]) >= 0.5)]


def _region_flat(table, box):
    """(std of grey levels, share of edge pixels) inside the box, or None when the image cannot be read"""
    try:
        import numpy as np
        from PIL import Image
        img = Image.open(table["image"]).convert("L")
        x, y, w, h = [int(v) for v in box[:4]]
        a = np.asarray(img.crop((max(0, x), max(0, y), min(img.width, x + w), min(img.height, y + h))), dtype=float)
        if a.size < 400: return None
        gx = np.abs(np.diff(a, axis=1)) > 18; gy = np.abs(np.diff(a, axis=0)) > 18
        return float(a.std()), float((gx.sum() + gy.sum()) / a.size)
    except Exception:
        return None


def check_display(d, table):
    """Display defects: kept when a measured fact supports them, removed when the measurement contradicts them,
    kept as "unverified" when the measurement cannot see that kind of defect in that region (so weak coverage never deletes true findings)."""
    t, box = d["type"], d.get("box")
    facts = [f for f in table["facts"] if f["rule"] == t]
    if t == "D5":
        return (True, {"facts": len(facts), "basis": "visual" if not facts else "row_correlation"})
    if box:
        for f in facts:
            boxes = f.get("boxes") or ([f["box"]] if f.get("box") else []) or ([f["band"]] if f.get("band") else [])
            if any(_iou(box, b) >= 0.1 for b in boxes):
                return (True, {"basis": "fact", "fact": f})
    elif facts:
        return (True, {"basis": "fact_without_box", "facts": len(facts)})
    if not box:
        return (True, {"basis": "unverified", "reason": "no box to check"})
    texts = _region_texts(box, table["rows"])
    if t == "D4" and texts:   # OCR read the text there: a null-like value must be among it
        import measure as M
        return (any(M.NULL_RE.search(r["text"]) for r in texts), {"basis": "ocr_text", "texts": [r["text"][:40] for r in texts[:4]]})
    if t == "D7" and texts:   # OCR read the text there: an ellipsis must end one of the runs
        return (any(r["text"].rstrip().endswith(("…", "...", "..")) for r in texts), {"basis": "ocr_text", "texts": [r["text"][:40] for r in texts[:4]]})
    if t == "D3":   # a missing image is a flat or placeholder area; a detailed picture there contradicts the claim
        fl = _region_flat(table, box)
        if fl is not None:
            std, edges = fl
            if std < 12 or edges < 0.01: return (True, {"basis": "flat_region", "std": round(std, 1), "edges": round(edges, 3)})
            if std > 40 and edges > 0.08: return (False, {"basis": "detailed_region", "std": round(std, 1), "edges": round(edges, 3)})
    return (True, {"basis": "unverified", "reason": "the measurement does not cover this kind of defect here"})


def verify(report, table):
    """report: the model's JSON; table: measure.measure() output. Returns the verified report."""
    kept, dropped = [], []
    for d in report.get("defects", []) or []:
        t = str(d.get("type", "")).upper()
        if t not in RULES:
            dropped.append(dict(d, why="unknown type")); continue
        d = dict(d, type=t)
        ok, ev = (check_ergonomic if t.startswith("E") else check_display)(d, table)
        d["verifier"] = ev
        (kept if ok else dropped).append(d)
    return {"defects": kept, "suspected": dropped, "screen_has_defect": bool(kept)}


def parse_report(text):
    """the model's JSON answer, tolerant of what models actually send: code fences, text before or after (B-CoT writes its analysis
    first), trailing commas, single quotes, and an answer cut off by the token limit (the complete defect objects are kept)"""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json|JSON)?\s*", "", t); t = re.sub(r"\s*```\s*$", "", t)
    starts = [m0.start() for m0 in re.finditer(r'\{\s*"defects"', t)]
    cands = [t[s0:] for s0 in reversed(starts)] or ([t[t.find("{"):]] if "{" in t else [])
    for c in cands:
        end = c.rfind("}")
        while end > 0:
            chunk = c[:end + 1]
            for variant in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk).replace("'", '"')):
                try:
                    obj = json.loads(variant)
                    if isinstance(obj, dict) and "defects" in obj: return obj
                except json.JSONDecodeError:
                    pass
            end = c.rfind("}", 0, end)
    # cut off: keep every complete defect object, and the screen flag if it made it
    defects = []
    for m0 in re.finditer(r'\{[^{}]*"type"\s*:\s*"[A-Za-z]\d+"[^{}]*\}', t):
        try: defects.append(json.loads(re.sub(r",\s*([}\]])", r"\1", m0.group(0))))
        except json.JSONDecodeError: pass
    if not defects and not re.search(r'"defects"\s*:\s*\[\s*\]', t):
        raise ValueError("no JSON object in the answer")
    flag = re.search(r'"screen_has_defect"\s*:\s*(true|false)', t)
    return {"defects": defects, "screen_has_defect": (flag.group(1) == "true") if flag else bool(defects), "salvaged": True}
