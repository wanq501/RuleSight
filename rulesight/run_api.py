"""RuleSight API runner (design §13.4–13.5).

    export OPENROUTER_API_KEY=...
    python run_api.py --manifest data/RuleSight_data.json --model anthropic/claude-haiku-5.5 --tag haiku \
        --settings s0,s3 --datasets test_real_400 --out runs/haiku.jsonl [--limit 20] [--dry-run]

One line per (sid, setting) in the output; re-running skips lines already present (resume). --dry-run uses a stub
answer so the whole pipeline (prompt → parse → verify → record) can be exercised without an API key.
"""
from __future__ import annotations
import argparse, base64, hashlib, io, json, os, random, re, sys, time
from pathlib import Path
import http.client, subprocess, urllib.error, urllib.request
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import measure as M          # noqa: E402
import verify as V           # noqa: E402

KB = V.KB
PROMPTS = {p.stem: (HERE / "prompts" / p.name).read_text(encoding="utf-8") for p in (HERE / "prompts").glob("*.txt")}
PIPELINE = "v2.1: tiles for every setting; S3 judges tool candidates with close-ups; in-domain exemplars; images as JPEG q92"   # part of the version: a change here retires earlier answers
PROMPT_VERSION = "pv3-" + hashlib.md5(("".join(PROMPTS[k] for k in sorted(PROMPTS)) + (HERE / "rulekb" / "rules.json").read_text(encoding="utf-8") + PIPELINE).encode("utf-8")).hexdigest()[:8]
UILENS = {p.stem: p.read_text(encoding="utf-8") for p in (HERE / "prompts_uilens").glob("*.txt")} if (HERE / "prompts_uilens").is_dir() else {}   # the per-category prompts of UI-Lens, kept outside PROMPT_VERSION
UILENS_CODE = {"Text Overflow": "TO", "Cropped Content": "CC", "Container Overlap": "CO", "Undisplayed Content": "UC", "Abnormal Text Ellipsis": "AE"}
PER_ROW = {"uilens"}   # settings whose prompt depends on the task row, so the rows of one page are not answered by one shared call
TYPE_LIST = "; ".join(f"{r['id']} {r['name']}" for r in KB["rules"])
DISPLAY_IDS = [r["id"] for r in KB["rules"] if r["id"].startswith("D")]
TYPE_LIST_DISPLAY = "; ".join(f"{r['id']} {r['name']}" for r in KB["rules"] if r["id"].startswith("D"))
DISPLAY_SETS = ("ui_lens", "glib", "rico_synth", "e_defect", "e_hard", "e_real")   # benchmarks whose labels are display defects only


def scope_of(item):
    ds = str(item.get("run_set") or item.get("dataset") or "")
    return "display" if any(ds == x or ds.startswith(x + "_") for x in DISPLAY_SETS) else "all"
PRICES_PER_M = {   # USD per million tokens, OpenRouter 2026-10 (input, output); update from the price page at run time
    "anthropic/claude-sonnet-5.5": (2.0, 10.0), "anthropic/claude-haiku-5.5": (0.10, 0.50),
    "openai/gpt-5.4-mini": (0.75, 4.50), "google/gemini-3.8-flash": (0.75, 3.75),
}
MAX_SIDE = 1568
VERIFIED = {"s3", "s3norule", "s3nomeas", "s3noex", "s3goldbox", "s3goldmeas"}         # settings that include the verifier
MEASURED = {"s3", "s3norule", "s3noex", "s3nover", "s3goldbox", "s3goldmeas"}          # settings whose prompt (and retrieval) see the measurement table


IMG_QUALITY = 92


def img_data_url(img):
    """the smaller of PNG and JPEG (quality 92) as a data URL: flat UI screens stay PNG, photo-rich screens become a fraction of the bytes"""
    p = io.BytesIO(); img.save(p, "PNG", optimize=True)
    j = io.BytesIO(); img.save(j, "JPEG", quality=IMG_QUALITY, optimize=True)
    data, mime = (p.getvalue(), "image/png") if len(p.getvalue()) <= len(j.getvalue()) else (j.getvalue(), "image/jpeg")
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def img_b64(path, max_side=MAX_SIDE, crop=None):
    """the image as sent (long side <= max_side) as a data URL, and its size; crop = [x, y, w, h] in original pixels"""
    img = Image.open(path).convert("RGB")
    if crop:
        x, y, w, h = [int(v) for v in crop[:4]]
        img = img.crop((max(0, x), max(0, y), min(img.width, x + w), min(img.height, y + h)))
    w, h = img.size
    s = max_side / max(w, h)
    if s < 1: img = img.resize((max(1, int(w * s)), max(1, int(h * s))))
    return img_data_url(img), img.size


UL_MAP = KB.get("category_map", {}).get("UI-Lens", {})


def tiles_b64(path, max_side=MAX_SIDE, overlap=60):
    """a tall screenshot that the model would otherwise see shrunk: its upper and lower halves at (near) full resolution, with the rows each covers"""
    img = Image.open(path).convert("RGB"); W, H = img.size
    if H <= max_side or H < 1.25 * W: return []
    out = []; half = H // 2
    for name, y0, y1 in (("upper half", 0, min(H, half + overlap)), ("lower half", max(0, half - overlap), H)):
        t = img.crop((0, y0, W, y1)); w, h = t.size; sc = min(1.0, max_side / max(w, h))
        if sc < 1: t = t.resize((max(1, int(w * sc)), max(1, int(h * sc))))
        out.append({"name": name, "b64": img_data_url(t), "rows": [y0, y1], "size": t.size})
    return out


def uniform_blocks(img_path, min_side=120, max_n=2):
    """large uniform blocks that differ from the page background: where content was expected but nothing is drawn (a D3 candidate)"""
    try:
        import numpy as np
        img = Image.open(img_path).convert("L"); W, H = img.size; a = np.asarray(img, dtype=float)
        cell = 40; bg = float(np.median(a)); hits = np.zeros((H // cell, W // cell), dtype=bool)
        for gy in range(H // cell):
            for gx in range(W // cell):
                c = a[gy * cell:(gy + 1) * cell, gx * cell:(gx + 1) * cell]
                if c.std() < 4 and abs(float(c.mean()) - bg) > 18: hits[gy, gx] = True
        seen = np.zeros_like(hits); blocks = []
        for gy in range(hits.shape[0]):
            for gx in range(hits.shape[1]):
                if not hits[gy, gx] or seen[gy, gx]: continue
                stack = [(gy, gx)]; cells = []
                while stack:
                    y, x = stack.pop()
                    if y < 0 or x < 0 or y >= hits.shape[0] or x >= hits.shape[1] or seen[y, x] or not hits[y, x]: continue
                    seen[y, x] = True; cells.append((y, x)); stack += [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]
                ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
                bw, bh = (max(xs) - min(xs) + 1) * cell, (max(ys) - min(ys) + 1) * cell
                if bw >= min_side and bh >= min_side and bw < 0.9 * W: blocks.append([min(xs) * cell, min(ys) * cell, bw, bh])
        blocks.sort(key=lambda b: -(b[2] * b[3]))
        return blocks[:max_n]
    except Exception:
        return []


def candidates(table, img_path, scope="display", max_n=6):
    """the regions the measurements flag, as numbered candidates for the model to judge: ellipsis (D7), text past the screen (D6),
    text runs that intersect (D1), placeholder values (D4), large uniform blocks (D3), elements past their parent (D8), element overlaps (D2)"""
    out = []
    pri = {"D7": 0, "D6": 1, "D1": 2, "D4": 3, "D3": 4, "D8": 5, "D2": 6}
    for f in table.get("facts") or []:
        r = f.get("rule")
        if r not in pri or (scope == "display" and not r.startswith("D")): continue
        box = f.get("box") or (f.get("boxes") or [None])[0]
        if not box: continue
        if r == "D7": fact = f"text '{str(f.get('text') or '')[-30:]}' ends with an ellipsis"
        elif r == "D6": fact = f"a text box runs {f.get('overflow_px')} px past the screen edge"
        elif r == "D1": fact = f"two text boxes intersect (IoS {f.get('ios')}): {f.get('boxes')}"
        elif r == "D4": fact = f"text '{str(f.get('text') or '')[:40]}' looks like a programming placeholder"
        elif r == "D8": fact = f"an element extends {f.get('overflow_px')} px beyond its parent or the screen"
        else: fact = f"two elements overlap (IoS {f.get('ios')})"
        out.append({"rule": r, "box": [int(v) for v in box[:4]], "fact": fact, "pri": pri[r] + (0 if r != "D7" else -min(0.5, (box[3] or 0) / 200))})
    for b in uniform_blocks(img_path):
        out.append({"rule": "D3", "box": b, "fact": f"a uniform block of {b[2]}x{b[3]} px that differs from the page background: no content is drawn there", "pri": pri["D3"]})
    out.sort(key=lambda c: c["pri"]); out = out[:max_n]
    for i, c in enumerate(out): c["id"] = f"C{i + 1}"; c.pop("pri", None)
    return out


def exemplar_pool(man):
    """exemplars for the prompt: the manifest's own (Rico-synth / ErgoHMI) plus real ones from the UI-Lens Dev split (expert boxes), never a test page"""
    pool = list(man.get("exemplars") or [])
    for s_ in man.get("screens") or []:
        if s_.get("dataset") != "ui_lens_dev": continue
        page = s_.get("md5") or s_.get("file")
        if s_.get("label") and s_.get("boxes") and s_.get("ui_lens_task") in UL_MAP:
            for b in s_["boxes"][:2]:
                if isinstance(b, (list, tuple)) and len(b) >= 4 and b[2] > 0 and b[3] > 0:
                    pool.append({"file": s_["file"], "box": [int(v) for v in b[:4]], "rule": UL_MAP[s_["ui_lens_task"]][0], "source": "ui_lens_dev", "page": page})
        elif s_.get("label") == 0:
            pool.append({"file": s_["file"], "box": None, "rule": None, "source": "ui_lens_dev", "page": page})
    return pool


def exemplar_crop(box, W, H, margin=1.0, min_side=256):
    """a region around an exemplar's defect: the box grown by its own size on every side, at least min_side square, inside the image"""
    x, y, w, h = [int(v) for v in box[:4]]
    cw, ch = max(min_side, int(w * (1 + 2 * margin))), max(min_side, int(h * (1 + 2 * margin)))
    cx, cy = x + w // 2, y + h // 2
    x0, y0 = max(0, min(W - cw, cx - cw // 2)), max(0, min(H - ch, cy - ch // 2))
    return [x0, y0, min(cw, W), min(ch, H)]


def rule_cards_text(rules, ids=None):
    out = []
    for r in rules:
        if ids and r["id"] not in ids: continue
        out.append(f"[{r['id']}] {r['name']}: {r['definition']} Decision: {r['decision']} (source: {r['source']})")
    return "\n".join(out)


def retrieve(table, k_rules=6, k_ex=3, exemplar_index=None, sid=None, use_facts=True, scope="all", prefer_rules=None, in_domain=False, page=None):
    """the rule cards: every display rule (D1..D8, short, and the core taxonomy), plus for benchmarks beyond display defects the three ergonomic
    rules that the screenshot's profile and measured facts point to; exemplars from Rico-synth / ErgoHMI only"""
    n_text = sum(1 for r in table["rows"] if r["kind"] == "text"); n_el = sum(1 for r in table["rows"] if r["kind"] == "element")
    touch = table["device"].get("ppi", 92) >= 200
    score = {rid: 0 for rid in V.RULES}
    for f in (table["facts"] if use_facts else []): score[f["rule"]] += 3   # a measured fact names its rule
    if n_text > 8: score["D1"] += 1; score["D6"] += 1; score["E1"] += 1; score["E3"] += 1
    if n_el > 4: score["D2"] += 1; score["E6"] += 1
    if touch: score["E4"] += 1; score["E6"] += 1
    if table["device"].get("lux", 0) >= 10000: score["E8"] += 3
    if table["device"].get("cvd", "none") != "none": score["E7"] += 3
    ergo = [rid for rid, _ in sorted(((r, v) for r, v in score.items() if r.startswith("E")), key=lambda kv: -kv[1])[:3]]
    ids = list(DISPLAY_IDS) + ([] if scope == "display" else ergo)
    ex = []
    if exemplar_index:   # never a test page: UI-Lens Dev crops for UI-Lens targets (not the target's own page), else Rico-synth / ErgoHMI
        rng = random.Random(hashlib.md5(str(sid).encode()).hexdigest())
        src = ("ui_lens_dev",) if in_domain and any(e.get("source") == "ui_lens_dev" for e in exemplar_index) else ("rico_synth", "s_night", "ergohmi")
        pos = [e for e in exemplar_index if e.get("rule") in ids and e.get("source") in src and (page is None or e.get("page") != page)]
        neg = [e for e in exemplar_index if not e.get("rule") and e.get("source") in src and (page is None or e.get("page") != page)]
        rng.shuffle(pos); rng.shuffle(neg)
        want_first = [r for r in (prefer_rules or []) if r in ids]
        pos.sort(key=lambda e: 0 if e["rule"] in want_first else 1)   # the rules the candidates point to come first
        seen, picked = set(), []
        for e in pos:   # different rules first
            if e["rule"] not in seen and len(picked) < max(0, k_ex - 1): picked.append(e); seen.add(e["rule"])
        for e in pos:
            if len(picked) >= max(0, k_ex - 1): break
            if e not in picked: picked.append(e)
        ex = picked + neg[:max(0, k_ex - len(picked))]
    return ids, ex


def build_messages(setting, item, table, data_dir, exemplar_index):
    dev = table["device"]
    device = f"ppi {dev['ppi']}, viewing distance {dev['distance_mm']} mm, ambient {dev.get('lux')} lx, colour vision {dev.get('cvd')}"
    sc = scope_of(item)
    img_path = Path(data_dir) / item["file"]
    cands = candidates(table, img_path, sc) if setting in MEASURED else []
    page_id = item.get("md5") or item.get("file")
    base = {"device": device, "type_list": TYPE_LIST_DISPLAY if sc == "display" else TYPE_LIST}
    ids, ex = retrieve(table, exemplar_index=exemplar_index, sid=item["sid"], use_facts=setting in MEASURED, scope=sc, prefer_rules=[c["rule"] for c in cands],
                       in_domain=str(item.get("run_set") or item.get("dataset") or "").startswith("ui_lens"), page=page_id)
    cards = rule_cards_text(KB["rules"], ids if setting in ("s2", "s3", "s3nomeas", "s3noex", "s3nover", "s3goldbox", "s3goldmeas") else None)
    W0, H0 = table["size"]
    b64, size = img_b64(Path(data_dir) / item["file"])   # the screenshot as the model sees it; its size defines the frame of every coordinate
    k = size[0] / max(1, W0)                              # original pixels -> pixels of the image the model sees
    meas = M.table_text(table, max_rows=40, scale=k, sent=size)
    cand_text = "\n".join(f"{c['id']} rule {c['rule']} box {[int(round(v * k)) for v in c['box']]}: {c['fact']}" for c in cands) or "(none flagged)"
    base["candidates"] = cand_text
    if setting == "s3goldmeas" and item.get("measured"):
        meas += "\nGROUND-TRUTH MEASUREMENTS: " + json.dumps(item["measured"])
    if setting in ("s3goldbox", "bzoom") and item.get("boxes"):
        meas += "\nREGION OF INTEREST (given, pixels of the image you see): " + json.dumps([[int(round(v * k)) for v in b[:4]] for b in item["boxes"]])
    key = {"uilens": "s0", "s0": "s0", "bcot": "bcot", "bzoom": "s0", "s1": "s1", "s2": "s2", "s3": "s3", "s3norule": "s1", "s3nomeas": "s2", "s3noex": "s3", "s3nover": "s3", "s3goldbox": "s3", "s3goldmeas": "s3"}[setting]
    text = PROMPTS[key].format(**base, rule_cards=cards, measurements=meas)
    if setting == "uilens":   # UI-Lens prompting: the published prompt of the row's category, answered in the JSON format of the system prompt
        task = item.get("ui_lens_task") or ""; ids = KB.get("category_map", {}).get("UI-Lens", {}).get(task, [])
        text = UILENS.get(UILENS_CODE.get(task, ""), "") + f"\n\nDevice: {device}.\nReport every {task} issue you find as one defect in the JSON format of the system prompt, with \"type\" and \"rule\" set to " + " or ".join(ids) + ", the box of the issue, one sentence of evidence and a confidence. If there is no such issue, return no defects."
    if setting == "s3norule":   # exemplars + measurements, no rule cards
        text = PROMPTS["s1"].format(**base) + "\nMeasurement table:\n" + meas
    text += f"\nReport every box as [x, y, w, h] in pixels of the screenshot you see ({size[0]}x{size[1]})."
    content = [{"type": "text", "text": text}]
    content.append({"type": "image_url", "image_url": {"url": b64}})
    for t in tiles_b64(img_path):   # the same screenshot at higher resolution, every setting alike
        y0, y1 = t["rows"]
        content.append({"type": "text", "text": f"Tile: the {t['name']} of the same screenshot at higher resolution (rows {int(round(y0 * k))}-{int(round(y1 * k))} of the first image). Report boxes in the first image's pixels."})
        content.append({"type": "image_url", "image_url": {"url": t['b64']}})
    for c in cands:   # a close look at each flagged region
        with Image.open(img_path) as im0: EW, EH = im0.size
        cb, _ = img_b64(img_path, 512, crop=exemplar_crop(c["box"], EW, EH, margin=0.6, min_side=200))
        content.append({"type": "text", "text": f"Candidate {c['id']} (rule {c['rule']}): a close-up of the flagged region."})
        content.append({"type": "image_url", "image_url": {"url": cb}})
    if setting in ("bzoom",) and item.get("boxes"):
        x, y, w, h = [int(v) for v in item["boxes"][0][:4]]
        img = Image.open(Path(data_dir) / item["file"]).convert("RGB")
        crop = img.crop((max(0, x - 20), max(0, y - 20), min(img.width, x + w + 20), min(img.height, y + h + 20)))
        content.append({"type": "text", "text": "Zoomed region:"}); content.append({"type": "image_url", "image_url": {"url": img_data_url(crop)}})
    if setting in ("s1", "s2", "s3", "s3norule", "s3nomeas", "s3nover", "s3goldbox", "s3goldmeas"):
        for e in ex:
            try:
                ep = Path(data_dir) / e["file"]
                if e.get("box"):   # the region around the defect, so the exemplar shows what the rule looks like
                    with Image.open(ep) as im0: EW, EH = im0.size
                    eb, _ = img_b64(ep, 512, crop=exemplar_crop(e["box"], EW, EH))
                    rule = next((r for r in KB["rules"] if r["id"] == e.get("rule")), {})
                    cap = f"Exemplar of {e.get('rule')} ({rule.get('name', '')}): the defect is in the middle of this crop."
                else:
                    eb, _ = img_b64(ep, 512); cap = "Exemplar of a screen without display defects."
                content.append({"type": "text", "text": cap})
                content.append({"type": "image_url", "image_url": {"url": eb}})
            except Exception:
                pass
    return [{"role": "system", "content": PROMPTS["system"]}, {"role": "user", "content": content}], ids, size


_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # ignores http_proxy / https_proxy from the environment


API_PROXY_PORT = int(os.environ.get("RS_API_PROXY_PORT", "17890"))   # the reverse tunnel from AutoDL to the proxy on the Windows PC (Clash)
_API_OPENER = [None]


def api_opener(base_url):
    """OpenRouter's providers refuse requests from mainland China (HTTP 403). The requests therefore go through the Windows PC's proxy, which
    Atelier's SSH tunnel makes reachable on AutoDL at 127.0.0.1:17890 (-R). Local endpoints (vLLM) are reached directly."""
    if "127.0.0.1" in base_url or "localhost" in base_url: return _DIRECT, "local"
    if _API_OPENER[0] is None:
        via = os.environ.get("RS_API_PROXY") or None   # the runner on the Windows PC passes the PC's proxy (Clash)
        route = "pc-proxy" if via else None
        if not via and os.environ.get("RS_ON_AUTODL"):
            import socket
            try:
                with socket.create_connection(("127.0.0.1", API_PROXY_PORT), timeout=3): via, route = f"http://127.0.0.1:{API_PROXY_PORT}", "windows-proxy"
            except OSError:
                via = None
        if via: _API_OPENER[0] = (urllib.request.build_opener(urllib.request.ProxyHandler({"http": via, "https": via})), route)
        else: _API_OPENER[0] = (urllib.request.build_opener(), "system")   # the environment's / the system's proxy settings
        try:
            sp = urllib.request.getproxies(); shown = via or sp.get("https") or sp.get("http") or "直连"
            print(f"模型请求线路：{'指定代理 ' + via if via else ('系统代理 ' + shown if shown != '直连' else '直连（系统未设代理）')}", flush=True)
        except Exception:
            pass
    return _API_OPENER[0]


def http_error_text(e):
    """the provider's own reason, not just 'HTTP Error 403: Forbidden'"""
    try:
        body = e.read().decode("utf-8", "replace")[:600]
        try: msg = (json.loads(body).get("error") or {}).get("message") or body
        except Exception: msg = body
        return f"HTTP {e.code}: {str(msg).strip()[:300]}"
    except Exception:
        return f"HTTP {getattr(e, 'code', '?')}"


def call_openrouter(model, messages, reasoning="low", timeout=180, base_url=os.environ.get("RS_OPENROUTER_BASE", "https://openrouter.ai/api/v1"), key_env="OPENROUTER_API_KEY"):
    """OpenRouter by default; any OpenAI-compatible endpoint with --base-url (vLLM for the local Qwen3-VL-8B: http://127.0.0.1:8000/v1)"""
    key = os.environ.get(key_env, "") or ("none" if "127.0.0.1" in base_url or "localhost" in base_url else "")
    if not key: raise RuntimeError(f"{key_env} is not set")
    body = {"model": model, "messages": messages, "temperature": 0, "top_p": 1, "max_tokens": 3000}
    if "openrouter" in base_url:
        if reasoning == "off": body["reasoning"] = {"enabled": False}
        elif reasoning: body["reasoning"] = {"effort": reasoning}
    elif reasoning == "off":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "HTTP-Referer": "https://atelier.local", "X-Title": "RuleSight"})
    t0 = time.time()
    opener, route = api_opener(base_url)
    data = None; last = None
    for attempt in range(4):   # a dropped connection, a timeout, a 429 or a 5xx is tried again (2, 4, 8 s); everything else is final
        try:
            with opener.open(req, timeout=timeout) as r:
                data = json.loads(r.read().decode())
            break
        except urllib.error.HTTPError as e:
            msg = http_error_text(e)
            if e.code == 403 and route == "direct": msg += "（AutoDL 直连 OpenRouter 被拒：模型方不服务中国大陆 IP。请开着 Windows 上的 Clash 再双击 AutoDL.bat，模型请求会经它转发）"
            if e.code in (408, 429) or e.code >= 500: last = RuntimeError(msg); time.sleep(2 * (attempt + 1) + random.random()); continue
            raise RuntimeError(msg)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, http.client.HTTPException, json.JSONDecodeError) as e:   # SSL EOF, timeouts, a reply cut short (IncompleteRead), a garbled body
            last = RuntimeError(f"网络：{str(e)[:160]}"); time.sleep(2 * (attempt + 1) + random.random()); continue
    if data is None: raise last or RuntimeError("no answer")
    lat = time.time() - t0
    text = data["choices"][0]["message"]["content"] or ""
    if isinstance(text, list): text = "".join(c.get("text", "") for c in text)
    usage = data.get("usage", {})
    return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), lat, data.get("model", model)


def measure_cached(data_dir, it, ocr):
    """the measurement table of a page under a device, computed once and kept next to the data (PaddleOCR is the slow part)"""
    key = f"{it.get('md5') or hashlib.md5(it['file'].encode()).hexdigest()}_{it.get('device_key', 'phone')}_{ocr}"
    cache = Path(data_dir) / "measure_cache" / f"{key}.json"
    if cache.exists():
        try:
            t = json.loads(cache.read_text(encoding="utf-8")); t["image"] = str(Path(data_dir) / it["file"]); return t
        except Exception:
            pass
    if os.environ.get("RS_REQUIRE_MEASURE_CACHE"): raise RuntimeError(f"这一页还没有测量表（{key}）：先在 AutoDL 上跑“测量”，再同步到本机")
    t = M.measure(Path(data_dir) / it["file"], it.get("device_key", "phone"), ocr)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True); cache.write_text(json.dumps(t, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return t


def stub_answer(item, table, k=1.0):
    """dry-run: claim the facts the table already shows, in the pixels of the image a model would see (so parse → scale back → verify → score are exercised)"""
    sb = lambda b: [int(round(v * k)) for v in b[:4]]
    defects = []
    for f in table["facts"][:3]:
        box = (f.get("boxes") or [f.get("box") or f.get("band") or [0, 0, 10, 10]])[0]
        defects.append({"type": f["rule"], "box": sb(box), "rule": f["rule"], "evidence": "table fact", "confidence": 0.8})
    for r in table["rows"]:
        if r["kind"] == "text" and r.get("contrast") is not None and r["contrast"] < 4.5:
            defects.append({"type": "E1", "box": sb(r["box"]), "rule": "E1", "evidence": r["id"], "confidence": 0.7}); break
    if not defects and item.get("boxes"):   # an unsupported guess, so the verifier has something to remove in S3 and to keep in S3-nover
        b = item["boxes"][0]; defects.append({"type": "D2", "box": sb(b), "rule": "D2", "evidence": "looks occluded", "confidence": 0.4})
    return json.dumps({"defects": defects, "screen_has_defect": bool(defects)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--settings", default="s0,s3"); ap.add_argument("--datasets", default="test_real_400"); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("RS_WORKERS", "4")), help="pages sent at once (each page's settings stay in order)")
    ap.add_argument("--limit", type=int, default=0, help="number of pages (all rows of a page are kept together)"); ap.add_argument("--reasoning", default="low"); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--ocr", default="auto")
    ap.add_argument("--base-url", default=os.environ.get("RS_OPENROUTER_BASE", "https://openrouter.ai/api/v1"), help="OpenAI-compatible endpoint; vLLM for the local model"); ap.add_argument("--key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--run-key", default="", help="the step this leg belongs to (several steps may run at once)")
    ap.add_argument("--leg", type=int, default=-1, help="index of this run in the step's plan (progress.json run.legs)"); ap.add_argument("--plan", default="", help="print the planned rows / calls as JSON and exit")
    ap.add_argument("--ledger", default="", help="JSON file with the spend of every run so far (budget guard)"); ap.add_argument("--cap", type=float, default=0, help="stop before the ledger total passes this many USD")
    a = ap.parse_args()
    man = json.loads(Path(a.manifest).read_text(encoding="utf-8")); data_dir = Path(a.manifest).parent
    wanted = [d.strip() for d in a.datasets.split(",") if d.strip()]
    items = []
    for s in man["screens"]:   # a dataset by name, or a subset (ui_lens_400) that items carry in "subsets"
        hit = s.get("dataset") if s.get("dataset") in wanted else next((x for x in (s.get("subsets") or []) if x in wanted), None)
        if hit: items.append(dict(s, run_set=hit))
    # one page's rows next to each other (UI-Lens has five task rows per page), so a page is answered once and --limit counts whole pages
    order = {}
    for k, s in enumerate(items): order.setdefault(s.get("md5") or s["file"], k)
    items.sort(key=lambda s: (order[s.get("md5") or s["file"]], s["sid"]))
    if a.limit:
        keep, pages = [], []
        for s in items:
            pk = s.get("md5") or s["file"]
            if pk not in pages:
                if len(pages) >= a.limit: break
                pages.append(pk)
            keep.append(s)
        items = keep
    if a.plan:   # rows and model calls this leg will make (a UI-Lens page answers all its task rows with one call per setting)
        n_set = len([x for x in a.settings.split(",") if x.strip()])
        pages = {(s.get("md5") or s["file"], s.get("device_key", "phone"), json.dumps(s.get("measured") or {}, sort_keys=True)) for s in items}
        calls = sum(len(items) if x.strip() in PER_ROW else len(pages) for x in a.settings.split(",") if x.strip())   # UI-Lens prompting asks once per task row
        print(json.dumps({"rows": len(items) * n_set, "calls": calls, "pages": len(pages)})); return
    if not items: print(f"没有截图属于 {wanted}（manifest 里的数据集：{sorted({s.get('dataset') for s in man['screens']})}）"); return
    exemplars = exemplar_pool(man)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and str(out) != os.devnull:   # answers made with other prompts or rule cards are kept aside, not mixed in or reused
        try:
            lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        except Exception:
            lines = []
        old = [l for l in lines if l.get("pv") != PROMPT_VERSION]
        if old:
            keep = [l for l in lines if l.get("pv") == PROMPT_VERSION]
            arch = out.with_name(out.stem + f".old-{(old[0].get('pv') or 'pv1')}.jsonl")
            with open(arch, "a", encoding="utf-8") as fa:
                for l in old: fa.write(json.dumps(l, ensure_ascii=False) + "\n")
            out.write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in keep), encoding="utf-8")
            print(f"提示词或规则卡已更新（{PROMPT_VERSION}）：{len(old)} 条旧记录移到 {arch.name}，这些行重新跑")
    done, prior = set(), {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                j = json.loads(line)
                if "error" not in j or re.search(r"prohibited content|content policy|moderation|flagged", str(j.get("error")), re.I):
                    done.add((j["sid"], j["setting"]))   # a failed row is tried again on the next run; a content refusal would only be refused again
                if j.get("parsed") and not j.get("shared_call"): prior[(j["sid"], j["setting"])] = j
            except Exception: pass
    price = PRICES_PER_M.get(a.model, (1.0, 4.0))
    ledger_p = Path(a.ledger) if a.ledger else None
    LEDGER_LAST = [None]   # the last ledger read in full: a read that fails mid-write must never reset the total
    def ledger_read():
        if not ledger_p: return {"total_usd": 0.0, "by_tag": {}}
        for k in range(6):
            try:
                if not ledger_p.exists():
                    if LEDGER_LAST[0] is not None: time.sleep(0.05); continue   # being replaced this instant
                    return {"total_usd": 0.0, "by_tag": {}}
                L = json.loads(ledger_p.read_text(encoding="utf-8"))
                if LEDGER_LAST[0] is not None and L.get("total_usd", 0) < LEDGER_LAST[0].get("total_usd", 0): L = LEDGER_LAST[0]   # never go backwards
                LEDGER_LAST[0] = L; return L
            except Exception:
                time.sleep(0.05 * (k + 1))
        return dict(LEDGER_LAST[0]) if LEDGER_LAST[0] is not None else {"total_usd": 0.0, "by_tag": {}}
    import threading
    LOCK = threading.Lock(); MEASURE_LOCK = threading.Lock(); LEDGER_WARNED = []
    def ledger_add(usd):
        if not ledger_p or usd <= 0: return
        with LOCK: _ledger_add(usd)
    def _ledger_add(usd):
        try:
            L = ledger_read(); L["total_usd"] = round(L.get("total_usd", 0) + usd, 6); L.setdefault("by_tag", {})[a.tag] = round(L["by_tag"].get(a.tag, 0) + usd, 6)
            LEDGER_LAST[0] = L
            ledger_p.parent.mkdir(parents=True, exist_ok=True); tmp = ledger_p.with_suffix(".tmp"); tmp.write_text(json.dumps(L), encoding="utf-8")
            for k in range(12):   # on Windows the rename is refused while the status reader has the file open: wait a little and try again
                try: tmp.replace(ledger_p); return
                except PermissionError: time.sleep(0.05 * (k + 1))
            ledger_p.write_text(json.dumps(L), encoding="utf-8")
        except Exception as e:
            if not LEDGER_WARNED: LEDGER_WARNED.append(1); print(f"账本没写上（不影响记录）：{str(e)[:120]}")
    C = {"new": 0, "spent": 0.0, "calls": 0, "shared": 0}
    import progress as P   # the step's plan in progress.json: this leg's rows, calls, spend, last activity
    if a.leg >= 0:
        try: P.init(Path(a.out).parent.parent)
        except Exception: pass
    n_set = len([x for x in a.settings.split(",") if x.strip()])
    rows_total = len(items) * n_set; rows_done0 = sum(1 for it in items for st in a.settings.split(",") if (it["sid"], st) in done)
    last_rep = [0.0]; t_start = time.time()
    last_line = [time.time()]; HB = [lambda: None]   # the lock heartbeat, set once the lock is taken below
    def report(force=False, **extra):
        if time.time() - last_line[0] > 60:   # one line a minute in the log: rows, calls, spend; and the lock's heartbeat
            last_line[0] = time.time()
            try: HB[0]()
            except Exception: pass
            print(f"进度 {a.tag} {a.settings}: {rows_done0 + C['new']:,} / {rows_total:,} 行 · {C['calls']} 次调用 · 已花 ${C['spent']:.3f} · 失败 {n_err[0]}", flush=True)
        if a.leg < 0 or (not force and time.time() - last_rep[0] < 2): return
        last_rep[0] = time.time(); r_done = rows_done0 + C["new"]; el = time.time() - t_start
        rate = C["new"] / el if el > 5 and C["new"] else None
        try: P.run_leg(a.leg, key=a.run_key or None, state=extra.pop("state", "running"), rows_done=r_done, rows_total=rows_total, calls_done=C["calls"], spent=round(C["spent"], 4), errors=n_err[0],
                       eta_s=int((rows_total - r_done) / rate) if rate else None, **extra)
        except Exception: pass
    n_err = [0]; first_err = []
    page_cache = {}   # UI-Lens has one row per (page, task): the page is sent once per setting, its defect list serves all five tasks
    by_sid = {s["sid"]: s for s in man["screens"]}
    for (sid, st), j in prior.items():   # answers from an earlier run of this file serve the remaining rows of the same page
        it0 = by_sid.get(sid)
        if it0: page_cache[((it0.get("md5") or it0["file"], it0.get("device_key", "phone"), json.dumps(it0.get("measured") or {}, sort_keys=True)), st)] = {k: v for k, v in j.items() if k not in ("sid", "dataset")}
    report(force=True, activity="测量第一页（OCR）" if items else "")
    STREAK = {"fail": 0, "pause_until": 0.0, "pauses": 0}   # consecutive failures across workers: pause on 8, give up after 3 pauses
    def note_result(ok):
        with LOCK:
            if ok: STREAK["fail"] = 0; return
            STREAK["fail"] += 1
            if STREAK["fail"] >= 8 and time.time() >= STREAK["pause_until"]:
                STREAK["pauses"] += 1; STREAK["fail"] = 0; STREAK["pause_until"] = time.time() + 60
                print(f"连续 8 次调用失败：暂停 60 秒等网络恢复（第 {STREAK['pauses']} 次）", flush=True)
                if STREAK["pauses"] >= 3: STOP["net"] = True; print("网络持续不可用：这一步停下，检查 Clash 和网络后点“重试”，只补失败的部分", flush=True)
    lock_p = out.with_suffix(".lock")
    def pid_alive(pid):
        try:
            if os.name == "nt":
                r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, timeout=10); return str(pid) in r.stdout
            os.kill(pid, 0); return True
        except Exception:
            return False
    try:
        lk = json.loads(lock_p.read_text(encoding="utf-8")) if lock_p.exists() else None
        if lk and lk.get("pid") != os.getpid() and time.time() - float(lk.get("t", 0)) < 180 and pid_alive(int(lk.get("pid", -1))):
            print(f"另一个执行器进程（PID {lk.get('pid')}）正在写 {out.name}：本进程退出，以免两个进程把记录写坏。关掉 Atelier 再双击 Atelier.bat 会清掉残留进程", flush=True); sys.exit(7)
    except SystemExit:
        raise
    except Exception:
        pass
    def heartbeat():
        try: lock_p.write_text(json.dumps({"pid": os.getpid(), "t": time.time(), "tag": a.tag}), encoding="utf-8")
        except Exception: pass
    heartbeat(); HB[0] = heartbeat
    pages = {}   # page -> its rows, in item order (UI-Lens: up to five task rows share one page)
    for it in items: pages.setdefault((it.get("md5") or it["file"], it.get("device_key", "phone"), json.dumps(it.get("measured") or {}, sort_keys=True)), []).append(it)
    STOP = {"budget": False, "net": False}
    fo = out.open("a", encoding="utf-8")
    def emit(rec):
        with LOCK: fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush(); C["new"] += 1
    def do_page(page_key, rows):
        table = None
        for it in rows:
            todo = [st for st in a.settings.split(",") if (it["sid"], st) not in done]
            for st in todo:
                if STOP["budget"] or STOP["net"]: return
                while time.time() < STREAK["pause_until"] and not STOP["net"]: time.sleep(1)
                ck = (page_key, st, it["sid"]) if st in PER_ROW else (page_key, st)
                if ck in page_cache:
                    core = page_cache[ck]
                    emit(dict(core, sid=it["sid"], dataset=it.get("run_set") or it.get("dataset"), cost_usd=0.0, latency_s=0.0, tokens_in=0, tokens_out=0, shared_call=True, ts=time.strftime("%Y-%m-%d %H:%M:%S")))
                    with LOCK: C["shared"] += 1
                    continue
                if a.cap and ledger_read().get("total_usd", 0) >= a.cap:
                    STOP["budget"] = True; return
                if table is None:
                    report(activity=f"测量 {it['sid']}")
                    try:
                        with MEASURE_LOCK: table = measure_cached(data_dir, it, a.ocr)
                    except Exception as e:   # no table for this page: every row of the page is a failed row, visible in the report
                        for it2 in rows:
                            for st2 in [x for x in a.settings.split(",") if (it2["sid"], x) not in done]:
                                emit({"sid": it2["sid"], "setting": st2, "model": a.tag, "pv": PROMPT_VERSION, "dataset": it2.get("run_set") or it2.get("dataset"), "error": f"measurement: {str(e)[:260]}", "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
                                with LOCK:
                                    n_err[0] += 1
                                    if not first_err: first_err.append(f"measurement: {str(e)[:200]}")
                        return
                if st in ("bzoom", "s3goldbox") and not it.get("boxes"): continue
                if st == "s3goldmeas" and not it.get("measured"): continue
                msgs, rule_ids, size = build_messages(st, it, table, data_dir, exemplars)
                try:
                    if a.dry_run:
                        if os.environ.get("RS_STUB_FAIL"): raise RuntimeError(os.environ["RS_STUB_FAIL"])   # test bench: every call fails with this message
                        text, tin, tout, lat, mid = stub_answer(it, table, size[0] / max(1, table["size"][0])), 3000, 150, 0.0, "stub"
                        if os.environ.get("RS_STUB_DELAY"): time.sleep(float(os.environ["RS_STUB_DELAY"]))   # test bench: a dry run that takes time, to watch the monitor
                    else:
                        text, tin, tout, lat, mid = call_openrouter(a.model, msgs, a.reasoning, base_url=a.base_url, key_env=a.key_env)
                    try: rep = V.parse_report(text); parsed = True
                    except Exception:
                        rep, parsed = {"defects": [], "screen_has_defect": False}, False
                    k_back = table["size"][0] / max(1, size[0])   # boxes from the image the model saw -> original pixels (measurements, ground truth)
                    for d in rep.get("defects") or []:
                        if isinstance(d, dict) and isinstance(d.get("box"), (list, tuple)) and len(d["box"]) >= 4:
                            try: d["box"] = [int(round(float(v) * k_back)) for v in d["box"][:4]]
                            except Exception: d["box"] = None
                    if st in VERIFIED:
                        ver = V.verify(rep, table)
                    else:   # S0, B-CoT, B-Zoom, S1, S2 and S3-without-verifier: the model's own claims, as the model gave them
                        claims = [dict(d, type=str(d.get("type", "")).upper()) for d in (rep.get("defects") or []) if isinstance(d, dict)]
                        ver = {"defects": [d for d in claims if d["type"] in V.RULES], "suspected": [], "screen_has_defect": bool([d for d in claims if d["type"] in V.RULES])}
                    cost = 0.0 if "127.0.0.1" in a.base_url or "localhost" in a.base_url else tin / 1e6 * price[0] + tout / 1e6 * price[1]
                    ledger_add(cost)
                    with LOCK: C["spent"] += cost; C["calls"] += 1
                    rec = {"sid": it["sid"], "setting": st, "model": a.tag, "pv": PROMPT_VERSION, "model_id": mid, "dataset": it.get("run_set") or it.get("dataset"), "pred": int(ver["screen_has_defect"]), "types": sorted({d["type"] for d in ver["defects"]}),
                           "verified": ver["defects"], "suspected": ver["suspected"], "raw": text[:4000], "parsed": parsed, "rules_retrieved": rule_ids, "image_size": size,
                           "tokens_in": tin, "tokens_out": tout, "cost_usd": round(cost, 6), "latency_s": round(lat, 2), "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
                    if parsed: page_cache[ck] = {k: v for k, v in rec.items() if k not in ("sid", "dataset")}
                    note_result(True)
                except Exception as e:
                    note_result(False)
                    rec = {"sid": it["sid"], "setting": st, "model": a.tag, "pv": PROMPT_VERSION, "dataset": it.get("run_set") or it.get("dataset"), "error": str(e)[:300], "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
                    with LOCK:
                        n_err[0] += 1
                        if not first_err: first_err.append(str(e)[:200])
                emit(rec)
                report(activity=f"{st} · {it['sid']}")
    try:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
            futs = [ex.submit(do_page, pk, rows) for pk, rows in pages.items()]
            for f in concurrent.futures.as_completed(futs):
                exc = f.exception()
                if exc: print(f"page worker failed: {str(exc)[:200]}"); n_err[0] += 1
    finally:
        fo.close()
        try: lock_p.unlink()
        except Exception: pass
    n_new, n_calls, n_shared, spent = C["new"], C["calls"], C["shared"], C["spent"]
    if STOP["net"]:
        report(force=True, state="failed", activity="", error="网络持续不可用：检查 Clash 和网络后点重试")
        print(f"wrote {n_new} records to {out} ({n_calls} model calls, {n_shared} rows reused the same page's answer); spent ≈ ${spent:.4f}"); sys.exit(6)
    if STOP["budget"]:
        report(force=True, state="budget", activity="预算到上限，已停")
        print(f"budget reached - stopping: {ledger_read().get('total_usd', 0):.2f} USD of {a.cap:.0f}")
        print(f"wrote {n_new} records to {out} ({n_calls} model calls, {n_shared} rows reused the same page's answer); spent ≈ ${spent:.4f}"); return
    n_bad = n_err[0]
    left = rows_total - rows_done0 - (n_new - n_bad)   # rows of this leg still without an answer after this attempt
    if n_bad and rows_total and left <= max(2, 0.02 * rows_total):   # a handful of rows that keep failing does not fail the leg
        report(force=True, state="done", activity="")
        print(f"wrote {n_new} records to {out} ({n_calls} model calls, {n_shared} rows reused the same page's answer); spent ≈ ${spent:.4f}")
        print(f"这一段还有 {left} / {rows_total} 行没有拿到回答（例：{first_err[0] if first_err else '?'}）：不到 2%，按完成处理，打分时不计这几行")
        return
    if n_new and n_bad and (n_bad == n_new or (n_new >= 10 and n_bad / n_new > 0.2)):
        why = first_err[0] if first_err else "?"
        report(force=True, state="failed", activity="", error=f"{n_bad} / {n_new} 次调用失败：{why}")
        print(f"wrote {n_new} records to {out} ({n_calls} model calls, {n_shared} rows reused the same page's answer); spent ≈ ${spent:.4f}")
        print(f"这一段 {n_bad} / {n_new} 次调用失败（例：{why}）。原因修好后点“重试”：只补失败的部分，成功的不重跑")
        sys.exit(6)
    report(force=True, state="done", activity="")
    print(f"wrote {n_new} records to {out} ({n_calls} model calls, {n_shared} rows reused the same page's answer); spent ≈ ${spent:.4f}")
    if n_bad: print(f"有 {n_bad} / {n_new} 次调用失败（例：{first_err[0] if first_err else '?'}）：再运行这一步只补失败的部分")


if __name__ == "__main__":
    main()
