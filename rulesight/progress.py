"""Progress of the RuleSight executor, for the console (rs_data/progress.json, served by the status server).

    stage / detail      what is happening right now, in plain words
    datasets            per dataset: state (waiting, downloading, building, done, skipped, failed), counts, bytes for downloads
    thumbs              preview samples per dataset: a small JPEG with the ground-truth boxes drawn, and a caption
"""
from __future__ import annotations
import json, os, threading, time
from pathlib import Path

_LOCK = threading.Lock()
_PATH = None


def init(rs_home):
    global _PATH
    _PATH = Path(rs_home) / "progress.json"
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    return _PATH


_LAST = [None]   # the last state read or written in full: a read that fails while another process touches the file must not look like an empty state


def _read():
    if not _PATH: return {}
    for k in range(6):
        try:
            if not _PATH.exists(): return dict(_LAST[0]) if _LAST[0] else {}
            d = json.loads(_PATH.read_text(encoding="utf-8")); _LAST[0] = d; return d
        except Exception:
            time.sleep(0.03 * (k + 1))
    return dict(_LAST[0]) if _LAST[0] else {}


def _write(d):
    if not _PATH: return
    d["t"] = time.time(); _LAST[0] = d
    try:
        tmp = _PATH.with_suffix(f".{os.getpid()}.tmp"); tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        for k in range(10):   # on Windows the rename is refused while a reader has the file open: wait a little and try again
            try: tmp.replace(_PATH); return
            except PermissionError: time.sleep(0.03 * (k + 1))
        _PATH.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass   # a full disk must not take the reporter down with it; the run itself reports the disk


def stage(name, detail=""):
    with _LOCK:
        d = _read(); d["stage"] = name; d["detail"] = detail; _write(d)


def ds(name, **fields):
    with _LOCK:
        d = _read(); cur = d.setdefault("datasets", {}).setdefault(name, {}); cur.update(fields)
        if fields.get("state") in ("done", "downloaded", "failed", "skipped"): cur.pop("eta_s", None); cur.pop("rate_bps", None)
        _write(d)


def _run_path(key):
    return _PATH.with_name(f"run_{key}.json") if _PATH else None


def _read_json(path):
    for k in range(6):
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        except Exception:
            time.sleep(0.03 * (k + 1))
    return None


def _write_json(path, d):
    try:
        tmp = path.with_suffix(f".{os.getpid()}.tmp"); tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        for k in range(10):
            try: tmp.replace(path); return
            except PermissionError: time.sleep(0.03 * (k + 1))
        path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def run_plan(step, legs, budget=None, key=None):
    """the whole experiment step up front: every leg (model × settings × datasets) listed as waiting, with its planned rows and calls.
    With a key (the PC runner: one per step), the plan lives in its own file run_<key>.json, so the two lanes (API models and local models)
    never overwrite each other's progress; without one (rs.sh on AutoDL) it lives under "run" in progress.json, as before"""
    run = {"step": step, "started": time.time(), "budget": budget, "legs": [dict(l, state="waiting", rows_done=0, calls_done=0, spent=0.0) for l in legs]}
    if key and _PATH:
        with _LOCK: _write_json(_run_path(key), run)
        return
    with _LOCK:
        d = _read(); d["run"] = run; _write(d)


_RUNS = {}   # key -> last full plan this process read or wrote


def run_leg(i, key=None, **fields):
    if key and _PATH:
        with _LOCK:
            path = _run_path(key); r = _read_json(path) or _RUNS.get(key)
            if not r: return
            legs = r.get("legs") or []
            if 0 <= i < len(legs):
                legs[i].update(fields); legs[i]["t"] = time.time(); _RUNS[key] = r; _write_json(path, r)
        return
    with _LOCK:
        d = _read(); r = d.get("run") or {}; legs = r.get("legs") or []
        if 0 <= i < len(legs): legs[i].update(fields); legs[i]["t"] = time.time(); _write(d)


def run_file(key):
    """the plan of one step, as the PC runner wrote it (None if absent)"""
    return _read_json(_run_path(key)) if _PATH else None


def current(name):
    """the dataset the following log lines belong to (main thread); GLIB's thread passes its name to note() itself"""
    _CURRENT[0] = name


def note(line, name=None):
    """keep the last lines of a dataset's log with the dataset, so its card can show them"""
    name = name or _CURRENT[0]
    if not name: return
    with _LOCK:
        d = _read(); cur = d.setdefault("datasets", {}).setdefault(name, {})
        lines = (cur.get("log") or [])[-7:] + [time.strftime("%H:%M:%S ") + str(line)[:200]]
        cur["log"] = lines; _write(d)


def reset_queue(order, skip=()):
    """at the start of a run: every dataset not yet done is 'waiting' with its place in the queue; a run that was cut off leaves no stale 'downloading'"""
    with _LOCK:
        d = _read(); ds_all = d.setdefault("datasets", {})
        pos = 0
        for k in order:
            cur = ds_all.setdefault(k, {})
            if k in skip: cur.update(state="skipped", reason="本次不处理"); continue
            if cur.get("state") in ("done", "downloaded"): continue
            pos += 1
            cur.update(state="waiting", queue=pos); cur.pop("eta_s", None); cur.pop("rate_bps", None); cur.pop("reason", None)
        d["stage"] = "fetch"; _write(d)


_RATE = {}


def count(name, done, total, **fields):
    """count-based progress (files, screens) with a smoothed rate and the time left"""
    now = time.time(); r = _RATE.setdefault(name, {"t0": now, "n0": done, "rate": None})
    el = now - r["t0"]
    if el > 2 and done > r["n0"]:
        cur = (done - r["n0"]) / el
        r["rate"] = cur if r["rate"] is None else 0.7 * r["rate"] + 0.3 * cur
    eta = int((total - done) / r["rate"]) if r["rate"] and total and done < total else None
    ds(name, done=done, total=total, eta_s=eta, **fields)


def dir_bytes(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try: total += os.path.getsize(os.path.join(root, f))
            except OSError: pass
    return total


class Watch:
    """while a download runs, measure the folder every few seconds and report the bytes"""
    def __init__(self, name, folder, expected, every=4.0, base_bytes=0):
        self.name, self.folder, self.expected, self.every, self.base = name, Path(folder), expected, every, base_bytes
        self._stop = threading.Event(); self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        t0, last, rate = time.time(), None, None
        while not self._stop.is_set():
            b = self.base + (dir_bytes(self.folder) if self.folder.exists() else 0)
            if last is not None:
                cur = max(0.0, (b - last) / self.every)
                rate = cur if rate is None else 0.7 * rate + 0.3 * cur   # smoothed, so the estimate does not jump
            last = b
            eta = int((self.expected - b) / rate) if rate and self.expected and b < self.expected else None
            ds(self.name, state="downloading", bytes=b, expected_bytes=self.expected, rate_bps=rate, eta_s=eta, elapsed_s=int(time.time() - t0))
            self._stop.wait(self.every)

    def __enter__(self): self._t.start(); return self

    def __exit__(self, *exc):
        self._stop.set(); self._t.join(timeout=10)
        ds(self.name, bytes=self.base + (dir_bytes(self.folder) if self.folder.exists() else 0))


RULE_ZH = {"D1": "文字重叠", "D2": "控件遮挡", "D3": "图片缺失", "D4": "空值", "D5": "花屏撕裂", "D6": "文字溢出", "D7": "异常省略号", "D8": "内容裁切",
           "E1": "文字对比度", "E2": "非文字对比度", "E3": "字太小", "E4": "点击目标小", "E5": "320px 重排", "E6": "间距拥挤", "E7": "色觉", "E8": "环境光"}
TASK_ZH = {"Text Overflow": "文字溢出", "Cropped Content": "内容裁切", "Container Overlap": "容器重叠", "Undisplayed Content": "内容未显示", "Abnormal Text Ellipsis": "异常省略号"}
ORDER = ["ui_lens", "ui_lens_dev", "uicrit", "glib", "rico_synth", "e_defect", "e_hard", "e_real", "f_wild"]


def caption(s):
    if s.get("ui_lens_task"):
        return f"{TASK_ZH.get(s['ui_lens_task'], s['ui_lens_task'])}：{'有' if s.get('label') else '无'}"
    if s.get("types"):
        return "、".join(RULE_ZH.get(t, t) for t in s["types"][:3])
    return "有缺陷" if s.get("label") else "无缺陷"


def make_thumbs(root, screens, rs_home, per_set=6, width=360):
    """a few samples per dataset, with the ground-truth boxes drawn, for the console's preview"""
    from PIL import Image, ImageDraw
    out_dir = Path(rs_home) / "thumbs"; out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        try: old.unlink()
        except OSError: pass
    by = {}
    for s in screens:
        by.setdefault(s["dataset"], []).append(s)
    thumbs = []
    for dsn, items in sorted(by.items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99):
        pos = [s for s in items if s.get("label")]; neg = [s for s in items if not s.get("label")]
        kinds = {}   # one positive of each task / defect kind first (round robin), then two clean ones
        for s in pos: kinds.setdefault(s.get("ui_lens_task") or (s.get("types") or ["?"])[0], []).append(s)
        spread, k = [], 0
        while len(spread) < max(1, per_set - 2) and any(len(v) > k for v in kinds.values()):
            for v in kinds.values():
                if len(v) > k and len(spread) < max(1, per_set - 2): spread.append(v[k])
            k += 1
        pick = (spread + neg[:per_set - len(spread)])[:per_set] if pos else neg[:per_set]
        for k, s in enumerate(pick):
            src = Path(root) / s["file"] if not os.path.isabs(s["file"]) else Path(s["file"])
            try:
                img = Image.open(src).convert("RGB")
                W, H = img.size; f = width / W
                th = img.resize((width, max(1, int(H * f))))
                d = ImageDraw.Draw(th)
                for b in (s.get("boxes") or [])[:12]:
                    x, y, w, h = [float(v) for v in b[:4]]
                    d.rectangle([x * f, y * f, (x + w) * f, (y + h) * f], outline=(220, 40, 40), width=3)
                p = out_dir / f"{dsn}_{k}.jpg"; th.save(p, "JPEG", quality=82)
                label = caption(s)
                thumbs.append({"dataset": dsn, "sid": s["sid"], "path": str(p), "caption": label, "boxes": len(s.get("boxes") or []), "app": s.get("app", ""), "size": [W, H]})
            except Exception:
                continue
    with _LOCK:
        d = _read(); d["thumbs"] = thumbs; _write(d)
    return thumbs
