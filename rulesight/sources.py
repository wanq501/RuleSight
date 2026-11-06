"""RuleSight data sources (design §13.1, v8).

    python sources.py check          test every source from this machine and say what is reachable and what needs a token

Sources
  UI-Lens   HuggingFace  wuhuohua/UI_lens        gated (accept the conditions on the page once) -> HF_TOKEN; CC BY 4.0; 6.85 GB; main test set
  UICrit    GitHub       google-research-datasets/uicrit   uicrit_public.csv (critiques with boxes, Rico screen ids); CC BY 4.0
  Rico      Google Storage unique_uis.tar.gz (6 GB) or HuggingFace rootsautomation/RICO-Screen2Words (22k screens with view hierarchies)
  GLIB      Zenodo 5081242 data.zip (10 GB): only testDataSet and a sample of Normal are read, by HTTP range requests
  ErgoHMI   the platform's own pages (autodl/ergohmi export)
"""
from __future__ import annotations
import http.client, io, json, os, socket, struct, sys, time, urllib.error, urllib.request, zlib
from pathlib import Path

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")      # AutoDL: export HF_ENDPOINT=https://hf-mirror.com
UI_LENS_REPO = "wuhuohua/UI_lens"
UICRIT_CSV = "https://raw.githubusercontent.com/google-research-datasets/uicrit/main/uicrit_public.csv"
RICO_TAR = "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz"
RICO_HF_FULL = "creative-graphic-design/Rico"          # all 66k screens with view hierarchies (parquet, config ui-screenshots-and-view-hierarchies)
RICO_HF_REPO = "rootsautomation/RICO-Screen2Words"     # 22k-screen subset with view hierarchies (second choice)
GLIB_ZIP = "https://zenodo.org/records/5081242/files/data.zip?download=1"
GLIB_RECORD, GLIB_KEY = "5081242", "data.zip"


def zenodo_file(record=GLIB_RECORD, key=GLIB_KEY, direct=True, opener=None, api_base="https://zenodo.org"):
    """(size, content url) of one file of a Zenodo record, from the records API — no HEAD, no range probe needed"""
    url = f"{api_base}/api/records/{record}"
    req = urllib.request.Request(url, headers={"User-Agent": "RuleSight/1.0", "Accept": "application/json"})
    with (opener.open(req, timeout=40) if opener else _DIRECT.open(req, timeout=40) if direct else _open(req, 40)) as r:
        j = json.loads(r.read().decode("utf-8", "replace"))
    files = j.get("files") or []
    if isinstance(files, dict):   # newer API shape: {"entries": {key: {...}}}
        ent = files.get("entries") or files
        files = list(ent.values()) if isinstance(ent, dict) else list(ent)
    for f in files:
        if f.get("key") == key or f.get("filename") == key:
            size = int(f.get("size") or 0)
            links = f.get("links") or {}
            link = links.get("content") or links.get("self") or f"https://zenodo.org/api/records/{record}/files/{key}/content"
            if "/api/records/" in link and not link.endswith("/content"): link = f"https://zenodo.org/api/records/{record}/files/{key}/content"
            return size, link
    raise RuntimeError(f"file {key} not in record {record}")
NIGHTHAWK_GIT = "https://github.com/20200501/Nighthawk.git"


class _NoAuthOnHostChange(urllib.request.HTTPRedirectHandler):
    """follow redirects, but never carry the HuggingFace token to another host: the mirror sends large files to a signed storage URL
    (cas-bridge.xethub.hf.co) that refuses a request with a second credential — the reason Rico stood still while curl ran at 5.6 MB/s"""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            import urllib.parse
            if urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
                for k in [k for k in new.headers if k.lower() == "authorization"]: del new.headers[k]
                new.unredirected_hdrs.pop("Authorization", None)
        return new


_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler(proxies={}), _NoAuthOnHostChange())


def _clone(req):
    """a fresh Request: urllib mutates req.host to the proxy, so the direct retry must not reuse the same object"""
    return urllib.request.Request(req.full_url, data=req.data, headers=dict(req.header_items()), method=req.get_method())


def _open(req, timeout):
    """through the environment's proxy first (AutoDL's academic proxy for GitHub / HuggingFace), then directly (Zenodo, Google Storage)."""
    have_proxy = bool(os.environ.get("http_proxy") or os.environ.get("https_proxy") or os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY"))
    if not have_proxy:
        return _DIRECT.open(req, timeout=timeout)
    try:
        return urllib.request.build_opener(_NoAuthOnHostChange()).open(_clone(req), timeout=timeout)   # the environment's proxy
    except Exception as e:
        try: return _DIRECT.open(_clone(req), timeout=timeout)                    # retry without it, fresh request
        except Exception: raise e


def _head(url, timeout=25, headers=None):
    req = urllib.request.Request(url, method="HEAD", headers=headers or {"User-Agent": "RuleSight/1.0"})
    with _open(req, timeout) as r:
        return r.status, dict(r.headers)


def _get(url, timeout=25, headers=None, rng=None):
    h = {"User-Agent": "RuleSight/1.0"}; h.update(headers or {})
    if rng: h["Range"] = f"bytes={rng[0]}-{rng[1]}"
    req = urllib.request.Request(url, headers=h)
    with _open(req, timeout) as r:
        return r.status, r.read(), dict(r.headers)


def hf_headers():
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def check_all(log=print):
    """One line per source: reachable / needs token / blocked. Returns a dict of booleans."""
    out = {}
    # UI-Lens: the dataset info endpoint answers for a gated repo only with a token that accepted the conditions
    try:
        st, body, _ = _get(f"{HF_ENDPOINT}/api/datasets/{UI_LENS_REPO}", headers=hf_headers())
        info = json.loads(body.decode("utf-8", "replace"))
        gated = info.get("gated", False); files = [s.get("rfilename") for s in info.get("siblings", [])]
        log(f"UI-Lens: 页面可访问（{HF_ENDPOINT}），列出文件 {len(files)} 个，gated={gated}")
        if not hf_headers():
            out["ui_lens"] = False; log("UI-Lens: 没有 HF_TOKEN，下载会被拒绝（先在页面点同意条款，再 export HF_TOKEN=hf_xxx）")
        else:
            probe = next((f for f in files if f.lower().endswith((".md", ".json", ".jsonl"))), "README.md")
            try:
                st2, h2 = _head(f"{HF_ENDPOINT}/datasets/{UI_LENS_REPO}/resolve/main/{probe}", headers={"User-Agent": "RuleSight/1.0", **hf_headers()})
                out["ui_lens"] = st2 == 200; log(f"UI-Lens: token 有效，可以下载（试读 {probe}）")
            except Exception as e2:
                out["ui_lens"] = False
                log(f"UI-Lens: token 不能下载（{str(e2)[:60]}）：多半是还没在数据集页面点同意条款，或 token 没有读权限")
    except Exception as e:
        msg = str(e)
        out["ui_lens"] = False
        if "401" in msg or "403" in msg: log("UI-Lens: 需要 HF_TOKEN，且账号要先在数据集页面点同意条款（https://huggingface.co/datasets/wuhuohua/UI_lens）")
        else: log(f"UI-Lens: 连不上 {HF_ENDPOINT}（{msg[:80]}）；AutoDL 上先 export HF_ENDPOINT=https://hf-mirror.com")
    try:
        st, h = _head(UICRIT_CSV); out["uicrit"] = st == 200; n = int(h.get('Content-Length') or 0); log("UICrit: 可访问（GitHub" + (f"，{n // 1024} KB" if n else "") + "）")
    except Exception as e:
        out["uicrit"] = False; log(f"UICrit: 连不上 GitHub（{str(e)[:80]}）；AutoDL 上先 source /etc/network_turbo")
    try:
        st, h = _head(RICO_TAR); out["rico_gs"] = st == 200; log(f"Rico（Google 存储）: 可访问（{int(h.get('Content-Length', 0)) // 1024 ** 2} MB）")
    except Exception as e:
        out["rico_gs"] = False; log(f"Rico（Google 存储）: 连不上（{str(e)[:60]}）；改用 HuggingFace 的 {RICO_HF_FULL}（全集），不行再用 {RICO_HF_REPO}")
    for repo, key in ((RICO_HF_FULL, "rico_hf_full"), (RICO_HF_REPO, "rico_hf")):
        try:
            _get(f"{HF_ENDPOINT}/api/datasets/{repo}", headers=hf_headers())
            out[key] = True; log(f"Rico（HuggingFace {repo}）: 可访问")
        except Exception as e:
            out[key] = False; log(f"Rico（HuggingFace {repo}）: 连不上（{str(e)[:60]}）")
    try:
        st, body, h = _get(GLIB_ZIP, rng=(0, 3))
        out["glib"] = st in (200, 206) and body[:2] == b"PK"
        log(f"GLIB（Zenodo）: 可访问，{'支持' if st == 206 else '不支持'}分段下载（只取需要的几百个文件）")
    except Exception as e:
        out["glib"] = False; log(f"GLIB（Zenodo）: 连不上（{str(e)[:60]}）")
    try:
        st, h = _head("https://github.com/20200501/Nighthawk"); out["nighthawk_code"] = st == 200; log("Nighthawk 代码: 可访问（GitHub）")
    except Exception as e:
        out["nighthawk_code"] = False; log(f"Nighthawk 代码: 连不上（{str(e)[:60]}）")
    return out


class SlowLine(RuntimeError):
    """the connection moves too slowly to be worth waiting for"""


def download_file(url, dest, expected=0, headers=None, piece=1 << 20, on_progress=None, min_rate=150e3, grace_s=90, direct=False):
    """Resumable download in 1 MB pieces with a stall guard. Raises SlowLine when, after grace_s seconds, the average rate stays under
    min_rate bytes/s — a line that would take days is given up at once so another route can be tried. Returns the bytes written."""
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part"); have = part.stat().st_size if part.exists() else 0
    h = {"User-Agent": "RuleSight/1.0", **(headers or {})}
    size = int(expected or 0)
    if not size:
        try:
            req = urllib.request.Request(url, method="HEAD", headers=h)
            with (_DIRECT.open(req, timeout=30) if direct else _open(req, 30)) as r: size = int(r.headers.get("Content-Length") or 0)
        except Exception:
            size = 0
    if size and have >= size: part.replace(dest); return 0
    t0 = time.time(); got0 = have; pos = have; fails = 0
    with open(part, "ab") as fo:
        while not size or pos < size:
            end = pos + piece - 1 if not size else min(size - 1, pos + piece - 1)
            try:
                req = urllib.request.Request(url, headers={**h, "Range": f"bytes={pos}-{end}"})
                with (_DIRECT.open(req, timeout=45) if direct else _open(req, 45)) as r:
                    if r.status == 200 and pos > 0: raise RuntimeError("the server ignores range requests")
                    data = r.read()
                    if not size and r.status == 206 and "/" in (r.headers.get("Content-Range") or ""): size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
            except http.client.IncompleteRead as e:
                data = bytes(e.partial)
            except (urllib.error.URLError, ConnectionError, socket.timeout, TimeoutError, http.client.HTTPException) as e:
                fails += 1
                if fails > 8: raise RuntimeError(f"gave up after 8 failures: {str(e)[:80]}")
                if time.time() - t0 > grace_s and (pos - got0) / (time.time() - t0) < min_rate: raise SlowLine(f"{(pos - got0) / 1e6:.1f} MB in {int(time.time() - t0)} s")
                time.sleep(min(20, 2 * fails)); continue
            if not data:
                if size and pos < size: fails += 1; time.sleep(2); continue
                break
            fo.write(data); pos += len(data); fails = 0
            if on_progress: on_progress(pos, size)
            el = time.time() - t0
            if el > grace_s and (pos - got0) / el < min_rate: raise SlowLine(f"{(pos - got0) / 1e6:.1f} MB in {int(el)} s")
            if r.status == 200: break
    part.replace(dest)
    return pos - have


class Aborted(RuntimeError):
    pass


def _range_once(url, a, b, headers, direct, timeout=60, abort=None, on_bytes=None):
    """bytes a..b of url; read in 64 KB steps so a give-up (abort()) takes effect within a step; an IncompleteRead is continued"""
    out = bytearray(); pos = a
    while pos <= b:
        req = urllib.request.Request(url, headers={**headers, "Range": f"bytes={pos}-{b}"})
        try:
            with (_DIRECT.open(req, timeout=timeout) if direct else _open(req, timeout)) as r:
                if r.status != 206: raise RuntimeError("the server ignores range requests")
                while True:
                    if abort and abort(): raise Aborted("given up")
                    chunk = r.read(1 << 16)
                    if not chunk: break
                    out += chunk
                    if on_bytes: on_bytes(len(chunk))
        except http.client.IncompleteRead as e:
            out += bytes(e.partial)
            if not e.partial: raise
        pos = a + len(out)
    return bytes(out)


def download_file_parallel(url, dest, size, headers=None, workers=12, piece=4 << 20, on_progress=None, min_rate=150e3, grace_s=90, direct=True, log=None, label=""):
    """One file through many range requests at once (mirrors cap the speed of one connection, not of a dozen). Resumable: finished pieces are
    recorded beside the file. Raises SlowLine when the combined rate stays under min_rate after grace_s seconds."""
    import threading, concurrent.futures
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    size = int(size)
    if dest.exists() and dest.stat().st_size == size: return 0
    part = dest.with_name(dest.name + ".part"); side = dest.with_name(dest.name + ".pieces")
    try: done = set(json.loads(side.read_text())) if side.exists() and part.exists() else set()
    except Exception: done = set()
    n = max(1, -(-size // piece)); h = {"User-Agent": "RuleSight/1.0", **(headers or {})}
    lock = threading.Lock(); st = {"bytes": len(done) * piece, "t0": time.time(), "b0": len(done) * piece, "err": None, "saved": 0, "live": 0, "fails": 0, "last_err": ""}
    def _live(n):
        with lock: st["live"] += n
    with open(part, "r+b" if part.exists() else "w+b") as fo:
        fo.truncate(size)
        def fetch(i):
            a = i * piece; b = min(size - 1, a + piece - 1); last = None
            for attempt in range(8):
                if st["err"]: return
                try:
                    before = st["live"]
                    data = _range_once(url, a, b, h, direct, abort=lambda: st["err"] is not None, on_bytes=_live)
                    if len(data) != b - a + 1: raise RuntimeError(f"short piece {len(data)} of {b - a + 1}")
                    with lock:
                        fo.seek(a); fo.write(data); st["bytes"] += len(data); done.add(i); st["saved"] += 1
                        if st["saved"] % 8 == 0: fo.flush(); side.write_text(json.dumps(sorted(done)))
                    if on_progress: on_progress(min(size, st["bytes"]), size)
                    return
                except Aborted:
                    return
                except Exception as e:
                    last = e
                    with lock:
                        st["fails"] += 1; st["last_err"] = f"{type(e).__name__}: {str(e)[:90]}"
                        if log and st["fails"] <= 5 or (log and st["fails"] % 20 == 0): log(f"  {label}块 {i} 第 {attempt + 1} 次失败（{st['last_err']}），重试")
                    time.sleep(min(20, 2 * (attempt + 1)))
            with lock:
                if not st["err"]: st["err"] = RuntimeError(f"piece {i} failed 8 times: {str(last)[:80]}")
        todo = [i for i in range(n) if i not in done]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(fetch, i) for i in todo]
            last_log = time.time()
            while not all(f.done() for f in futs):
                time.sleep(2)
                el = time.time() - st["t0"]; got = st["live"]   # bytes that actually arrived, also inside unfinished pieces
                if on_progress: on_progress(min(size, st["b0"] + got), size, got / max(1e-6, el))
                if log and time.time() - last_log >= 20:
                    last_log = time.time()
                    log(f"  {label}已下 {(st['b0'] + got) / 1e6:.0f} / {size / 1e6:.0f} MB，{got / max(1e-6, el) / 1e6:.2f} MB/s，{workers} 个连接，失败重试 {st['fails']} 次" + (f"（最近：{st['last_err']}）" if st['last_err'] else ""))
                if el > grace_s and got / el < min_rate:
                    with lock: st["err"] = SlowLine(f"{got / 1e6:.1f} MB in {int(el)} s with {workers} connections" + (f"; last error: {st['last_err']}" if st['last_err'] else ""))
                    for f in futs: f.cancel()
                    break
            for f in futs:
                try: f.result()
                except concurrent.futures.CancelledError: pass
        fo.flush()
    if st["err"]:
        try: side.write_text(json.dumps(sorted(done)))
        except Exception: pass
        raise st["err"]
    part.replace(dest)
    try: side.unlink()
    except OSError: pass
    return st["bytes"] - st["b0"]


class _NoFollow(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl): return None


def resolve_url(url, headers=None, direct=True):
    """the final address behind a mirror's redirect (the signed storage URL): asked once, then every range request goes straight there —
    a dozen pieces asking the mirror again and again is what gets throttled"""
    h = {"User-Agent": "RuleSight/1.0", **(headers or {}), "Range": "bytes=0-0"}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoFollow()) if direct else urllib.request.build_opener(_NoFollow())
    cur = url
    for _ in range(5):
        try:
            with opener.open(urllib.request.Request(cur, headers=h), timeout=40) as r:
                return cur   # no redirect: this is the file itself
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308) and e.headers.get("Location"):
                cur = urllib.parse.urljoin(cur, e.headers["Location"]); continue
            if e.code in (200, 206): return cur
            raise
    return cur


def probe_speed(url, headers=None, direct=True, seconds=8, workers=4, piece=2 << 20):
    """(bytes per second, error) reading ranges of url for a few seconds through `workers` connections"""
    import threading
    h = {"User-Agent": "RuleSight/1.0", **(headers or {})}; st = {"n": 0, "err": None, "stop": False}; lock = threading.Lock()
    def go(k):
        try:
            a = k * piece
            _range_once(url, a, a + piece - 1, h, direct, timeout=seconds + 5, abort=lambda: st["stop"], on_bytes=lambda n: (lock.acquire(), st.__setitem__("n", st["n"] + n), lock.release()))
        except Aborted:
            pass
        except Exception as e:
            st["err"] = f"{type(e).__name__}: {str(e)[:70]}"
    ts = [threading.Thread(target=go, args=(k,), daemon=True) for k in range(workers)]
    t0 = time.time()
    for t in ts: t.start()
    time.sleep(seconds); st["stop"] = True
    for t in ts: t.join(timeout=3)
    rate = st["n"] / max(1e-6, time.time() - t0)
    return rate, (st["err"] if not st["n"] else None)


class RemoteZip:
    """Read single members of a zip on an HTTP server that supports range requests (Zenodo does), without downloading the whole file."""
    def __init__(self, url, headers=None, size=0, direct=False, opener=None):
        self.url, self.headers, self.direct, self.opener = url, headers or {}, direct, opener   # direct: no proxy from the environment; opener: a given one
        self.size = int(size or 0)
        if not self.size:
            try:   # HEAD first; Zenodo's download redirect does not always carry a length
                req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "RuleSight/1.0", **self.headers})
                with (self.opener.open(req, timeout=25) if self.opener else _DIRECT.open(req, timeout=25) if self.direct else _open(req, 25)) as r: self.size = int(r.headers.get("Content-Length") or 0)
            except Exception:
                pass
        if not self.size:   # a one-byte range request answers with "Content-Range: bytes 0-0/<total>"
            st, body, h = self._get(url, rng=(0, 0))
            cr = h.get("Content-Range") or ""
            if st == 206 and "/" in cr and cr.rsplit("/", 1)[1].isdigit(): self.size = int(cr.rsplit("/", 1)[1])
        if not self.size: raise RuntimeError("the server reports neither a file size nor a byte range")
        self.entries = self._central_directory()

    def _get(self, url, rng=None, timeout=60):
        """one range request; a 200 with a full body means ranges are not honoured (a proxy in between often does that)"""
        h = {"User-Agent": "RuleSight/1.0", **self.headers}
        if rng is not None: h["Range"] = f"bytes={rng[0]}-{rng[1]}"
        req = urllib.request.Request(url, headers=h)
        r = self.opener.open(req, timeout=timeout) if self.opener else _DIRECT.open(req, timeout=timeout) if self.direct else _open(req, timeout)
        with r:
            if rng is not None and r.status == 200:
                cl = r.headers.get("Content-Length")
                if cl is None or int(cl) > (rng[1] - rng[0] + 1) * 4 + 1024:
                    raise RuntimeError("the server (or a proxy) ignores range requests")
            return r.status, r.read(), dict(r.headers)

    CHUNK = 1 << 20   # long ranges are read in 1 MB pieces: a connection that drops mid-way costs one piece, not the whole read

    def _range(self, a, b, timeout=120):
        """bytes a..b inclusive, piece by piece; a dropped connection keeps what arrived and continues from there (up to 8 tries per piece)"""
        out = bytearray(); pos = a
        while pos <= b:
            end = min(b, pos + self.CHUNK - 1); got = None; last = None
            for attempt in range(8):
                try:
                    st, data, _ = self._get(self.url, rng=(pos, end), timeout=timeout)
                    if st != 206: raise RuntimeError("the server ignores range requests")
                    got = data; break
                except http.client.IncompleteRead as e:   # keep the part that arrived
                    if e.partial: got = bytes(e.partial); break
                    last = e
                except (urllib.error.URLError, ConnectionError, socket.timeout, TimeoutError, http.client.HTTPException) as e:
                    last = e
                time.sleep(min(20, 2 * (attempt + 1)))
            if not got: raise RuntimeError(f"piece {pos}-{end} failed 8 times: {str(last)[:80]}")
            out += got; pos += len(got)
        return bytes(out)

    def _central_directory(self):
        tail_len = min(self.size, 66000)
        tail = self._range(self.size - tail_len, self.size - 1)
        i = tail.rfind(b"PK\x05\x06")
        if i < 0: raise RuntimeError("end of central directory not found")
        eocd = tail[i:i + 22]   # signature, disk, cd disk, entries on disk, total entries (10:12), cd size (12:16), cd offset (16:20), comment length
        n_total = struct.unpack("<H", eocd[10:12])[0]; cd_size = struct.unpack("<I", eocd[12:16])[0]; cd_off = struct.unpack("<I", eocd[16:20])[0]
        if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or n_total == 0xFFFF:   # zip64 (GLIB's 10 GB archive is one)
            j = tail.rfind(b"PK\x06\x06")
            if j < 0: raise RuntimeError("zip64 end record not found")
            z64 = tail[j:j + 56]
            n_total = struct.unpack("<Q", z64[32:40])[0]; cd_size = struct.unpack("<Q", z64[40:48])[0]; cd_off = struct.unpack("<Q", z64[48:56])[0]
        cd = self._range(cd_off, cd_off + cd_size - 1)
        entries, p = {}, 0
        while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
            comp = struct.unpack("<H", cd[p + 10:p + 12])[0]
            csize = struct.unpack("<I", cd[p + 20:p + 24])[0]; usize = struct.unpack("<I", cd[p + 24:p + 28])[0]
            nlen, xlen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
            lho = struct.unpack("<I", cd[p + 42:p + 46])[0]
            name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
            extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
            if 0xFFFFFFFF in (csize, usize, lho):   # zip64 extra field
                q = 0
                while q + 4 <= len(extra):
                    hid, hsz = struct.unpack("<HH", extra[q:q + 4]); data = extra[q + 4:q + 4 + hsz]
                    if hid == 1:
                        vals = [struct.unpack("<Q", data[k:k + 8])[0] for k in range(0, len(data) - 7, 8)]
                        it = iter(vals)
                        if usize == 0xFFFFFFFF: usize = next(it, usize)
                        if csize == 0xFFFFFFFF: csize = next(it, csize)
                        if lho == 0xFFFFFFFF: lho = next(it, lho)
                    q += 4 + hsz
            entries[name] = (comp, csize, usize, lho)
            p += 46 + nlen + xlen + clen
        return entries

    def names(self): return list(self.entries)

    def read(self, name):
        comp, csize, usize, lho = self.entries[name]
        head = self._range(lho, lho + 29)
        nlen, xlen = struct.unpack("<HH", head[26:30])
        start = lho + 30 + nlen + xlen
        data = self._range(start, start + csize - 1) if csize else b""
        if comp == 0: return data
        if comp == 8:
            return zlib.decompress(data, -15)
        raise RuntimeError(f"unsupported compression {comp} for {name}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        res = check_all()
        print(json.dumps(res))
    else:
        print(__doc__)
