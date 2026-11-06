import http.client, io, json, os, random, socket, struct, sys, time, threading, urllib.error, urllib.parse, urllib.request, zlib, concurrent.futures
from pathlib import Path


class _NoAuthOnHostChange(urllib.request.HTTPRedirectHandler):
    pass


_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler(proxies={}))
_PROXY = urllib.request.build_opener()
USE_PROXY = [False]


def _open(req, timeout):
    return (_PROXY if USE_PROXY[0] else _DIRECT).open(req, timeout=timeout)


class SlowLine(RuntimeError):
    pass


class Aborted(RuntimeError):
    pass


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


SEED = 20261101


def main():
    D = Path(sys.argv[1]); dest = D / "glib"; dest.mkdir(parents=True, exist_ok=True)
    api = os.environ.get("ZENODO", "https://zenodo.org")
    rz = None; errs = []
    for name, proxy in (("直连", False), ("学术加速", True)):
        USE_PROXY[0] = proxy
        try:
            size, link = zenodo_file(direct=not proxy, api_base=api)
            rz = RemoteZip(link, size=size, direct=not proxy); print(f"GLIB：{name}连上 Zenodo，压缩包 {rz.size / 1e9:.1f} GB，{len(rz.names())} 个文件", flush=True); break
        except Exception as e:
            errs.append(f"{name}：{str(e)[:80]}")
    if rz is None: print("GLIB：Zenodo 两条路都不通（" + "；".join(errs) + "）"); sys.exit(1)
    names = rz.names()
    tests = [n for n in names if "/testDataSet/" in n and n.lower().endswith((".png", ".jpg", ".jpeg"))]
    normals = [n for n in names if "/images/normal/test/" in n.lower() and n.lower().endswith((".png", ".jpg", ".jpeg"))]
    csvs = [n for n in names if "data_csv" in n and "testDataSet" in n and n.lower().endswith(".csv")]
    rng = random.Random(SEED); rng.shuffle(normals)
    targets = tests + normals[:200]
    for n in csvs:
        try: (dest / Path(n).name).write_bytes(rz.read(n))
        except Exception: pass
    todo = []
    for n in targets:
        p = dest / ("test_" if n in tests else "normal_") / Path(n).name
        if not (p.exists() and p.stat().st_size == rz.entries[n][2]): todo.append((n, p))   # a file already here counts only at its true size
    print(f"GLIB：要 {len(targets)} 张（缺陷 {len(tests)} + 正常 {min(200, len(normals))}），已有 {len(targets) - len(todo)} 张，还要下 {len(todo)} 张", flush=True)
    lock = threading.Lock(); st = {"k": 0, "b": 0, "t0": time.time(), "fail": 0}
    def one(item):
        n, p = item; p.parent.mkdir(exist_ok=True)
        for attempt in range(3):
            try:
                data = rz.read(n); tmp = p.with_suffix(p.suffix + ".part"); tmp.write_bytes(data); tmp.replace(p)
                with lock:
                    st["k"] += 1; st["b"] += len(data)
                    if st["k"] % 20 == 0 or st["k"] == len(todo):
                        el = time.time() - st["t0"]; print(f"  已下 {st['k']} / {len(todo)} 张，{st['b'] / 1e6:.0f} MB，{st['b'] / max(el, 1e-6) / 1e3:.0f} KB/s", flush=True)
                return
            except Exception as e:
                last = e; time.sleep(3)
        with lock: st["fail"] += 1; print(f"  没下到：{Path(n).name}（{str(last)[:60]}）", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex: list(ex.map(one, todo))
    n_test = len(list((dest / "test_").glob("*.*"))); n_norm = len(list((dest / "normal_").glob("*.*")))
    ok = n_test >= len(tests) - 2 and n_norm >= min(200, len(normals)) - 2
    if ok: (dest / ".complete").write_text(str(n_test + n_norm))
    print(f"GLIB：缺陷 {n_test}/{len(tests)} 张，正常 {n_norm}/{min(200, len(normals))} 张" + ("，齐了" if ok else "，没齐：再运行一次会补齐"), flush=True)
    sys.exit(0 if ok else 1)


main()
