"""Measure every page of the manifest once, here on AutoDL (PaddleOCR lives here), and keep the tables in data/measure_cache.
The experiment runner on the Windows PC reads these tables and never needs OCR itself.
    python precompute.py --manifest rs_data/data/RuleSight_data.json [--ocr auto] [--workers 8]
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
import multiprocessing as mp
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RA = None


def _init():
    global RA
    import run_api as _ra   # imports measure; PaddleOCR loads on the first page
    RA = _ra


def _one(args):
    data_dir, it, ocr = args
    try:
        RA.measure_cached(data_dir, it, ocr); return True, ""
    except Exception as e:
        return False, f"{it.get('sid')}: {str(e)[:120]}"


def cache_name(it, ocr):
    return f"{it.get('md5') or hashlib.md5(it['file'].encode()).hexdigest()}_{it.get('device_key', 'phone')}_{ocr}.json"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", required=True); ap.add_argument("--ocr", default="auto")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    a = ap.parse_args()
    man_p = Path(a.manifest); data_dir = man_p.parent; man = json.loads(man_p.read_text(encoding="utf-8"))
    import progress as P; P.init(data_dir.parent)
    seen, todo = set(), []
    for s in man["screens"]:
        name = cache_name(s, a.ocr)
        if name in seen: continue
        seen.add(name)
        if not (data_dir / "measure_cache" / name).exists(): todo.append(s)
    total = len(seen); done0 = total - len(todo)
    print(f"measure: {total} pages, {done0} already measured, {len(todo)} to go ({a.workers} workers)", flush=True)
    P.stage("measure", f"测量 {done0} / {total} 页"); P.ds("measure", state="building" if todo else "done", done=done0, total=total)
    if todo:
        fails = []; t0 = time.time()
        with mp.get_context("fork").Pool(a.workers, initializer=_init) as pool:
            for k, (ok, err) in enumerate(pool.imap_unordered(_one, [(data_dir, s, a.ocr) for s in todo], chunksize=4), 1):
                if not ok: fails.append(err)
                if k % 25 == 0 or k == len(todo):
                    el = time.time() - t0; eta = int(el / k * (len(todo) - k))
                    P.count("measure", done0 + k, total); P.stage("measure", f"测量 {done0 + k} / {total} 页")
                    if k % 250 == 0 or k == len(todo): print(f"  measured {done0 + k} / {total} ({el / k:.2f} s/page, about {eta // 60} min left)", flush=True)
        if fails: print(f"  {len(fails)} pages failed, e.g. {fails[0]}", flush=True)
    n_ok = sum(1 for s in seen if (data_dir / "measure_cache" / s).exists())
    P.ds("measure", state="done" if n_ok == total else "failed", done=n_ok, total=total, reason="" if n_ok == total else f"{total - n_ok} 页没测出来，见日志")
    P.stage("measure_done" if n_ok == total else "measure_failed", f"测量 {n_ok} / {total} 页")
    print(f"measure done: {n_ok} / {total}", flush=True)
    sys.exit(0 if n_ok == total else 8)


if __name__ == "__main__":
    main()
