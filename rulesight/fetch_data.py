"""RuleSight data script (design §13.1, v8 — the datasets that can really be obtained).

    python sources.py check                                        # first: what this machine can reach, and whether HF_TOKEN works
    python fetch_data.py --root data --get-ui-lens --get-rico      # download UI-Lens (HF_TOKEN) and the Rico screens
    python fetch_data.py --root data [--ergohmi <ErgoHMI export>] [--skip ui_lens,glib,...]

Builds data/RuleSight_data.json (the frozen manifest, md5 of every image) and data/MANIFEST_MD5.txt from:
  ui_lens      HuggingFace wuhuohua/UI_lens (gated: accept the conditions on the page, set HF_TOKEN; HF_ENDPOINT may point at a mirror).
               Single-UI pages with per-task labels and boxes. 300 pages (stratified by app) become the Dev split; the rest is Test.
  uicrit       github google-research-datasets/uicrit: expert critiques with normalized boxes on Rico screens -> real labels for E1–E6 (and some D).
  glib         Zenodo 5081242 data.zip: testDataSet (real display issues from game test reports) + a matched sample of Normal, read by HTTP ranges.
  rico_synth   inject_defects.py on Rico screens (screenshot + view hierarchy) -> D1–D8 with exact boxes: exemplar pool, detector training, Dev.
  e_defect / e_hard / e_real   ErgoHMI pages (physical labels) when --ergohmi points at the export.
  f_wild       data/fwild/fwild.json written by fwild_capture.py.
Rico screens are read from --rico (<id>.jpg + <id>.json). get_rico() fills that folder from Google Storage (unique_uis.tar.gz) or, when that is
not reachable, from HuggingFace rootsautomation/RICO-Screen2Words (needs pyarrow); UICrit screens that neither source has are reported.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, json, os, random, re, sys, tarfile, time, urllib.parse, urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sources as S  # noqa: E402
import progress as P  # noqa: E402

SEED = 20261101
KB = json.loads((HERE / "rulekb" / "rules.json").read_text(encoding="utf-8"))
UI_LENS_MAP = KB["category_map"]["UI-Lens"]
UI_LENS_ALIASES = {   # the category as the label files may spell it -> the card's English name
    "text overflow": "Text Overflow", "文字溢出": "Text Overflow", "overflow": "Text Overflow",
    "cropped content": "Cropped Content", "内容裁切": "Cropped Content", "cropped": "Cropped Content", "crop": "Cropped Content",
    "container overlap": "Container Overlap", "容器重叠": "Container Overlap", "overlap": "Container Overlap",
    "undisplayed content": "Undisplayed Content", "内容未显示": "Undisplayed Content", "undisplayed": "Undisplayed Content", "missing": "Undisplayed Content",
    "abnormal text ellipsis": "Abnormal Text Ellipsis", "异常省略号": "Abnormal Text Ellipsis", "ellipsis": "Abnormal Text Ellipsis",
}
UICRIT_KINDS = [   # critique wording -> our rule ids (the E-rules are measured, the D-rules geometric)
    ("E1", r"contrast|hard to read|low visibility|faint|washed|legib|readab"),
    ("E3", r"(text|font|label)[^.]{0,40}(too small|small|tiny)|small (text|font)"),
    ("E4", r"(button|icon|target|tap|toggle|checkbox)[^.]{0,40}(too small|small|tiny)|small (button|icon|tap)"),
    ("E6", r"too close|crowd|clutter|spacing|cramped|padding"),
    ("D1", r"text[^.]{0,30}overlap|overlapping text"),
    ("D2", r"overlap|hidden behind|covers|obscur"),
    ("D6", r"cut off|cut-off|truncat"),
    ("D8", r"cropped"),
]


_MD5 = {}


def md5(path, chunk=1 << 20):
    """content hash, computed once per file per run (a UI-Lens page carries five task rows)"""
    key = str(path)
    if key in _MD5: return _MD5[key]
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""): h.update(b)
    _MD5[key] = h.hexdigest()
    return _MD5[key]


def rel(root, p):
    p = Path(p).resolve()
    try: return str(p.relative_to(root.resolve())).replace(os.sep, "/")
    except ValueError: return str(p)


# ----------------------------------------------------------------------------- UI-Lens
def canon_task(name):
    n = (name or "").strip().lower()
    for k, v in UI_LENS_ALIASES.items():
        if k in n: return v
    return (name or "").strip()


def load_ui_lens(root, log):
    """UI-Lens as downloaded by huggingface_hub into root/ui_lens: label_for_single_UIs/*.json(l) + single_UIs/.
    One manifest item per (page, task); a page is positive for its task when it carries at least one box."""
    base = root / "ui_lens"
    # the annotation folders as they really are on the hub: label_for_single_UIs_cn (and _en later), at any depth
    labs = sorted(p for p in base.rglob("label_for_single_UIs*") if p.is_dir()) if base.exists() else []
    if not labs:
        log("UI-Lens：还没下载（data/ui_lens 下没有 label_for_single_UIs* 目录）。先运行：python fetch_data.py --get-ui-lens"); return [], {}
    records = []
    label_files = sorted(f for lab in labs for f in list(lab.rglob("*.json")) + list(lab.rglob("*.jsonl")))
    P.stage("fetch", f"解析 UI-Lens 标注（{len(label_files)} 个文件）"); P.ds("ui_lens", state="building", detail=f"解析标注：{len(label_files)} 个文件")
    for f in label_files:
        txt = f.read_text(encoding="utf-8", errors="replace")
        try:
            j = json.loads(txt)
            rows = j if isinstance(j, list) else j.get("data") or j.get("items") or j.get("annotations") or [j]
        except Exception:
            rows = [json.loads(l) for l in txt.splitlines() if l.strip().startswith("{")]
        for r in rows:
            if not isinstance(r, dict): continue
            info = r.get("infos") if isinstance(r.get("infos"), dict) else r
            img = info.get("image_path") or r.get("image_path") or ""
            if not img: continue
            boxes = info.get("box_list") or []
            task = canon_task(info.get("target_problem") or r.get("target_problem") or f.stem)
            records.append({"id": str(r.get("id", len(records))), "image": img, "app": info.get("app_name") or "", "task": task,
                            "boxes": [[int(float(v)) for v in b[:4]] for b in boxes if isinstance(b, (list, tuple)) and len(b) >= 4], "labels": info.get("label_list") or [], "size": info.get("image_size") or None})
    if not records:
        log("UI-Lens：标注文件里没有读到条目（结构和数据卡不同？把 label_for_single_UIs 里的一个文件发我）"); return [], {}
    items, missing, by_task, seen_sids, by_name = [], 0, defaultdict(int), set(), None
    for r in records:
        p = base / "single_UIs" / r["image"]
        if not p.exists(): p = base / r["image"]
        if not p.exists():
            if by_name is None: by_name = {q.name: q for q in base.rglob("*") if q.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")}
            p = by_name.get(Path(r["image"]).name, p)
        if not p.exists(): missing += 1; continue
        pos = len(r["boxes"]) > 0
        types = sorted(UI_LENS_MAP.get(r["task"], [])) if pos else []
        by_task[r["task"]] += 1
        tcode = {"Text Overflow": "TO", "Cropped Content": "CC", "Container Overlap": "CO", "Undisplayed Content": "UC", "Abnormal Text Ellipsis": "AE"}.get(r["task"]) or ("T" + hashlib.md5(r["task"].encode()).hexdigest()[:4])
        sid = f"UL-{tcode}-{r['id']}"
        while sid in seen_sids: sid += "x"   # ids repeat across label files
        seen_sids.add(sid)
        items.append({"sid": sid, "dataset": "ui_lens", "split": "", "file": rel(root, p), "label": 1 if pos else 0, "types": types,
                      "boxes": [b + [r["task"]] for b in r["boxes"]], "device_key": "phone", "ui_lens_task": r["task"], "app": r["app"], "image_size": r["size"]})
    log(f"UI-Lens：{len(items)} 条（页 × 任务），{len({i['file'] for i in items})} 张图，按任务：{dict(by_task)}" + (f"，{missing} 条找不到图片" if missing else ""))
    # Dev split: 300 pages stratified by app; every (page, task) row of a Dev page goes to Dev
    rng = random.Random(SEED)
    pages = sorted({i["file"] for i in items}); by_app = defaultdict(set)
    for i in items: by_app[i["app"]].add(i["file"])
    dev_target = min(300, max(1, len(pages) // 7))   # 300 pages of the real set (4,566 pages); never more than a seventh of a small set
    dev_pages = set(); apps = sorted(by_app); per_app = max(1, dev_target // max(1, len(apps)))
    for a in apps:
        cand = sorted(by_app[a]); rng.shuffle(cand); dev_pages.update(cand[:per_app])
    while len(dev_pages) < dev_target: dev_pages.add(rng.choice(pages))
    for i in items:
        i["split"] = "dev" if i["file"] in dev_pages else "test"
        if i["split"] == "dev": i["dataset"] = "ui_lens_dev"
    log(f"UI-Lens：Dev {len(dev_pages)} 页（只用来调参），Test {len(pages) - len(dev_pages)} 页")
    pages_list = sorted({i["file"] for i in items}); P.ds("ui_lens", state="building", detail="计算每页的校验和", done=0, total=len(pages_list))
    for n_done, fp in enumerate(pages_list, 1):
        md5(root / fp if (root / fp).exists() else fp)
        if n_done % 200 == 0 or n_done == len(pages_list): P.count("ui_lens", n_done, len(pages_list), detail=f"计算校验和：{n_done} / {len(pages_list)} 页")
    for i in items: i["md5"] = md5(root / i["file"]) if (root / i["file"]).exists() else md5(i["file"])
    return items, dict(by_task)


def get_ui_lens(root, log):
    """download the single-UI part of UI-Lens with huggingface_hub (needs HF_TOKEN; HF_ENDPOINT may be a mirror)"""
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        log("需要 huggingface_hub：pip install huggingface_hub"); return False
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok and not (root / "ui_lens" / ".complete").exists(): log("没有 HF_TOKEN：先在 https://huggingface.co/datasets/wuhuohua/UI_lens 点同意条款，再 export HF_TOKEN=hf_xxx"); return False
    dest = root / "ui_lens"; dest.mkdir(parents=True, exist_ok=True)
    if not (dest / ".complete").exists() and any(dest.rglob("label_for_single_UIs*")):   # downloaded here in an earlier run: check it, no network
        n_lab0, n_img0 = ui_lens_fixup(dest, lambda *a: None)
        if n_lab0 and n_img0 >= int(os.environ.get("RS_UL_MIN_IMAGES", "4000")):
            (dest / ".complete").write_text(json.dumps({"files": n_img0, "source": "autodl"}))
            log(f"UI-Lens 已在这里（标注文件 {n_lab0} 个，图片 {n_img0} 张），不再下载")
            P.ds("ui_lens", state="downloaded", reason="", label_files=n_lab0, images=n_img0, source="AutoDL"); return True
    if (dest / ".complete").exists():   # uploaded from Windows (downloaded there through huggingface.co): nothing to download here
        log("UI-Lens 已从本机下载并上传到这里，不再下载")
        P.ds("ui_lens", state="unpacking", source="本机上传"); n_lab, n_img = ui_lens_fixup(dest, log)
        if n_lab and n_img:
            P.ds("ui_lens", state="downloaded", reason="", label_files=n_lab, images=n_img, source="本机上传"); return True
        log("  本机上传的 UI-Lens 不完整（没有标注或图片）：作废，改为在这里下载"); (dest / ".complete").unlink()
    pats = None   # everything except the sequence part (Seq_UIs*), which this study does not use
    ignore = ["Seq_UIs*", "Seq_UIs*/**", "seq_UIs*", "seq_UIs*/**"]
    expected = 6.85e9
    try:   # the real size of what will be downloaded, when the hub answers
        from huggingface_hub import HfApi
        import fnmatch
        info = HfApi(endpoint=S.HF_ENDPOINT, token=tok).dataset_info(S.UI_LENS_REPO, files_metadata=True)
        tot = sum((f.size or 0) for f in info.siblings if not f.rfilename.lower().startswith("seq_uis") and (pats is None or any(fnmatch.fnmatch(f.rfilename, pt) for pt in pats)))
        if tot: expected = tot
    except Exception as e:
        log(f"  （读不到 UI-Lens 的文件清单：{str(e)[:80]}；按 6.85 GB 估计）")
    log(f"下载 UI-Lens 单页部分到 {dest}（{S.HF_ENDPOINT}，约 {expected / 1e9:.2f} GB）…")
    P.stage("fetch", "下载 UI-Lens（主测试集，约 %.1f GB）" % (expected / 1e9)); P.ds("ui_lens", state="downloading", expected_bytes=expected)
    try:
        with P.Watch("ui_lens", dest, expected):
            snapshot_download(repo_id=S.UI_LENS_REPO, repo_type="dataset", local_dir=str(dest), token=tok, endpoint=S.HF_ENDPOINT, allow_patterns=pats, ignore_patterns=ignore, max_workers=8)
    except Exception as e:
        msg = str(e)
        why = "token 没有下载权限：多半是还没在 UI-Lens 页面点同意条款" if ("401" in msg or "403" in msg or "gated" in msg.lower()) else f"下载出错：{msg[:120]}"
        log(f"UI-Lens 下载失败：{why}"); P.ds("ui_lens", state="failed", reason=why); return False
    P.ds("ui_lens", state="unpacking"); n_lab, n_img = ui_lens_fixup(dest, log)
    if n_lab > 0 and n_img >= 4000: (dest / ".complete").write_text(json.dumps({"files": n_img, "source": "autodl"}))   # next runs skip the download
    P.ds("ui_lens", state="downloaded" if (n_lab > 0 and n_img > 0) else "failed", label_files=n_lab, images=n_img, reason="" if n_lab and n_img else "下载完成但没有找到标注或图片")
    return n_lab > 0 and n_img > 0


def ui_lens_fixup(dest, log):
    """after the download: unpack any archives in place (their own folders decide where files land), and report what is there"""
    import zipfile
    for z in sorted(dest.rglob("*.zip")):
        marker = z.with_suffix(".unzipped")
        if marker.exists(): continue
        log(f"  解压 {z.relative_to(dest)} …"); zipfile.ZipFile(z).extractall(z.parent); marker.write_text("ok")
        try: z.unlink()   # the archive is not needed once unpacked: it would double the space
        except OSError: pass
    lab = dest / "label_for_single_UIs"   # kept for older uploads; the loader finds label_for_single_UIs* anywhere
    labs = [p for p in dest.rglob("label_for_single_UIs*") if p.is_dir()]
    n_lab = sum(len(list(l.rglob("*.json"))) + len(list(l.rglob("*.jsonl"))) for l in labs)
    n_img = sum(1 for p in dest.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    log(f"UI-Lens 下载完成：标注文件 {n_lab} 个，图片 {n_img} 张")
    return n_lab, n_img


# ----------------------------------------------------------------------------- UICrit (+ Rico screens)
BOX_RE = re.compile(r"Bounding Box:\s*\[([^\]]+)\]")


def load_uicrit(root, rico_dir, log, max_screens=0):
    """critiques with boxes -> one manifest item per screen with the rule ids its critiques support (strong label: named by >= 2 of the 3 designers)"""
    csv_p = root / "uicrit_public.csv"
    if not csv_p.exists():
        try:
            st, body, _ = S._get(S.UICRIT_CSV, timeout=120); csv_p.write_bytes(body)
        except Exception as e:
            log(f"UICrit：下载 CSV 失败（{str(e)[:80]}）"); return []
    rows = list(csv.DictReader(open(csv_p, encoding="utf-8", errors="replace")))
    per_screen = defaultdict(lambda: {"annotators": 0, "hits": defaultdict(list), "ratings": []})
    for r in rows:
        sid = r["rico_id"]; d = per_screen[sid]; d["annotators"] += 1
        try: d["ratings"].append(float(r.get("design_quality_rating") or 0))
        except Exception: pass
        raw = r.get("comments") or ""
        try: items = ast.literal_eval(raw)
        except Exception: items = [raw]
        for it in (items if isinstance(items, list) else [items]):
            txt = str(it); low = txt.lower()
            m = BOX_RE.search(txt); box = None
            if m:
                try:
                    v = [float(x) for x in m.group(1).split(",")]
                    if len(v) == 4: box = v   # normalized x1, y1, x2, y2
                except Exception:
                    pass
            for rid, pat in UICRIT_KINDS:
                if re.search(pat, low): d["hits"][rid].append(box)
    items, missing = [], 0
    from PIL import Image
    for sid, d in sorted(per_screen.items()):
        img = rico_dir / f"{sid}.jpg"
        if not img.exists(): img = rico_dir / f"{sid}.png"
        if not img.exists(): missing += 1; continue
        W, H = Image.open(img).size
        types, boxes = [], []
        for rid, bl in d["hits"].items():
            types.append(rid)
            for b in bl:
                if b: boxes.append([int(b[0] * W), int(b[1] * H), max(1, int((b[2] - b[0]) * W)), max(1, int((b[3] - b[1]) * H)), rid])
        strong = [rid for rid, bl in d["hits"].items() if len(bl) >= 2]
        items.append({"sid": f"UC-{sid}", "dataset": "uicrit", "split": "test", "file": rel(root, img), "label": 1 if types else 0, "types": sorted(types), "strong_types": sorted(strong),
                      "boxes": boxes, "device_key": "phone", "rico_id": sid, "annotators": d["annotators"], "design_quality": (sum(d["ratings"]) / len(d["ratings"])) if d["ratings"] else None, "md5": md5(img)})
        if max_screens and len(items) >= max_screens: break
    log(f"UICrit：{len(items)} 张界面有截图" + (f"，{missing} 张缺 Rico 截图（先 python fetch_data.py --get-rico）" if missing else "") + f"；带判据类型的 {sum(1 for i in items if i['types'])} 张")
    return items


def _csv_ids(p):
    try:
        return {r.get("rico_id") for r in csv.DictReader(open(p, encoding="utf-8", errors="replace")) if r.get("rico_id")}
    except Exception:
        return set()


def ensure_uicrit_csv(root, log=print):
    """uicrit_public.csv with at least 900 screen ids: the file already there, else the GitHub download, else the copy shipped with the executor"""
    csv_p = root / "uicrit_public.csv"; bundled = HERE / "bundled" / "uicrit_public.csv"
    if csv_p.exists() and len(_csv_ids(csv_p)) >= 900: return csv_p
    if csv_p.exists(): log(f"UICrit 列表文件不完整（{len(_csv_ids(csv_p))} 个屏号），重新获取")
    try:
        st, body, _ = S._get(S.UICRIT_CSV, timeout=120)
        tmp = csv_p.with_suffix(".tmp"); tmp.write_bytes(body)
        if len(_csv_ids(tmp)) >= 900: tmp.replace(csv_p); log(f"UICrit 列表：GitHub 下载，{len(_csv_ids(csv_p))} 个屏"); return csv_p
        tmp.unlink(missing_ok=True)
    except Exception as e:
        log(f"UICrit 列表：GitHub 不可用（{str(e)[:60]}），用执行器自带的副本")
    if bundled.exists():
        csv_p.write_bytes(bundled.read_bytes()); log(f"UICrit 列表：执行器自带副本，{len(_csv_ids(csv_p))} 个屏"); return csv_p
    return csv_p


def uicrit_rico_ids(root):
    csv_p = ensure_uicrit_csv(root, log=lambda *a: None)
    return _csv_ids(csv_p) if csv_p.exists() else set()


# ----------------------------------------------------------------------------- Rico
def rico_from_voxel51(root, rico_dir, want, n_extra, have, log):
    """Voxel51/rico (FiftyOne export: data/data_0 … data_66, each file named by its Rico id; samples.json with the element boxes).
    Only the needed files are fetched: UICrit's screens plus n_extra others, 16 at a time with curl (what ran on the AutoDL instance)."""
    import subprocess, hashlib, threading, concurrent.futures
    V = "Voxel51/rico"; base = S.HF_ENDPOINT
    def get_json(url):
        last = None
        for _ in range(4):
            try:
                with S._DIRECT.open(urllib.request.Request(url, headers={"User-Agent": "RuleSight/1.0"}), timeout=60) as r: return json.loads(r.read().decode("utf-8"))
            except Exception as e:
                last = e; time.sleep(2)
        raise last
    P.ds("rico", state="downloading", detail="清点 Voxel51/rico 的文件", source="Voxel51/rico")
    top = get_json(f"{base}/api/datasets/{V}/tree/main/data")
    folders = sorted([x["path"] for x in top if x.get("type") == "directory"], key=lambda s: int(re.sub(r"\D", "", s.rsplit("/", 1)[-1]) or 0))
    if not folders: log("Rico（Voxel51/rico）：没有列出目录"); return False
    id2 = {}
    for k, fo in enumerate(folders):
        for x in get_json(f"{base}/api/datasets/{V}/tree/main/{fo}"):
            if x.get("type") != "file": continue
            name = x["path"].rsplit("/", 1)[-1]; stem, ext = os.path.splitext(name)
            if stem.isdigit() and ext.lower() in (".jpg", ".jpeg", ".png"): id2[stem] = (x["path"], int(x.get("size") or 0), ext.lower().replace(".jpeg", ".jpg"))
        if k == 0 and not id2: log("Rico（Voxel51/rico）：文件名不是屏号"); return False
        if k % 10 == 0: P.ds("rico", detail=f"清点 Voxel51/rico：{k + 1}/{len(folders)} 个目录")
    cover = want & set(id2)
    log(f"Rico（Voxel51/rico）：共 {len(id2)} 张带屏号；UICrit 需要的 {len(want)} 屏里有 {len(cover)} 屏")
    if len(cover) < 0.9 * len(want): return False
    imgs = {p.stem for p in rico_dir.glob("*.jpg")} | {p.stem for p in rico_dir.glob("*.png")}
    others = sorted(set(id2) - want, key=lambda s: hashlib.md5((s + str(SEED)).encode()).hexdigest())
    need_extra = max(0, n_extra - len(imgs - want))
    pick = [s for s in sorted(cover) if s not in imgs] + [s for s in others if s not in imgs][:need_extra]
    total = sum(id2[s][1] for s in pick); env0 = {k_: v_ for k_, v_ in os.environ.items() if k_.lower() not in ("http_proxy", "https_proxy", "all_proxy")}
    log(f"Rico（Voxel51/rico）：要下载 {len(pick)} 张（{total / 1e6:,.0f} MB），16 张同时下")
    st = {"n": 0, "b": 0, "fail": 0}; lock = threading.Lock(); t0 = time.time()
    def one(s):
        path, size, ext = id2[s]; out = rico_dir / f"{s}{ext}"; tmp = out.with_suffix(out.suffix + ".part")
        r = subprocess.run(["curl", "-sfL", "--retry", "6", "--retry-delay", "2", "--connect-timeout", "30", "-o", str(tmp), f"{base}/datasets/{V}/resolve/main/{path}"], env=env0, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        with lock:
            if r.returncode == 0 and tmp.exists(): tmp.replace(out); st["n"] += 1; st["b"] += size
            else:
                st["fail"] += 1
                if st["fail"] <= 5: log(f"  没下到 {s}（{(r.stderr or b'').decode('utf-8', 'replace').strip()[-80:] or r.returncode}）")
    stop = threading.Event()
    def report():
        last_log = 0.0
        while not stop.wait(2):
            el = max(1e-6, time.time() - t0); rate = st["b"] / el
            P.ds("rico", state="downloading", detail=f"Voxel51/rico：{st['n']} / {len(pick)} 张", bytes=st["b"], expected_bytes=total, rate_bps=rate, eta_s=int((total - st["b"]) / rate) if rate > 1e3 else None, reason="")
            if time.time() - last_log >= 15: last_log = time.time(); log(f"  已下 {st['n']} / {len(pick)} 张，{st['b'] / 1e6:,.0f} / {total / 1e6:,.0f} MB，{rate / 1e6:.1f} MB/s")
    rep_t = threading.Thread(target=report, daemon=True); rep_t.start()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex: list(ex.map(one, pick))
    stop.set(); rep_t.join(timeout=3)
    log(f"Rico（Voxel51/rico）：下完 {st['n']} 张" + (f"，{st['fail']} 张没下到" if st["fail"] else ""))
    # element boxes from samples.json; a screen without one gets an empty hierarchy (the injector then finds elements in the image)
    idx = {}
    try:
        tree = get_json(f"{base}/api/datasets/{V}/tree/main"); sj = next((x for x in tree if x.get("path") == "samples.json"), None)
        if sj and 0 < int(sj.get("size") or 0) < 3e9:
            tmpd = root / "rico_tmp"; tmpd.mkdir(exist_ok=True); sjp = tmpd / "samples.json"
            log(f"Rico：下载元素框 samples.json（{int(sj['size']) / 1e6:,.0f} MB）")
            subprocess.run(["curl", "-sfL", "--retry", "6", "-o", str(sjp), f"{base}/datasets/{V}/resolve/main/samples.json"], env=env0, check=True)
            data = json.load(open(sjp, encoding="utf-8"))
            for s_ in (data.get("samples", []) if isinstance(data, dict) else data):
                stem = Path(str(s_.get("filepath", ""))).stem
                if stem.isdigit(): idx[stem] = s_
            sjp.unlink(missing_ok=True)
    except Exception as e:
        log(f"Rico：samples.json 用不了（{str(e)[:80]}），层级留空")
    n_box = 0
    for img in list(rico_dir.glob("*.jpg")) + list(rico_dir.glob("*.png")):
        js = rico_dir / f"{img.stem}.json"
        if js.exists(): continue
        kids = []
        for val in (idx.get(img.stem) or {}).values():
            if isinstance(val, dict) and val.get("_cls") == "Detections":
                for det in val.get("detections") or []:
                    bb = det.get("bounding_box") or []
                    if len(bb) != 4: continue
                    x, y, w, h = bb; lab = str(det.get("label") or "")
                    kids.append({"bounds": [int(x * 1440), int(y * 2560), int((x + w) * 1440), int((y + h) * 2560)], "class": str(det.get("type") or lab), "text": "",
                                 "componentLabel": lab, "clickable": bool(det.get("clickable")), "visible-to-user": True, "children": []})
        if kids: n_box += 1
        js.write_text(json.dumps({"activity": {"root": {"bounds": [0, 0, 1440, 2560], "class": "root", "children": kids}}}), encoding="utf-8")
    have2 = {p.stem for p in rico_dir.glob("*.json")} & ({p.stem for p in rico_dir.glob("*.jpg")} | {p.stem for p in rico_dir.glob("*.png")})
    log(f"Rico（Voxel51/rico）：UICrit 的屏 {len(want & have2)}/{len(want)}，底图 {len(have2 - want)} 屏；{n_box} 屏带元素框")
    return len(want & have2) >= 0.98 * len(want)


def get_rico(root, rico_dir, want_ids, n_extra, log):
    P.stage("fetch", "下载 Rico 截图（UICrit 的 1,000 屏 + 合成用的屏）"); P.ds("rico", state="downloading", wanted=len(want_ids) + n_extra)
    try:
        _get_rico(root, rico_dir, want_ids, n_extra, log)
    finally:
        rd = Path(rico_dir); have = ({p.stem for p in rd.glob("*.json")} & ({p.stem for p in rd.glob("*.jpg")} | {p.stem for p in rd.glob("*.png")})) if rd.exists() else set()
        w = set(want_ids); cov = len(w & have); ok = bool(w) and cov >= 0.98 * len(w)
        P.ds("rico", state="done" if ok else "failed", screens=len(have), uicrit_have=cov, uicrit_total=len(w),
             reason="" if ok else f"Rico 只拿到 UICrit 的 {cov}/{len(w)} 屏：再点继续会补齐")


def _get_rico(root, rico_dir, want_ids, n_extra, log):
    """Fill rico_dir with <id>.jpg + <id>.json: the UICrit screens plus n_extra screens for synthesis.
    Google Storage tarball when reachable (streamed, only the wanted members kept), else HuggingFace RICO-Screen2Words parquet."""
    rico_dir.mkdir(parents=True, exist_ok=True)
    stems_j = {p.stem for p in rico_dir.glob("*.json")}; stems_i = {p.stem for p in rico_dir.glob("*.jpg")} | {p.stem for p in rico_dir.glob("*.png")}
    broken = (stems_j ^ stems_i)
    for st in broken:   # a screen needs both its screenshot and its hierarchy; leftovers of an interrupted run are cleared
        for ext in (".json", ".jpg", ".png"):
            try: (rico_dir / f"{st}{ext}").unlink()
            except OSError: pass
    if broken: log(f"Rico：清掉 {len(broken)} 个只有一半文件的屏")
    have = stems_j & stems_i
    need = set(want_ids) - have
    P.ds("rico", screens=len(have), uicrit_have=len(set(want_ids) & have), uicrit_total=len(want_ids))
    if not need and len(have) >= len(want_ids) + n_extra: log(f"Rico：已有 {len(have)} 屏"); return
    if not need and len(have) >= len(want_ids) + 1500: log(f"Rico：已有 {len(have)} 屏（UICrit 的都在，注入用的够了）"); return
    try:
        if rico_from_voxel51(root, rico_dir, set(want_ids), n_extra, have, log): return
    except Exception as e:
        log(f"Rico（Voxel51/rico）：{str(e)[:120]}；改用其他来源")
    have = {p.stem for p in rico_dir.glob("*.json")} & ({p.stem for p in rico_dir.glob("*.jpg")} | {p.stem for p in rico_dir.glob("*.png")}); need = set(want_ids) - have
    try:
        st0, h0 = S._head(S.RICO_TAR); total = int(h0.get("Content-Length") or 6.0e9)
        log(f"Rico：从 Google 存储流式读取 unique_uis.tar.gz（{total / 1e9:.1f} GB），只保留需要的屏…")
        req = urllib.request.Request(S.RICO_TAR, headers={"User-Agent": "RuleSight/1.0"})

        class _Count:   # counts the bytes read, for the progress bar and the time left
            def __init__(self, f): self.f, self.n, self.t = f, 0, 0.0
            def read(self, k=-1):
                b = self.f.read(k); self.n += len(b)
                if time.time() - self.t > 3:
                    self.t = time.time(); P.count("rico", self.n, total, unit="bytes", screens=sum(1 for _ in rico_dir.glob("*.json")))
                return b
        with urllib.request.urlopen(req, timeout=120) as r0:
            r = _Count(r0)
            with tarfile.open(fileobj=r, mode="r|gz") as tf:
                extra_taken = 0
                for m in tf:
                    name = Path(m.name).name; stem = Path(name).stem
                    if not (name.endswith(".jpg") or name.endswith(".json")): continue
                    take = stem in need or (extra_taken < n_extra * 2 and stem not in have)
                    if not take: continue
                    f = tf.extractfile(m)
                    if f is None: continue
                    (rico_dir / name).write_bytes(f.read())
                    if stem not in need: extra_taken += 1
        log(f"Rico：现有 {len(list(rico_dir.glob('*.json')))} 屏"); return
    except Exception as e:
        log(f"Rico（Google 存储）不可用（{str(e)[:60]}），改用 HuggingFace {S.RICO_HF_REPO}")
    for repo, patterns in ((S.RICO_HF_FULL, ["ui-screenshots-and-view-hierarchies/*.parquet", "*.md"]), (S.RICO_HF_REPO, ["*.parquet", "**/*.parquet", "*.md"])):
        try:
            from huggingface_hub import snapshot_download
            dest = root / ("rico_hf_" + repo.split("/")[0]); dest.mkdir(parents=True, exist_ok=True)
            expected, names, sizes = None, [], {}
            from huggingface_hub import HfApi
            import fnmatch
            for ep in (S.HF_ENDPOINT, "https://huggingface.co"):   # the file list from the mirror, else from the official site (through the academic proxy)
                try:
                    info = HfApi(endpoint=ep, token=os.environ.get("HF_TOKEN")).dataset_info(repo, files_metadata=True)
                    names = [f.rfilename for f in info.siblings if any(fnmatch.fnmatch(f.rfilename, pt) for pt in patterns)]
                    sizes = {f.rfilename: int(f.size or 0) for f in info.siblings if f.rfilename in names}
                    expected = sum(sizes.values()) or None
                    if names: log(f"Rico 文件清单（{ep}）：{len(names)} 个分片，共 {(expected or 0) / 1e9:.2f} GB"); break
                except Exception as e:
                    log(f"  （{ep} 没有给出文件清单：{str(e)[:60]}）")
            if not expected and names:   # sizes missing from the listing: ask the mirror for each shard's length
                tot = 0
                for nm in names:
                    try:
                        st0, h0 = S._head(f"{S.HF_ENDPOINT}/datasets/{repo}/resolve/main/{nm}", headers={"User-Agent": "RuleSight/1.0", **S.hf_headers()}); tot += int(h0.get("Content-Length") or h0.get("X-Linked-Size") or 0)
                    except Exception:
                        pass
                expected = tot or None
            P.ds("rico", state="downloading", bytes=0, expected_bytes=expected or 0, source=f"HuggingFace {repo.split('/')[0]}", shards=len(names) or None)
            log(f"Rico：从 HuggingFace {repo} 逐个分片下载（{S.HF_ENDPOINT}" + (f"，共约 {expected / 1e9:.1f} GB" if expected else "") + "），每片取出需要的屏后即删，磁盘上同时只放一片…")
            P.stage("fetch", f"下载 Rico（HuggingFace 镜像" + (f"，约 {expected / 1e9:.1f} GB，{len(names)} 个分片" if expected else "") + "）")
            import shutil
            got = set(); done_bytes = 0; t0 = time.time()
            # one file at a time with curl, straight through the mirror: the way that ran at 5–23 MB/s on the AutoDL instance (Washu's own log,
            # 18 shards in 8 minutes). Progress is read from the growing file every 2 s; a line goes to the log every 15 s; a file that stops
            # growing for 2 minutes is resumed (-C -), three times at most.
            if not names:   # no listing: let snapshot_download fetch everything at once (the old way)
                with P.Watch("rico", dest, expected or 0):
                    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(dest), endpoint=S.HF_ENDPOINT, token=os.environ.get("HF_TOKEN"), allow_patterns=patterns, max_workers=8)
                got = rico_from_parquet(dest, rico_dir, need, n_extra, have, log)
            import subprocess
            env0 = {k_: v_ for k_, v_ in os.environ.items() if k_.lower() not in ("http_proxy", "https_proxy", "all_proxy")}
            hist = []   # (time, total bytes) for a speed over the last ~10 s
            for k, nm in enumerate(names):
                shard_dir = dest / f"shard_{k}"; shard_dir.mkdir(parents=True, exist_ok=True); out = shard_dir / Path(nm).name
                url = f"{S.HF_ENDPOINT}/datasets/{repo}/resolve/main/{nm}"; sz = sizes.get(nm) or 0
                log(f"Rico 分片 {k + 1}/{len(names)}：{nm}（{sz / 1e6:,.0f} MB）")
                ok = False; last_err = ""
                for attempt in range(1, 4):
                    proc = subprocess.Popen(["curl", "-sSfL", "--retry", "8", "--retry-delay", "3", "--connect-timeout", "30", "-C", "-", "-o", str(out), url],
                                            env=env0, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    t_last_grow = time.time(); last_size = out.stat().st_size if out.exists() else 0; t_log = time.time()
                    while proc.poll() is None:
                        time.sleep(2)
                        cur = out.stat().st_size if out.exists() else 0; now = time.time()
                        if cur > last_size: last_size = cur; t_last_grow = now
                        tot = done_bytes + cur; hist.append((now, tot)); hist[:] = [x for x in hist if now - x[0] <= 12]
                        rate = (hist[-1][1] - hist[0][1]) / max(1e-6, hist[-1][0] - hist[0][0]) if len(hist) > 1 else None
                        eta = int(((expected or 0) - tot) / rate) if expected and rate and rate > 1e3 else None
                        P.ds("rico", state="downloading", detail=f"分片 {k + 1}/{len(names)}", bytes=tot, expected_bytes=expected or 0, rate_bps=rate, eta_s=eta, shards_done=k, reason="")
                        if now - t_log >= 15:
                            t_log = now; log(f"  已下 {tot / 1e6:,.0f} / {(expected or 0) / 1e6:,.0f} MB" + (f"，{rate / 1e6:.1f} MB/s" if rate else "") + (f"，最多还需约 {max(1, eta // 60)} 分钟" if eta else ""))
                        if now - t_last_grow > 120:   # stalled: stop this try, resume in the next
                            proc.kill(); last_err = "两分钟没有数据"; log(f"  分片 {k + 1}：两分钟没有数据，续传"); break
                    rc = proc.wait()
                    if rc == 0 and out.exists() and (not sz or out.stat().st_size == sz): ok = True; break
                    if rc != 0 and not last_err: last_err = (proc.stderr.read() or b"").decode("utf-8", "replace").strip()[-160:] or f"curl 退出码 {rc}"
                    log(f"  分片 {k + 1} 第 {attempt} 次没下完（{last_err}）" + ("，重试" if attempt < 3 else "")); last_err = ""
                if not ok:
                    msg = f"Rico 分片 {k + 1}/{len(names)} 三次都没下完：请看日志"
                    log(f"Rico：{msg}"); P.ds("rico", state="failed", reason=msg); break
                done_bytes += out.stat().st_size
                g = rico_from_parquet(shard_dir, rico_dir, need - got, max(0, n_extra - sum(1 for x in got if x not in need)), have | got, log)
                got |= g; shutil.rmtree(shard_dir, ignore_errors=True)
                n_cov = len(set(want_ids) & (have | got)); n_ext = len((have | got) - set(want_ids))
                P.ds("rico", state="downloading", bytes=done_bytes, expected_bytes=expected or 0, shards_done=k + 1, screens=n_cov + n_ext, uicrit_have=n_cov, uicrit_total=len(want_ids))
                log(f"  分片 {k + 1}/{len(names)} 完成：UICrit 的屏 {n_cov}/{len(want_ids)}，底图 {n_ext} 屏")
                if not (need - got) and sum(1 for x in got if x not in need) >= n_extra: log("  需要的屏都齐了，不再下载其余分片"); break
            have |= got; need -= got
            shutil.rmtree(dest, ignore_errors=True)
            log(f"Rico（{repo}）：取到 {len(got)} 屏；现有 {len(list(rico_dir.glob('*.json')))} 屏；UICrit 还缺 {len(need)} 屏")
            if not need: return
        except Exception as e:
            log(f"Rico（HuggingFace {repo}）失败（{str(e)[:100]}）：需要 pip install huggingface_hub pyarrow，并能访问 {S.HF_ENDPOINT}")


def hf_rico_json(row):
    """creative-graphic-design/Rico (ui-screenshots-and-view-hierarchies) stores the hierarchy as activity.root + activity.children
    (nodes grouped by level, no text field): rebuild the Rico-style JSON our readers expect (bounds, class, componentLabel, children)"""
    act = row.get("activity") or {}
    def conv(n):
        k = str(n.get("klass") or "")
        lab = "Text Button" if "Button" in k else "Text" if ("TextView" in k or "EditText" in k) else "Image" if "Image" in k else ""
        return {"bounds": list(n.get("bounds") or []), "class": k, "text": "", "componentLabel": lab, "clickable": bool(n.get("clickable")),
                "visible-to-user": bool(n.get("visible_to_user", True)), "visibility": n.get("visibility"), "content-desc": n.get("content_desc"),
                "resource-id": n.get("resource_id"), "children": []}
    root = conv(act.get("root") or {})
    for level in act.get("children") or []:
        for n in (level or []):
            if isinstance(n, dict): root["children"].append(conv(n))
    return {"activity_name": row.get("activity_name"), "activity": {"root": root}, "is_keyboard_deployed": row.get("is_keyboard_deployed"), "request_id": row.get("request_id")}


def rico_from_parquet(src, rico_dir, need, n_extra, have, log):
    """write <id>.jpg + <id>.json for the wanted screens out of Rico parquet shards. The screen id is an id column when there is one
    (Screen2Words: screenId), else the screenshot's file name (creative-graphic-design/Rico keeps "<ui_number>.jpg" as the image path)."""
    import pyarrow.parquet as pq
    got, extra = set(), 0
    for pf in sorted(Path(src).rglob("*.parquet")):
        f = pq.ParquetFile(pf); cols = f.schema_arrow.names
        idc = next((c for c in cols if c.lower() in ("screenid", "screen_id", "ui_number", "id", "rico_id")), None)
        imc = next((c for c in cols if c.lower() in ("image", "screenshot", "img")), None)
        vhc = next((c for c in cols if "hierarchy" in c.lower()), None)
        hf_act = "activity" in cols
        if not imc: log(f"  {pf.name}：列 {cols[:8]} 里没有图片列，跳过"); continue
        read = [c for c in (idc, imc, vhc) if c] + ([c for c in ("activity", "activity_name", "is_keyboard_deployed", "request_id") if c in cols] if hf_act else [])
        shown = False; n_rows = 0; no_id = 0
        for batch in f.iter_batches(batch_size=128, columns=read):
            for row in batch.to_pylist():
                n_rows += 1
                img = row[imc]; data = img.get("bytes") if isinstance(img, dict) else img
                if idc: sid = str(row[idc])
                else:
                    path = img.get("path") if isinstance(img, dict) else None
                    sid = Path(str(path)).stem if path else ""
                    if not sid.isdigit(): no_id += 1; continue
                if not shown:
                    log(f"  {pf.name}：屏号取自" + (f"列 {idc}" if idc else f"截图文件名（例：{img.get('path')} → {sid}）") + f"，层级取自" + ("activity" if hf_act else (vhc or "无")))
                    shown = True
                keep = sid in need or (extra < n_extra and sid not in have and sid not in got)
                if not keep or not data: continue
                (rico_dir / f"{sid}.jpg").write_bytes(data)
                if hf_act: vh = json.dumps(hf_rico_json(row), ensure_ascii=False)
                else:
                    vh = row.get(vhc) if vhc else None; vh = vh if isinstance(vh, str) else json.dumps(vh or {})
                (rico_dir / f"{sid}.json").write_text(vh, encoding="utf-8")
                got.add(sid)
                if sid not in need: extra += 1
        if no_id: log(f"  {pf.name}：{no_id} / {n_rows} 行拿不到屏号（截图没有文件名）")
    return got


# ----------------------------------------------------------------------------- GLIB
def load_glib(root, log, n_normal=200, url=None):
    """testDataSet (real display issues from game test reports) + a matched sample of Normal, read straight out of the Zenodo zip"""
    dest = root / "glib"; dest.mkdir(parents=True, exist_ok=True)
    man_p = dest / "glib_items.json"
    if man_p.exists() and (dest / ".complete").exists():
        items = json.loads(man_p.read_text(encoding="utf-8")); log(f"GLIB：已有 {len(items)} 张"); P.ds("glib", state="done", files=len(items), defective=sum(1 for i in items if i["label"]), reason=""); return items
    # files already here (uploaded from Windows, or a run that stopped after fetching them): no network needed
    have_test = sorted((dest / "test_").glob("*")) if (dest / "test_").exists() else []
    have_norm = sorted((dest / "normal_").glob("*")) if (dest / "normal_").exists() else []
    complete_marker = dest / ".complete"
    if complete_marker.exists() and len(have_test) >= 180 and len(have_norm) >= 180:   # a finished set (uploaded from Windows or fetched here)
        items = [{"sid": f"GL-{p.stem}", "dataset": "glib", "split": "test", "file": rel(root, p), "label": 1 if p.parent.name == "test_" else 0, "types": [], "boxes": [], "device_key": "desktop", "md5": md5(p)} for p in have_test + have_norm[:n_normal] if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        man_p.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        log(f"GLIB：用已有的 {len(items)} 张（{len(have_test)} 有缺陷）"); P.ds("glib", state="done", files=len(items), expected=len(items), defective=len(have_test), reason=""); return items
    if have_test or have_norm: log(f"GLIB：已有 {len(have_test)} 张缺陷、{len(have_norm)} 张正常，补齐其余的")
    rz = None; errs = []
    tries = ([("test-url", lambda: (0, url), False)] if url else []) + [("api-direct", lambda: S.zenodo_file(direct=True), True), ("api-proxy", lambda: S.zenodo_file(direct=False), False), ("download-direct", lambda: (0, S.GLIB_ZIP), True), ("download-proxy", lambda: (0, S.GLIB_ZIP), False)]
    for name, info, direct in tries:   # the records API gives the size; ranges are tried without the proxy first (it often strips them)
        try:
            size, link = info(); rz = S.RemoteZip(link, size=size, direct=direct); log(f"GLIB：{name}，{rz.size / 1e9:.1f} GB，{len(rz.names())} 个成员"); break
        except Exception as e:
            errs.append(f"{name}: {str(e)[:70]}"); rz = None
    if rz is None:
        why = "；".join(errs)[:220]
        log(f"GLIB：Zenodo 不可用（{why}）"); P.ds("glib", state="failed", reason=f"Zenodo 几条路都不通（{why}）。可在数据集页点“改用本机下载”，从你的电脑取这 392 张再传上来"); return []
    names = rz.names()
    tests = [n for n in names if "/testDataSet/" in n and n.lower().endswith((".png", ".jpg", ".jpeg"))]
    normals = [n for n in names if "/images/normal/test/" in n.lower() and n.lower().endswith((".png", ".jpg", ".jpeg"))]
    csvs = [n for n in names if "data_csv" in n and "testDataSet" in n and n.lower().endswith(".csv")]
    rng = random.Random(SEED); rng.shuffle(normals)
    log(f"GLIB：压缩包里 testDataSet {len(tests)} 张、Normal {len(normals)} 张、测试 CSV {len(csvs)} 个；取 testDataSet 全部 + Normal {n_normal} 张")
    items = []
    for n in csvs:
        try: (dest / Path(n).name).write_bytes(rz.read(n))
        except Exception: pass
    targets = tests + normals[:n_normal]
    P.ds("glib", state="downloading", files=0, expected=len(targets))
    import concurrent.futures, threading
    lock = threading.Lock(); counter = {"k": 0}
    def one(n):
        p = dest / ("test_" if n in tests else "normal_") / Path(n).name; p.parent.mkdir(exist_ok=True)
        try:
            if not p.exists(): p.write_bytes(rz.read(n))
            item = {"sid": f"GL-{p.stem}", "dataset": "glib", "split": "test", "file": rel(root, p), "label": 1 if n in tests else 0, "types": [], "boxes": [], "device_key": "desktop", "md5": md5(p)}
        except Exception as e:
            log(f"GLIB：{n} 读取失败（{str(e)[:60]}）"); item = None
        with lock:
            counter["k"] += 1
            if counter["k"] % 10 == 0: P.count("glib", counter["k"], len(targets), files=counter["k"], expected=len(targets))
        return item
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:   # eight members at once: one connection to Zenodo is slow, eight are not
        for item in ex.map(one, targets):
            if item: items.append(item)
    n_def = sum(1 for i in items if i["label"]); n_norm = len(items) - n_def
    complete = len(items) >= len(targets) - 2   # a couple of unreadable members at most
    log(f"GLIB：{len(items)} / {len(targets)} 张（{n_def} 有缺陷、{n_norm} 正常）" + ("" if complete else "：没取全，下次运行补齐其余的"))
    if complete:
        man_p.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8"); (dest / ".complete").write_text(str(len(items)))
        P.ds("glib", state="done", files=len(items), defective=n_def, expected=len(targets), reason="")
    else:   # partial: not written as the dataset's list, so the next run completes it instead of taking it as done
        P.ds("glib", state="failed", files=len(items), defective=n_def, expected=len(targets), reason=f"只取到 {len(items)} / {len(targets)} 张（Zenodo 连接不稳）：再点继续会补齐其余的，已取到的不重下")
    return items if complete else []


# ----------------------------------------------------------------------------- ErgoHMI, F-Wild, Rico-synth
def load_ergohmi(root, eg, log):
    items = []
    csvp = next(iter(Path(eg).rglob("pages.csv")), None)
    if not csvp: log("ErgoHMI：导出里没有 pages.csv"); return items
    with open(csvp, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img = Path(eg) / (row.get("render") or row.get("file") or "")
            if not img.exists(): continue
            viol = {k: int(v) for k, v in row.items() if k in ("TC", "NC", "VA", "TS", "RF", "TI", "CV", "AC") and str(v).isdigit()}
            types = [{"TC": "E1", "NC": "E2", "VA": "E3", "TS": "E4", "RF": "E5", "TI": "E6", "CV": "E7", "AC": "E8"}[k] for k, v in viol.items() if v > 0]
            dataset = "e_real" if (row.get("group") or "").startswith("real") else ("e_hard" if row.get("hard") == "1" else "e_defect")
            items.append({"sid": f"EH-{row.get('page_id', len(items))}", "dataset": dataset, "split": row.get("condition", ""), "file": str(img), "label": 1 if types else 0, "types": types, "boxes": [],
                          "device_key": "desktop" if "D1" in (row.get("condition") or "") else "phone", "measured": {k: row.get(k) for k in row if k.endswith(("_mm", "_arcmin", "_ratio", "_de"))}, "md5": md5(img)})
    log(f"ErgoHMI：{len(items)} 页"); return items


def load_fwild(root, log):
    p = root / "fwild" / "fwild.json"
    if not p.exists(): return []
    items = json.loads(p.read_text(encoding="utf-8"))
    for it in items:
        f = Path(it["file"]); it["file"] = rel(root, f) if f.is_absolute() else it["file"]; it["md5"] = md5(root / it["file"])
    log(f"F-Wild：{len(items)} 张"); return items


def load_rico_synth(root, rico_dir, n, log):
    import inject_defects as I
    out = root / "rico_synth"; man = out / "rico_synth.json"
    test_ids = uicrit_rico_ids(root)   # UICrit screens are a test set: never injected
    P.stage("fetch", f"在 Rico 截图上注入缺陷（{n} 屏）"); P.ds("rico_synth", state="building", done=0, total=n)
    items = json.loads(man.read_text(encoding="utf-8")) if man.exists() else I.synthesize(rico_dir, out, n, SEED, exclude=test_ids, on_progress=lambda k, t: P.count("rico_synth", k, t))
    leaked = [it for it in items if str(it.get("rico_id")) in test_ids]
    if leaked: raise SystemExit(f"Rico-synth 含 {len(leaked)} 张 UICrit 测试截图：删掉 {out} 后重新运行")
    for it in items:
        it.update({"dataset": "rico_synth", "split": "synth", "file": rel(root, out / it["file"]), "device_key": "phone"}); it["md5"] = md5(root / it["file"])
    log(f"Rico-synth：{len(items)} 屏（{sum(1 for i in items if i['label'])} 有注入缺陷）")
    P.ds("rico_synth", state="done", done=len(items), total=len(items), defective=sum(1 for i in items if i["label"])); return items


def _disk_guard(fn):
    """a full data disk ends the run with a plain sentence instead of a traceback"""
    def wrapper(*a, **k):
        try: return fn(*a, **k)
        except OSError as e:
            if e.errno in (28, 122):
                print("数据盘满了（Disk quota exceeded）：请在 AutoDL 控制台给数据盘扩容，或删掉 models/ 下不用的模型，再点“继续下载并构建数据”")
                try: P.stage("fetch_failed", "数据盘满了：扩容数据盘或删掉不用的模型后再继续")
                except Exception: pass
                sys.exit(7)
            raise
    return wrapper


@_disk_guard
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data"); ap.add_argument("--rico", default=""); ap.add_argument("--ergohmi", default="")
    ap.add_argument("--skip", default=""); ap.add_argument("--synth-n", type=int, default=2000); ap.add_argument("--rico-extra", type=int, default=2500)
    ap.add_argument("--get-ui-lens", action="store_true"); ap.add_argument("--get-rico", action="store_true"); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    root = Path(a.root).resolve(); root.mkdir(parents=True, exist_ok=True)
    def log(*m):   # to the run log, and to the card of the dataset being worked on
        msg = " ".join(str(x) for x in m); print(msg, flush=True)
        try:
            if msg.startswith("[GLIB] "): P.note(msg[7:], "glib")
            else: P.note(msg)
        except Exception:
            pass
    if a.check: S.check_all(log); return
    P.init(root.parent); P.stage("fetch", "开始准备数据")
    rico_dir = Path(a.rico).resolve() if a.rico else root / "rico"
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    P.reset_queue(["ui_lens", "rico", "uicrit", "glib", "rico_synth"], skip)   # the order of the run; what is done stays done
    glib_job, glib_out = None, []
    if "glib" not in skip:   # Zenodo is another host: GLIB downloads alongside UI-Lens and Rico instead of after them
        import threading
        def _glib():
            try: glib_out.extend(load_glib(root, lambda *m: log("[GLIB] " + " ".join(str(x) for x in m))))
            except Exception as e: log(f"[GLIB] 失败：{str(e)[:80]}")
        glib_job = threading.Thread(target=_glib, daemon=True); glib_job.start()
    P.current("ui_lens")
    if a.get_ui_lens and not get_ui_lens(root, log):
        log("UI-Lens 没有下载成功（原因见上一行）：先把其余数据准备好，主测试集缺失会在清单和控制台标红；修好后再点“下载并构建数据”")
    elif not a.get_ui_lens and "ui_lens" not in skip and not (root / "ui_lens" / "label_for_single_UIs").exists():
        P.ds("ui_lens", state="skipped", reason="没有 HuggingFace token")
    if a.get_rico:
        P.current("rico")
        ensure_uicrit_csv(root, log); want = uicrit_rico_ids(root)
        if len(want) < 900: log(f"UICrit 的屏号列表只有 {len(want)} 个：Rico 的选取会不完整"); P.ds("uicrit", state="failed", reason=f"UICrit 屏号列表只有 {len(want)} 个，GitHub 和自带副本都没拿到完整列表")
        get_rico(root, rico_dir, want, a.rico_extra, log)
    screens, counts, ul_tasks = [], {}, {}
    P.current("ui_lens")
    if "ui_lens" not in skip:
        ul, ul_tasks = load_ui_lens(root, log); screens += ul
        if ul:
            ul_bytes = P.dir_bytes(root / "ui_lens")   # the finished card keeps a full bar with the size on disk
            P.ds("ui_lens", state="done", reason="", pages=len({i["file"] for i in ul}), rows=len(ul), tasks=ul_tasks, bytes=ul_bytes, expected_bytes=ul_bytes)
        if not ul and a.get_ui_lens:
            log("UI-Lens：这次没有下载到任何页（HuggingFace 连不上或没点同意条款）；清单里不会有主测试集，修好后重试")
    P.current("uicrit")
    if "uicrit" not in skip:
        P.stage("fetch", "读取 UICrit 的设计师批评"); uc = load_uicrit(root, rico_dir, log); screens += uc
        P.ds("uicrit", state="done" if uc else "failed", screens=len(uc), with_types=sum(1 for i in uc if i["types"]), reason="" if uc else "没有对应的 Rico 截图")
    if "glib" not in skip:
        if glib_job is not None:
            glib_job.join(); screens += glib_out
        else:
            P.stage("fetch", "从 Zenodo 取 GLIB 的约 400 张截图"); screens += load_glib(root, log)
    P.current("rico_synth")
    if "rico_synth" not in skip and rico_dir.exists() and any(rico_dir.glob("*.json")): screens += load_rico_synth(root, rico_dir, a.synth_n, log)
    if a.ergohmi and "ergohmi" not in skip: screens += load_ergohmi(root, a.ergohmi, log)
    if "f_wild" not in skip: screens += load_fwild(root, log)
    for k in ("ui_lens", "ui_lens_dev", "uicrit", "glib", "rico_synth", "e_defect", "e_hard", "e_real", "f_wild"): counts[k] = sum(1 for s in screens if s["dataset"] == k)
    # UI-Lens-400: a stratified subset for the cross-model table (200 positive / 200 negative rows)
    rng = random.Random(SEED)
    ul_test = [s for s in screens if s["dataset"] == "ui_lens"]
    pos = [s for s in ul_test if s["label"] == 1]; neg = [s for s in ul_test if s["label"] == 0]
    sub = rng.sample(pos, min(200, len(pos))) + rng.sample(neg, min(200, len(neg)))
    for s in sub: s["subsets"] = ["ui_lens_400"]
    # exemplars for the rule cards come only from Rico-synth (never from a test set)
    exemplars = [{"file": s["file"], "rule": s["types"][0], "label": s["types"][0], "box": s["boxes"][0][:4] if s["boxes"] else None, "source": "rico_synth"} for s in screens if s["dataset"] == "rico_synth" and s["types"]]
    exemplars += [{"file": s["file"], "rule": "", "label": "no defect", "source": "rico_synth"} for s in screens if s["dataset"] == "rico_synth" and not s["types"]][:200]
    man = {"name": "RuleSight data v8", "built": time.strftime("%Y-%m-%d"), "seed": SEED, "rulekb": {"version": KB["version"], "rules": len(KB["rules"])}, "counts": counts, "ui_lens_tasks": ul_tasks,
           "n_ui_lens_400": len(sub), "exemplars": exemplars, "screens": screens}
    if "ui_lens" not in skip and counts["ui_lens"] == 0:
        if (root / "ui_lens" / "label_for_single_UIs").exists() or any((root / "ui_lens").glob("*.zip")):
            # downloaded but nothing parsed: a format problem that must be fixed before any run
            log("UI-Lens 下载了但一条都没读出：不写 manifest。把 data/ui_lens/label_for_single_UIs 里的一个文件发来核对格式"); sys.exit(2)
        log("提醒：这次没有 UI-Lens（没有 HuggingFace token，或没有下载）。其余数据照常写入；有 token 后再运行一次下载即可补上")
    if counts["rico_synth"] == 0: log("提醒：Rico-synth 为 0（Rico 截图没拿到），范例库为空，S1/S2/S3 的范例会缺失")
    if counts["uicrit"] == 0: log("提醒：UICrit 为 0（Rico 截图没拿到），E 判据的真实标注集缺失")
    try:
        (root / "RuleSight_data.json").write_text(json.dumps(man, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        if e.errno in (28, 122): log("数据盘满了，清单写不进去：请扩容数据盘或删掉不用的模型后重跑"); P.stage("fetch_failed", "数据盘满了"); sys.exit(7)
        raise
    m = md5(root / "RuleSight_data.json"); (root / "MANIFEST_MD5.txt").write_text(m + "\n", encoding="utf-8")
    log(json.dumps(counts, ensure_ascii=False) + f" | ui_lens_400: {len(sub)} | exemplars: {len(exemplars)} | manifest md5: {m}")
    P.stage("fetch", "生成样张")
    try: P.make_thumbs(root, screens, root.parent)
    except Exception as e: log(f"样张生成失败：{str(e)[:80]}")
    P.stage("fetch_done", "数据准备完成：" + "，".join(f"{k} {v}" for k, v in counts.items() if v))


if __name__ == "__main__":
    main()
