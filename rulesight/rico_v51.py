# Rico screens by their id, for RuleSight. plan: list Voxel51/rico (file per screenshot) and write the download plan;
# hier: build hierarchy files from its samples.json (detections); status: what is in rs_data/data/rico.
import sys, json, csv, os, re, hashlib, urllib.request
from pathlib import Path
D = Path(sys.argv[1]); mode = sys.argv[2]
MIRROR = os.environ.get('MIRROR', 'https://hf-mirror.com'); V = 'Voxel51/rico'
rico = D / 'rico'; tmp = D / 'rico_tmp'; tmp.mkdir(parents=True, exist_ok=True)
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def get_json(url):
    for k in range(4):
        try:
            with op.open(urllib.request.Request(url, headers={'User-Agent': 'RuleSight/1.0'}), timeout=60) as r: return json.loads(r.read().decode('utf-8'))
        except Exception as e:
            last = e
    raise last
want = {r['rico_id'] for r in csv.DictReader(open(D / 'uicrit_public.csv', encoding='utf-8', errors='replace')) if r.get('rico_id')}
def have_now():
    imgs = {p.stem for p in rico.glob('*.jpg')} | {p.stem for p in rico.glob('*.png')}
    return {p.stem for p in rico.glob('*.json')} & imgs, imgs
if mode == 'status':
    j = {p.stem for p in rico.glob('*.json')}; _, imgs = have_now()
    for st in j - imgs: (rico / f'{st}.json').unlink(missing_ok=True)   # a hierarchy without its screenshot
    have, imgs = have_now()
    print(f'  现有：UICrit 的屏 {len(want & imgs)}/{len(want)}，注入底图 {len(imgs - want)} 屏'); sys.exit(0)
if mode == 'plan':
    top = get_json(f'{MIRROR}/api/datasets/{V}/tree/main/data')
    folders = sorted([x['path'] for x in top if x.get('type') == 'directory'], key=lambda s: int(re.sub(r'\D', '', s.rsplit('/', 1)[-1]) or 0))
    if not folders: print(f'  Voxel51/rico 没有列出目录（{str(top)[:120]}）'); sys.exit(3)
    id2path = {}; examples = []
    for k, fo in enumerate(folders):
        for x in get_json(f'{MIRROR}/api/datasets/{V}/tree/main/{fo}'):
            if x.get('type') != 'file': continue
            name = x['path'].rsplit('/', 1)[-1]; stem, ext = os.path.splitext(name)
            if len(examples) < 2: examples.append(name)
            if stem.isdigit() and ext.lower() in ('.jpg', '.jpeg', '.png'): id2path[stem] = x['path']
        if k == 0 and not id2path: print(f'  文件名不是屏号（例：{examples}）'); sys.exit(3)
        if k % 10 == 0 or k == len(folders) - 1: print(f'  清点文件：{k + 1}/{len(folders)} 个目录，{len(id2path)} 张带屏号', flush=True)
    cover = want & set(id2path)
    print(f'  Voxel51/rico 共 {len(id2path)} 张（文件名例：{examples}）；UICrit 需要的 {len(want)} 屏里有 {len(cover)} 屏')
    if len(cover) < 0.9 * len(want): print('  覆盖不到九成'); sys.exit(3)
    _, imgs = have_now()
    others = sorted(set(id2path) - want, key=lambda s: hashlib.md5((s + '20261101').encode()).hexdigest())
    need_extra = max(0, 2500 - len(imgs - want))
    pick = [s for s in sorted(cover) if s not in imgs] + [s for s in others if s not in imgs][:need_extra]
    with open(tmp / 'plan.tsv', 'w') as f:
        for s in pick:
            ext = os.path.splitext(id2path[s])[1].lower().replace('.jpeg', '.jpg')
            f.write(f"{MIRROR}/datasets/{V}/resolve/main/{id2path[s]}\t{rico}/{s}{ext}\n")
    print(f'  要下载 {len(pick)} 张：UICrit {sum(1 for s in pick if s in want)} + 底图 {sum(1 for s in pick if s not in want)}'); sys.exit(0)
if mode == 'hier':   # hierarchy files from samples.json (FiftyOne): detections → nodes; screens without one get an empty hierarchy (the injector then finds elements in the image)
    idx = {}
    try:
        data = json.load(open(tmp / 'samples.json', encoding='utf-8'))
        for s in (data.get('samples', []) if isinstance(data, dict) else data):
            stem = Path(str(s.get('filepath', ''))).stem
            if stem.isdigit(): idx[stem] = s
        print(f'  samples.json：{len(idx)} 条带屏号的标注')
    except Exception as e:
        print(f'  samples.json 用不了（{str(e)[:80]}）：层级留空，注入时从图像里找元素')
    n = m = 0
    _, imgs = have_now()
    for sid in imgs:
        js = rico / f'{sid}.json'
        if js.exists(): continue
        kids = []
        for key, val in (idx.get(sid) or {}).items():
            if isinstance(val, dict) and val.get('_cls') == 'Detections':
                for det in val.get('detections') or []:
                    bb = det.get('bounding_box') or [0, 0, 0, 0]
                    if len(bb) != 4: continue
                    x, y, w, h = bb; lab = str(det.get('label') or '')
                    kids.append({'bounds': [int(x * 1440), int(y * 2560), int((x + w) * 1440), int((y + h) * 2560)], 'class': str(det.get('type') or lab), 'text': '',
                                 'componentLabel': lab, 'clickable': bool(det.get('clickable')), 'visible-to-user': True, 'children': []})
        if kids: m += 1
        js.write_text(json.dumps({'activity': {'root': {'bounds': [0, 0, 1440, 2560], 'class': 'root', 'children': kids}}}), encoding='utf-8'); n += 1
    print(f'  写好 {n} 个层级文件（{m} 个带元素框）'); sys.exit(0)
if mode == 'untar':   # route B: the original archive, only the needed screens kept
    import tarfile
    _, imgs = have_now()
    need_extra = max(0, 2500 - len(imgs - want)); extra = 0; got = 0; pending = {}; done = set(imgs)
    with tarfile.open(tmp / 'unique_uis.tar.gz', 'r|gz') as tf:
        for mbr in tf:
            name = Path(mbr.name).name; stem, ext = os.path.splitext(name)
            if not mbr.isfile() or not stem.isdigit() or ext not in ('.jpg', '.json'): continue
            if stem in done: continue
            if not (stem in want or stem in pending or extra < need_extra): continue
            (rico / name).write_bytes(tf.extractfile(mbr).read())
            parts = pending.setdefault(stem, set()); parts.add(ext)
            if parts == {'.jpg', '.json'}:
                pending.pop(stem); got += 1; done.add(stem)
                if stem not in want: extra += 1
                if got % 500 == 0: print(f'  已取出 {got} 屏（UICrit {len(want & done)}/{len(want)}）', flush=True)
            if want <= done and extra >= need_extra and not pending: break   # everything needed is out: the rest of the archive is not read
    for stem in pending:
        for ext in ('.jpg', '.json'): (rico / f'{stem}{ext}').unlink(missing_ok=True)
    print(f'  从原始包取出 {got} 屏'); sys.exit(0)
