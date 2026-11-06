"""What a step produced: per model x setting x dataset the rows, failures, tokens, latency and cost; the distinct errors; sample answers;
the metrics table; the spend. Used by the status server (AutoDL) and by the experiment runner on the Windows PC."""
from __future__ import annotations
import json, os, re

RS_STEP_TAGS = {'dev': ['haiku_dev'], 'trial': ['haiku', 'haiku_r2', 'haiku_r3'], 'main': ['haiku', 'gpt54mini', 'gemini38flash', 'sonnet55'], 'local': ['qwen3vl8b', 'internvl35']}
RS_STEP_RESULTS = {'dev': 'dev_results.json', 'trial': 'trial_results.json', 'main': 'results.json', 'local': 'results.json'}


def rs_apitest(key, models):
    """from AutoDL, exactly as run_api calls: one short request per model; route = windows-proxy (reverse tunnel up) or direct"""
    import socket, urllib.error, urllib.request
    via = None
    try:
        with socket.create_connection(('127.0.0.1', 17890), timeout=3): via = 'http://127.0.0.1:17890'
    except OSError:
        via = None
    op = urllib.request.build_opener(urllib.request.ProxyHandler({'http': via, 'https': via} if via else {}))
    out = {'route': 'windows-proxy' if via else 'direct', 'models': []}
    for m in models:
        body = json.dumps({'model': m, 'messages': [{'role': 'user', 'content': 'Reply with the single word OK.'}], 'max_tokens': 5, 'temperature': 0}).encode()
        req = urllib.request.Request(os.environ.get('RS_OPENROUTER_BASE', 'https://openrouter.ai/api/v1') + '/chat/completions', data=body, headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
        t0 = time.time()
        try:
            with op.open(req, timeout=60) as r: j = json.loads(r.read().decode('utf-8', 'replace'))
            out['models'].append({'model': m, 'ok': True, 'ms': int((time.time() - t0) * 1000), 'text': ((j.get('choices') or [{}])[0].get('message') or {}).get('content', '')[:40]})
        except urllib.error.HTTPError as e:
            try:
                b = e.read().decode('utf-8', 'replace')[:500]
                try: msg = (json.loads(b).get('error') or {}).get('message') or b
                except Exception: msg = b
            except Exception:
                msg = ''
            out['models'].append({'model': m, 'ok': False, 'code': e.code, 'message': f'HTTP {e.code}: {str(msg)[:240]}'})
        except Exception as e:
            out['models'].append({'model': m, 'ok': False, 'message': str(e)[:240]})
    return out


def rs_report(step, RS_HOME, pv=None):
    """what a step produced, read on the instance so the console can show it: per model × setting × dataset the rows, failures,
    tokens, latency and cost; the distinct error messages with counts; two answers with their verification; the metrics table; the spend"""
    import collections, statistics
    tags = RS_STEP_TAGS.get(step, [])
    out = {'step': step, 'groups': [], 'errors': [], 'samples': [], 'table': [], 'files': []}
    for tag in tags:
        fp = os.path.join(RS_HOME, 'runs', f'{tag}.jsonl')
        if not os.path.isfile(fp): continue
        recs = list(latest_records(fp, pv).values()); out['files'].append({'tag': tag, 'rows': len(recs), 'mtime': int(os.path.getmtime(fp))})
        grp = collections.defaultdict(list)
        for r in recs: grp[(r.get('setting'), r.get('dataset'))].append(r)
        for (st, ds), rs in sorted(grp.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
            ok = [r for r in rs if 'error' not in r]
            lat = [r.get('latency_s') or 0 for r in ok if not r.get('shared_call')]
            out['groups'].append({'tag': tag, 'setting': st, 'dataset': ds, 'rows': len(rs), 'ok': len(ok), 'errors': len(rs) - len(ok),
                                  'parsed': sum(1 for r in ok if r.get('parsed')), 'positive': sum(1 for r in ok if r.get('pred')),
                                  'tokens_in': round(statistics.mean([r.get('tokens_in') or 0 for r in ok]), 0) if ok else 0,
                                  'tokens_out': round(statistics.mean([r.get('tokens_out') or 0 for r in ok]), 0) if ok else 0,
                                  'latency': round(statistics.mean(lat), 1) if lat else 0, 'cost': round(sum(r.get('cost_usd') or 0 for r in ok), 4)})
        errc = collections.Counter(re.sub(r'(req|gen|id)[-_:][A-Za-z0-9-]{8,}', r'\1-…', str(r.get('error'))[:220]) for r in recs if 'error' in r)
        out['errors'] += [{'tag': tag, 'message': m, 'count': c} for m, c in errc.most_common(5)]
        good = [r for r in recs if 'error' not in r and r.get('parsed')]
        for r in ([x for x in good if x.get('setting') == 's3'] or good)[:2]:
            out['samples'].append({'tag': tag, 'sid': r.get('sid'), 'setting': r.get('setting'), 'pred': r.get('pred'), 'types': r.get('types'),
                                   'verified': [{'type': v.get('type'), 'box': v.get('box'), 'evidence': str(v.get('evidence') or '')[:120], 'basis': (v.get('verifier') or {}).get('basis')} for v in (r.get('verified') or [])[:3]],
                                   'rules': r.get('rules_retrieved'), 'raw': str(r.get('raw') or '')[:400]})
    try:
        res = json.loads(_read(os.path.join(RS_HOME, 'results', RS_STEP_RESULTS.get(step, 'results.json'))) or '{}')
        res_pv = (res.get('executor') or {}).get('prompt_version')
        if pv and res and res_pv != pv: out['table'] = []; out['stale_results'] = res_pv or 'older'   # scored under earlier prompts (or before versions were recorded): not shown
        else: out['table'] = [t for t in (res.get('table') or []) if t.get('model') in tags]
    except Exception:
        pass
    # UI-Lens per task: the metrics, and what the positives were reported as (where the type goes astray)
    try:
        res2 = json.loads(_read(os.path.join(RS_HOME, 'results', RS_STEP_RESULTS.get(step, 'results.json'))) or '{}')
        st = res2.get('stats') or {}
        out['tasks'] = [] if out.get('stale_results') else [t for t in (st.get('ui_lens_tasks') or []) if t.get('model') in tags]
        out['tau_curve'] = [] if out.get('stale_results') else [t for t in (st.get('tau_curve_dev') or []) if t.get('model') in tags]
        out['refused'] = {m: len(v) for m, v in (st.get('refused_by_provider') or {}).items() if m in tags}
    except Exception:
        pass
    try:
        items = manifest_items(os.path.join(RS_HOME, 'data', 'RuleSight_data.json'))
        import collections
        conf = collections.defaultdict(collections.Counter)
        for tag in tags:
            fp = os.path.join(RS_HOME, 'runs', f'{tag}.jsonl')
            if not os.path.isfile(fp): continue
            for r in latest_records(fp, pv).values():
                it = items.get(r.get('sid')) or {}
                if 'error' in r or not it.get('ui_lens_task') or not it.get('label'): continue
                ts = r.get('types') or []
                for t in (ts or ['none']): conf[(tag, r.get('setting'), it['ui_lens_task'])][t] += 1
        out['positives_reported_as'] = [{'tag': k[0], 'setting': k[1], 'task': k[2], 'types': dict(v.most_common(4))} for k, v in sorted(conf.items())]
    except Exception:
        out['positives_reported_as'] = []
    try: out['spent'] = json.loads(_read(os.path.join(RS_HOME, 'ledger.json')) or '{}').get('total_usd', 0)
    except Exception: out['spent'] = None
    return out




def _read(p):
    try:
        with open(p, encoding="utf-8") as f: return f.read()
    except Exception:
        return "{}"

_CACHE = {}   # run file -> {"size": bytes read, "latest": {(sid, setting): compact record}}
_MAN = {"path": None, "mtime": None, "items": {}}
KEEP = ("sid", "setting", "dataset", "pv", "error", "parsed", "pred", "types", "tokens_in", "tokens_out", "latency_s", "shared_call", "cost_usd", "rules_retrieved")


def _compact(r):
    c = {k: r.get(k) for k in KEEP if k in r}
    if r.get("verified"): c["verified"] = [{"type": v.get("type"), "box": v.get("box"), "evidence": str(v.get("evidence") or "")[:160], "confidence": v.get("confidence"), "verifier": {"basis": (v.get("verifier") or {}).get("basis")}} for v in r["verified"][:4]]
    if r.get("raw"): c["raw"] = str(r["raw"])[:500]
    return c


def latest_records(fp, pv=None):
    """the latest record per (sid, setting) of one run file, read incrementally (only the bytes appended since the last call)"""
    try: size = os.path.getsize(fp)
    except OSError: return {}
    st = _CACHE.get(fp)
    if not st or size < st["size"]: st = _CACHE[fp] = {"size": 0, "latest": {}}
    if size > st["size"]:
        with open(fp, "rb") as f:
            f.seek(st["size"]); chunk = f.read(size - st["size"])
        tail = chunk.rfind(b"\n") + 1
        dec = json.JSONDecoder()
        for raw in chunk[:tail].splitlines():
            line = raw.decode("utf-8", "replace").strip(); pos = 0
            while pos < len(line):   # a line may carry two records glued together (two writers on one file); take each
                try: r, end = dec.raw_decode(line, pos)
                except ValueError:
                    nxt = line.find("{", pos + 1)
                    if nxt < 0: break
                    pos = nxt; continue
                pos = end
                while pos < len(line) and line[pos] in " \t\r,": pos += 1
                if not isinstance(r, dict): continue
                k = (r.get("sid"), r.get("setting")); old = st["latest"].get((k, r.get("pv")))
                if "error" not in r or old is None or "error" in old: st["latest"][(k, r.get("pv"))] = _compact(r)
        st["size"] += tail
    out = {}
    for (k, rpv), r in st["latest"].items():
        if pv and rpv != pv: continue
        if k not in out or "error" in out[k]: out[k] = r
    return out


def manifest_items(path):
    try: mt = os.path.getmtime(path)
    except OSError: return {}
    if _MAN["path"] != path or _MAN["mtime"] != mt:
        try: _MAN.update(path=path, mtime=mt, items={x["sid"]: x for x in (json.loads(_read(path) or "{}").get("screens") or [])})
        except Exception: _MAN.update(path=path, mtime=mt, items={})
    return _MAN["items"]
