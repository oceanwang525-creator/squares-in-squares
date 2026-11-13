"""Stream public Kaggle kernel output CSVs; keep per-n packings strictly better than our best.

Data only: CSVs are parsed as text, never executed. Each candidate n must pass the
unchanged official per-puzzle evaluator on (a) the literal strings and (b) the strings
after an actual pandas read_csv/to_csv round trip, with both certified L strictly below
the current best certified L.
"""
import csv, io, json, os, sys, time, urllib.request, urllib.parse
from multiprocessing import Pool
from decimal import Decimal
import numpy as np, pandas as pd
from squares._official_metric import evaluate_single_puzzle

STATE = 'state'
os.makedirs(STATE, exist_ok=True)
UA = {'User-Agent': 'curl/8.4.0'}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)
    with open(f'{STATE}/merge.log', 'a') as f:
        print(time.strftime('%H:%M:%S'), *a, file=f)


def read_rows(text):
    r = csv.DictReader(io.StringIO(text))
    cols = set(r.fieldnames or [])
    if not {'n', 'square_id', 'x', 'y', 'angle'} <= cols:
        return None
    by = {}
    for row in r:
        try:
            n = int(row['n'])
        except Exception:
            continue
        by.setdefault(n, []).append({'n': row['n'], 'square_id': row['square_id'],
                                     'x': row['x'], 'y': row['y'], 'angle': row['angle']})
    return by


def float_L(rows):
    try:
        x = np.array([float(r['x']) for r in rows]); y = np.array([float(r['y']) for r in rows])
        a = np.array([float(r['angle']) for r in rows])
    except Exception:
        return np.inf
    e = 0.5 * (np.abs(np.cos(a)) + np.abs(np.sin(a)))
    return max((x + e).max() - (x - e).min(), (y + e).max() - (y - e).min())


def pandas_transport(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=['id', 'n', 'square_id', 'x', 'y', 'angle'])
    w.writeheader()
    for i, r in enumerate(rows):
        w.writerow({'id': i, **r})
    df = pd.read_csv(io.StringIO(buf.getvalue()))
    out = df.to_csv(index=False)
    return list(csv.DictReader(io.StringIO(out)))


def certify(args):
    n, rows = args
    try:
        if len(rows) != n:
            return n, None, 'count'
        _, _, L1, why1 = evaluate_single_puzzle(n, rows)
        if L1 is None:
            return n, None, why1
        t = pandas_transport(rows)
        t = [{k: r[k] for k in ('n', 'square_id', 'x', 'y', 'angle')} for r in t]
        _, _, L2, why2 = evaluate_single_puzzle(n, t)
        if L2 is None:
            return n, None, 'pandas:' + why2
        return n, (L1, L2), 'ok'
    except Exception as e:
        return n, None, f'err {e}'


class Best:
    def __init__(self, merged_csv, lit_json, pd_json):
        lit = {r['n']: r['L'] for r in json.load(open(lit_json))['results']}
        pdl = {r['n']: r['L'] for r in json.load(open(pd_json))['results']}
        self.L = {n: max(Decimal(lit[n]), Decimal(pdl[n])) for n in lit}
        self.base = dict(self.L)
        self.rows = {}
        self.src = {}
        p = f'{STATE}/best.json'
        if os.path.exists(p):
            d = json.load(open(p))
            for k, v in d.items():
                n = int(k)
                if Decimal(v['L']) >= self.L[n]:
                    continue          # stale: not below the current baseline
                self.L[n] = Decimal(v['L']); self.rows[n] = v['rows']; self.src[n] = v['src']
        self.fL = {n: float(v) for n, v in self.L.items()}

    def save(self):
        d = {str(n): {'L': str(self.L[n]), 'rows': self.rows[n], 'src': self.src[n]} for n in self.rows}
        json.dump(d, open(f'{STATE}/best.json.tmp', 'w'))
        os.replace(f'{STATE}/best.json.tmp', f'{STATE}/best.json')

    def gain(self):
        return sum(float(self.base[n] - self.L[n]) / n ** 0.5 for n in self.rows)


def scan_text(best, text, src, pool):
    by = read_rows(text)
    if not by:
        return 0
    cand = []
    for n, rows in by.items():
        if n not in best.fL or len(rows) != n:
            continue
        if float_L(rows) < best.fL[n] - 1e-11:
            cand.append((n, rows))
    if not cand:
        return 0
    won = 0
    for n, res, why in pool.imap_unordered(certify, cand):
        if res is None:
            continue
        L = max(Decimal(res[0]), Decimal(res[1]))
        if L < best.L[n]:
            old = best.L[n]
            best.L[n] = L; best.fL[n] = float(L); best.src[n] = src
            best.rows[n] = by[n]
            won += 1
            log(f'  win n={n} L {old:.12f} -> {L:.12f} ({src})')
    if won:
        best.save()
    return won


def fetch(url, total=420, chunk=1 << 20):
    """Download with a hard overall deadline (a slow trickle cannot hang the scan)."""
    t0 = time.time(); parts = []
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        while True:
            b = r.read(chunk)
            if not b:
                break
            parts.append(b)
            if time.time() - t0 > total:
                raise TimeoutError(f'download exceeded {total}s')
    return b''.join(parts)


def kernel_files(u, s):
    files, tok = [], None
    while True:
        url = f'https://www.kaggle.com/api/v1/kernels/output?userName={u}&kernelSlug={s}' + (
            f'&pageToken={urllib.parse.quote(tok)}' if tok else '')
        o = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120))
        files += o.get('files', []); tok = o.get('nextPageToken')
        if not tok:
            return files


def done_urls_unversioned(done, k):
    # first-pass entries were plain URLs; treat them as scanned only for runs before this session's first listing
    return {k['url']} if (k['url'] in done and (k.get('run') or '') < '2026-11-12T04:40') else set()


def main():
    best = Best(*sys.argv[1:4])
    ks = json.load(open(sys.argv[4]))
    done_p = f'{STATE}/done.json'
    done = set(json.load(open(done_p))) if os.path.exists(done_p) else set()
    with Pool(8) as pool:
        for k in ks:
            key = f"{k['url']}@{k.get('run')}"
            if not k.get('out') or key in done or k['url'] in done_urls_unversioned(done, k):
                continue
            u, s = k['url'].split('/')[2:4]
            try:
                files = kernel_files(u, s)
            except Exception as e:
                log('list fail', u, s, e); continue
            csvs = [f for f in files if f['fileName'].lower().endswith('.csv') and f.get('size', 0) < 120e6]
            log(f'{k["score"]} {u}/{s}: {len(csvs)} csv')
            for f in csvs:
                try:
                    data = fetch(f['url'])
                    text = data.decode('utf-8', 'replace'); del data
                except Exception as e:
                    log('  dl fail', f['fileName'], e); continue
                w = scan_text(best, text, f'kaggle:{u}/{s}:{f["fileName"]}', pool)
                if w:
                    log(f'  {f["fileName"]}: {w} wins; total gain {best.gain():.6f}')
            done.add(key); json.dump(sorted(done), open(done_p, 'w'))
    log('DONE total gain', best.gain(), 'n improved', len(best.rows))


if __name__ == '__main__':
    main()
