"""Official whole-file verification, twice: literal CSV and after pandas read_csv/to_csv.
Lists every n whose L is strictly lower than the reference merged.csv in BOTH transports.
usage: verify_full.py candidate.csv ref_lit.json ref_pandas.json outdir
"""
import json, os, subprocess, sys, time
from decimal import Decimal
import pandas as pd

cand, ref_lit_p, ref_pd_p, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
pd_csv = os.path.join(out, 'candidate-pandas.csv')
pd.read_csv(cand).to_csv(pd_csv, index=False)
print('pandas', pd.__version__, '->', pd_csv, flush=True)

reports = {}
for tag, path in (('literal', cand), ('pandas', pd_csv)):
    rep = os.path.join(out, f'official-{tag}.json')
    t = time.time()
    r = subprocess.run(['squares', 'score', path, '--report', rep], capture_output=True, text=True)
    print(f'[{tag}] squares score exit={r.returncode} in {time.time()-t:.0f}s', flush=True)
    print(r.stdout.strip()[-2000:]); print(r.stderr.strip()[-2000:])
    d = json.load(open(rep))
    reports[tag] = d
    print(f'[{tag}] score={d["score"]!r} accepted={d["accepted"]} missing={d["missing"]} rejected={d["rejected"]}', flush=True)

ref = {'literal': {r['n']: r for r in json.load(open(ref_lit_p))['results']},
       'pandas': {r['n']: r for r in json.load(open(ref_pd_p))['results']}}
new = {t: {r['n']: r for r in reports[t]['results']} for t in reports}
wins, worse = [], []
for n in range(1, 1001):
    a = [Decimal(new[t][n]['L']) for t in ('literal', 'pandas')]
    b = [Decimal(ref[t][n]['L']) for t in ('literal', 'pandas')]
    if a[0] < b[0] and a[1] < b[1]:
        wins.append((n, b[0], a[0], b[1], a[1]))
    elif a[0] > b[0] or a[1] > b[1]:
        worse.append((n, b[0], a[0], b[1], a[1]))
rows = [dict(n=n, merged_L_literal=str(b0), new_L_literal=str(a0), merged_L_pandas=str(b1), new_L_pandas=str(a1),
             score_gain=float((b0 - a0) / Decimal(n).sqrt())) for n, b0, a0, b1, a1 in wins]
json.dump(dict(score_literal=reports['literal']['score'], score_pandas=reports['pandas']['score'],
               accepted_literal=reports['literal']['accepted'], accepted_pandas=reports['pandas']['accepted'],
               ref_score_literal=json.load(open(ref_lit_p))['score'], wins=rows,
               not_strictly_better=[dict(n=w[0]) for w in worse]),
          open(os.path.join(out, 'wins_vs_merged.json'), 'w'), indent=1)
print(f'strict wins in both transports: {len(wins)}; n worse/not-strict: {len(worse)} {[w[0] for w in worse][:20]}')
print('sum score gain', sum(r['score_gain'] for r in rows))
