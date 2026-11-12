"""Union the wins of public_resweep_kernels and public_resweep_other, keep the
per-n best (by pandas-roundtrip L, only pandas-stable ones), swap into merged.csv."""
import json, sys
from decimal import Decimal, getcontext
import pandas as pd
getcontext().prec = 60
R = 'poc/'
TIES = {919}  # official L equal to ours (no strict gain); kept as incumbent
inc = pd.read_csv(R+'pell_corrugation/data/border-scan/merged.csv')
cands = {}  # n -> list of (L_pandas Decimal, source, rows_df)
for d in ['public_resweep_kernels', 'public_resweep_other']:
    w = json.load(open(f'{R}{d}/docs/wins.json'))
    rows = pd.read_csv(f'{R}{d}/out/wins_rows.csv', dtype={'id': int, 'n': int})
    for ns, info in w.items():
        n = int(ns)
        if d.endswith('kernels'):
            if not info.get('pandas_roundtrip_ok'): continue
            L = Decimal(info['pandas_roundtrip_L']); src = info['source_kernel']
        else:
            L = Decimal(str(info['new_L_pandas'])); src = info['source']
        our = Decimal(info['our_L'])
        if L >= our or n in TIES: continue
        cands.setdefault(n, []).append((L, d + ':' + src, rows[rows.n == n]))
out = inc.copy()
chosen = {}
for n, lst in cands.items():
    L, src, r = min(lst, key=lambda t: t[0])
    chosen[n] = (str(L), src)
    out = out[out.n != n]
    out = pd.concat([out, r])
out = out.sort_values(['n', 'square_id']).reset_index(drop=True)
assert len(out) == 500500 and list(out.columns) == list(inc.columns)
out.to_csv('poc/public_resweep_merge/out/merged_candidate.csv', index=False)
json.dump(chosen, open('poc/public_resweep_merge/docs/chosen.json', 'w'), indent=1)
print(len(chosen), 'n swapped')
