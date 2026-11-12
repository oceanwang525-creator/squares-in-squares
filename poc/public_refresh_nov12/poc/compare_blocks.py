"""Map stage-2 results (gain g, size k, c*) onto n = K^2 + g (K >= k, by grid wrapping) and
report where c* beats our current c(K, g). Also tries pairs of blocks placed in disjoint
row/column bands (gain g1+g2, c = max, needs K >= k1 + k2 - overlap margin, conservatively k1+k2)."""
import glob, json, math, re, sys
import numpy as np

B = '../repo/poc/pell_corrugation/data/border-scan/'
L = {r['n']: float(r['L']) for r in json.load(open(B + 'merged-literal-official.json'))['results']}
for k, v in json.load(open('state/best.json')).items():
    L[int(k)] = float(v['L'])

res = []
for f in sys.argv[1:]:
    for line in open(f):
        m = re.search(r'gain=(\d+) block=(\S+) n=(\d+) k=(\d+) c_seed=\S+ -> c\*=([\d.]+)', line)
        if m:
            res.append((int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4)), float(m.group(5))))
# best c* per (gain, k)
best = {}
for g, blk, n, k, c in res:
    if (g, k) not in best or c < best[(g, k)][0]:
        best[(g, k)] = (c, blk, n)
wins = {}
for (g, k), (c, blk, n0) in best.items():
    for K in range(k, 32):
        n = K * K + g
        if n > 1000 or g >= 2 * K + 1:
            continue
        ours = L[n] - K
        if c < ours - 1e-9:
            gain = (ours - c) / math.sqrt(n)
            if n not in wins or gain > wins[n][0]:
                wins[n] = (gain, K, g, c, ours, blk, n0)
print('single-block results:', len(res), 'distinct (g,k):', len(best))
tot = sum(w[0] for w in wins.values())
print('n improvable (by wrapping):', len(wins), 'score gain', round(tot, 6))
for n, w in sorted(wins.items()):
    print(f'  n={n} K={w[1]} j={w[2]} c*={w[3]:.6f} ours={w[4]:.6f} gain={w[0]:.2e} block={w[5]} (from n0={w[6]})')
