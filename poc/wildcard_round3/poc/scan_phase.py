"""Scan all rigid 45deg band cores (w,a1,a2,b1,b2,t) over every n. For each config (count c at L), list n<=c whose
L_ours(n) > L - allow, split by staircase phase class (s=a1+a2 odd -> dphi=0.5). Needs c>=n and L < L_ours(n)+allow."""
import sys, math, csv, json
sys.path.insert(0, '/Users/lucaszhang/squares-in-squares/poc/k2p10_band_search/poc')
from enum45 import count
I2 = 1/math.sqrt(2)
Lo = {}
for r in csv.DictReader(open('/Users/lucaszhang/squares-in-squares/poc/leader_curve_analysis/docs/per_n_comparison.csv')):
    Lo[int(r['n'])] = (float(r['L_ours']), float(r['L_lb']))
hits = []
best = {}
for w in range(1, 40):
  for t in range(0, 30):
    for s in range(0, 64):
      L = 0.5*s + 1 + w*I2 + t
      if L > 32.1: continue
      K = int(math.floor(L + 1e-9)); L0 = L - t
      for a1 in range((s+1)//2, s+1):
        a2 = s - a1
        if a1 > L0 + 1e-9: continue
        for b1 in range(0, int(L0)+1):
          for b2 in range(b1, int(L0)+1):
            r = count(w, a1, a2, b1, b2, t, K)
            if r is None: continue
            c = r[0]
            for n in (c, c-1, c-2, c-3):   # n just below count suffices to test; higher n handled by n<=c monotone gap
              pass
            # best config per (parity,n): lowest L with count>=n
            par = s % 2
            key = (par, c)
            if key not in best or L < best[key][0]:
                best[key] = (L, w, a1, a2, b1, b2, t, c)
out = {}
for par in (0, 1):
    cs = sorted(c for (p, c) in best if p == par)
    # for each n: min L over configs with count >= n
    for n in range(1, 1001):
        m = None
        for c in cs:
            if c >= n:
                v = best[(par, c)]
                if m is None or v[0] < m[0]: m = v
        if m: out[(par, n)] = m
rows = []
for n in range(1, 1001):
    lo, lb = Lo[n]
    e = out.get((0, n)); o = out.get((1, n))
    rows.append((n, lo, lb, e[0] if e else None, o[0] if o else None, o))
json.dump([(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows], open('/Users/lucaszhang/squares-in-squares/poc/wildcard_round3/docs/scan_phase.json', 'w'))
print("odd-s (dphi=.5) configs with count>=n and L < L_ours+0.015:")
for n, lo, lb, le, lod, o in rows:
    if lod is not None and lod < lo + 0.015:
        print(n, 'ours %.5f lb %.5f even-rigid %s odd-rigid %.5f' % (lo, lb, le and round(le, 5), lod), o)
print("even-s sanity: n where rigid even-s L < ours -1e-6:", [r[0] for r in rows if r[3] is not None and r[3] < r[1]-1e-6][:20])
