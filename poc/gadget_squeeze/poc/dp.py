"""Staircase composition DP over the pocket library, numba-free (numpy knapsack).

Pocket i: a (w_i + f) x (h_i + f) rectangle holding w_i*h_i + g_i squares. In a K+f box, m
pockets fit along the BL/TR seam iff  sum w <= K,  sum h <= K,  sum (w + h + 1) <= 2K + 1
(adjacent pockets need one extra column OR one extra row); then n = K^2 + sum g, side K + f.
Same rule as poc/rectangle_gadgets/poc/gaddp.py and poc/pocket_squeeze/poc/compose.py; only
the inner knapsack is re-implemented with numpy (this venv has no numba).

The knapsack is *incremental*: items enter in increasing f, and adding one unbounded item is
one ascending sweep over the state array (each item raises sum(w+h+1) by at least 2, so the
`s` slice order is exact). That makes the whole per-K frontier cost O(#items) sweeps plus one
max() per level, instead of a full DP per level.

usage: dp.py lib.json [oursL.json] [--out pred.json]
"""
import json, math, sys
import numpy as np

KMAX = 31


class Knap:
    """best[x, y, s] = max gain with sum w = x, sum h = y, sum (w + h + 1) = s."""

    def __init__(self, K):
        self.K = K
        S = 2 * K + 1
        self.best = -np.ones((K + 1, K + 1, S + 1), np.int32)
        self.choice = -np.ones((K + 1, K + 1, S + 1), np.int64)
        self.best[0, 0, 0] = 0
        self.items = []                     # (w, h, g)
        self.tags = []                      # caller's label for each stored item

    def add(self, w, h, g, tag=None):
        K = self.K
        if w > K or h > K:
            return
        idx = len(self.items)
        self.items.append((w, h, g))
        self.tags.append(tag if tag is not None else idx)
        S = 2 * K + 1
        for s in range(w + h + 1, S + 1):
            src = self.best[:K + 1 - w, :K + 1 - h, s - w - h - 1]
            dst = self.best[w:, h:, s]
            cand = np.where(src >= 0, src + g, -1)
            better = cand > dst
            if better.any():
                dst[better] = cand[better]
                ch = self.choice[w:, h:, s]
                ch[better] = idx

    def gmax(self):
        return int(self.best.max())

    def parts(self):
        b = self.best
        x, y, s = np.unravel_index(int(b.argmax()), b.shape)
        out = []
        guard = 0
        while self.choice[x, y, s] >= 0 and guard < 500:
            guard += 1
            k = int(self.choice[x, y, s])
            w, h, g = self.items[k]
            out.append(self.tags[k])
            x, y, s = x - w, y - h, s - w - h - 1
        return out


def items_of(lib):
    """(w,h,g) -> (best f, lib index, transposed?) over both orientations."""
    d = {}
    for i, g in enumerate(lib):
        f = float(g['f'])
        for tr, (w, h) in enumerate(((g['w'], g['h']), (g['h'], g['w']))):
            k = (w, h, g['g'])
            if k not in d or f < d[k][0] - 1e-15:
                d[k] = (f, i, tr)
    return d


def frontier(lib, K, verbose=False):
    """[(f, gmax, [(lib index, transposed)])] at each level where the reachable gain grows."""
    d = items_of(lib)
    keys = sorted(d, key=lambda k: d[k][0])
    kn = Knap(K)
    out = []
    gprev = 0
    i = 0
    while i < len(keys):
        f = d[keys[i]][0]
        while i < len(keys) and d[keys[i]][0] <= f + 1e-15:
            k = keys[i]
            kn.add(k[0], k[1], k[2], tag=k)
            i += 1
        gm = kn.gmax()
        if gm > gprev:
            parts = [(d[k][1], d[k][2]) for k in kn.parts()]
            out.append((f, gm, parts))
            gprev = gm
        if gprev >= 2 * K:
            break
    return out


def predict(lib, Kmin=2, Kmax=KMAX, show=True):
    """n -> (K, f, parts): smallest f whose composition reaches gain >= n - K^2."""
    pred = {}
    for K in range(Kmin, Kmax + 1):
        fr = frontier(lib, K)
        for j in range(1, 2 * K + 1):
            n = K * K + j
            if n > 1000:
                break
            for f, gmax, parts in fr:
                if gmax >= j:
                    pred[n] = (K, f, parts)
                    break
        if show:
            print(f'  K={K}: frontier {[(round(f, 6), g) for f, g, _ in fr][:8]}'
                  f'{" ..." if len(fr) > 8 else ""}', flush=True)
    return pred


def best_at(lib, K, f):
    """Max gain reachable at one level f (all items with f_i <= f), plus the composition."""
    d = items_of(lib)
    keys = [k for k in d if d[k][0] <= f + 1e-15]
    kn = Knap(K)
    for k in keys:
        kn.add(k[0], k[1], k[2], tag=k)
    gm = kn.gmax()
    if gm <= 0:
        return 0, []
    return gm, [(d[k][1], d[k][2]) for k in kn.parts()]


def load_lib(path):
    lib = json.load(open(path))
    for g in lib:
        if 'f' not in g:
            g['f'] = max(g['fx'], g['fy'])
    return lib


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    lib = load_lib(args[0])
    pred = predict(lib)
    out = {}
    if len(args) > 1:
        L = {int(k): float(v) for k, v in json.load(open(args[1])).items()}
        rows = []
        for n, (K, f, parts) in sorted(pred.items()):
            if n in L:
                d = L[n] - (K + f)
                if d > 1e-7:
                    rows.append((d / math.sqrt(n), n, L[n], K + f, K, f, parts))
        rows.sort(key=lambda r: -r[0])
        print(f'DP strictly better at {len(rows)} n; score gain {sum(r[0] for r in rows):.6f}')
        for sg, n, ours, lp, K, f, parts in rows[:60]:
            desc = ' + '.join(f'{lib[i]["w"]}x{lib[i]["h"]}+{lib[i]["g"]}{"T" if tr else ""}'
                              f'[{lib[i].get("src")}@{lib[i]["f"]:.6f}]' for i, tr in parts)
            print(f'  n={n} K={K}: {ours:.6f} -> {lp:.6f} (dL {ours-lp:+.2e}, gain {sg:.2e}) {desc}')
            out[n] = dict(K=K, f=f, L=lp, parts=[[i, tr] for i, tr in parts])
    if '--out' in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index('--out') + 1], 'w'))
        print('wrote', sys.argv[sys.argv.index('--out') + 1], len(out), 'cells')
