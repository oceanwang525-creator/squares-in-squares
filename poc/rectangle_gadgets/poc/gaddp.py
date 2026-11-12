"""Staircase composition DP over the sub-gadget library (corrected rule).

Gadget i: pocket (w_i + f) x (h_i + f) holding w_i*h_i + g_i squares. m gadgets along the BL/TR seam of
a (K + f) box are feasible iff  sum w <= K,  sum h <= K,  sum (w + h + 1) <= 2K + 1
(each adjacent pair needs one extra row OR one extra column). Then n = K^2 + sum g.

usage: gaddp.py gadlib.json report.json [out.json]
"""
import json, math, sys
import numpy as np
import numba as nb

KMAX = 31


@nb.njit(cache=True)
def knap(W, H, G, K):
    """best[x, y, s] = max gain with sum w = x, sum h = y, sum (w+h+1) = s (exact), -1 if unreachable."""
    S = 2 * K + 1
    best = -np.ones((K + 1, K + 1, S + 1), np.int32)
    choice = -np.ones((K + 1, K + 1, S + 1), np.int32)
    best[0, 0, 0] = 0
    for x in range(K + 1):
        for y in range(K + 1):
            for s in range(S + 1):
                b = best[x, y, s]
                if b < 0:
                    continue
                for k in range(W.shape[0]):
                    nx = x + W[k]; ny = y + H[k]; ns = s + W[k] + H[k] + 1
                    if nx <= K and ny <= K and ns <= S:
                        if b + G[k] > best[nx, ny, ns]:
                            best[nx, ny, ns] = b + G[k]; choice[nx, ny, ns] = k
    return best, choice


def run(lib, L):
    # unique items per (w, h, g) with best f; both orientations
    items = {}
    for i, g in enumerate(lib):
        f = max(g['fx'], g['fy'])
        for (w, h) in ((g['w'], g['h']), (g['h'], g['w'])):
            k = (w, h, g['g'])
            if k not in items or f < items[k][0]:
                items[k] = (f, i)
    keys = sorted(items, key=lambda k: items[k][0])
    fs = sorted(set(round(items[k][0], 12) for k in keys))
    pred = {}
    for K in range(2, KMAX + 1):
        done = set()
        for f in fs:
            sel = [k for k in keys if items[k][0] <= f + 1e-12 and k[0] <= K and k[1] <= K]
            if not sel:
                continue
            W = np.array([k[0] for k in sel], np.int64); H = np.array([k[1] for k in sel], np.int64)
            G = np.array([k[2] for k in sel], np.int64)
            best, choice = knap(W, H, G, K)
            gmax = int(best.max())
            if gmax <= 0:
                continue
            # backtrack the argmax
            x, y, s = np.unravel_index(int(best.argmax()), best.shape)
            chosen = []
            while choice[x, y, s] >= 0:
                k = int(choice[x, y, s]); chosen.append(sel[k])
                x, y, s = x - W[k], y - H[k], s - W[k] - H[k] - 1
            for j in range(1, gmax + 1):
                n = K * K + j
                if n <= 1000 and n not in pred:
                    pred[n] = dict(K=K, f=f, gain=gmax, parts=[(k, items[k][1]) for k in chosen])
            if all((K * K + j > 1000) or (K * K + j in pred) for j in range(1, 2 * K + 1)):
                break
    return pred


if __name__ == '__main__':
    lib = json.load(open(sys.argv[1]))
    L = {r['n']: float(r['L']) for r in json.load(open(sys.argv[2]))['results']}
    pred = run(lib, L)
    rows = []
    for n, p in sorted(pred.items()):
        d = L[n] - (p['K'] + p['f'])
        if d > 1e-7:
            rows.append((d / math.sqrt(n), n, L[n], p['K'] + p['f'], p))
    rows.sort(key=lambda r: -r[0])
    print(f'library {len(lib)} gadgets; predicted cells {len(pred)}; DP strictly better at {len(rows)} n; score gain {sum(r[0] for r in rows):.6f}')
    for sg, n, ours, Lp, p in rows[:60]:
        parts = ', '.join(f'({k[0]}x{k[1]},+{k[2]} from n={lib[i]["src"]})' for k, i in p['parts'])
        print(f'  n={n} K={p["K"]} j={n-p["K"]**2}: ours {ours:.6f} -> {Lp:.6f} (gain {sg:.2e}) total gain {p["gain"]}: {parts}')
    if len(sys.argv) > 3:
        json.dump({str(r[1]): dict(L=r[3], K=r[4]['K'], f=r[4]['f'], gain=r[4]['gain'], parts=[[list(k), i] for k, i in r[4]['parts']]) for r in rows}, open(sys.argv[3], 'w'))
