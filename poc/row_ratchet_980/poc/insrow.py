"""Insert one square into row n-1 and squeeze inside a loosened box: candidates for row n.
usage: insrow.py n_src K L_target workers out.json [n_candidates]
"""
import json, math, sys, time
import numpy as np
import numba as nb
from multiprocessing import Pool
from row980 import load_row, build_model, full_viol
import pslp2
from pslp2 import squeeze, feas


@nb.njit(cache=True)
def pen_at(x, y, t, X, Y, T, L):
    """Total penetration of a unit square at (x, y, t) against all squares and the walls."""
    e = 0.5 * (abs(math.cos(t)) + abs(math.sin(t)))
    tot = max(0.0, e - x) + max(0.0, e - y) + max(0.0, x + e - L) + max(0.0, y + e - L)
    for j in range(X.shape[0]):
        dx = X[j] - x; dy = Y[j] - y
        if dx * dx + dy * dy >= 2.0: continue
        best = -1e9
        for a in range(4):
            if a < 2:
                al = t + a * (math.pi / 2); phi = T[j] - al
            else:
                al = T[j] + (a - 2) * (math.pi / 2); phi = t - al
            g = abs(dx * math.cos(al) + dy * math.sin(al)) - 0.5 - 0.5 * (abs(math.cos(phi)) + abs(math.sin(phi)))
            if g > best: best = g
        if best < 0: tot += -best
    return tot


def work(args):
    a, L0, K, x, y, t, tag = args
    b = np.vstack([a, [[x, y, t]]])
    try:
        idx, act, stat, mov, mov_full = build_model(b, L0, K, act_reach=3.2)
        new = len(b) - 1
        if new not in idx:
            return tag, None, None
        sub = b[idx]
        f, Xn, Yn, Tn, v, it = squeeze(sub[:, 0].copy(), sub[:, 1].copy(), sub[:, 2].copy(), K, K, L0 - K, frozen=stat, moving=mov, max_it=500)
    except Exception as e:
        return tag, None, None
    full = b.copy(); full[idx, 0] = Xn; full[idx, 1] = Yn; full[idx, 2] = Tn
    rest = np.ones(len(b), bool); rest[idx] = False
    full[rest & mov_full, 0] += f - (L0 - K); full[rest & mov_full, 1] += f - (L0 - K)
    Ln = K + f
    if v > 1e-12 or full_viol(full, Ln) > 1e-12:
        return tag, None, v
    return tag, Ln, full.tolist()


if __name__ == '__main__':
    n_src, K, Lt, W_, out = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    ncand = int(sys.argv[6]) if len(sys.argv) > 6 else 40
    a, L = load_row(n_src)
    L0 = Lt + 0.03
    idx, act, stat, mov, mov_full = build_model(a, L, K)
    a0 = a.copy(); a0[mov_full, 0] += L0 - L; a0[mov_full, 1] += L0 - L      # loosen: far block moves out with the wall
    X, Y, T = a0[:, 0].copy(), a0[:, 1].copy(), a0[:, 2].copy()
    # candidate positions: grid 0.1, upright and two tilts taken from the arms
    td = np.degrees(np.mod(a[:, 2], np.pi / 2)); til = np.minimum(td, 90 - td) > 2
    angs = [0.0] + [float(np.median(a[til & (a[:, 0] < L / 3), 2]) if (til & (a[:, 0] < L / 3)).any() else 0.3),
                    float(np.median(a[til & (a[:, 0] >= L / 3), 2]) if (til & (a[:, 0] >= L / 3)).any() else 0.3)]
    cands = []
    for gx in np.arange(0.5, L0 - 0.49, 0.1):
        for gy in np.arange(0.5, L0 - 0.49, 0.1):
            for t in angs:
                p = pen_at(gx, gy, t, X, Y, T, L0)
                cands.append((p, gx, gy, t))
    cands.sort()
    chosen = []
    for p, gx, gy, t in cands:
        if all(math.hypot(gx - c[1], gy - c[2]) > 0.8 or abs(t - c[3]) > 0.05 for c in chosen):
            chosen.append((p, gx, gy, t))
        if len(chosen) >= ncand: break
    print(f'insert into n={n_src} (L={L:.6f}) at box {L0:.6f}; {len(chosen)} sites, overlap of best sites: {[round(c[0],3) for c in chosen[:8]]}', flush=True)
    res = []; best = (9e9, None); t0 = time.time()
    with Pool(W_) as pool:
        for tag, Ln, full in pool.imap_unordered(work, [(a0, L0, K, c[1], c[2], c[3], i) for i, c in enumerate(chosen)]):
            if Ln is None:
                res.append((9e9, tag)); continue
            res.append((Ln, tag))
            if Ln < best[0]:
                best = (Ln, tag); json.dump(dict(n=n_src + 1, L=Ln, site=chosen[tag][1:], sq=full), open(out, 'w'))
    ok = sorted(r for r in res if r[0] < 9e8)
    print(f'{len(ok)}/{len(res)} feasible in {time.time()-t0:.0f}s; best L={ok[0][0] if ok else None}; target {Lt:.9f}; top: {[round(l,5) for l,_ in ok[:8]]}', flush=True)
