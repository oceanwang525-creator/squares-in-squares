"""Reduced model of one row: arms active, grid blocks rigid (bottom-left static, top-right moving with the wall)."""
import json, math
import numpy as np
from pslp import squeeze, viol, feas


def load_row(n, path='rows_975_992.json'):
    rows = json.load(open(path))[str(n)]
    a = np.array([(float(r['x']), float(r['y']), float(r['angle'])) for r in rows])
    e = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
    a[:, 0] -= (a[:, 0] - e).min(); a[:, 1] -= (a[:, 1] - e).min()
    L = max((a[:, 0] + e).max(), (a[:, 1] + e).max())
    return a, L


def classify(a, L, K, reach=1.6):
    """active: within `reach` of any strongly tilted square (>2 deg). static: other squares whose nearest corner
    block is bottom-left. moving: other squares in the top-right block. Returns masks and the kept layer."""
    t = np.degrees(np.mod(a[:, 2], np.pi / 2)); til = np.minimum(t, 90 - t) > 2.0
    d = np.full(len(a), 1e9)
    P = a[til, :2]
    for i in range(0, len(a), 256):
        d[i:i + 256] = np.min(np.hypot(a[i:i + 256, None, 0] - P[None, :, 0], a[i:i + 256, None, 1] - P[None, :, 1]), axis=1)
    return til, d


def build_model(a, L, K, act_reach=2.6, layer=1.6, tr_corner=None):
    """tr_corner=None picks the moving block automatically."""
    til, d = classify(a, L, K)
    active = d < act_reach
    # rigid squares: decide block by flood from corners is overkill; top-right block = not active and closer to (L, L)
    # than to (0, 0) in the max-norm sense relative to the arms' inner corner
    if tr_corner is not None:
        cx, cy = tr_corner
        moving = (~active) & (a[:, 0] > cx) & (a[:, 1] > cy)
    else:
        # top-right block = rigid squares with no strongly tilted square up-right of them
        P = a[til, :2]
        upright_of = np.zeros(len(a), bool)
        for i in range(0, len(a), 256):
            upright_of[i:i + 256] = np.any((P[None, :, 0] > a[i:i + 256, None, 0]) & (P[None, :, 1] > a[i:i + 256, None, 1]), axis=1)
        moving = (~active) & ~upright_of
    static = (~active) & ~moving
    # keep only rigid squares within `layer` of an active square
    A = a[active, :2]
    dl = np.full(len(a), 1e9)
    for i in range(0, len(a), 256):
        dl[i:i + 256] = np.min(np.hypot(a[i:i + 256, None, 0] - A[None, :, 0], a[i:i + 256, None, 1] - A[None, :, 1]), axis=1)
    keep = active | (dl < layer)
    idx = np.nonzero(keep)[0]
    return idx, active[idx], static[idx], moving[idx], moving


def run_squeeze(a, L, K, idx, stat, mov, mov_full, X=None, Y=None, T=None, **kw):
    """Squeeze the reduced model; returns the full updated configuration and the new L."""
    sub = a[idx]
    X0 = sub[:, 0].copy() if X is None else X; Y0 = sub[:, 1].copy() if Y is None else Y; T0 = sub[:, 2].copy() if T is None else T
    f0 = L - K
    f, Xn, Yn, Tn, v, it = squeeze(X0, Y0, T0, K, K, f0, frozen=stat, moving=mov, **kw)
    full = a.copy()
    full[idx, 0] = Xn; full[idx, 1] = Yn; full[idx, 2] = Tn
    rest = np.ones(len(a), bool); rest[idx] = False
    shift = f - f0
    full[rest & mov_full, 0] += shift; full[rest & mov_full, 1] += shift
    return full, K + f, v, it


def full_viol(full, Lnew):
    mp, mw = feas(full[:, 0].copy(), full[:, 1].copy(), full[:, 2].copy(), Lnew, Lnew)
    return max(0.0, -min(mp, mw))
