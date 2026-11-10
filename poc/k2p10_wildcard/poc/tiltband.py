"""Counting model: w rows of tilted unit squares (rows slide independently, optional upright
corner caps at SW/NE) between two upright grids (each grid: column phase 0 or f=frac(L), one split).
best_count(L, theta) -> (count, params). Pure counting; packings built in build.py."""
import numpy as np, itertools, sys

def row_interval(d, th, L, caps):
    s, co = np.sin(th), np.cos(th); h = s + co
    pl = np.maximum((h/2 + s*d)/co, (h/2 - co*d)/s)
    ph = np.minimum((L - h/2 + s*d)/co, (L - h/2 - co*d)/s)
    for corner in caps:
        c0 = np.array([.5, .5]) if corner == "SW" else np.array([L-.5, L-.5])
        a = np.full_like(d, -1e9); b = np.full_like(d, 1e9)
        nrm = np.array([-s, co]); t = np.array([co, s])
        for u in (np.array([1., 0]), np.array([0, 1.]), t, nrm):
            r = .5*(abs(u[0])+abs(u[1])) + .5*(abs(u@t)+abs(u@nrm))
            # |u.(d nrm + p t - c0)| < r
            k = u@t; base = d*(u@nrm) - u@c0
            if abs(k) < 1e-12:
                bad = np.abs(base) >= r
                a = np.where(bad, 1e9, a); b = np.where(bad, -1e9, b)
            else:
                p1 = (-r - base)/k; p2 = (r - base)/k
                a = np.maximum(a, np.minimum(p1, p2)); b = np.minimum(b, np.maximum(p1, p2))
        ov = a < b
        if corner == "SW": pl = np.where(ov, np.maximum(pl, b), pl)
        else: ph = np.where(ov, np.minimum(ph, a), ph)
    return pl, ph

def best_count(L, th, step_den=50, wmax=28, ret_all=False):
    s, co = np.sin(th), np.cos(th)
    f = L - np.floor(L); m = int(np.floor(L + 1e-12))
    cs = np.arange(-s*L - 1, co*L + 1, 1.0/step_den); N = len(cs); eps = 1.0/step_den
    def mk(xs, top):
        X, Y = np.meshgrid(np.asarray(xs, float), (f if top else 0.0) + np.arange(m), indexing="ij")
        return X.ravel(), Y.ravel()
    def cfgs(top):
        A = np.arange(m); B = f + np.arange(m)
        out = [A, B]
        for k in range(1, m):
            out.append(np.concatenate([np.arange(k), f + np.arange(k, m)]))
            out.append(np.concatenate([f + np.arange(k), np.arange(k+1, m)]))
        return [mk(xs[xs + 1 <= L + 1e-9], top) for xs in out]
    best = (0, None)
    for caps in ([], ["SW"], ["NE"], ["SW", "NE"]):
        pl, ph = row_interval(cs + .5, th, L, caps)
        rc = np.where(ph - pl >= -1e-9, np.floor(ph - pl + 1e-9) + 1, 0).astype(int)
        def keep(x, y):
            k = np.ones(len(x), bool)
            if "SW" in caps: k &= ~((x < 1 - 1e-9) & (y < 1 - 1e-9))
            if "NE" in caps: k &= ~((x + 1 > L - 1 + 1e-9) & (y + 1 > L - 1 + 1e-9))
            return k
        NB = np.zeros(N, int)
        for x, y in cfgs(False):
            k = keep(x, y); v = np.sort(-s*x[k] + co*(y[k] + 1))
            NB = np.maximum(NB, np.searchsorted(v, cs + eps + 1e-9, side="right"))
        NTv = []
        for x, y in cfgs(True):
            k = keep(x, y); NTv.append(np.sort(-s*(x[k] + 1) + co*y[k]))
        ext = np.arange(-s*L - 1, co*L + 1 + wmax + 1, 1.0/step_den)
        TC = np.zeros(len(ext), int)
        for v in NTv: TC = np.maximum(TC, len(v) - np.searchsorted(v, ext - 1e-9, side="left"))
        acc = np.zeros(N, int)
        for w in range(0, wmax):
            if w > 0:
                sh = np.zeros(N, int)
                if (w-1)*step_den < N: sh[:N-(w-1)*step_den] = rc[(w-1)*step_den:]
                acc = acc + sh
            tot = NB + TC[w*step_den:w*step_den+N] + acc + len(caps)
            k = int(np.argmax(tot))
            if tot[k] > best[0]: best = (int(tot[k]), (w, float(cs[k]), tuple(caps)))
    return best

if __name__ == "__main__":
    K = int(sys.argv[1]); L = float(sys.argv[2])
    out = []
    for th in list(np.arange(0.20, 1.37, 0.01)) + [np.pi/4]:
        out.append((best_count(L, th), th))
    out.sort(key=lambda z: -z[0][0])
    print(K, L, K*K, "need", sys.argv[3] if len(sys.argv) > 3 else "")
    for o in out[:8]: print(o)
    print("45deg", best_count(L, np.pi/4))
