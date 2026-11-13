"""Find minimal clean-cut rectangles around tilted clusters: (W, H, count) -> gadget (a, b, g, f)."""
import csv, math, sys, collections
import numpy as np

def load(path, ns):
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        n = int(r['n'])
        if n in ns: by[n].append((float(r['x']), float(r['y']), float(r['angle'])))
    out = {}
    for n, rows in by.items():
        a = np.array(rows); e = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
        a[:, 0] -= (a[:, 0] - e).min(); a[:, 1] -= (a[:, 1] - e).min()
        out[n] = (a, e, max((a[:, 0] + e).max(), (a[:, 1] + e).max()))
    return out

def clusters(a, mask, link=1.5):
    idx = np.nonzero(mask)[0]; lab = {}; c = 0
    for s in idx:
        if s in lab: continue
        st = [s]; lab[s] = c
        while st:
            i = st.pop()
            d = np.hypot(a[idx, 0] - a[i, 0], a[idx, 1] - a[i, 1])
            for j in idx[d < link]:
                if j not in lab: lab[j] = c; st.append(j)
        c += 1
    return [np.array([i for i in idx if lab[i] == q]) for q in range(c)]

def clean_rect(a, e, x0, x1, y0, y1, tol=1e-7):
    """Grow the rectangle until no square straddles its boundary."""
    for _ in range(200):
        lo_x = a[:, 0] - e; hi_x = a[:, 0] + e; lo_y = a[:, 1] - e; hi_y = a[:, 1] + e
        inside = (hi_x > x0 + tol) & (lo_x < x1 - tol) & (hi_y > y0 + tol) & (lo_y < y1 - tol)   # overlaps rect
        nx0 = min(x0, lo_x[inside].min()); nx1 = max(x1, hi_x[inside].max())
        ny0 = min(y0, lo_y[inside].min()); ny1 = max(y1, hi_y[inside].max())
        if (nx0, nx1, ny0, ny1) == (x0, x1, y0, y1): break
        x0, x1, y0, y1 = nx0, nx1, ny0, ny1
    return x0, x1, y0, y1, int(inside.sum())

if __name__ == '__main__':
    path = sys.argv[1]; ns = [int(x) for x in sys.argv[2:]]
    for n, (a, e, L) in sorted(load(path, set(ns)).items()):
        t = np.degrees(np.mod(a[:, 2], np.pi / 2)); tilt = np.minimum(t, 90 - t) > 5
        K = int(L); print(f'n={n} L={L:.6f} K={K} j={n-K*K} c={L-K:.6f} strongly tilted={int(tilt.sum())}')
        for cl in clusters(a, tilt):
            if len(cl) < 6: continue
            x0 = (a[cl, 0] - e[cl]).min(); x1 = (a[cl, 0] + e[cl]).max(); y0 = (a[cl, 1] - e[cl]).min(); y1 = (a[cl, 1] + e[cl]).max()
            X0, X1, Y0, Y1, cnt = clean_rect(a, e, x0, x1, y0, y1)
            W, H = X1 - X0, Y1 - Y0
            A, Bq = int(W + 1e-9), int(H + 1e-9)
            ang = np.median(np.minimum(t[cl], 90 - t[cl]))
            print(f'   cluster {len(cl)} tilted @~{ang:.2f}deg bbox {x1-x0:.3f}x{y1-y0:.3f} -> clean rect {W:.4f} x {H:.4f} at ({X0:.3f},{Y0:.3f}) holds {cnt}: (a,b,g,f)=({A},{Bq},{cnt-A*Bq},{W-A:.4f}/{H-Bq:.4f})')
