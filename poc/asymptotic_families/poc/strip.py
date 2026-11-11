"""Tilted-column strip: strip height h=j+f (f=L-floor(L)), width L, columns of m=j+1 squares tilted by theta
(touching top and bottom: m cos t + sin t = h), columns spaced 1/cos t, left/right ends are wasted triangles.
Count N*m vs upright j*floor(L). Overlap-checked with k2p10_fill2 a_overlap."""
import sys, math, numpy as np
sys.path.insert(0, '/Users/lucaszhang/squares-in-squares/poc/k2p10_fill2/poc')
from a_overlap import overlap_pairs, wall_violation
from scipy.optimize import brentq

def theta_for(m, h):
    g = lambda t: m*math.cos(t)+math.sin(t)-h
    # smallest t>t* where g<=0 : g decreasing after peak at atan(1/m)
    t0 = math.atan(1/m)
    if g(t0) <= 0: return t0
    if g(math.pi/2) > 0: return None
    return brentq(g, t0, math.pi/2)

def strip_count(L, j, m=None):
    f = L - math.floor(L); h = j + f; m = m or j+1
    t = theta_for(m, h)
    if t is None: return None
    c, s = math.cos(t), math.sin(t)
    # column k: bottom-left corner of first square at (x0+k/c, 0)+..., column leans right by m*s
    N = int(math.floor((L - m*s - c)*c + 1 + 1e-12)) if L >= m*s+c else 0
    return t, N, N*m, j*math.floor(L), h

def build(L, j, m=None):
    r = strip_count(L, j, m); t, N, cnt, up, h = r; m = m or j+1
    c, s = math.cos(t), math.sin(t)
    # column leaning right: squares rotated by -t.. use angle=-t so up-axis (-sin(-t)?)=...; place by axis u=(sin t,cos t)
    # square i centre = origin_k + (i+.5)*u + .5*v, v=(cos t,-sin t) ; base corner chosen so lowest vertex is y=0
    q = []
    for k in range(N):
        bx = k/c
        for i in range(m):
            cx = bx + (i+.5)*s + .5*c
            cy = (i+.5)*c - .5*s + s   # shift so min y>=0 (column's lowest corner at y=0)
            q.append((cx, cy, -t))
    return np.array(q)

if __name__ == '__main__':
    L = float(sys.argv[1]) if len(sys.argv) > 1 else 25.7071067812
    print('L', L)
    print('j m theta N tilted upright(j*floor L) gain')
    for j in range(1, 26):
        for m in (j, j+1, j+2):
            r = strip_count(L, j, m) if m != j else None
            if r is None: continue
            t, N, cnt, up, h = r
            print(j, m, round(t, 4), N, cnt, up, cnt-up)
    # overlap check of the j with best gain
    best = max(((strip_count(L, j) or (0,0,-99,0,0))[2] - j*math.floor(L), j) for j in range(1, 26))
    print('best (gain, j):', best)
    q = build(L, best[1]); P, d = overlap_pairs(q, 1e-9); v, _ = wall_violation(q, 1e9)
    lo = (q[:, :2] - 0.5*(abs(np.cos(q[:,2]))+abs(np.sin(q[:,2])))[:,None]).min()
    h = best[1] + L - math.floor(L)
    top = (q[:,1] + 0.5*(abs(np.cos(q[:,2]))+abs(np.sin(q[:,2])))).max()
    right = (q[:,0] + 0.5*(abs(np.cos(q[:,2]))+abs(np.sin(q[:,2])))).max()
    print('overlaps', len(P), 'min coord', lo, 'top', top, 'h', h, 'right', right, 'L', L)
