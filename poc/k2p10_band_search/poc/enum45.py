"""Exhaustive enumeration of 45deg staircase-band cores + t-layer L-border (w,a1,a2,b1,b2,t) at L in [target, ours].
Count = exact squares of pell_corrugation.build_staircase_band_45 (+border), with max column count l."""
import sys, math, json, itertools
S2 = math.sqrt(2); I2 = 1/S2
T = lambda a: a*(a+1)//2
def count(w, a1, a2, b1, b2, t, K):
    L0 = 0.5*(a1+a2) + 1 + w*I2
    c = T(a1)+T(a2)+T(b1)+T(b2)
    if a1 >= 1 and a1+b1 > L0+1e-9: c -= 1
    if a1 >= 1 and a1+b2 > L0+1e-9: c -= 1
    if a2 >= 1 and a2+b1 > L0+1e-9: c -= 1
    if a2 >= 1 and a2+b2 > L0+1e-9: c -= 1
    u_wall = 0.5*(w-1) + 0.5*abs(a1-a2)*I2 + I2
    u_lo = max((b1+1)*I2+.5, u_wall); u_hi = S2*L0 - max((b2+1)*I2+.5, u_wall)
    span = u_hi - u_lo
    if span < -1e-9: return None
    l = int(math.floor(span + 1e-9)) + 1
    f = math.floor(L0+1e-9)+1
    if max(a1+b1,a1+b2,a2+b1,a2+b2) > f: return None
    k_core = K - t
    border = t*K + k_core*t
    return c + w*l + border, l, L0 + t
def run(n, target, ours, K):
    out = []
    for w in range(1, 40):
        for t in range(0, K):
            for s in range(0, 2*K):
                L = 0.5*s + 1 + w*I2 + t
                if not (target - 1e-6 <= L <= ours + 1e-6): continue
                if int(math.floor(L)) != K: continue
                for a1 in range(0, s+1):
                    a2 = s - a1
                    if a2 > a1: continue
                    # staircases must fit: a<=L0, band must fit the wall
                    L0 = L - t
                    if a1 > L0 + 1e-9: continue
                    for b1 in range(0, int(L0)+1):
                        for b2 in range(b1, int(L0)+1):
                            r = count(w, a1, a2, b1, b2, t, K)
                            if r is None: continue
                            out.append((r[0], L, w, a1, a2, b1, b2, t, r[1]))
    out.sort(reverse=True)
    return out
if __name__ == "__main__":
    n, target, ours, K = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    res = run(n, target, ours, K)
    print(n, "configs", len(res), "best counts", [r[0] for r in res[:8]])
    best = {}
    for r in res:
        k = round(r[1], 5)
        if k not in best or r[0] > best[k][0]: best[k] = r
    for k in sorted(best): print(k, best[k])
    json.dump(res[:200], open(f"docs/enum45_{n}.json", "w"))
