"""Like enum45 but container side L is a free real (L >= L45 of the config): extra transverse slack is wasted,
but longitudinal span grows (more columns) and corner drops relax. Reports best count per tested L."""
import sys, math, json
S2 = math.sqrt(2); I2 = 1/S2
T = lambda a: a*(a+1)//2
def count(w, a1, a2, b1, b2, t, K, L):
    L0 = L - t
    c = T(a1)+T(a2)+T(b1)+T(b2)
    for a in (a1, a2):
        for b in (b1, b2):
            if a >= 1 and a+b > L0+1e-9: c -= 1
    if max(a1+b1,a1+b2,a2+b1,a2+b2) > math.floor(L0+1e-9)+1: return None
    if a1 > L0+1e-9: return None
    u_wall = 0.5*(w-1) + 0.5*abs(a1-a2)*I2 + I2
    u_lo = max((b1+1)*I2+.5, u_wall); u_hi = S2*L0 - max((b2+1)*I2+.5, u_wall)
    span = u_hi-u_lo
    if span < -1e-9: return None
    l = int(math.floor(span+1e-9))+1
    return c + w*l + t*K + (K-t)*t, l
def best_at(L, K):
    best = (0,)
    for w in range(1, 36):
      for t in range(0, 4):
        for s in range(0, 2*K):
            L45 = 0.5*s + 1 + w*I2
            if L45 > L - t + 1e-9 or L - t - L45 > 0.75: continue   # keep transverse slack < 0.75
            for a1 in range((s+1)//2, s+1):
                a2 = s-a1
                for b1 in range(0, int(L)+1):
                    for b2 in range(b1, int(L)+1):
                        r = count(w, a1, a2, b1, b2, t, K, L)
                        if r and r[0] > best[0]: best = (r[0], w, a1, a2, b1, b2, t, r[1], round(L45+t, 5))
    return best
if __name__ == "__main__":
    n, K, lo, hi = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    for i in range(7):
        L = lo + (hi-lo)*i/6 - (1e-6 if i == 6 else 0)
        print(n, round(L, 5), best_at(L, K), flush=True)
