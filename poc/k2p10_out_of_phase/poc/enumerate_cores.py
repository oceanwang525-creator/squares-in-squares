"""Enumerate (t, w, a1, a2, b1, b2) staircase-band cores + t-layer L-border for K^2+10; list L45 and square count."""
import math, sys
S2 = math.sqrt(2)
T = lambda a: a*(a+1)//2
def run(K, n, Linc, tol=0.004):
    out = []
    for t in range(0, K-3):
        for w in range(1, 30):
            for a1 in range(0, K+1):
                for a2 in range(0, a1+1):
                    s = a1+a2
                    L = s/2 + 1 + w/S2 + t
                    if L > Linc + tol or L < K - 0.3: continue
                    for b1 in range(0, 12):
                        for b2 in range(b1, 12):
                            L0 = L - t
                            if max(a1,a2)+max(b1,b2) > L0+1e-9: continue
                            u_wall = 0.5*(w-1)+0.5*abs(a1-a2)/S2+1/S2
                            u_lo = max((b1+1)/S2+0.5, u_wall)
                            u_hi = S2*L0 - max((b2+1)/S2+0.5, u_wall)
                            if u_hi < u_lo: continue
                            ell = 1+int(u_hi-u_lo+1e-9)
                            D = s-b1-b2
                            core = T(a1)+T(a2)+T(b1)+T(b2)+w*ell
                            border = K*K-(K-t)**2 if t else 0
                            # border uses k_total = ceil-ish; here L_total=L so k_total=K
                            tot = core + border
                            out.append((tot, L, s % 2, t, w, a1, a2, b1, b2, ell, D))
    return out
if __name__ == "__main__":
    for K, n, Linc in ((25,635,25.7071067812),(26,686,26.7071067812)):
        rs = run(K, n, Linc)
        print(f"== K={K} n={n} Linc={Linc} configs {len(rs)}")
        for par in (0,1):
            sub = [r for r in rs if r[2]==par and r[1] <= Linc+0.004]
            sub.sort(key=lambda r:(-r[0], r[1]))
            print(" parity", "odd(out-of-phase)" if par else "even(in-phase)", "best counts (count,L45,t,w,a1,a2,b1,b2,ell,D):")
            for r in sub[:6]: print("  ", r[0], round(r[1],5), r[3:])
            ok = [r for r in sub if r[0] >= n]
            print("  configs with count>=n:", len(ok), "min L45:", min((r[1] for r in ok), default=None))
