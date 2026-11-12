"""Exact 45-degree tilted m x p rectangle + four corner staircases (a=SW, b=NE, s=SE, t=NW).
For each target n, find the smallest L for which such a configuration holds >= n squares,
then build it explicitly and check every pair with SAT.
"""
import itertools, math, sys
import numpy as np
from bandgen import sat_overlap

R2 = math.sqrt(2)
T = lambda x: x * (x + 1) // 2


def min_L(m, p, a, b, s, t):
    """Smallest L satisfying the face constraints; tips/walls checked in build()."""
    return max((a + b + 2 + R2 * p) / 2, (s + t + 2 + R2 * m) / 2, m / R2 + p / R2)  # bbox >= (m+p)/sqrt2


def build(L, m, p, a, b, s, t, eps=1e-12):
    # choose u0 / v faces in the middle of their feasible ranges, then check tips
    U1 = (a + 1) / R2 + 0.5; U2 = (2 * L - b - 1) / R2 - p + 0.5
    Vhi_max = (L - 1 - s) / R2; Vlo_min = (t + 1 - L) / R2
    best = None
    for fu in np.linspace(0, 1, 21):
        u0 = U1 + fu * (U2 - U1)
        for fv in np.linspace(0, 1, 21):
            vlo = Vlo_min + fv * (Vhi_max - m - Vlo_min)
            ok = True
            for uu in (u0 - 0.5, u0 + p - 0.5):
                for vv in (vlo, vlo + m):
                    x = (uu + vv) / R2; y = (uu - vv) / R2
                    if x < -eps or y < -eps or x > L + eps or y > L + eps:
                        ok = False
            if ok:
                best = (u0, vlo); break
        if best: break
    if best is None:
        return None
    u0, vlo = best
    P = []
    for r in range(m):
        v = vlo + 0.5 + r
        for c in range(p):
            u = u0 + c
            P.append(((u + v) / R2, (u - v) / R2, math.pi / 4))
    for (n_, ax, ay) in ((a, 0, 0), (b, 1, 1), (s, 1, 0), (t, 0, 1)):
        for i in range(n_):
            for j in range(n_ - i):
                x = L - 0.5 - i if ax else 0.5 + i
                y = L - 0.5 - j if ay else 0.5 + j
                P.append((x, y, 0.0))
    return np.array(P)


def check(P, L, tol=1e-9):
    e = 0.5 * (np.abs(np.cos(P[:, 2])) + np.abs(np.sin(P[:, 2])))
    if (P[:, 0] - e).min() < -tol or (P[:, 1] - e).min() < -tol or (P[:, 0] + e).max() > L + tol or (P[:, 1] + e).max() > L + tol:
        return False
    for i in range(len(P)):
        d = np.hypot(P[i + 1:, 0] - P[i, 0], P[i + 1:, 1] - P[i, 1])
        for j in np.nonzero(d < 1.415)[0] + i + 1:
            if sat_overlap(P[i, 0], P[i, 1], P[i, 2], P[j, 0], P[j, 1], P[j, 2], -tol):
                return False
    return True


def search(n, Lmax):
    k = math.isqrt(n)
    res = []
    for m in range(1, 2 * k + 2):
        for p in range(m, 2 * k + 2):
            if (m + p) / R2 > Lmax:
                break
            if m * p > n:
                break
            Su = math.floor(2 * Lmax - 2 - R2 * p + 1e-12); Sv = math.floor(2 * Lmax - 2 - R2 * m + 1e-12)
            if Su < 0 or Sv < 0:
                continue
            bestc = None
            for a in range(0, min(Su, k) + 1):
                for b in range(0, min(Su - a, k) + 1):
                    for s_ in range(0, min(Sv, k - a, k - b) + 1):
                        t_ = min(Sv - s_, k - a, k - b)
                        if t_ < 0: continue
                        cnt = m * p + T(a) + T(b) + T(s_) + T(t_)
                        if cnt >= n:
                            L = min_L(m, p, a, b, s_, t_)
                            if L <= Lmax and (bestc is None or L < bestc[0]):
                                bestc = (L, m, p, a, b, s_, t_, cnt)
            if bestc:
                res.append(bestc)
    res.sort()
    return res


if __name__ == "__main__" and sys.argv[1] != "exact":
    n = int(sys.argv[1]); Lmax = float(sys.argv[2])
    out = search(n, Lmax)
    print(len(out), 'candidate shapes with L <=', Lmax)
    shown = 0
    for L, m, p, a, b, s_, t_, cnt in out:
        P = build(L + 1e-9, m, p, a, b, s_, t_)
        ok = P is not None and len(P) == cnt and check(P, L + 1e-9)
        print(f'L={L:.6f} m={m} p={p} a={a} b={b} s={s_} t={t_} count={cnt} built_ok={ok}')
        shown += 1
        if shown >= 12: break


from scipy.optimize import linprog


def lp_L(m, p, a, b, s, t):
    """min L over (u0, vlo, L) subject to face, staircase and corner constraints."""
    A = []; B = []
    # u0 >= (a+1)/R2 + .5            -> -u0 <= -(...)
    A.append([-1, 0, 0]); B.append(-((a + 1) / R2 + 0.5))
    # u0 <= (2L - b - 1)/R2 - p + .5  -> u0 - 2L/R2 <= -(b+1)/R2 - p + .5
    A.append([1, 0, -2 / R2]); B.append(-(b + 1) / R2 - p + 0.5)
    # vlo >= (t + 1 - L)/R2           -> -vlo - L/R2 <= -(t+1)/R2
    A.append([0, -1, -1 / R2]); B.append(-(t + 1) / R2)
    # vlo + m <= (L - 1 - s)/R2       -> vlo - L/R2 <= -(1+s)/R2 - m
    A.append([0, 1, -1 / R2]); B.append(-(1 + s) / R2 - m)
    for du in (-0.5, p - 0.5):
        for dv in (0, m):
            # x = (u0+du+vlo+dv)/R2 in [0,L]; y = (u0+du-vlo-dv)/R2 in [0,L]
            A.append([-1, -1, 0]); B.append(du + dv)
            A.append([1, 1, -R2]); B.append(-(du + dv))
            A.append([-1, 1, 0]); B.append(du - dv)
            A.append([1, -1, -R2]); B.append(-(du - dv))
    r = linprog([0, 0, 1], A_ub=A, b_ub=B, bounds=[(None, None)] * 3, method='highs')
    return (r.x[2], r.x[0], r.x[1]) if r.success else None


def build_at(L, u0, vlo, m, p, a, b, s, t):
    P = []
    for rr in range(m):
        v = vlo + 0.5 + rr
        for c in range(p):
            u = u0 + c
            P.append(((u + v) / R2, (u - v) / R2, math.pi / 4))
    for (n_, ax, ay) in ((a, 0, 0), (b, 1, 1), (s, 1, 0), (t, 0, 1)):
        for i in range(n_):
            for j in range(n_ - i):
                P.append((L - 0.5 - i if ax else 0.5 + i, L - 0.5 - j if ay else 0.5 + j, 0.0))
    return np.array(P)


def search_exact(n, Lmax):
    k = math.isqrt(n); out = []
    for m in range(1, 2 * k + 2):
        for p in range(m, 2 * k + 2):
            if (m + p) / R2 > Lmax or m * p > n + 2 * k:
                break
            Su = math.floor(2 * Lmax - 2 - R2 * p + 1e-12); Sv = math.floor(2 * Lmax - 2 - R2 * m + 1e-12)
            if Su < 0 or Sv < 0:
                continue
            for a in range(0, min(Su, k) + 1):
                for b in range(0, min(Su - a, k) + 1):
                    for s_ in range(0, min(Sv, k) + 1):
                        for t_ in range(0, min(Sv - s_, k) + 1):
                            if a + s_ > k or a + t_ > k or b + s_ > k or b + t_ > k:
                                continue
                            cnt = m * p + T(a) + T(b) + T(s_) + T(t_)
                            if cnt < n or cnt > n + 3:
                                continue
                            r = lp_L(m, p, a, b, s_, t_)
                            if r and r[0] <= Lmax:
                                out.append((r[0], m, p, a, b, s_, t_, cnt, r[1], r[2]))
    out.sort()
    return out


def main2():
    n = int(sys.argv[2]); Lmax = float(sys.argv[3])
    out = search_exact(n, Lmax)
    print(n, len(out), 'LP-feasible shapes with L <=', Lmax)
    shown = 0
    for L, m, p, a, b, s_, t_, cnt, u0, vlo in out:
        P = build_at(L + 1e-9, u0, vlo, m, p, a, b, s_, t_)
        ok = check(P, L + 1e-9)
        if ok:
            print(f'  VALID L={L:.6f} m={m} p={p} a={a} b={b} s={s_} t={t_} count={cnt}')
            shown += 1
            if shown >= 5: break
if __name__ == "__main__" and sys.argv[1] == "exact": main2()
