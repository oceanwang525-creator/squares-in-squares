import sys, math, numpy as np
sys.path.insert(0, "/Users/lucaszhang/squares-in-squares")
sys.path.insert(0, "/Users/lucaszhang/squares-in-squares/poc/baseline/poc/src")
from poc.pell_corrugation.poc.pell_corrugation import build_staircase_band_45, add_l_border_exact, extent, sat_check, polish_band
def build(t,w,a1,a2,b1,b2,ell,K):
    q, L0 = build_staircase_band_45(w, ell, a1, a2, b1, b2)
    if t: q = add_l_border_exact(q, L0, t, K)
    return q, L0 + t
if __name__ == "__main__":
    t,w,a1,a2,b1,b2,ell,K = map(int, sys.argv[1:9])
    q, L = build(t,w,a1,a2,b1,b2,ell,K)
    print("n", len(q), "L", L, "extent", extent(q), "sat", sat_check(q))
