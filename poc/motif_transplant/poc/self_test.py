"""Self-test: fix only the off-lattice squares of a packing, greedily refill the rest with the
four corner-anchored upright lattices, and count how many squares come back."""
import sys
from motif import load, off_lattice, clusters, fill

path = sys.argv[1]; ns = [int(x) for x in sys.argv[2:]]
for n, (a, L) in sorted(load(path, set(ns)).items()):
    m = off_lattice(a, L); cl = clusters(a, m)
    for tol in (-1e-9, -1e-7):
        P = fill(L + 1e-9, a[m], tol)
        print(f'n={n} L={L:.6f} off-lattice={int(m.sum())} clusters={[len(c) for c in cl]} tol={tol:g} rebuilt={len(P)}/{n}')
