"""Basin hopping from a saved b_fill result: noise on all squares, re-relax. usage: b_hop.py file.npy Lc hops sigma seed"""
import sys, numpy as np
sys.path.insert(0, "/Users/lucaszhang/squares-in-squares/poc/k2p10_fill2/poc")
import b_fill as f, b_overlap as bo
P = np.load(sys.argv[1]); Lc = float(sys.argv[2]); hops = int(sys.argv[3]); sg = float(sys.argv[4])
rng = np.random.default_rng(int(sys.argv[5]))
P, E = f.relax(P, Lc, 5000); best = (E, P.copy())
print("start", E, flush=True)
for h in range(hops):
    Q = best[1].copy(); Q += rng.normal(0, sg, Q.shape) * np.array([1, 1, 0.5])
    Q, E = f.relax(Q, Lc + 0.02, 1500); Q, E = f.relax(Q, Lc, 5000)
    print(h, E, bo.energy(Q, Lc)[2]["maxd"], flush=True)
    if E < best[0]: best = (E, Q.copy()); np.save(sys.argv[1].replace(".npy", "_hop.npy"), Q)
print("BEST", best[0], bo.energy(best[1], Lc)[2])
