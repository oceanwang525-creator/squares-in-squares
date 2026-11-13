#!/usr/bin/env python3
"""Shadow-priced deletion + free-angle SLP compression.

For a target n take the packing of n+k (k = 1..KMAX), delete k squares and
compress the box with the free-angle SLP of `free_slp.py`.  Unlike
`poc/neighbour_deletion` (scale down + L-BFGS penalty, 24 hand-picked deletion
sets per target) the squares to delete are chosen by their LP shadow price: for
every square of the donor one SLP step is solved with that square removed, and
the predicted first-order drop of the box side ranks the candidates.  Only the
top TOP candidates get the full SLP.  For k > 1 the deletions are greedy
(delete the best, recompute prices on the compressed result).
DELETE_PREFILTER=m prices only the m squares with the largest LP dual mass.

usage: delete_slp.py FULL.csv N_LIST JOBS OUT_ROWS.csv REPORT.json [KMAX] [TOP] [ITERS]
A row is emitted only if both official transports are strictly below the
incumbent group for the target n.
"""
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rows as R  # noqa: E402
import tight_lp as TL  # noqa: E402
import free_slp as FS  # noqa: E402

G = {}
PREFILTER = int(os.environ.get("DELETE_PREFILTER", "0"))   # 0 = price every square


def arr(grp):
    return (np.array([[float(r["x"]), float(r["y"])] for r in grp]), np.array([float(r["angle"]) for r in grp]))


def price(args):
    key, q = args
    c, th = G[key]
    keep = np.arange(len(c)) != q
    out = FS.slp_step(c[keep], th[keep], 1e-3, 5e-4)
    return q, (out[2] if out else 0.0)


def compress(args):
    key, q, iters = args
    c, th = G[key]
    keep = np.arange(len(c)) != q
    c2, th2, base, cur, log = FS.refine(c[keep], th[keep], iters, 2e-3)
    return q, c2, th2, cur


def main():
    path, ns, jobs, out_rows, out_rep = sys.argv[1], [int(v) for v in sys.argv[2].split(",")], int(sys.argv[3]), sys.argv[4], sys.argv[5]
    kmax = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    top = int(sys.argv[7]) if len(sys.argv) > 7 else 6
    iters = int(sys.argv[8]) if len(sys.argv) > 8 else 60
    need = set(ns) | {n + k for n in ns for k in range(1, kmax + 1)}
    groups = R.read_groups(path, need)
    wins, rep = {}, []
    for n in ns:
        c_t, th_t = arr(groups[n])
        target = FS.measure(c_t, th_t)[0]
        for k in range(1, kmax + 1):
            c, th = arr(groups[n + k])
            donor = FS.measure(c, th)[0]
            if donor <= target + 1e-12:
                continue
            state = [(c, th)]
            for depth in range(k):
                best = None
                for si, (cs, ths) in enumerate(state):
                    G["cur"] = (cs, ths)
                    order = range(len(cs))
                    if PREFILTER:
                        # cheap prefilter: squares carrying the most dual mass in the donor's own LP
                        # are the ones sitting on the chains that hold the box side
                        full = FS.slp_step(cs, ths, 1e-3, 5e-4, want_duals=True)
                        if full is not None:
                            order = np.argsort(-full[3])[:PREFILTER]
                    with mp.get_context("fork").Pool(jobs) as pool:
                        prices = sorted(pool.imap_unordered(price, [("cur", int(q)) for q in order], chunksize=4), key=lambda t: t[1])
                        cand = [q for q, p in prices[:top] if p < -1e-9]
                        res = pool.map(compress, [("cur", q, iters) for q in cand])
                    for q, c2, th2, cur in res:
                        if best is None or cur < best[2]:
                            best = (c2, th2, cur, prices[0][1], len(cand))
                if best is None:
                    break
                state = [(best[0], best[1])]
            if best is None:
                rep.append(dict(n=n, k=k, target=target, donor=donor, status="no-priced-deletion"))
                print(rep[-1], flush=True)
                continue
            info = dict(n=n, k=k, target=target, donor=donor, reached=best[2], gap=best[2] - target,
                        best_price=best[3], tried=best[4])
            if best[2] < target - 1e-12:
                c2, th2 = best[0], best[1]
                cand_rows = [dict(n=str(n), square_id=str(t), x=repr(float(c2[t, 0])), y=repr(float(c2[t, 1])),
                                  angle=repr(float(th2[t]))) for t in range(n)]
                old, new = R.both(n, groups[n]), R.both(n, cand_rows)
                if new is None:
                    mid = 0.5 * (c2.min(0) + c2.max(0))
                    for dil in (2e-13, 2e-12, 2e-11, 2e-10):
                        cd = mid + (c2 - mid) * (1 + dil)
                        cand_rows = [dict(n=str(n), square_id=str(t), x=repr(float(cd[t, 0])), y=repr(float(cd[t, 1])),
                                          angle=repr(float(th2[t]))) for t in range(n)]
                        new = R.both(n, cand_rows)
                        if new is not None:
                            break
                if new is not None and new[0] < old[0] and new[1] < old[1]:
                    info.update(status="WIN", old_literal=str(old[0]), new_literal=str(new[0]),
                                old_pandas=str(old[1]), new_pandas=str(new[1]))
                    wins[n] = cand_rows
                    groups[n] = cand_rows
                else:
                    info["status"] = "official-not-strict"
            else:
                info["status"] = "above-target"
            rep.append(info)
            print(info, flush=True)
            json.dump(rep, open(out_rep, "w"), indent=1)
            R.write_rows(out_rows, wins)
    json.dump(rep, open(out_rep, "w"), indent=1)
    R.write_rows(out_rows, wins)


if __name__ == "__main__":
    main()
