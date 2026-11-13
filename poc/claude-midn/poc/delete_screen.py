#!/usr/bin/env python3
"""Whole-range screen for `delete_slp`: one worker per target n, k = 1.

For target n: price every square of the (n+1) packing with one SLP step
(square removed), then compress the TOP best-priced deletions for ITERS
steps and report how close the box gets to L(n).  States are saved so that
promising targets can be continued with `delete_slp.py`-style deep runs.

usage: delete_screen.py FULL.csv LO-HI|N,N,.. JOBS OUT_DIR [TOP] [ITERS]
writes OUT_DIR/screen.jsonl (one line per n) and OUT_DIR/state_<n>.npz
A hit (reached < L(n)) is gated by the official metric in both transports and
written to OUT_DIR/win_<n>.csv.
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
import free_slp as FS  # noqa: E402

G = {}


def arr(grp):
    return (np.array([[float(r["x"]), float(r["y"])] for r in grp]), np.array([float(r["angle"]) for r in grp]))


def gate(n, c, th, inc_rows):
    old = R.both(n, inc_rows)
    mid = 0.5 * (c.min(0) + c.max(0))
    for dil in (0.0, 2e-13, 2e-12, 2e-11, 2e-10):
        cd = mid + (c - mid) * (1 + dil)
        cand = [dict(n=str(n), square_id=str(t), x=repr(float(cd[t, 0])), y=repr(float(cd[t, 1])),
                     angle=repr(float(th[t]))) for t in range(n)]
        new = R.both(n, cand)
        if new is not None:
            return (new[0] < old[0] and new[1] < old[1]), cand, old, new
    return False, None, old, None


def work(n):
    groups, out, top, iters = G["groups"], G["out"], G["top"], G["iters"]
    try:
        ct, tht = arr(groups[n])
        target = FS.measure(ct, tht)[0]
        c, th = arr(groups[n + 1])
        donor = FS.measure(c, th)[0]
        prices = []
        for q in range(n + 1):
            keep = np.arange(n + 1) != q
            o = FS.slp_step(c[keep], th[keep], 1e-3, 5e-4)
            prices.append(o[2] if o else 0.0)
        prices = np.array(prices)
        info = dict(n=n, target=target, donor=donor, priced=int((prices < -1e-9).sum()), best_price=float(prices.min()))
        if info["priced"] == 0:
            info["status"] = "rigid"
            return info
        # distinct candidates: skip squares whose price ties one already taken and that sit next to it
        order = [int(q) for q in np.argsort(prices) if prices[q] < -1e-9]
        picked = []
        for q in order:
            if all(np.hypot(*(c[q] - c[p])) > 2.5 for p in picked):
                picked.append(q)
            if len(picked) == top:
                break
        best = None
        for q in picked:
            keep = np.arange(n + 1) != q
            c2, th2, _b, cur, _log = FS.refine(c[keep], th[keep], iters, 2e-3)
            if best is None or cur < best[2]:
                best = (c2, th2, cur, q)
        np.savez(out / f"state_{n}.npz", c=best[0], th=best[1])
        info.update(reached=best[2], gap=best[2] - target, deleted=best[3], tried=len(picked),
                    recovered=(donor - best[2]) / max(donor - target, 1e-300))
        if best[2] < target - 1e-12:
            ok, cand, old, new = gate(n, best[0], best[1], groups[n])
            if ok:
                R.write_rows(out / f"win_{n}.csv", {n: cand})
                info.update(status="WIN", old_literal=str(old[0]), new_literal=str(new[0]),
                            old_pandas=str(old[1]), new_pandas=str(new[1]))
            else:
                info["status"] = "official-not-strict"
        else:
            info["status"] = "above-target"
        return info
    except Exception as err:  # noqa: BLE001
        return dict(n=n, status="error", error=repr(err)[:300])


def main():
    path, spec, jobs, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    G["top"] = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    G["iters"] = int(sys.argv[6]) if len(sys.argv) > 6 else 40
    if "-" in spec:
        lo, hi = map(int, spec.split("-")); ns = list(range(lo, hi + 1))
    else:
        ns = [int(v) for v in spec.split(",")]
    out.mkdir(parents=True, exist_ok=True)
    G["out"] = out
    G["groups"] = R.read_groups(path, set(ns) | {n + 1 for n in ns})
    with mp.get_context("fork").Pool(jobs) as pool, open(out / "screen.jsonl", "a") as f:
        for info in pool.imap_unordered(work, ns, chunksize=1):
            f.write(json.dumps(info) + "\n"); f.flush()
            if info.get("status") in ("WIN", "official-not-strict") or info.get("gap", 1) < 5e-4:
                print({k: v for k, v in info.items() if k not in ("old_pandas", "new_pandas")}, flush=True)


if __name__ == "__main__":
    main()
