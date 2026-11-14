"""Iterated (transitive) closure of the corner lift.

`scan_lift.py` lifts the *tracked* rows.  A lift produces a new packing with a
new bounding box, which is itself a legal donor for a further lift, so the
operation must be closed by iteration:

    library := tracked rows
    repeat:  for every (target n, donor p, j, i) with L < library[n].L
                 build + certify the lift; if certified, library[n] := lift

Each accepted lift is a *strict certified* improvement on the tracked row for
that `n` (dL < -1e-9 in the literal transport), so the loop can only build
packings that are at least as good as anything tracked.

    python closure.py --rounds 6 --jobs 16
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mproc
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "poc" / "baseline" / "poc" / "src"))
sys.path.insert(0, str(HERE))

import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

import scan_lift as SL  # noqa: E402

DATA = HERE.parent / "data"


def library_bboxes(lib):
    out = {}
    for n, rows in lib.items():
        q = np.array([(float(x), float(y), float(t)) for x, y, t in rows])
        h = 0.5 * (np.abs(np.cos(q[:, 2])) + np.abs(np.sin(q[:, 2])))
        out[n] = (float((q[:, 0] + h).max() - (q[:, 0] - h).min()),
                  float((q[:, 1] + h).max() - (q[:, 1] - h).min()))
    return out


def candidates(lib, bboxes, sides):
    """All lifts with a float L strictly below the library side."""
    Ps = np.array(sorted(bboxes), dtype=int)
    W = np.array([bboxes[p][0] for p in Ps])
    H = np.array([bboxes[p][1] for p in Ps])
    C = np.floor(W)
    out = []
    for n, L0 in sides.items():
        K = math.isqrt(n)
        if K < 2:
            continue
        need = n - Ps
        for j in range(0, K + 1):
            rest = need - j * K
            with np.errstate(divide="ignore", invalid="ignore"):
                i = np.ceil(rest / C)
            i = np.where(np.asarray(i) < 0, 0, i)
            i = np.nan_to_num(i, nan=0.0, posinf=1e9)
            L = np.maximum.reduce([W + j, H + i, np.full(len(Ps), float(K))])
            ok = L < L0 - 1e-9
            for idx in np.nonzero(ok)[0]:
                out.append((float(L[idx]), n, int(Ps[idx]), j, int(i[idx])))
    out.sort()
    best = {}
    for L, n, p, j, i in out:
        best.setdefault(n, [])
        if len(best[n]) < 3:            # keep the 3 cheapest shapes per n
            best[n].append((L, n, p, j, i))
    return [c for v in best.values() for c in v]


def verify(job):
    """build + certify one lift; returns (n, rows, L_lit, L_pandas) or None."""
    n, donor_rows, K, j, i = job
    try:
        rows = SL.build_lift(donor_rows, K, j, i, n)
        lit, pan = SL.certify(n, rows)
        return n, rows, lit, pan
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()

    lib = SL.load_rows()
    (DATA / "library_round0.json").write_text(
        json.dumps({str(n): len(r) for n, r in lib.items()}))
    gains = []
    for rnd in range(1, a.rounds + 1):
        bboxes = library_bboxes(lib)
        sides = {n: max(wh) for n, wh in bboxes.items()}
        cand = candidates(lib, bboxes, sides)
        print(f"round {rnd}: {len(cand)} candidate lifts", flush=True)
        if not cand:
            break
        jobs = [(n, lib[p], math.isqrt(n), j, i) for _, n, p, j, i in cand]
        with mp.workdps(70):
            with mproc.get_context("spawn").Pool(a.jobs) as pool:
                res = pool.map(verify, jobs)
        changed = 0
        for (L, n, p, j, i), r in zip(cand, res):
            if r is None:
                continue
            n2, rows, lit, pan = r
            cur = mp.mpf(SL.certify(n, [dict(square_id=str(k), x=x, y=y, angle=t)
                                        for k, (x, y, t) in enumerate(lib[n])])[0])
            d = mp.mpf(lit) - cur
            if d <= mp.mpf("-1e-9"):
                lib[n] = [(x["x"], x["y"], x["angle"]) for x in rows]
                changed += 1
                gains.append(dict(round=rnd, n=n, donor=p, j=j, i=i,
                                  L_lit=lit, L_pandas=pan, dL_lit=mp.nstr(d, 8),
                                  dL_pandas=mp.nstr(mp.mpf(pan) - cur, 8)))
                print(f"  n={n} donor={p} j={j} i={i} dL={mp.nstr(d, 6)}", flush=True)
        print(f"round {rnd}: {changed} accepted lifts", flush=True)
        pickle.dump({n: lib[n] for n in {g['n'] for g in gains}},
                    open(DATA / "closure_wins.pkl", "wb"))
        json.dump(gains, open(DATA / "closure_wins.json", "w"), indent=1)
        if not changed:
            break
    print(f"total {len(gains)} lifts; best dL "
          f"{min((mp.mpf(g['dL_lit']) for g in gains), default=0)}")


if __name__ == "__main__":
    main()
