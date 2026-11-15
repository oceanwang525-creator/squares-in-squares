#!/usr/bin/env python3
"""H126 gate: harvest, then rank m=21..31, r=+9..+17 targets by best predicted pair gap; optionally solve.

Usage:
  python run_gate.py rank                    # harvest + rank, writes docs/rank.json
  python run_gate.py solve N [N ...] [--pairs K] [--slack S]
"""
from __future__ import annotations

import json
import math
import sys
import time
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import band3ext_lib as bx  # noqa: E402

M_RANGE = range(21, 32)
R_RANGE = range(9, 18)


def all_targets() -> list[int]:
    return [m * m + r for m in M_RANGE for r in R_RANGE]


def rank(slack: float = 0.008) -> list[dict]:
    t0 = time.time()
    if not (bx.DATA_DIR / "harvest").exists() or "--reharvest" in sys.argv:
        hv = bx.harvest_all()
        print(f"harvested {len(hv)} bricks in {time.time() - t0:.1f}s", flush=True)
    bricks = bx.load_pool()
    print(f"pool={len(bricks)} bricks", flush=True)
    ns = all_targets()
    fr = bx.load_frontier(ns)
    out = []
    for n in ns:
        inc = fr[n]["L_dec"]
        m = int(math.floor(float(inc)))
        r = n - m * m
        f_inc = float(inc) - m
        pairs = bx.enumerate_pairs(bricks, m, r, f_inc, slack=slack)
        top = [
            {"score": round(p[0], 5), "fm": round(p[1], 5), "d": p[2], "ct": p[3],
             "b1": f"{p[4]['tag']}({p[4]['w']}x{p[4]['h']}g{p[4]['g']} f={p[4]['f']:.4f})",
             "b2": f"{p[5]['tag']}({p[5]['w']}x{p[5]['h']}g{p[5]['g']} f={p[5]['f']:.4f})"}
            for p in pairs[:5]
        ]
        out.append({"n": n, "m": m, "r": r, "inc": str(inc)[:20], "f_inc": round(f_inc, 5),
                    "src": fr[n].get("source"), "npairs": len(pairs), "top": top})
    out.sort(key=lambda d: d["top"][0]["score"] if d["top"] else 9.0)
    bx.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (bx.DOCS_DIR / "rank.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for d in out:
        if d["top"]:
            t = d["top"][0]
            print(f"n={d['n']} m={d['m']} r=+{d['r']} f_inc={d['f_inc']} npairs={d['npairs']} best score={t['score']:+.5f} "
                  f"fm={t['fm']} d={t['d']} ct={t['ct']} {t['b1']} + {t['b2']}", flush=True)
    print(f"rank done {time.time() - t0:.1f}s; {sum(1 for d in out if d['top'])} targets with pairs", flush=True)
    return out


def solve(ns: list[int], n_pairs: int = 3, slack: float = 0.008) -> None:
    t0 = time.time()
    bricks = bx.load_pool()
    fr = bx.load_frontier(ns)
    log_p = bx.DOCS_DIR / f"solve_{'_'.join(map(str, ns))}.jsonl"
    for n in ns:
        inc = fr[n]["L_dec"]
        wn = bx.DATA_DIR / f"win_n{n}.json"
        if wn.exists():
            wd = json.loads(wn.read_text(encoding="utf-8"))
            inc = min(inc, max(Decimal(wd["L_lit"]), Decimal(wd["L_pandas"])))
        m = int(math.floor(float(inc)))
        r = n - m * m
        f_inc = float(inc) - m
        pairs = bx.enumerate_pairs(bricks, m, r, f_inc, slack=slack)
        print(f"\n--- n={n} m={m} r=+{r} inc={str(inc)[:18]} pairs={len(pairs)} ---", flush=True)
        best = None
        for idx, (score, fm, d, ct, b1, b2) in enumerate(pairs[:n_pairs]):
            tag = f"n{n}_p{idx}_{b1['w']}x{b1['h']}g{b1['g']}+{b2['w']}x{b2['h']}g{b2['g']}_d{d}"
            ts = time.time()
            try:
                cert, diag = bx.solve_pair_ext(n, m, b1, b2, d, inc, tag, base_seed=91000 + 10 * n + idx)
            except Exception as exc:  # noqa: BLE001
                print(f"  {tag}: ERROR {exc}", flush=True)
                continue
            rec = {"n": n, "tag": tag, "fm": fm, "score": score, "b1": b1["tag"], "b2": b2["tag"],
                   "t": round(time.time() - ts, 1), "diag": diag}
            if cert is not None:
                rec.update({"L": cert["L_dec_str"], "dL": cert["dL"], "win": cert["is_real_win"], "label": cert["label"]})
                print(f"  {tag}: L={cert['L_float']:.9f} dL={cert['dL']:+.6e} win={cert['is_real_win']} ({rec['t']}s)", flush=True)
                if best is None or cert["L_float"] < best["L_float"]:
                    best = cert
                if cert["is_real_win"]:
                    bx.save_win_locked(cert)
                    print(f"*** CERTIFIED WIN SAVED n={n} dL={cert['dL']:+.6e} L={cert['L_dec_str']} ***", flush=True)
                    inc = Decimal(cert["L_dec_str"])
            else:
                print(f"  {tag}: None diag={diag[:2]} ({rec['t']}s)", flush=True)
            with log_p.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
    print(f"solve done {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    argv = sys.argv[1:]
    slack = 0.008
    n_pairs = 3
    if "--slack" in argv:
        slack = float(argv[argv.index("--slack") + 1])
    if "--pairs" in argv:
        n_pairs = int(argv[argv.index("--pairs") + 1])
    if not argv or argv[0] == "rank":
        rank(slack)
    elif argv[0] == "solve":
        nums = []
        skip = False
        for a in argv[1:]:
            if skip:
                skip = False
                continue
            if a in ("--slack", "--pairs"):
                skip = True
                continue
            if a.isdigit():
                nums.append(int(a))
        solve(nums, n_pairs=n_pairs, slack=slack)
