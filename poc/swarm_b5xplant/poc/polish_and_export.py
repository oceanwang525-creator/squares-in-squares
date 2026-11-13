"""Standalone mechwalk polish + origin/main (37be2a8) export verification for H40 wins."""
from __future__ import annotations

import fcntl
import json
import math
import multiprocessing as mp
import pathlib
import sys
from decimal import Decimal

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
B5_DIR = HERE.parent
REPO = B5_DIR.parent.parent
sys.path.insert(0, str(REPO / ".github" / "scripts"))
sys.path.insert(0, str(REPO / "poc" / "auto_union" / "poc"))
sys.path.insert(0, str(HERE))

import b5_lib as bl
import check_final_csv
import union as auto_union


def _polish_saved_worker(args: tuple[str, str]) -> tuple[int, bool, str]:
    wp_str, inc_str = args
    wp = pathlib.Path(wp_str)
    d = json.loads(wp.read_text(encoding="utf-8"))
    n = int(d["n"])
    rows = sorted(d["rows"], key=lambda r: int(r["square_id"]))
    x = np.array([float(r["x"]) for r in rows], dtype=np.float64)
    y = np.array([float(r["y"]) for r in rows], dtype=np.float64)
    th = np.radians(np.array([float(r["angle"]) for r in rows], dtype=np.float64))
    cur_dec = max(Decimal(d["L_lit"]), Decimal(d["L_pd"]))
    cert = bl.polish_and_certify(n, x, y, th, cur_dec, f"{d['label']}_mw", do_mechwalk=True)
    if cert.get("ok") and cert.get("is_real_win"):
        saved = bl.save_win_b5(n, cert, cert["label"])
        return n, saved, f"improved dL={cert['dL']:+.6e} L={cert['L_lit'][:16]}"
    return n, False, f"kept L={d['L_lit'][:16]}"


def export_and_verify(do_polish: bool = True) -> list[dict]:
    win_paths = sorted(bl.DATA_DIR.glob("win_n*.json"))
    if not win_paths:
        raise RuntimeError("No win_n*.json files found")
    ns = [int(json.loads(p.read_text(encoding="utf-8"))["n"]) for p in win_paths]
    _, inc_map = bl.fast_load_frontier(target_ns=ns)

    if do_polish:
        tasks = [(str(p), str(inc_map[int(json.loads(p.read_text(encoding="utf-8"))["n"])])) for p in win_paths]
        with mp.Pool(processes=min(4, len(tasks))) as pool:
            for n, saved, msg in pool.map(_polish_saved_worker, tasks):
                print(f"[polish n={n}] saved={saved} {msg}", flush=True)

    # Atomic strict-win export against origin/main inc_map under .final.lock
    win_paths = sorted(bl.DATA_DIR.glob("win_n*.json"))
    ns = [int(json.loads(p.read_text(encoding="utf-8"))["n"]) for p in win_paths]
    _, inc_map = bl.fast_load_frontier(target_ns=ns)

    groups: dict[int, list[dict]] = {}
    meta: dict[int, dict] = {}
    lock_path = bl.DATA_DIR / ".final.lock"
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        for wp in sorted(bl.DATA_DIR.glob("win_n*.json")):
            d = json.loads(wp.read_text(encoding="utf-8"))
            n = int(d["n"])
            L_lit = Decimal(d["L_lit"])
            L_pd = Decimal(d["L_pd"])
            L_inc = inc_map[n]
            dL_dec = max(L_lit, L_pd) - L_inc
            if L_lit == L_pd and dL_dec <= Decimal("-1e-9"):
                groups[n] = d["rows"]
                meta[n] = {"L_inc": L_inc, "L_lit": L_lit, "L_pd": L_pd, "dL": float(dL_dec), "label": d["label"], "rows": d["rows"]}
            else:
                print(f"[SKIP n={n}] not strict win vs origin/main: dL={dL_dec}", flush=True)
        text = auto_union.final_text(groups)
        auto_union.atomic(bl.FINAL_CSV, text.encode("utf-8"))

    # Full independent certification of final.csv
    check_final_csv.validate(bl.FINAL_CSV.read_text(encoding="utf-8"))
    lit, pd_t = auto_union.load(bl.FINAL_CSV)
    assert set(lit.keys()) == set(groups.keys()) and set(pd_t.keys()) == set(groups.keys())

    summary = []
    total_ds = Decimal(0)
    for n in sorted(groups.keys()):
        res = auto_union.check((n, lit[n], pd_t[n]))
        assert res is not None, f"literal != pandas for n={n}"
        L_lit_s, L_pd_s = res
        L_lit = Decimal(L_lit_s)
        L_pd = Decimal(L_pd_s)
        L_inc = inc_map[n]
        dL = float(L_lit - L_inc)
        ds = (L_lit * L_lit - L_inc * L_inc) / Decimal(n)
        assert dL <= -1e-9, f"n={n} failed strict win dL={dL}"
        total_ds += ds
        angs = [float(r["angle"]) % 90.0 for r in meta[n]["rows"]]
        non0_90 = sum(1 for a in angs if min(a, 90.0 - a) > 1e-6)
        summary.append({
            "n": n,
            "L_inc": str(L_inc),
            "L_lit": L_lit_s,
            "L_pd": L_pd_s,
            "dL": dL,
            "ds": float(ds),
            "non0_90": non0_90,
            "label": meta[n]["label"],
        })
        print(
            f"[CERT n={n}] L_inc={str(L_inc)[:16]} L_lit={L_lit_s[:22]} lit==pd={L_lit == L_pd} "
            f"dL={dL:+.6e} ds={float(ds):+.6e} non0/90={non0_90}/{n} ({meta[n]['label']})",
            flush=True,
        )
    print(f"[TOTAL] wins={len(summary)} total_ds={float(total_ds):+.6e} check_final_csv=PASS", flush=True)
    return summary


if __name__ == "__main__":
    do_polish = "--no-polish" not in sys.argv
    export_and_verify(do_polish=do_polish)
