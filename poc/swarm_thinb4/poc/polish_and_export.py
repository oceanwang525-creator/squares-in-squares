#!/usr/bin/env python3
"""Certify, polish (optional), prune superseded wins, and export poc/swarm_thinb4/final.csv."""
from __future__ import annotations

import fcntl
import json
import multiprocessing as mp
import pathlib
import sys
from decimal import Decimal

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
THIN_DIR = HERE.parent
REPO = THIN_DIR.parent.parent
sys.path.insert(0, str(REPO / ".github" / "scripts"))
sys.path.insert(0, str(REPO / "poc" / "auto_union" / "poc"))
sys.path.insert(0, str(HERE))

import check_final_csv  # noqa: E402
import thinb4_lib as tl  # noqa: E402
import union as auto_union  # noqa: E402


def load_full_portfolio_inc(target_ns: list[int]) -> dict[int, Decimal]:
    """Load exact certified minimum L across merged.csv + all other poc/*/final.csv files."""
    need = set(target_ns)
    inc: dict[int, Decimal] = {}
    groups = auto_union.parse_text((REPO / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv").read_text("utf-8"))
    for n in need:
        if n in groups:
            inc[n] = Decimal(auto_union.certify(n, groups[n]))
    for fp in sorted((REPO / "poc").glob("*/final.csv")):
        if fp.parent.name == "swarm_thinb4" or fp.stat().st_size <= 20:
            continue
        g = auto_union.parse_text(fp.read_text("utf-8"))
        for n in need:
            if n in g:
                Ld = Decimal(auto_union.certify(n, g[n]))
                if n not in inc or Ld < inc[n]:
                    inc[n] = Ld
    return inc


def _polish_saved_worker(wp_str: str) -> tuple[int, bool, str]:
    wp = pathlib.Path(wp_str)
    d = json.loads(wp.read_text(encoding="utf-8"))
    n = int(d["n"])
    rows = sorted(d["rows"], key=lambda r: int(r["square_id"]))
    x = np.array([float(r["x"]) for r in rows], dtype=np.float64)
    y = np.array([float(r["y"]) for r in rows], dtype=np.float64)
    th = np.array([float(r["angle"]) for r in rows], dtype=np.float64)
    cur_dec = max(Decimal(d["L_lit"]), Decimal(d["L_pd"]))
    cert = tl.polish_and_certify(n, x, y, th, cur_dec, f"{d['label']}_mw", do_mechwalk=True)
    if cert.get("ok") and cert.get("is_real_win"):
        saved = tl.save_win_thinb4(n, cert, cert["label"])
        return n, saved, f"improved dL={cert['dL']:+.6e} L={cert['L_lit'][:16]}"
    return n, False, f"kept L={d['L_lit'][:16]}"


def export_and_verify(do_polish: bool = False, prune_superseded: bool = False) -> list[dict]:
    win_paths = sorted(tl.DATA_DIR.glob("win_n*.json"))
    if not win_paths:
        raise RuntimeError("No win_n*.json files found")

    if do_polish:
        with mp.Pool(processes=min(4, len(win_paths))) as pool:
            for n, saved, msg in pool.map(_polish_saved_worker, [str(p) for p in win_paths]):
                print(f"[polish n={n}] saved={saved} {msg}", flush=True)

    win_paths = sorted(tl.DATA_DIR.glob("win_n*.json"))
    ns = [int(json.loads(p.read_text(encoding="utf-8"))["n"]) for p in win_paths]
    inc_map = load_full_portfolio_inc(ns)

    groups: dict[int, list[dict]] = {}
    meta: dict[int, dict] = {}
    lock_path = tl.DATA_DIR / ".final.lock"
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        for wp in sorted(tl.DATA_DIR.glob("win_n*.json")):
            d = json.loads(wp.read_text(encoding="utf-8"))
            n = int(d["n"])
            L_lit = Decimal(d["L_lit"])
            L_pd = Decimal(d["L_pd"])
            L_inc = inc_map[n]
            dL_dec = max(L_lit, L_pd) - L_inc
            if L_lit == L_pd and dL_dec <= Decimal("-1e-9"):
                groups[n] = d["rows"]
                meta[n] = {
                    "L_inc": L_inc, "L_lit": L_lit, "L_pd": L_pd,
                    "dL": float(dL_dec), "label": d["label"], "rows": d["rows"],
                }
            else:
                print(f"[SKIP n={n}] superseded/not strict win vs origin/main: dL={float(dL_dec):+.6e}", flush=True)
                if prune_superseded:
                    wp.unlink()
                    print(f"  -> pruned {wp.name}", flush=True)
        text = auto_union.final_text(groups)
        auto_union.atomic(tl.FINAL_CSV, text.encode("utf-8"))

    check_final_csv.validate(tl.FINAL_CSV.read_text(encoding="utf-8"))
    lit, pd_t = auto_union.load(tl.FINAL_CSV)
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
        angs = [abs(float(r["angle"])) % (0.5 * np.pi) for r in meta[n]["rows"]]
        non0_90 = sum(1 for a in angs if min(a, 0.5 * np.pi - a) > 1e-6)
        summary.append({
            "n": n, "L_inc": str(L_inc), "L_lit": L_lit_s, "L_pd": L_pd_s,
            "dL": dL, "ds": float(ds), "non0_90": non0_90, "label": meta[n]["label"],
        })
        print(
            f"[CERT n={n}] L_inc={str(L_inc)[:16]} L_lit={L_lit_s[:22]} lit==pd={L_lit == L_pd} "
            f"dL={dL:+.6e} ds={float(ds):+.6e} non0/90={non0_90}/{n} ({meta[n]['label']})",
            flush=True,
        )
    print(f"[TOTAL] wins={len(summary)} total_ds={float(total_ds):+.6e} check_final_csv=PASS", flush=True)
    return summary


if __name__ == "__main__":
    do_polish = "--polish" in sys.argv
    prune = "--prune" in sys.argv
    export_and_verify(do_polish=do_polish, prune_superseded=prune)
