#!/usr/bin/env python3
"""Extract gadlib pockets from union_incumbent and validate/merge base pocket libraries against donors_all.csv."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc" / "faultline_dp2" / "poc"))
sys.path.insert(0, str(REPO / "poc" / "pocket_lift_scan3" / "poc"))
sys.path.insert(0, str(REPO / "poc" / "swarm_softseamdp" / "poc"))

import gadlib as gl  # noqa: E402
from scan_pockets import Row, load_rows  # noqa: E402
import dp_soft  # noqa: E402

DATA_DIR = REPO / "poc" / "swarm_highdpocket" / "data"
TOL = 1e-9


def extract_gadlib(union_csv: Path, improved_ns: set[int]) -> list[dict]:
    D = gl.load_all(str(union_csv))
    out = []
    for n in sorted(D):
        a, e, L = D[n]
        t = np.degrees(np.mod(a[:, 2], np.pi / 2))
        tilt = np.minimum(t, 90 - t) > 0.5
        if tilt.sum() == 0:
            continue
        cls = gl.clusters(a, tilt)
        seeds = [c for c in cls]
        for i in range(len(cls)):
            for j in range(i + 1, len(cls)):
                seeds.append(np.concatenate([cls[i], cls[j]]))
        seen = set()
        src_id = (10000 + n) if n in improved_ns else n
        for cl in seeds:
            x0 = float((a[cl, 0] - e[cl]).min())
            x1 = float((a[cl, 0] + e[cl]).max())
            y0 = float((a[cl, 1] - e[cl]).min())
            y1 = float((a[cl, 1] + e[cl]).max())
            X0, X1, Y0, Y1, idx = gl.clean_rect(a, e, x0, x1, y0, y1)
            W, H = float(X1 - X0), float(Y1 - Y0)
            if W > L - 1e-6 and H > L - 1e-6:
                continue
            w_i, h_i = int(W + 1e-7), int(H + 1e-7)
            if w_i < 1 or h_i < 1:
                continue
            g = int(len(idx) - w_i * h_i)
            if g <= 0:
                continue
            key = (round(X0, 4), round(Y0, 4), round(W, 4), round(H, 4))
            if key in seen:
                continue
            seen.add(key)
            f_val = max(W - w_i, H - h_i)
            out.append({
                "src": src_id,
                "w": w_i,
                "h": h_i,
                "g": g,
                "f": float(f_val),
                "cnt": int(len(idx)),
                "x0": float(X0),
                "x1": float(X1),
                "y0": float(Y0),
                "y1": float(Y1),
            })
    return out


def validate_against_donors(records: list[dict], rows: dict[int, np.ndarray], cache: dict[int, Row]) -> list[dict]:
    out = []
    dropped = 0
    remapped = 0
    for g in records:
        orig_src = int(g["src"])
        if orig_src >= 2000 and orig_src != 2006 and orig_src < 10000:
            cands = [20000 + (orig_src - 2000), orig_src]
        elif orig_src <= 1000:
            cands = [orig_src, 10000 + orig_src]
        else:
            cands = [orig_src]

        matched = None
        for sid in cands:
            if sid not in rows:
                continue
            if sid not in cache:
                cache[sid] = Row(sid, rows[sid])
            row = cache[sid]
            m = (
                (row.lx >= g["x0"] - TOL)
                & (row.hx <= g["x1"] + TOL)
                & (row.ly >= g["y0"] - TOL)
                & (row.hy <= g["y1"] + TOL)
            )
            k = int(m.sum())
            if k < 2:
                continue
            X0, X1 = float(row.lx[m].min()), float(row.hx[m].max())
            Y0, Y1 = float(row.ly[m].min()), float(row.hy[m].max())
            W, H = X1 - X0, Y1 - Y0
            w = int(W + 1e-6)
            h = int(H + 1e-6)
            gg = k - w * h
            if gg < 1 or w < 1 or h < 1:
                continue
            f = max(W - w, H - h)
            # For remapped candidates, require exact match on cnt and (w,h,g)
            if sid != orig_src and (k != g["cnt"] or w != g["w"] or h != g["h"] or gg != g["g"] or abs(f - g["f"]) > 1e-6):
                continue
            rec = dict(g)
            rec.update(src=sid, w=w, h=h, g=gg, f=f, cnt=k, x0=X0, x1=X1, y0=Y0, y1=Y1)
            if sid != orig_src:
                remapped += 1
            matched = rec
            break

        if matched is not None:
            out.append(matched)
        else:
            dropped += 1
    print(f"Validated {len(records)} -> {len(out)} kept ({remapped} remapped, {dropped} dropped)", flush=True)
    return out


def main() -> None:
    report = json.loads((DATA_DIR / "union_report.json").read_text())["results"]
    improved_ns = {int(r["n"]) for r in report if r["src"] != "merged.csv"}

    gad = extract_gadlib(DATA_DIR / "union_incumbent.csv", improved_ns)
    (DATA_DIR / "gadlib_pockets.json").write_text(json.dumps(gad, indent=0))
    print(f"Extracted {len(gad)} gadlib pockets", flush=True)

    rows = load_rows(str(DATA_DIR / "donors_all.csv"))
    cache: dict[int, Row] = {}

    e3 = json.loads((REPO / "poc" / "pocket_lift_scan3" / "data" / "pockets_pareto_e3_sane.json").read_text())
    pb = json.loads((REPO / "poc" / "swarm_liftpolish3" / "data" / "pockets_partb_sane.json").read_text())
    br = json.loads((DATA_DIR / "bricks_pockets.json").read_text())

    all_base = validate_against_donors(e3 + pb + gad + br, rows, cache)
    (DATA_DIR / "base_pockets_sane.json").write_text(json.dumps(all_base, indent=0))
    print(f"Wrote {len(all_base)} validated base pockets -> data/base_pockets_sane.json", flush=True)


if __name__ == "__main__":
    main()
