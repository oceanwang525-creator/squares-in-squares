#!/usr/bin/env python3
"""Extract clean-cut proper sub-gadgets from union_incumbent.csv as pocket records (with exact boxes)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc" / "faultline_dp2" / "poc"))
import gadlib as gl  # noqa: E402

SOFT_DIR = REPO / "poc" / "swarm_softseamdp"
DATA_DIR = SOFT_DIR / "data"


def main() -> None:
    report = json.loads((DATA_DIR / "union_report.json").read_text())["results"]
    diff_ns = {int(r["n"]) for r in report if r["src"] != "merged.csv"}

    # Append 5000 + n rows to donors_all.csv if not already there
    existing_ns = set()
    with (DATA_DIR / "donors_all.csv").open("r", newline="") as f:
        for r in csv.DictReader(f):
            existing_ns.add(int(r["n"]))

    union_rows: dict[int, list[dict[str, str]]] = {}
    max_rid = 0
    with (DATA_DIR / "union_incumbent.csv").open("r", newline="") as f:
        for r in csv.DictReader(f):
            union_rows.setdefault(int(r["n"]), []).append(r)
    with (DATA_DIR / "donors_all.csv").open("r", newline="") as f:
        for r in csv.DictReader(f):
            max_rid = max(max_rid, int(r["id"]) + 1)

    added_donors = 0
    with (DATA_DIR / "donors_all.csv").open("a", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        for n in sorted(diff_ns):
            dn = 5000 + n
            if dn in existing_ns:
                continue
            for r in union_rows[n]:
                w.writerow([max_rid, dn, r["square_id"], r["x"], r["y"], r["angle"]])
                max_rid += 1
            added_donors += 1

    D = gl.load_all(str(DATA_DIR / "union_incumbent.csv"))
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
        src_id = (5000 + n) if n in diff_ns else n
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

    (DATA_DIR / "gadlib_pockets.json").write_text(json.dumps(out, indent=0))
    print(f"Extracted {len(out)} proper gadlib pockets (added {added_donors} donor groups to donors_all.csv)")


if __name__ == "__main__":
    main()
