#!/usr/bin/env python3
"""Build exact origin/main union incumbent and donor catalog for swarm_highdpocket."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import csv
import glob
import io
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc" / "baseline" / "poc" / "src"))
sys.path.insert(0, str(REPO / "poc" / "auto_union" / "poc"))

from squares.core import certify, read_submission  # noqa: E402
import union as auto_union  # noqa: E402

OUT_DIR = REPO / "poc" / "swarm_highdpocket"
DATA_DIR = OUT_DIR / "data"
MERGED_CSV = REPO / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
REPORT_NOW = REPO / "poc" / "pocket_lift_scan3" / "data" / "report_now.json"


def _strip_s(v: str) -> str:
    return v[1:] if v.startswith("s") else v


def _fast_side(rows: list[dict[str, str]]) -> float:
    x = np.array([float(_strip_s(str(r["x"]))) for r in rows], dtype=np.float64)
    y = np.array([float(_strip_s(str(r["y"]))) for r in rows], dtype=np.float64)
    a = np.array([float(_strip_s(str(r["angle"]))) for r in rows], dtype=np.float64)
    e = 0.5 * (np.abs(np.cos(a)) + np.abs(np.sin(a)))
    return float(max((x + e).max() - (x - e).min(), (y + e).max() - (y - e).min()))


def _cert_cand(task: tuple[int, str, list[dict[str, str]]]) -> tuple[int, str, str | None, str | None]:
    n, src, rows = task
    try:
        text = auto_union.final_text({n: rows})
        lit = auto_union.parse_text(text)
        pd_t = auto_union.parse_text(pd.read_csv(io.StringIO(text)).to_csv(index=False))
        res = auto_union.check((n, lit[n], pd_t[n]))
        if res is None:
            return n, src, None, None
        return n, src, res[0], res[1]
    except Exception:
        return n, src, None, None


def main() -> None:
    t0 = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rep_now = {int(r["n"]): r["L"] for r in json.loads(REPORT_NOW.read_text())["results"]}
    merged = read_submission(MERGED_CSV)

    # Re-certify merged.csv rows that changed since REPORT_NOW
    merged_tasks = []
    for n in range(1, 1001):
        s_fast = _fast_side(merged[n])
        if abs(s_fast - float(rep_now[n])) > 1e-12:
            merged_tasks.append((n, "merged.csv", merged[n]))
    if merged_tasks:
        print(f"Re-certifying {len(merged_tasks)} updated merged.csv rows...", flush=True)
        with ProcessPoolExecutor(max_workers=2) as pool:
            for n, src, l_lit, l_pd in pool.map(_cert_cand, merged_tasks, chunksize=4):
                if l_lit is not None and l_pd is not None:
                    rep_now[n] = str(max(Decimal(l_lit), Decimal(l_pd)))

    cands_by_n: dict[int, list[tuple[float, str, list[dict[str, str]]]]] = {}
    for n, rows in merged.items():
        cands_by_n[n] = [(_fast_side(rows), "merged.csv", rows)]

    for p in sorted((REPO / "poc").glob("*/final.csv")):
        if p.parent.name == "swarm_highdpocket" or p.stat().st_size <= 50:
            continue
        try:
            raw_lines_by_n: dict[int, list[str]] = {}
            fast_coords: dict[int, list[tuple[float, float, float]]] = {}
            with p.open("r", encoding="utf-8") as fh:
                hdr = fh.readline()
                if not hdr.startswith("id,x,y,deg"):
                    continue
                for line in fh:
                    if not line.strip():
                        continue
                    parts = line.rstrip("\r\n").split(",")
                    if len(parts) != 4:
                        continue
                    n = int(parts[0].split("_", 1)[0])
                    raw_lines_by_n.setdefault(n, []).append(line)
                    xv = float(_strip_s(parts[1]))
                    yv = float(_strip_s(parts[2]))
                    dv = float(_strip_s(parts[3]))
                    fast_coords.setdefault(n, []).append((xv, yv, dv * np.pi / 180.0))
        except Exception as exc:
            print(f"skip {p}: {exc}", flush=True)
            continue

        rel = str(p.relative_to(REPO))
        need_ns = []
        for n, arr in fast_coords.items():
            if len(arr) != n or n not in rep_now:
                continue
            a = np.array(arr, dtype=np.float64)
            e = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
            s = float(max((a[:, 0] + e).max() - (a[:, 0] - e).min(), (a[:, 1] + e).max() - (a[:, 1] - e).min()))
            if s < float(rep_now[n]) - 1e-10:
                need_ns.append((s, n))
        if not need_ns:
            continue
        sub_txt = "id,x,y,deg\n" + "".join("".join(raw_lines_by_n[n]) for _, n in need_ns)
        try:
            groups = auto_union.parse_text(sub_txt)
        except Exception as exc:
            print(f"parse skip {p}: {exc}", flush=True)
            continue
        for s, n in need_ns:
            if n in groups and len(groups[n]) == n:
                cands_by_n[n].append((s, rel, groups[n]))

    to_cert = []
    for n in range(1, 1001):
        if len(cands_by_n[n]) > 1:
            challengers = sorted(cands_by_n[n][1:], key=lambda t: t[0])[:4]
            for _, src, rows in challengers:
                to_cert.append((n, src, rows))

    print(f"Found {len(to_cert)} candidate groups across {sum(1 for n in cands_by_n if len(cands_by_n[n]) > 1)} ns to dual-certify", flush=True)

    cert_results: dict[int, list[tuple[Decimal, str, str, str, list[dict[str, str]]]]] = {}
    if to_cert:
        cand_map = {(n, src): rows for n, src, rows in to_cert}
        with ProcessPoolExecutor(max_workers=2) as pool:
            for n, src, l_lit, l_pd in pool.map(_cert_cand, to_cert, chunksize=4):
                if l_lit is None or l_pd is None:
                    continue
                l_worst = max(Decimal(l_lit), Decimal(l_pd))
                cert_results.setdefault(n, []).append((l_worst, src, l_lit, l_pd, cand_map[(n, src)]))

    best_rows: dict[int, list[dict[str, str]]] = {}
    report_results = []
    improved_vs_merged = []
    for n in range(1, 1001):
        m_L = Decimal(rep_now[n])
        chosen_src = "merged.csv"
        chosen_L_lit = rep_now[n]
        chosen_L_pd = rep_now[n]
        chosen_rows = merged[n]
        best_worst = m_L

        for l_worst, src, l_lit, l_pd, rows in cert_results.get(n, []):
            if l_worst < best_worst:
                best_worst = l_worst
                chosen_src = src
                chosen_L_lit = l_lit
                chosen_L_pd = l_pd
                chosen_rows = rows

        if chosen_src != "merged.csv" and best_worst < m_L - Decimal("1e-9"):
            improved_vs_merged.append((n, chosen_src, float(m_L - best_worst), float(best_worst)))
        best_rows[n] = chosen_rows
        report_results.append({
            "n": n,
            "K": math.isqrt(n),
            "D": n - math.isqrt(n) ** 2,
            "L": str(best_worst),
            "L_lit": chosen_L_lit,
            "L_pd": chosen_L_pd,
            "src": chosen_src,
        })

    print(f"Union built in {time.time() - t0:.1f}s; {len(improved_vs_merged)} ns strictly beat merged.csv", flush=True)
    for n, src, d, lf in sorted(improved_vs_merged):
        K = math.isqrt(n)
        D = n - K * K
        if 12 <= D <= 35:
            print(f"  [D={D:2d}, K={K:2d}] n={n:3d}: L={lf:.9f} (-{d:.6e}) from {src}", flush=True)

    (DATA_DIR / "union_report.json").write_text(json.dumps({"results": report_results}, indent=1))

    # Write union_incumbent.csv
    with (DATA_DIR / "union_incumbent.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", "n", "square_id", "x", "y", "angle"])
        rid = 0
        for n in range(1, 1001):
            for r in sorted(best_rows[n], key=lambda x: int(x["square_id"])):
                w.writerow([rid, n, r["square_id"], r["x"], r["y"], r["angle"]])
                rid += 1

    # Write donors_all.csv:
    # 1..1000: merged.csv
    # 10000+n: improved union_incumbent rows (for n where src != merged.csv)
    # 2006: pocket_lift_scan3/data/built/n740.json
    # 20000+i: bricks from swarm_gstep, swarm_b7xplant, swarm_thinb4
    improved_ns = sorted(r["n"] for r in report_results if r["src"] != "merged.csv")
    brick_pockets = []
    with (DATA_DIR / "donors_all.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", "n", "square_id", "x", "y", "angle"])
        rid = 0
        for n in range(1, 1001):
            for r in sorted(merged[n], key=lambda x: int(x["square_id"])):
                w.writerow([rid, n, r["square_id"], r["x"], r["y"], r["angle"]])
                rid += 1
        for n in improved_ns:
            for r in sorted(best_rows[n], key=lambda x: int(x["square_id"])):
                w.writerow([rid, 10000 + n, r["square_id"], r["x"], r["y"], r["angle"]])
                rid += 1
        b2006 = REPO / "poc" / "pocket_lift_scan3" / "data" / "built" / "n740.json"
        if b2006.exists():
            rows2006 = json.loads(b2006.read_text())
            for r in sorted(rows2006, key=lambda x: int(x["square_id"])):
                w.writerow([rid, 2006, r["square_id"], r["x"], r["y"], r["angle"]])
                rid += 1

        def _add_brick(bdict: dict, tag: str, bn: int) -> None:
            nonlocal rid
            xs = np.array(bdict["x"], dtype=np.float64)
            ys = np.array(bdict["y"], dtype=np.float64)
            ths = np.array(bdict["th"], dtype=np.float64)
            ext = 0.5 * (np.abs(np.cos(ths)) + np.abs(np.sin(ths)))
            x0 = float((xs - ext).min())
            y0 = float((ys - ext).min())
            xs = xs - x0
            ys = ys - y0
            W = float((xs + ext).max())
            H = float((ys + ext).max())
            ww = int(bdict["w"])
            hh = int(bdict["h"])
            gg = int(bdict["g"])
            cnt = len(xs)
            if cnt != ww * hh + gg:
                return
            f_val = max(W - ww, H - hh)
            for sid in range(cnt):
                w.writerow([rid, bn, sid, repr(float(xs[sid])), repr(float(ys[sid])), repr(float(ths[sid]))])
                rid += 1
            brick_pockets.append({
                "src": bn,
                "name": tag,
                "w": ww,
                "h": hh,
                "g": gg,
                "f": f_val,
                "cnt": cnt,
                "x0": 0.0,
                "x1": W,
                "y0": 0.0,
                "y1": H,
            })

        bn = 20000
        for bp in sorted(glob.glob(str(REPO / "poc" / "swarm_gstep" / "data" / "brick_*.json"))):
            _add_brick(json.loads(Path(bp).read_text()), f"gstep_{Path(bp).stem}", bn)
            bn += 1
        for bp in sorted(glob.glob(str(REPO / "poc" / "swarm_b7xplant" / "data" / "brick_*.json"))):
            _add_brick(json.loads(Path(bp).read_text()), f"b7_{Path(bp).stem}", bn)
            bn += 1
        thin_cache = REPO / "poc" / "swarm_thinb4" / "data" / "compressed_bricks_cache.json"
        if thin_cache.exists():
            for name, bd in sorted(json.loads(thin_cache.read_text()).items()):
                _add_brick(bd, f"thin_{name}", bn)
                bn += 1
        for bp in sorted(glob.glob(str(REPO / "poc" / "swarm_peelstep" / "data" / "brick_*.json"))):
            _add_brick(json.loads(Path(bp).read_text()), f"peelstep_{Path(bp).stem}", bn)
            bn += 1

    (DATA_DIR / "bricks_pockets.json").write_text(json.dumps(brick_pockets, indent=1))
    print(f"Wrote union_report.json, union_incumbent.csv, donors_all.csv ({len(improved_ns)} improved ns, {len(brick_pockets)} bricks)", flush=True)


if __name__ == "__main__":
    main()
