#!/usr/bin/env python3
"""H101: 2-D Overhang (kx + ky <= 2, |I| = 0) Pinwheel Composer Library."""
from __future__ import annotations

import os
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_v] = "1"

import csv
import glob
import json
import math
import sys
from decimal import Decimal
from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
INTERLOCK2_DIR = REPO / "poc" / "swarm_interlock2"
DATA_DIR = INTERLOCK2_DIR / "data"
DOCS_DIR = INTERLOCK2_DIR / "docs"
FINAL_CSV = INTERLOCK2_DIR / "final.csv"

for _p in (
    REPO / "poc" / "baseline" / "poc" / "src",
    REPO / "poc" / "auto_union" / "poc",
    REPO / "poc" / "swarm_lpslack" / "poc",
    REPO / "poc" / "swarm_mechwalk" / "poc",
    REPO / "poc" / "swarm_highd" / "poc",
    REPO / "poc" / "swarm_brickslp" / "poc",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from squares import core  # noqa: E402
import union as auto_union  # noqa: E402
import lpslack  # noqa: E402
import brick_slp  # noqa: E402
import utils as hu  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, full_to_free_sym, run_cpp_masked_melt  # noqa: E402

EPS = mp.mpf("1e-12")
SPREAD = mp.mpf("1e-12")


def norm_L_rows(rows: list[dict[str, str]]) -> float:
    a = np.array([[float(r["x"]), float(r["y"]), float(r["angle"])] for r in rows], dtype=np.float64)
    e = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
    return float(max((a[:, 0] + e).max() - (a[:, 0] - e).min(), (a[:, 1] + e).max() - (a[:, 1] - e).min()))


def load_portfolio_best() -> tuple[dict[int, list[dict[str, str]]], dict[int, float]]:
    """Load fastest float64 best packings across merged.csv and poc/*/final.csv (excl. swarm_interlock2)."""
    best_rows: dict[int, list[dict[str, str]]] = {}
    best_L: dict[int, float] = {}

    merged = REPO / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
    with merged.open(newline="", encoding="utf-8") as f:
        tmp: dict[int, list[dict[str, str]]] = {}
        for r in csv.DictReader(f):
            n = int(r["n"])
            tmp.setdefault(n, []).append(
                {"square_id": r["square_id"], "x": r["x"], "y": r["y"], "angle": r["angle"]}
            )
        for n, rs in tmp.items():
            rs.sort(key=lambda z: int(z["square_id"]))
            best_rows[n] = rs
            best_L[n] = norm_L_rows(rs)

    for fp in sorted((REPO / "poc").glob("*/final.csv")):
        if fp.parent.name == "swarm_interlock2" or fp.stat().st_size <= 20:
            continue
        tmp_deg: dict[int, list[tuple[int, str, str, str]]] = {}
        try:
            with fp.open(newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    n_s, sid_s = r["id"].split("_")
                    tmp_deg.setdefault(int(n_s), []).append(
                        (int(sid_s), r["x"].lstrip("s"), r["y"].lstrip("s"), r["deg"].lstrip("s"))
                    )
        except Exception:
            continue
        for n, rs in tmp_deg.items():
            if len(rs) != n:
                continue
            rs.sort(key=lambda z: z[0])
            # Quick float filter first before expensive 60-dps deg->rad conversion
            a = np.array([[float(x), float(y), math.radians(float(d))] for _, x, y, d in rs], dtype=np.float64)
            e = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
            Lf = float(max((a[:, 0] + e).max() - (a[:, 0] - e).min(), (a[:, 1] + e).max() - (a[:, 1] - e).min()))
            if Lf < best_L.get(n, 1e9) - 1e-12:
                with mp.workdps(60):
                    conv = [
                        {
                            "square_id": str(sid),
                            "x": xs,
                            "y": ys,
                            "angle": mp.nstr(mp.mpf(ds) * mp.pi / 180, 50),
                        }
                        for sid, xs, ys, ds in rs
                    ]
                best_rows[n] = conv
                best_L[n] = Lf
    return best_rows, best_L


def load_merged_rows() -> dict[int, list[dict[str, str]]]:
    merged_rows: dict[int, list[dict[str, str]]] = {}
    merged = REPO / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
    with merged.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            merged_rows.setdefault(int(r["n"]), []).append(
                {"square_id": r["square_id"], "x": r["x"], "y": r["y"], "angle": r["angle"]}
            )
    for n in merged_rows:
        merged_rows[n].sort(key=lambda z: int(z["square_id"]))
    return merged_rows


def exact_incumbent_L(n: int, best_rows_by_n: dict[int, list[dict[str, str]]]) -> Decimal:
    """Certify exact Decimal L_inc for n against merged.csv and poc/*/final.csv."""
    cands: list[list[dict[str, str]]] = []
    merged = REPO / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
    with merged.open(newline="", encoding="utf-8") as f:
        rs = [
            {"square_id": r["square_id"], "x": r["x"], "y": r["y"], "angle": r["angle"]}
            for r in csv.DictReader(f)
            if int(r["n"]) == n
        ]
        if rs:
            rs.sort(key=lambda z: int(z["square_id"]))
            cands.append(rs)

    pref = f"{n:03}_" if n < 1000 else "1000_"
    for fp in sorted((REPO / "poc").glob("*/final.csv")):
        if fp.parent.name == "swarm_interlock2" or fp.stat().st_size <= 20:
            continue
        text = fp.read_text(encoding="utf-8")
        lines = text.splitlines()
        sub = [lines[0]] + [ln for ln in lines[1:] if ln.startswith(pref)]
        if len(sub) != n + 1:
            continue
        sub_text = "\n".join(sub) + "\n"
        try:
            parsed = auto_union.parse_text(sub_text)
            if n in parsed and len(parsed[n]) == n:
                cands.append(parsed[n])
        except Exception:
            continue

    best_dec: Decimal | None = None
    for rows in cands:
        res = lpslack.certify_pair(n, rows)
        if res is None:
            continue
        L_cur = max(res[0], res[1])
        if best_dec is None or L_cur < best_dec:
            best_dec = L_cur
    if best_dec is None:
        return Decimal(repr(norm_L_rows(best_rows_by_n[n])))
    return best_dec


def _prep_float(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = np.array([[float(r["x"]), float(r["y"]), float(r["angle"])] for r in rows], dtype=np.float64)
    e = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
    a[:, 0] -= (a[:, 0] - e).min()
    a[:, 1] -= (a[:, 1] - e).min()
    return a[:, 0] - e, a[:, 0] + e, a[:, 1] - e, a[:, 1] + e


def extract_pocket_50d(
    rows: list[dict[str, str]], pk: dict, transpose: bool = False
) -> tuple[list[dict], mp.mpf, mp.mpf] | None:
    """Extract exact 50-digit local squares (bbox min at 0) from donor decimal strings."""
    lx, hx, ly, hy = _prep_float(rows)
    tol = 1e-8
    m = (
        (lx >= pk["x0"] - tol)
        & (hx <= pk["x1"] + tol)
        & (ly >= pk["y0"] - tol)
        & (hy <= pk["y1"] + tol)
    )
    idx = np.nonzero(m)[0]
    if len(idx) != pk["cnt"]:
        return None
    with mp.workdps(50):
        ext = [
            mp.mpf("0.5") * (abs(mp.cos(mp.mpf(rows[i]["angle"]))) + abs(mp.sin(mp.mpf(rows[i]["angle"]))))
            for i in idx
        ]
        xs = [mp.mpf(rows[i]["x"]) for i in idx]
        ys = [mp.mpf(rows[i]["y"]) for i in idx]
        X0 = min(x - t for x, t in zip(xs, ext))
        Y0 = min(y - t for y, t in zip(ys, ext))
        X1 = max(x + t for x, t in zip(xs, ext))
        Y1 = max(y + t for y, t in zip(ys, ext))
        sq = [{"x": x - X0, "y": y - Y0, "angle": rows[i]["angle"]} for i, x, y in zip(idx, xs, ys)]
        W, H = X1 - X0, Y1 - Y0
        if transpose:
            sq = [{"x": H - s["y"], "y": s["x"], "angle": s["angle"]} for s in sq]
            W, H = H, W
    return sq, W, H


def brick_to_50d(b: dict, transpose: bool = False) -> tuple[list[dict], mp.mpf, mp.mpf]:
    """Convert float64 brick dict to 50-digit local squares (bbox min at 0)."""
    x = np.asarray(b["x"], dtype=np.float64)
    y = np.asarray(b["y"], dtype=np.float64)
    th = np.asarray(b["th"], dtype=np.float64)
    with mp.workdps(50):
        xs = [mp.mpf(repr(float(v))) for v in x]
        ys = [mp.mpf(repr(float(v))) for v in y]
        ts = [mp.mpf(repr(float(v))) for v in th]
        ext = [mp.mpf("0.5") * (abs(mp.cos(t)) + abs(mp.sin(t))) for t in ts]
        X0 = min(xv - e for xv, e in zip(xs, ext))
        Y0 = min(yv - e for yv, e in zip(ys, ext))
        X1 = max(xv + e for xv, e in zip(xs, ext))
        Y1 = max(yv + e for yv, e in zip(ys, ext))
        sq = [{"x": xv - X0, "y": yv - Y0, "angle": mp.nstr(tv, 45)} for xv, yv, tv in zip(xs, ys, ts)]
        W, H = X1 - X0, Y1 - Y0
        if transpose:
            sq = [{"x": H - s["y"], "y": s["x"], "angle": s["angle"]} for s in sq]
            W, H = H, W
    return sq, W, H


def load_all_pockets(
    best_rows: dict[int, list[dict[str, str]]],
    extra_scan_path: Path | None = None,
) -> dict[tuple[int, int, int], list[dict]]:
    """Load e3_sane + scanned_5rows + gstep/b7xplant/thinb4 bricks, grouped by (w, h, g)."""
    merged_rows: dict[int, list[dict[str, str]]] = {}
    merged_csv = REPO / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
    with merged_csv.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            merged_rows.setdefault(int(r["n"]), []).append(
                {"square_id": r["square_id"], "x": r["x"], "y": r["y"], "angle": r["angle"]}
            )
    for n in merged_rows:
        merged_rows[n].sort(key=lambda z: int(z["square_id"]))
    prep_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    by_whg: dict[tuple[int, int, int], list[dict]] = {}

    def _add_pocket_entry(pk: dict, rows_map: dict[int, list[dict[str, str]]], tag_prefix: str) -> None:
        src = int(pk["src"])
        if src not in rows_map:
            return
        if src not in prep_cache:
            prep_cache[src] = _prep_float(rows_map[src])
        lx, hx, ly, hy = prep_cache[src]
        m = (
            (lx >= pk["x0"] - 1e-8)
            & (hx <= pk["x1"] + 1e-8)
            & (ly >= pk["y0"] - 1e-8)
            & (hy <= pk["y1"] + 1e-8)
        )
        if int(m.sum()) != int(pk["cnt"]):
            return
        w, h, g, f = int(pk["w"]), int(pk["h"]), int(pk["g"]), float(pk["f"])
        if f >= 0.999 or g < 1 or w < 1 or h < 1:
            return
        fx = float(hx[m].max() - lx[m].min()) - w
        fy = float(hy[m].max() - ly[m].min()) - h
        for tr in ((False, True) if w != h else (False,)):
            rw, rh = (h, w) if tr else (w, h)
            rfx, rfy = (fy, fx) if tr else (fx, fy)
            rf = max(rfx, rfy)
            by_whg.setdefault((rw, rh, g), []).append(
                {
                    "uid": f"{tag_prefix}_n{src}_{rw}x{rh}_g{g}",
                    "kind": "donor_pocket",
                    "src": src,
                    "rows_map": "best" if rows_map is best_rows else "merged",
                    "pk": pk,
                    "transpose": tr,
                    "w": rw,
                    "h": rh,
                    "g": g,
                    "fx": rfx,
                    "fy": rfy,
                    "f": rf,
                }
            )

    e3_path = REPO / "poc" / "pocket_lift_scan3" / "data" / "pockets_pareto_e3_sane.json"
    if e3_path.exists():
        for pk in json.loads(e3_path.read_text()):
            _add_pocket_entry(pk, merged_rows, "e3")

    if extra_scan_path is not None and extra_scan_path.exists():
        # Invalidate prep_cache for best_rows keys
        prep_cache.clear()
        for pk in json.loads(extra_scan_path.read_text()):
            _add_pocket_entry(pk, best_rows, "scan5")

    def _add_brick(b: dict, name: str) -> None:
        w, h, g = int(b["w"]), int(b["h"]), int(b["g"])
        if g < 1 or w < 1 or h < 1:
            return
        x = np.asarray(b["x"], dtype=np.float64)
        y = np.asarray(b["y"], dtype=np.float64)
        th = np.asarray(b["th"], dtype=np.float64)
        if len(x) != w * h + g:
            return
        hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
        W = float((x + hx).max() - (x - hx).min())
        H = float((y + hx).max() - (y - hx).min())
        fx, fy = W - w, H - h
        if max(fx, fy) >= 0.999:
            return
        pen, _ = cpp_eval_sat_pen_bbox(x - float((x - hx).min()), y - float((y - hx).min()), th)
        if pen > 1e-6:
            return
        for tr in ((False, True) if w != h else (False,)):
            rw, rh = (h, w) if tr else (w, h)
            rfx, rfy = (fy, fx) if tr else (fx, fy)
            by_whg.setdefault((rw, rh, g), []).append(
                {
                    "uid": f"{name}_{rw}x{rh}_g{g}",
                    "kind": "brick",
                    "brick": b,
                    "transpose": tr,
                    "w": rw,
                    "h": rh,
                    "g": g,
                    "fx": rfx,
                    "fy": rfy,
                    "f": max(rfx, rfy),
                }
            )

    for pat in (
        "poc/swarm_gstep/data/brick_*.json",
        "poc/swarm_b7xplant/data/brick_*.json",
        "poc/swarm_thinb4/data/brick_*.json",
    ):
        for jp in sorted(glob.glob(str(REPO / pat))):
            b = json.loads(Path(jp).read_text())
            _add_brick(b, Path(jp).stem)

    thin_cache = REPO / "poc" / "swarm_thinb4" / "data" / "compressed_bricks_cache.json"
    if thin_cache.exists():
        for name, b in json.loads(thin_cache.read_text()).items():
            _add_brick(b, f"thin_{name}")

    # Keep top-2 distinct entries per (w, h, g) sorted by f
    pruned: dict[tuple[int, int, int], list[dict]] = {}
    for k, lst in by_whg.items():
        lst.sort(key=lambda r: r["f"])
        kept = []
        seen_uid = set()
        for it in lst:
            if it["uid"] in seen_uid:
                continue
            seen_uid.add(it["uid"])
            kept.append(it)
            if len(kept) >= 2:
                break
        pruned[k] = kept
    return pruned


def get_item_squares_50d(
    item: dict,
    merged_rows: dict[int, list[dict[str, str]]],
    best_rows: dict[int, list[dict[str, str]]],
) -> tuple[list[dict], mp.mpf, mp.mpf] | None:
    if item["kind"] == "donor_pocket":
        rmap = best_rows if item["rows_map"] == "best" else merged_rows
        return extract_pocket_50d(rmap[item["src"]], item["pk"], transpose=item["transpose"])
    return brick_to_50d(item["brick"], transpose=item["transpose"])


def sym_squares_50d(sq: list[dict], W: mp.mpf, H: mp.mpf, s: int) -> list[dict]:
    """Apply D2 reflection/180-rotation s in {0,1,2,3} preserving (W, H) at 50 digits."""
    out = []
    with mp.workdps(50):
        for r in sq:
            x, y, a = r["x"], r["y"], r["angle"]
            neg = False
            if s & 1:
                x = W - x
                neg = not neg
            if s & 2:
                y = H - y
                neg = not neg
            if neg:
                a = "-" + a[1:] if isinstance(a, str) and a.startswith("-") else ("-" + a if isinstance(a, str) else -a)
            out.append({"x": x, "y": y, "angle": a})
    return out


def find_overhang_candidates(
    by_whg: dict[tuple[int, int, int], list[dict]],
    inc_L: dict[int, float],
    regime: str = "zero_gap",
    dL_margin: float = 0.004,
) -> list[dict]:
    """Find all (K, D) pairs with kx + ky in {1, 2}, kx * ky == 0.

    regime='zero_gap': h1+h2 <= K-1 (if kx>0) or w1+w2 <= K-1 (if ky>0), D = g1+g2 - (kx+ky).
    regime='seam_touch': h1+h2 == K (if kx>0) or w1+w2 == K (if ky>0), D = g1+g2.
    """
    best_items = [lst[0] for lst in by_whg.values()]
    hits: dict[int, list[dict]] = {}
    for K in range(16, 32):
        cand = [it for it in best_items if it["w"] < K and it["h"] < K]
        for i in range(len(cand)):
            p1 = cand[i]
            for j in range(i, len(cand)):
                p2 = cand[j]
                kx = max(0, p1["w"] + p2["w"] - K)
                ky = max(0, p1["h"] + p2["h"] - K)
                k = kx + ky
                if k not in (1, 2) or kx * ky != 0:
                    continue
                if regime == "zero_gap":
                    if kx > 0 and p1["h"] + p2["h"] > K - 1:
                        continue
                    if ky > 0 and p1["w"] + p2["w"] > K - 1:
                        continue
                    D = p1["g"] + p2["g"] - k
                elif regime == "seam_touch":
                    if kx > 0 and p1["h"] + p2["h"] != K:
                        continue
                    if ky > 0 and p1["w"] + p2["w"] != K:
                        continue
                    D = p1["g"] + p2["g"]
                else:
                    raise ValueError(regime)
                if D < 1:
                    continue
                n = K * K + D
                if n > 1000 or n not in inc_L:
                    continue
                L_pred = K + max(p1["f"], p2["f"])
                dL = L_pred - inc_L[n]
                if dL <= dL_margin:
                    hits.setdefault(n, []).append(
                        {
                            "n": n,
                            "K": K,
                            "D": D,
                            "kx": kx,
                            "ky": ky,
                            "regime": regime,
                            "p1": p1,
                            "p2": p2,
                            "L_pred": L_pred,
                            "L_inc_est": inc_L[n],
                            "dL_pred": dL,
                        }
                    )
    out = []
    for n, lst in hits.items():
        lst.sort(key=lambda r: r["dL_pred"])
        out.append(lst[0])
    out.sort(key=lambda r: r["dL_pred"])
    return out


def assemble_zero_gap_50d(
    K: int,
    n: int,
    sq1: list[dict],
    W1: mp.mpf,
    H1: mp.mpf,
    w1: int,
    h1: int,
    sq2: list[dict],
    W2: mp.mpf,
    H2: mp.mpf,
    w2: int,
    h2: int,
) -> list[dict[str, str]] | None:
    """Assemble exact 50-digit zero-overlap 2-D overhang pinwheel (P1 in BL, P2 in TR)."""
    kx = max(0, w1 + w2 - K)
    ky = max(0, h1 + h2 - K)
    if kx * ky != 0 or (kx + ky) not in (1, 2):
        return None
    if kx > 0 and h1 + h2 > K - 1:
        return None
    if ky > 0 and w1 + w2 > K - 1:
        return None

    with mp.workdps(50):
        fmax = max(W1 - w1, H1 - h1, W2 - w2, H2 - h2)
        pad = mp.mpf("1e-9")
        S = mp.mpf(K) + fmax + pad

        out: list[dict] = []
        # P1 in BL
        for s in sq1:
            out.append({"x": s["x"] + EPS, "y": s["y"] + EPS, "angle": s["angle"]})
        # P2 in TR
        dx2 = S - W2 - EPS
        dy2 = S - H2 - EPS
        for s in sq2:
            out.append({"x": s["x"] + dx2, "y": s["y"] + dy2, "angle": s["angle"]})

        # TL upright block: (K - w2) cols x (K - h1) rows anchored at (0, S)
        for i in range(K - w2):
            x_c = mp.mpf(i) + mp.mpf("0.5") + (i + 1) * SPREAD
            for b in range(K - h1):
                y_c = S - mp.mpf("0.5") - mp.mpf(b) - (b + 1) * SPREAD
                out.append({"x": x_c, "y": y_c, "angle": "0"})

        # BR upright block: (K - w1) cols x (K - h2) rows anchored at (S, 0)
        for a in range(K - w1):
            x_c = S - mp.mpf("0.5") - mp.mpf(a) - (a + 1) * SPREAD
            for j in range(K - h2):
                y_c = mp.mpf(j) + mp.mpf("0.5") + (j + 1) * SPREAD
                out.append({"x": x_c, "y": y_c, "angle": "0"})

        # Central strip if t >= 2
        n_pk = len(sq1) + len(sq2)
        if kx > 0:
            t = K - (h1 + h2)
            if t >= 2:
                x_lo = mp.mpf(K - w2) + mp.mpf(K - w2 + 2) * SPREAD + EPS
                x_hi = S - mp.mpf(K - w1) - mp.mpf(K - w1 + 2) * SPREAD - EPS
                y_lo = H1 + mp.mpf("2") * EPS
                y_hi = S - H2 - mp.mpf("2") * EPS
                avail_y = y_hi - y_lo
                nrows = min(t, int(mp.floor(avail_y - mp.mpf("1e-9"))))
                if nrows > 0 and x_hi - x_lo >= mp.mpf(kx) + mp.mpf("1e-9"):
                    gap_x = (x_hi - x_lo - mp.mpf(kx)) / mp.mpf(kx + 1)
                    gap_y = (avail_y - mp.mpf(nrows)) / mp.mpf(nrows + 1)
                    for ix in range(kx):
                        xc = x_lo + (ix + 1) * gap_x + mp.mpf(ix) + mp.mpf("0.5")
                        for iy in range(nrows):
                            yc = y_lo + (iy + 1) * gap_y + mp.mpf(iy) + mp.mpf("0.5")
                            out.append({"x": xc, "y": yc, "angle": "0"})
        elif ky > 0:
            t = K - (w1 + w2)
            if t >= 2:
                x_lo = W1 + mp.mpf("2") * EPS
                x_hi = S - W2 - mp.mpf("2") * EPS
                y_lo = mp.mpf(K - h1) + mp.mpf(K - h1 + 2) * SPREAD + EPS
                y_hi = S - mp.mpf(K - h2) - mp.mpf(K - h2 + 2) * SPREAD - EPS
                avail_x = x_hi - x_lo
                ncols = min(t, int(mp.floor(avail_x - mp.mpf("1e-9"))))
                if ncols > 0 and y_hi - y_lo >= mp.mpf(ky) + mp.mpf("1e-9"):
                    gap_x = (avail_x - mp.mpf(ncols)) / mp.mpf(ncols + 1)
                    gap_y = (y_hi - y_lo - mp.mpf(ky)) / mp.mpf(ky + 1)
                    for ix in range(ncols):
                        xc = x_lo + (ix + 1) * gap_x + mp.mpf(ix) + mp.mpf("0.5")
                        for iy in range(ky):
                            yc = y_lo + (iy + 1) * gap_y + mp.mpf(iy) + mp.mpf("0.5")
                            out.append({"x": xc, "y": yc, "angle": "0"})

        if len(out) < n:
            return None
        if len(out) > n:
            # Drop surplus upright grid squares farthest from center or in central strip
            extra = len(out) - n
            grid_idxs = sorted(
                range(n_pk, len(out)),
                key=lambda idx: -(abs(float(out[idx]["x"]) - float(S) * 0.5) + abs(float(out[idx]["y"]) - float(S) * 0.5)),
            )
            drop = set(grid_idxs[:extra])
            out = [s for idx, s in enumerate(out) if idx not in drop]

        rows = []
        for idx, s in enumerate(out):
            ang = s["angle"] if isinstance(s["angle"], str) else mp.nstr(s["angle"], 45)
            rows.append(
                {
                    "square_id": str(idx),
                    "x": mp.nstr(s["x"], 45),
                    "y": mp.nstr(s["y"], 45),
                    "angle": ang,
                }
            )
    return rows


def assemble_seam_touch_float(
    K: int,
    n: int,
    sq1: list[dict],
    W1: mp.mpf,
    H1: mp.mpf,
    w1: int,
    h1: int,
    sq2: list[dict],
    W2: mp.mpf,
    H2: mp.mpf,
    w2: int,
    h2: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, tuple[float, float, float, float]] | None:
    """Assemble seam-touching 2-D overhang pinwheel (h1+h2==K or w1+w2==K, D=g1+g2)."""
    kx = max(0, w1 + w2 - K)
    ky = max(0, h1 + h2 - K)
    if kx * ky != 0 or (kx + ky) not in (1, 2):
        return None
    if kx > 0 and h1 + h2 != K:
        return None
    if ky > 0 and w1 + w2 != K:
        return None

    fmax = float(max(W1 - w1, H1 - h1, W2 - w2, H2 - h2))
    L = float(K) + fmax
    W1f, H1f, W2f, H2f = float(W1), float(H1), float(W2), float(H2)

    x1 = np.array([float(s["x"]) for s in sq1], dtype=np.float64)
    y1 = np.array([float(s["y"]) for s in sq1], dtype=np.float64)
    t1 = np.array([float(s["angle"]) for s in sq1], dtype=np.float64)

    x2 = np.array([float(s["x"]) for s in sq2], dtype=np.float64) + (L - W2f)
    y2 = np.array([float(s["y"]) for s in sq2], dtype=np.float64) + (L - H2f)
    t2 = np.array([float(s["angle"]) for s in sq2], dtype=np.float64)

    # Check tilted cores of P1 and P2 are collision-free
    def _tilt(th):
        d = np.degrees(np.mod(th, np.pi / 2))
        return np.minimum(d, 90.0 - d) > 2.0

    m1, m2 = _tilt(t1), _tilt(t2)
    if np.any(m1) and np.any(m2):
        tpen, _ = cpp_eval_sat_pen_bbox(
            np.concatenate([x1[m1], x2[m2]]),
            np.concatenate([y1[m1], y2[m2]]),
            np.concatenate([t1[m1], t2[m2]]),
        )
        if tpen > 0.05:
            return None

    gx, gy = [], []
    # TL: (K - w2) x (K - h1) anchored at (0, L)
    for i in range(K - w2):
        for b in range(K - h1):
            gx.append(i + 0.5)
            gy.append(L - 0.5 - b)
    # BR: (K - w1) x (K - h2) anchored at (L, 0)
    for a in range(K - w1):
        for j in range(K - h2):
            gx.append(L - 0.5 - a)
            gy.append(j + 0.5)

    ax = np.concatenate([x1, x2, np.array(gx, dtype=np.float64)])
    ay = np.concatenate([y1, y2, np.array(gy, dtype=np.float64)])
    at = np.concatenate([t1, t2, np.zeros(len(gx), dtype=np.float64)])
    if len(ax) != n:
        return None

    if kx > 0:
        seam_box = (max(0.0, L - W2f), min(L, W1f), max(0.0, L - H2f), min(L, H1f))
    else:
        seam_box = (max(0.0, L - W2f), min(L, W1f), max(0.0, L - H2f), min(L, H1f))
    return ax, ay, at, L, seam_box


def polish_rows_lpslack_mechwalk(
    n: int,
    base_rows: list[dict[str, str]],
    L_inc_dec: Decimal,
) -> dict:
    """Run multi-schedule lpslack + soft-mode mechwalk + lpslack and certify."""
    best_rows = base_rows
    best_dec: Decimal | None = None
    res0 = lpslack.certify_pair(n, best_rows)
    if res0 is not None:
        best_dec = max(res0[0], res0[1])
    else:
        # Repair tiny float64 brick overlaps via fixed-angle LP before lpslack
        x0 = np.array([float(r["x"]) for r in best_rows], dtype=np.float64)
        y0 = np.array([float(r["y"]) for r in best_rows], dtype=np.float64)
        t0 = np.array([float(r["angle"]) for r in best_rows], dtype=np.float64)
        xf, yf, _, okf = hu.fixed_angle_lp_tighten(x0, y0, t0, rounds=5, margin=1e-7)
        if okf:
            best_rows = lpslack.rows_from_arrays(np.column_stack([xf, yf]), t0)

    # Pass 1: wider-angle lpslack
    r1 = lpslack.optimize_group(
        n,
        best_rows,
        "999.0",
        outer_iters=6,
        fixed_rounds=6,
        slp_modes=("dual_non90", "all"),
        tr_th_schedule=(2e-3, 3e-4, 5e-5),
        cycles=1,
        time_limit=10.0,
    )
    if r1.get("rows") is not None:
        d1 = Decimal(r1["new_L"])
        if best_dec is None or d1 < best_dec:
            best_dec = d1
            best_rows = r1["rows"]

    # Pass 2: tighter refinement if promising (within +1.5e-3 of incumbent or already winning)
    if best_dec is not None and (best_dec - L_inc_dec) < Decimal("1.5e-3"):
        r2 = lpslack.optimize_group(
            n,
            best_rows,
            "999.0",
            outer_iters=8,
            fixed_rounds=10,
            slp_modes=("dual_non90", "non90", "all"),
            tr_th_schedule=(3e-4, 5e-5, 1e-5),
            cycles=1,
            time_limit=12.0,
        )
        if r2.get("rows") is not None:
            d2 = Decimal(r2["new_L"])
            if d2 < best_dec:
                best_dec = d2
                best_rows = r2["rows"]

    # Final exact certification check
    final_pair = lpslack.certify_pair(n, best_rows)
    if final_pair is None:
        return {"ok": False, "reason": "final_cert_fail"}
    lit, pan = final_pair
    dL_lit = float(lit - L_inc_dec)
    dL_pan = float(pan - L_inc_dec)
    is_win = (lit <= L_inc_dec - Decimal("1e-9")) and (pan <= L_inc_dec - Decimal("1e-9"))
    return {
        "ok": True,
        "n": n,
        "L_lit": str(lit),
        "L_pd": str(pan),
        "L_inc": str(L_inc_dec),
        "dL_lit": dL_lit,
        "dL_pd": dL_pan,
        "is_real_win": bool(is_win),
        "rows": best_rows,
    }
