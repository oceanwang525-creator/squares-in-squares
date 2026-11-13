"""H35: Interlocking Corner-Shared Sub-Brick Pinwheel Assembly + Masked Seam Melt."""
from __future__ import annotations

import csv
import json
import math
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
BRICKSLP_POC = Path(
    "/home/lucas/.gemini/antigravity-cli/worktrees/a4fb0b93-b63b-426d-aa12-67454e99bc08/swarm-brickslp-1e87a2dd/poc/swarm_brickslp/poc"
)
BRICKSLP_DATA = BRICKSLP_POC.parent / "data"
HIGHD_POC = REPO / "poc" / "swarm_highd" / "poc"
HOMO_POC = REPO / "poc" / "swarm_homology" / "poc"
NEW159_POC = REPO / "poc" / "swarm_new159" / "poc"
GAD_POC = REPO / "poc" / "rectangle_gadgets" / "poc"
UNION_POC = REPO / "poc" / "auto_union" / "poc"

for _p in (UNION_POC, HOMO_POC, HIGHD_POC, NEW159_POC, GAD_POC, BRICKSLP_POC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import brick_slp  # noqa: E402
import frontier_io  # noqa: E402
import gadlib  # noqa: E402
import utils as u  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, full_to_free_sym, run_cpp_masked_melt  # noqa: E402

INTERLOCK_DIR = REPO / "poc" / "swarm_interlock"
DATA_DIR = INTERLOCK_DIR / "data"
DOCS_DIR = INTERLOCK_DIR / "docs"
FINAL_CSV = INTERLOCK_DIR / "final.csv"
MERGED_CSV = REPO / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"


def fast_load_frontier(target_ns: list[int] | None = None) -> tuple[dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]], dict[int, Decimal]]:
    """Fast load minimum-L packings across merged.csv, poc/*/final.csv, and brickslp wins (~1s).

    Exact Decimal L_inc is computed via union.certify for `target_ns` (or float64 fallback for others).
    """
    packs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    best_L_dec: dict[int, Decimal] = {}

    df = pd.read_csv(MERGED_CSV, usecols=["n", "square_id", "x", "y", "angle"])
    for n_val, sub in df.groupby("n"):
        n = int(n_val)
        sub = sub.sort_values("square_id")
        x, y, th, L = frontier_io.norm_pack(
            sub["x"].to_numpy(np.float64),
            sub["y"].to_numpy(np.float64),
            sub["angle"].to_numpy(np.float64),
        )
        packs[n] = (x, y, th, L)
        best_L_dec[n] = Decimal(f"{L:.17g}")

    union_json = REPO / "poc" / "auto_union" / "docs" / "union.json"
    if union_json.exists():
        try:
            rep = json.loads(union_json.read_text())
            for r in rep.get("reports", [{}])[0].get("results", []):
                if r.get("L") is not None:
                    best_L_dec[int(r["n"])] = Decimal(str(r["L"]))
        except Exception:
            pass

    need = set(int(n) for n in target_ns) if target_ns else set()
    best_raw_rows: dict[int, list[dict]] = {}

    for fp in sorted((REPO / "poc").glob("*/final.csv")):
        if fp.parent.name == "swarm_interlock" or fp.stat().st_size <= 20:
            continue
        groups: dict[int, list[tuple[int, float, float, float, dict]]] = {}
        try:
            with fp.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    n_str, sid_str = row["id"].split("_")
                    n = int(n_str)
                    sid = int(sid_str)
                    xs = row["x"].lstrip("s")
                    ys = row["y"].lstrip("s")
                    ds = row["deg"].lstrip("s")
                    rad = math.radians(float(ds))
                    raw = {"square_id": sid, "x": xs, "y": ys, "deg": ds}
                    groups.setdefault(n, []).append((sid, float(xs), float(ys), rad, raw))
        except Exception:
            continue
        for n, rows in groups.items():
            if len(rows) != n:
                continue
            rows.sort(key=lambda z: z[0])
            nx, ny, nt, L = frontier_io.norm_pack(
                np.array([z[1] for z in rows], dtype=np.float64),
                np.array([z[2] for z in rows], dtype=np.float64),
                np.array([z[3] for z in rows], dtype=np.float64),
            )
            if L < packs[n][3] + 1e-9:
                if n in need:
                    # Exact Decimal check via auto_union
                    parsed = frontier_io.auto_union.parse_text(fp.read_text(encoding="utf-8"))
                    if n in parsed:
                        Ld = Decimal(frontier_io.auto_union.certify(n, parsed[n]))
                        if Ld < best_L_dec[n]:
                            best_L_dec[n] = Ld
                            packs[n] = (nx, ny, nt, L)
                            best_raw_rows[n] = parsed[n]
                elif L < packs[n][3] - 1e-12:
                    packs[n] = (nx, ny, nt, L)
                    best_L_dec[n] = Decimal(f"{L:.17g}")

    # Also incorporate brickslp win JSONs for packs (without overriding clean brick sources if needed)
    extra_brick_packs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    if BRICKSLP_DATA.exists():
        for jp in sorted(BRICKSLP_DATA.glob("*win_n*.json")):
            rec = json.loads(jp.read_text())
            n = int(rec["n"])
            rows = rec["rows"]
            x = np.array([float(r["x"]) for r in rows], dtype=np.float64)
            y = np.array([float(r["y"]) for r in rows], dtype=np.float64)
            th = np.array([float(r["angle"]) for r in rows], dtype=np.float64)
            nx, ny, nt, L = frontier_io.norm_pack(x, y, th)
            extra_brick_packs[n] = (nx, ny, nt, L)
            if n in need:
                Ld = max(Decimal(rec["L_lit"]), Decimal(rec["L_pd"]))
                if Ld < best_L_dec[n]:
                    best_L_dec[n] = Ld
                    packs[n] = (nx, ny, nt, L)
            elif L < packs[n][3] - 1e-12:
                packs[n] = (nx, ny, nt, L)

    return packs, best_L_dec


def extract_all_bricks(packs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]], compress_key_bricks: bool = True) -> dict[tuple[int, int, int], dict]:
    """Extract sub-bricks from frontier packings + brickslp win JSONs + optional SLP tightening."""
    bricks = brick_slp.extract_subbricks(packs)

    def _add_rect(x: np.ndarray, y: np.ndarray, th: np.ndarray, X0: float, X1: float, Y0: float, Y1: float, w: int, h: int, src_n: int, tag: str) -> None:
        e = frontier_io.hx_of(th)
        idx = np.where((x - e >= X0 - 0.05) & (x + e <= X1 + 0.05) & (y - e >= Y0 - 0.05) & (y + e <= Y1 + 0.05))[0]
        g = len(idx) - w * h
        if g <= 0:
            return
        nx, ny, W, H = brick_slp.rect_bbox(x[idx], y[idx], th[idx])
        pen, _ = cpp_eval_sat_pen_bbox(nx, ny, th[idx])
        if pen > 1e-7:
            return
        fx, fy = W - w, H - h
        f = max(fx, fy)
        if f >= 1.0 - 1e-7:
            return
        for rot in (0, 1):
            if rot == 0:
                rw, rh, rx, ry, rt, rfx, rfy = w, h, nx.copy(), ny.copy(), th[idx].copy(), fx, fy
            else:
                rw, rh = h, w
                rx, ry, rt = ny.copy(), (W - nx).copy(), (th[idx] + 0.5 * math.pi).copy()
                rx, ry, W2, H2 = brick_slp.rect_bbox(rx, ry, rt)
                rfx, rfy = W2 - rw, H2 - rh
            k = (rw, rh, g)
            if k not in bricks or max(rfx, rfy) < bricks[k]["f"] - 1e-12:
                bricks[k] = {
                    "w": rw, "h": rh, "g": g, "f": max(rfx, rfy), "fx": rfx, "fy": rfy,
                    "src_n": src_n, "tag": tag, "x": rx, "y": ry, "th": rt,
                }

    # Forced extraction from brickslp win JSONs (since seam melt prevents gadlib.clean_rect from stopping)
    if BRICKSLP_DATA.exists():
        forced_specs = [
            ("win_n910.json", [(0, 10.67, 9.0, 30.67, 10, 21, "w910_L"), (11.0, 30.67, 0.0, 9.67, 19, 9, "w910_B")]),
            ("win_n911.json", [(0, 10.69, 9.0, 30.69, 10, 21, "w911_L"), (13.0, 30.69, 0.0, 9.69, 17, 9, "w911_B")]),
            ("win_n971.json", [(0, 10.66, 10.0, 31.66, 10, 21, "w971_L"), (15.0, 31.66, 0.0, 10.66, 16, 10, "w971_B")]),
            ("win_n538.json", [(0, 18.71, 14.0, 23.71, 18, 9, "w538_T"), (18.0, 23.71, 0.0, 13.71, 5, 13, "w538_R")]),
        ]
        for fname, rects in forced_specs:
            fp = BRICKSLP_DATA / fname
            if not fp.exists():
                continue
            rec = json.loads(fp.read_text())
            rows = rec["rows"]
            x = np.array([float(r["x"]) for r in rows], dtype=np.float64)
            y = np.array([float(r["y"]) for r in rows], dtype=np.float64)
            th = np.array([float(r["angle"]) for r in rows], dtype=np.float64)
            x, y, th, _ = frontier_io.norm_pack(x, y, th)
            for X0, X1, Y0, Y1, w, h, tag in rects:
                _add_rect(x, y, th, X0, X1, Y0, Y1, w, h, int(rec["n"]), tag)

    if compress_key_bricks:
        for key in [(19, 9, 4), (10, 21, 6), (24, 8, 6), (17, 9, 5), (16, 10, 4), (11, 16, 4)]:
            if key not in bricks:
                continue
            b = bricks[key]
            w, h, g = key
            rx, ry, rt, rf, rok = brick_slp.joint_xytheta_slp_rect(b["x"], b["y"], b["th"], w, h, rounds=10)
            if rok and rf < b["f"] - 1e-8:
                nx, ny, W, H = brick_slp.rect_bbox(rx, ry, rt)
                _add_rect(nx, ny, rt, -0.1, W + 0.1, -0.1, H + 0.1, w, h, b["src_n"], b["tag"] + "_slp")

    return bricks


def sym_brick(b: dict, s: int) -> dict:
    """Apply D2 reflection/180-rotation s in {0,1,2,3} preserving (w, h)."""
    x, y, th = b["x"].copy(), b["y"].copy(), b["th"].copy()
    w, h = b["w"], b["h"]
    W, H = w + b["fx"], h + b["fy"]
    if s & 1:
        x = W - x
        th = -th
    if s & 2:
        y = H - y
        th = -th
    nx, ny, W2, H2 = brick_slp.rect_bbox(x, y, th)
    return dict(b, x=nx, y=ny, th=th, fx=W2 - w, fy=H2 - h, f=max(W2 - w, H2 - h))


def tilted_mask(th: np.ndarray, thresh_deg: float = 2.0) -> np.ndarray:
    t = np.degrees(np.mod(th, np.pi / 2))
    return np.minimum(t, 90.0 - t) > thresh_deg


def count_tilted_in_box(x: np.ndarray, y: np.ndarray, th: np.ndarray, box: tuple[float, float, float, float]) -> int:
    """Count tilted squares whose centers fall within `box = (x0, x1, y0, y1)` padded by 0.5."""
    t = tilted_mask(th)
    if not np.any(t):
        return 0
    x0, x1, y0, y1 = box
    hx = 0.5 * (np.abs(np.cos(th[t])) + np.abs(np.sin(th[t])))
    tx, ty = x[t], y[t]
    hit = (tx + hx > x0 + 0.05) & (tx - hx < x1 - 0.05) & (ty + hx > y0 + 0.05) & (ty - hx < y1 - 0.05)
    return int(np.sum(hit))


def assemble_interlock(
    m: int,
    b1: dict,
    b2: dict,
    s1: int,
    s2: int,
    drop_from: str = "b1",
    align: str = "outer",
    shift_delta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, tuple[float, float, float, float]] | None:
    """Assemble interlocking pinwheel in [0, m+f]^2 and return (x, y, th, L, I_bbox) or None."""
    sb1 = sym_brick(b1, s1)
    sb2 = sym_brick(b2, s2)
    w1, h1, g1 = sb1["w"], sb1["h"], sb1["g"]
    w2, h2, g2 = sb2["w"], sb2["h"], sb2["g"]
    kx = w1 + w2 - m
    ky = h1 + h2 - m
    if kx < 0 or ky < 0 or (kx == 0 and ky == 0):
        return None
    n_exp = m * m + g1 + g2
    f = max(sb1["f"], sb2["f"])
    L = float(m) + f

    if align == "outer":
        dx1, dy1 = 0.0, L - (h1 + sb1["fy"])
        dx2, dy2 = L - (w2 + sb2["fx"]), 0.0
    elif align == "inner":
        dx1, dy1 = f - sb1["fx"], float(m - h1)
        dx2, dy2 = float(m - w2), f - sb2["fy"]
    else:  # center
        dx1 = 0.5 * (f - sb1["fx"])
        dy1 = (m - h1) + 0.5 * (f - sb1["fy"])
        dx2 = (m - w2) + 0.5 * (f - sb2["fx"])
        dy2 = 0.5 * (f - sb2["fy"])

    x1, y1, t1 = sb1["x"] + dx1, sb1["y"] + dy1, sb1["th"].copy()
    x2, y2, t2 = sb2["x"] + dx2, sb2["y"] + dy2, sb2["th"].copy()

    # Check tilted cores of sb1 and sb2 are collision-free!
    tilt1 = tilted_mask(t1)
    tilt2 = tilted_mask(t2)
    if np.any(tilt1) and np.any(tilt2):
        tpen, _ = cpp_eval_sat_pen_bbox(
            np.concatenate([x1[tilt1], x2[tilt2]]),
            np.concatenate([y1[tilt1], y2[tilt2]]),
            np.concatenate([t1[tilt1], t2[tilt2]]),
        )
        if tpen > 1e-6:
            return None

    k_drop = kx * ky
    I_box = (float(m - w2), float(w1) + f, float(m - h1), float(h2) + f)

    if k_drop > 0:
        if drop_from == "b1":
            cand = np.where((~tilt1) & (x1 > m - w2 - 0.35) & (y1 < h2 + f + 0.35))[0]
            if len(cand) < k_drop:
                return None
            score = x1[cand] - y1[cand]
            drop_idx = cand[np.argsort(-score)[:k_drop]]
            keep = np.ones(len(x1), dtype=bool)
            keep[drop_idx] = False
            x1, y1, t1 = x1[keep], y1[keep], t1[keep]
        elif drop_from == "b2":
            cand = np.where((~tilt2) & (x2 < w1 + f + 0.35) & (y2 > m - h1 - 0.35))[0]
            if len(cand) < k_drop:
                return None
            score = -x2[cand] + y2[cand]
            drop_idx = cand[np.argsort(-score)[:k_drop]]
            keep = np.ones(len(x2), dtype=bool)
            keep[drop_idx] = False
            x2, y2, t2 = x2[keep], y2[keep], t2[keep]
        elif drop_from == "regrid":
            # Drop kx*ky from BOTH and place kx*ky centered grid in I
            c1 = np.where((~tilt1) & (x1 > m - w2 - 0.35) & (y1 < h2 + f + 0.35))[0]
            c2 = np.where((~tilt2) & (x2 < w1 + f + 0.35) & (y2 > m - h1 - 0.35))[0]
            if len(c1) < k_drop or len(c2) < k_drop:
                return None
            d1 = c1[np.argsort(-(x1[c1] - y1[c1]))[:k_drop]]
            d2 = c2[np.argsort(-(-x2[c2] + y2[c2]))[:k_drop]]
            k1 = np.ones(len(x1), dtype=bool); k1[d1] = False
            k2 = np.ones(len(x2), dtype=bool); k2[d2] = False
            x1, y1, t1 = x1[k1], y1[k1], t1[k1]
            x2, y2, t2 = x2[k2], y2[k2], t2[k2]
            gx_i, gy_i = [], []
            for ix in range(kx):
                for iy in range(ky):
                    gx_i.append((m - w2) + 0.5 * f + ix + 0.5)
                    gy_i.append((m - h1) + 0.5 * f + iy + 0.5)
            x1 = np.concatenate([x1, np.array(gx_i)])
            y1 = np.concatenate([y1, np.array(gy_i)])
            t1 = np.concatenate([t1, np.zeros(len(gx_i))])
        else:
            return None

    # Fill BL [0, m-w2] x [0, m-h1] and TR [w1+f, m+f] x [h2+f, m+f]
    gx, gy = [], []
    for i in range(m):
        for j in range(m):
            if i < w1 and j >= m - h1:
                continue
            if i >= m - w2 and j < h2:
                continue
            if i < m - w2 and j < m - h1:
                gx.append(i + 0.5)
                gy.append(j + 0.5 + (shift_delta if sb1["f"] < sb2["f"] else 0.0))
            elif i >= w1 and j >= h2:
                gx.append(i + f + 0.5 - (shift_delta if sb1["f"] < sb2["f"] else 0.0))
                gy.append(j + f + 0.5)

    ax = np.concatenate([x1, x2, np.array(gx, dtype=np.float64)])
    ay = np.concatenate([y1, y2, np.array(gy, dtype=np.float64)])
    at = np.concatenate([t1, t2, np.zeros(len(gx), dtype=np.float64)])
    if len(ax) != n_exp:
        return None
    return ax, ay, at, L, I_box


def build_interlock_melt_seeds(
    n: int,
    xb: np.ndarray,
    yb: np.ndarray,
    tb: np.ndarray,
    Lb: float,
    I_box: tuple[float, float, float, float],
    rng: np.random.Generator,
    L_expand: float = 0.045,
    L_inc: float | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[float]], list[str]]:
    """Build C++ masked-melt seeds centered on the interlocking overlap I and adjacent seams."""
    L0 = max(Lb, L_inc if L_inc is not None else Lb) + L_expand
    sc = L0 / Lb
    x0, y0, t0 = xb * sc, yb * sc, tb.copy()
    ix0, ix1, iy0, iy1 = [v * sc for v in I_box]
    icx, icy = 0.5 * (ix0 + ix1), 0.5 * (iy0 + iy1)

    L_t1 = min(Lb + 0.015, (L_inc - 0.0015) if L_inc is not None else (Lb + 0.015))
    L_t1 = max(L_t1, Lb + 0.002)
    sched_list = [
        [L0, 0.5 * (L0 + L_t1), L_t1 + 0.004, L_t1],
        [L0, Lb + 0.020, Lb + 0.008, Lb + 0.001],
        [L0, Lb + 0.012, Lb + 0.002, Lb - 0.002],
    ]

    q_list, m_list, ls_list, labels = [], [], [], []

    centers = [
        (icx, icy, "I_ctr"),
        (ix0, iy0, "I_bl"),
        (ix1, iy1, "I_tr"),
        (ix0, iy1, "I_tl"),
        (ix1, iy0, "I_br"),
        (icx, iy0 - 1.5, "I_seam_bot"),
        (icx, iy1 + 1.5, "I_seam_top"),
        (ix0 - 1.5, icy, "I_seam_left"),
        (ix1 + 1.5, icy, "I_seam_right"),
    ]

    for s_idx in range(14):
        x, y, th = x0.copy(), y0.copy(), t0.copy()
        cx, cy, cname = centers[s_idx % len(centers)]
        rad = float(rng.uniform(4.2, 8.2))
        near = ((x - cx) ** 2 + (y - cy) ** 2) <= rad * rad
        in_I = (x >= ix0 - 1.5) & (x <= ix1 + 1.5) & (y >= iy0 - 1.5) & (y <= iy1 + 1.5)
        near = near | in_I
        nm = int(near.sum())
        if nm > 0:
            x[near] += rng.uniform(-0.18, 0.18, nm)
            y[near] += rng.uniform(-0.18, 0.18, nm)
            th[near] += rng.uniform(-0.28, 0.28, nm)
        q_list.append(full_to_free_sym(x, y, th, L0))
        m_list.append(np.concatenate([near.astype(np.uint8)] * 3))
        ls_list.append(sched_list[s_idx % len(sched_list)])
        labels.append(f"imelt_{cname}_{s_idx}")

    # 10 unit-focused melt seeds around I (preserving tilted cores outside core radius)
    t_deg = np.degrees(np.mod(t0, np.pi / 2))
    is_unit = np.minimum(t_deg, 90.0 - t_deg) < 2.0
    for s_idx in range(10):
        x, y, th = x0.copy(), y0.copy(), t0.copy()
        rad = float(rng.uniform(5.5, 11.5))
        cx, cy, _ = centers[s_idx % 5]
        near = (((x - cx) ** 2 + (y - cy) ** 2) <= rad * rad) & (
            is_unit | (((x - icx) ** 2 + (y - icy) ** 2) <= 3.8 ** 2)
        )
        nm = int(near.sum())
        if nm > 0:
            x[near] += rng.uniform(-0.16, 0.16, nm)
            y[near] += rng.uniform(-0.16, 0.16, nm)
            th[near] += rng.uniform(-0.20, 0.20, nm)
        q_list.append(full_to_free_sym(x, y, th, L0))
        m_list.append(np.concatenate([near.astype(np.uint8)] * 3))
        ls_list.append(sched_list[s_idx % len(sched_list)])
        labels.append(f"umelt_{s_idx}")

    return q_list, m_list, ls_list, labels
