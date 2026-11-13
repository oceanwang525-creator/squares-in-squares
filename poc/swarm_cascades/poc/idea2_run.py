"""Idea 2: Direct Joint SLP Audit + Min-Overlap Cross-n Cuts + Void-Guided +1 Insertions."""
from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

POC_DIR = Path(__file__).resolve().parent
DOCS_DIR = POC_DIR.parent / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import donor_surgery as ds  # noqa: E402
import slp_utils as su  # noqa: E402

ALL_TARGETS = [902, 912, 738, 843, 786, 904, 582, 263, 646, 593, 499, 695, 749, 863, 640]

CUT_PAIRS = [
    (582, [631, 682]),
    (499, [544, 592, 690]),
    (640, [744, 799]),
    (593, [801, 979]),
    (695, [979]),
    (738, [971, 793]),
    (912, [972, 973]),
]

PLUS1_PAIRS = [
    (646, 645),
    (749, 748),
    (863, 862),
    (593, 592),
    (738, 737),
    (263, 262),
    (904, 903),
]


def get_norm(df: pd.DataFrame, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    sub = df[df["n"] == n]
    x = sub["x"].to_numpy(copy=True)
    y = sub["y"].to_numpy(copy=True)
    th = sub["angle"].to_numpy(copy=True)
    return ds.normalize_coords(x, y, th)


def find_min_pen_cuts(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    n_target: int,
    inc_L: float,
    top_k: int = 4,
    seam_radius: float = 4.0,
) -> list[dict]:
    """Scan all (cx, cy) cross-cuts of width dk returning tightest cuts sorted by exact SAT pen."""
    n_d = len(x)
    x, y, th, L = ds.normalize_coords(x, y, th)
    dk = math.isqrt(n_d) - math.isqrt(n_target)
    if dk <= 0:
        return []

    xs = np.linspace(0.4, L - dk - 0.1, int((L - dk) / 0.35) + 1)
    ys = np.linspace(0.4, L - dk - 0.1, int((L - dk) / 0.35) + 1)
    cands = []
    seen = set()
    for cx in xs:
        kx = (x < cx) | (x >= cx + dk)
        if int(kx.sum()) < n_target:
            continue
        for cy in ys:
            keep = kx & ((y < cy) | (y >= cy + dk))
            if int(keep.sum()) != n_target:
                continue
            sig = keep.tobytes()
            if sig in seen:
                continue
            seen.add(sig)
            nx = x[keep].copy()
            ny = y[keep].copy()
            nth = th[keep].copy()
            nx[nx >= cx + dk] -= dk
            ny[ny >= cy + dk] -= dk
            nx, ny, nth, Lb = ds.normalize_coords(nx, ny, nth)
            if Lb > inc_L + 0.008:
                continue
            pen, Lb2 = su.sat_max_overlap_and_bbox(nx, ny, nth)
            sm = ((np.abs(nx - cx) < seam_radius) | (np.abs(ny - cy) < seam_radius)).astype(np.uint8)
            cands.append({
                "pen": float(pen),
                "Lb": float(Lb2),
                "cx": float(cx),
                "cy": float(cy),
                "x": nx,
                "y": ny,
                "th": nth,
                "seam": sm,
            })

    cands.sort(key=lambda d: (d["pen"], d["Lb"]))
    selected = []
    for c in cands:
        # Keep spatial diversity across (cx, cy)
        if all(abs(c["cx"] - s["cx"]) + abs(c["cy"] - s["cy"]) > 2.5 for s in selected):
            selected.append(c)
            if len(selected) >= top_k:
                break
    return selected


def find_void_plus1_seeds(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    top_k: int = 4,
    seam_radius: float = 4.5,
) -> list[dict]:
    """Find largest empty voids via exact rotated-square SDF and insert +1 square."""
    x, y, th, L = ds.normalize_coords(x, y, th)
    gx, gy = np.meshgrid(np.linspace(0.5, L - 0.5, 65), np.linspace(0.5, L - 0.5, 65))
    px, py = gx.ravel(), gy.ravel()
    c, s = np.cos(th)[None, :], np.sin(th)[None, :]
    dx = px[:, None] - x[None, :]
    dy = py[:, None] - y[None, :]
    lx = np.abs(dx * c + dy * s)
    ly = np.abs(-dx * s + dy * c)
    min_d = (np.maximum(lx, ly) - 0.5).min(axis=1)
    order = np.argsort(min_d)[::-1]

    picked = []
    for idx in order:
        vx, vy, vd = float(px[idx]), float(py[idx]), float(min_d[idx])
        if all((vx - q[0]) ** 2 + (vy - q[1]) ** 2 > 4.0 for q in picked):
            picked.append((vx, vy, vd))
            if len(picked) >= top_k:
                break

    seeds = []
    for i, (vx, vy, vd) in enumerate(picked):
        d2 = (x - vx) ** 2 + (y - vy) ** 2
        nearest = int(np.argmin(d2))
        ang = float(th[nearest])
        if i % 2 == 1:
            ang += math.radians(10.0)
        nx = np.append(x, vx)
        ny = np.append(y, vy)
        nth = np.append(th, ang)
        sm = (((nx - vx) ** 2 + (ny - vy) ** 2) <= seam_radius ** 2).astype(np.uint8)
        seeds.append({"x": nx, "y": ny, "th": nth, "seam": sm, "vx": vx, "vy": vy, "vd": vd})
    return seeds


def worker_slp(args: tuple) -> dict:
    n, name, x, y, th, pen_cpp, L_cpp, rounds = args
    t0 = time.time()
    bx, by, bth, bL, bok = su.joint_xytheta_slp(x, y, th, rounds=rounds, tr_th0=0.025, tr_xy0=0.12)
    dt = time.time() - t0
    if not bok:
        return {"n": n, "name": name, "ok": False, "pen_cpp": pen_cpp, "L_cpp": L_cpp, "L_slp": None, "dt": dt}
    v = su.verify_official(n, bx, by, bth)
    return {
        "n": n,
        "name": name,
        "ok": v["ok"],
        "pen_cpp": pen_cpp,
        "L_cpp": L_cpp,
        "L_slp": v.get("L_float"),
        "L_lit": v.get("L_lit"),
        "L_pandas": v.get("L_pandas"),
        "dt": dt,
        "x": bx if v["ok"] else None,
        "y": by if v["ok"] else None,
        "th": bth if v["ok"] else None,
    }


def record_and_check_win(res: dict, inc_L: float, front_L: float, wins: list[dict], records: list[dict]) -> None:
    n = res["n"]
    bx, by, bth = res.pop("x", None), res.pop("y", None), res.pop("th", None)
    res["inc_L"] = inc_L
    res["front_L"] = front_L
    L_slp = res.get("L_slp")
    print(f"  slp n={n} {res['name']}: ok={res['ok']} L_slp={L_slp} (inc={inc_L:.6f}, front={front_L:.6f}) dt={res['dt']:.1f}s", flush=True)
    if res["ok"] and L_slp is not None and L_slp < inc_L - 1e-11 and bx is not None:
        w = su.save_win_to_final_csv(n, bx, by, bth)
        res["win_check"] = w
        if w.get("win"):
            print(f"  [WIN!] n={n} {res['name']}: L_lit={w['L_lit']} vs inc={w['inc_L']} (dL=-{w['dL']:.6e})", flush=True)
            wins.append(w)
    records.append(res)


def main() -> None:
    t_all = time.time()
    df, info = su.load_merged_and_gap(ALL_TARGETS)
    print("Targets:", {n: round(info[n]["L_ours"], 6) for n in ALL_TARGETS}, flush=True)

    records: list[dict] = []
    wins: list[dict] = []

    # 1. Track A: Direct Joint SLP audit on all 15 targets
    print("\n=== TRACK A: Direct Joint (x,y,dtheta) SLP Audit (4 workers) ===", flush=True)
    direct_tasks = []
    for n in ALL_TARGETS:
        x, y, th, L = get_norm(df, n)
        direct_tasks.append((n, f"{n}_direct_slp", x, y, th, 0.0, L, 12))
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(worker_slp, t) for t in direct_tasks]
        for fut in as_completed(futs):
            res = fut.result()
            n = res["n"]
            record_and_check_win(res, info[n]["L_ours"], info[n]["L_front"], wins, records)

    # 2. Track B: Min-overlap cross-n interior cuts
    print("\n=== TRACK B: Min-Overlap Cross-n Cuts + 4-Thread C++ a-Melt + SLP ===", flush=True)
    for target, donors in CUT_PAIRS:
        inc_L = info[target]["L_ours"]
        front_L = info[target]["L_front"]
        coords_list, sched_list, masks_list, names = [], [], [], []
        for donor in donors:
            xd, yd, thd, _ = get_norm(df, donor)
            cuts = find_min_pen_cuts(xd, yd, thd, target, inc_L, top_k=4, seam_radius=4.0)
            for idx, c in enumerate(cuts):
                Lb = c["Lb"]
                L_goal = min(Lb, inc_L - 0.002)
                L_sched = [max(Lb + 0.04, inc_L + 0.03), max(Lb + 0.015, inc_L + 0.008), Lb + 0.002, L_goal]
                coords_list.append((c["x"], c["y"], c["th"]))
                sched_list.append(L_sched)
                masks_list.append(c["seam"])
                names.append(f"{donor}_to_{target}_c{idx}_pen{c['pen']:.2f}")
                print(f"  seed {names[-1]}: raw_pen={c['pen']:.3f} Lb={Lb:.6f} cx={c['cx']:.1f} cy={c['cy']:.1f}", flush=True)

        if not coords_list:
            continue
        t0 = time.time()
        cpp_out = su.cpp_batch_melt(
            coords_list, sched_list, sym_mode=1, a_start=0.85, n_melt_stages=6,
            seam_masks=masks_list, seam_iters=260, full_iters=450, n_ruin_iters=10,
            n_threads=4, base_seed=target * 17,
        )
        print(f"  [n={target}] C++ batch melt ({len(coords_list)} seeds) in {time.time()-t0:.1f}s", flush=True)
        ranked = sorted(zip(names, cpp_out), key=lambda z: (0 if z[1]["pen"] < 1e-6 else 1, z[1]["pen"], z[1]["L_bbox"]))
        to_slp = []
        for name, r in ranked[:4]:
            print(f"    cpp {name}: pen={r['pen']:.3e} Lb={r['L_bbox']:.6f} feas={r['best_L_feas']}", flush=True)
            if r["pen"] < 0.03:
                to_slp.append((target, name, r["x"], r["y"], r["th"], r["pen"], r["L_bbox"], 12))
        if to_slp:
            with ProcessPoolExecutor(max_workers=4) as ex:
                for fut in as_completed([ex.submit(worker_slp, t) for t in to_slp]):
                    record_and_check_win(fut.result(), inc_L, front_L, wins, records)

    # 3. Track C: Void-guided +1 insertions
    print("\n=== TRACK C: Void-Guided +1 Insertions + 4-Thread C++ a-Melt + SLP ===", flush=True)
    for target, donor in PLUS1_PAIRS:
        inc_L = info[target]["L_ours"]
        front_L = info[target]["L_front"]
        xd, yd, thd, Ld = get_norm(df, donor)
        vseeds = find_void_plus1_seeds(xd, yd, thd, top_k=4, seam_radius=4.5)
        coords_list, sched_list, masks_list, names = [], [], [], []
        for idx, vs in enumerate(vseeds):
            L_sched = [inc_L + 0.035, inc_L + 0.010, inc_L - 0.001, max(Ld, inc_L - 0.005)]
            coords_list.append((vs["x"], vs["y"], vs["th"]))
            sched_list.append(L_sched)
            masks_list.append(vs["seam"])
            names.append(f"{donor}_plus1_v{idx}_d{vs['vd']:.2f}")
        t0 = time.time()
        cpp_out = su.cpp_batch_melt(
            coords_list, sched_list, sym_mode=1, a_start=0.85, n_melt_stages=6,
            seam_masks=masks_list, seam_iters=260, full_iters=450, n_ruin_iters=10,
            n_threads=4, base_seed=target * 31,
        )
        print(f"  [n={target} +1] C++ batch melt in {time.time()-t0:.1f}s", flush=True)
        ranked = sorted(zip(names, cpp_out), key=lambda z: (0 if z[1]["pen"] < 1e-6 else 1, z[1]["pen"], z[1]["L_bbox"]))
        to_slp = []
        for name, r in ranked[:3]:
            print(f"    cpp {name}: pen={r['pen']:.3e} Lb={r['L_bbox']:.6f} feas={r['best_L_feas']}", flush=True)
            if r["pen"] < 0.03:
                to_slp.append((target, name, r["x"], r["y"], r["th"], r["pen"], r["L_bbox"], 12))
        if to_slp:
            with ProcessPoolExecutor(max_workers=4) as ex:
                for fut in as_completed([ex.submit(worker_slp, t) for t in to_slp]):
                    record_and_check_win(fut.result(), inc_L, front_L, wins, records)

    out_path = DOCS_DIR / "idea2_results.json"
    out_path.write_text(json.dumps(records, indent=2))
    print(f"\nIdea 2 finished in {time.time()-t_all:.1f}s. Wins={len(wins)}. Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
