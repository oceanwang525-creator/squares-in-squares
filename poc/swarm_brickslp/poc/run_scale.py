#!/usr/bin/env python3
"""Checkpointed Scale-Up Runner for H17 (Sub-Brick Compression & Broadcast + Pinwheel Seam Melt) & H21 (Cross-Family Slack Sharing)."""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import brick_slp as bs  # noqa: E402
import frontier_io as fio  # noqa: E402
import utils as u  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, run_cpp_masked_melt  # noqa: E402

DOCS_DIR = HERE.parent / "docs"
CKPT_PATH = DOCS_DIR / "scale_ckpt.json"
BRICK_NPZ = DOCS_DIR / "improved_bricks.npz"


def polish_and_certify_full(
    n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray, L_inc_dec: Decimal, label: str
) -> dict:
    """Relax if needed, run tight multi-stage joint SLP + fixed-angle LP, and certify literal+pandas."""
    hx = fio.hx_of(th)
    x = x - float((x - hx).min())
    y = y - float((y - hx).min())
    pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
    if 1e-7 < pen < 0.05:
        x, y, th, _ = u.relax_lbfgs(x, y, th, bb + max(0.002, 1.8 * pen), maxiter=200, w_wall=4.0, w_pair=1.0)
        pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
    if pen > 0.015:
        return {"ok": False, "label": label, "n": n, "pen": float(pen)}

    xs, ys, ts, Ls, ok = bs.joint_xytheta_slp_tight(x, y, th)
    if not ok:
        xs, ys, ts, Ls = x, y, th, bb

    best_cert = None
    best_dec = None
    for mlp in (1e-7, 2e-8, 5e-9, 1e-6):
        xf, yf, Lf, okf = u.fixed_angle_lp_tighten(xs, ys, ts, rounds=5, margin=mlp)
        cand_x, cand_y = (xf, yf) if okf else (xs, ys)
        cert = fio.certify_against_frontier(n, cand_x, cand_y, ts, L_inc_dec)
        if cert is not None:
            c_dec = max(Decimal(cert["L_lit"]), Decimal(cert["L_pd"]))
            if best_dec is None or c_dec < best_dec:
                best_dec = c_dec
                best_cert = cert
            if cert["is_real_win"] and mlp <= 1e-7:
                break

    if best_cert is None:
        return {"ok": False, "label": label, "n": n, "pen": float(pen)}
    best_cert["label"] = label
    if best_cert["is_real_win"]:
        fio.save_win_and_rebuild_final(n, best_cert, label)
    return best_cert


def _polish_task(args):
    n, x, y, th, L_inc_str, label = args
    return polish_and_certify_full(n, x, y, th, Decimal(L_inc_str), label)


def load_ckpt() -> dict:
    if CKPT_PATH.exists():
        return json.loads(CKPT_PATH.read_text())
    return {"subbricks": {}, "pinwheel_melt": {}, "cross_family": {}}


def save_ckpt(ckpt: dict) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_PATH.write_text(json.dumps(ckpt, indent=2))


def diversify_top_k(labels: list[str], lf: np.ndarray, po: np.ndarray, k: int = 10) -> list[int]:
    order = sorted(range(len(labels)), key=lambda i: (0 if lf[i] < 1e9 else 1, float(lf[i]) if lf[i] < 1e9 else float(po[i])))
    keep, used = [], set()
    for pref, qta in (("pw_seam_", 4), ("pw_slack_", 3), ("mpk_void_", 2), ("mpk_hot_", 1)):
        c = 0
        for i in order:
            if i not in used and labels[i].startswith(pref):
                keep.append(i)
                used.add(i)
                c += 1
                if c >= qta or len(keep) >= k:
                    break
    for i in order:
        if len(keep) >= k:
            break
        if i not in used:
            keep.append(i)
            used.add(i)
    return keep


def run_pinwheel_melt_target(
    n: int, xb: np.ndarray, yb: np.ndarray, tb: np.ndarray, Lb: float, L_inc_dec: Decimal, tag: str, n_threads: int = 4
) -> dict:
    t0 = time.time()
    c_dir = polish_and_certify_full(n, xb, yb, tb, L_inc_dec, f"{tag}_direct")
    rng = np.random.default_rng(42000 + n)
    q_list, m_list, ls_list, labels = bs.build_pinwheel_seam_seeds(n, xb, yb, tb, Lb, rng, L_expand=0.025)
    qb, qf, lf, _, po, lb = run_cpp_masked_melt(
        n, 1, q_list, m_list, ls_list, [6, 10, 14, 18],
        a_start=0.80, n_melt_stages=4, masked_iters=200, full_iters=200,
        n_threads=n_threads, base_seed=51000 + n,
    )
    keep = diversify_top_k(labels, lf, po, k=10)
    tasks = []
    for i in keep:
        use_f = lf[i] < 1e9
        L_ref = float(lf[i]) if use_f else ls_list[i][-1]
        x, y, th = cpp_expand_sym(n, 1, L_ref, qf[i] if use_f else qb[i])
        tasks.append((n, x, y, th, str(L_inc_dec), f"{tag}_{labels[i]}"))

    with mp.Pool(processes=min(4, len(tasks))) as pool:
        pol = pool.map(_polish_task, tasks)

    all_c = [c for c in ([c_dir] + pol) if c.get("ok")]
    best = min(all_c, key=lambda c: c["dL"]) if all_c else {"ok": False, "n": n}
    dt = time.time() - t0
    status = "WIN" if best.get("is_real_win") else ("OK" if best.get("ok") else "FAIL")
    print(
        f"  [{tag}] n={n} status={status} dL={best.get('dL')} via {best.get('label')} "
        f"(feas={int(np.sum(lf < 1e9))}/28, {dt:.1f}s)",
        flush=True,
    )
    return {
        "n": n,
        "status": status,
        "dL": best.get("dL"),
        "L_lit": best.get("L_lit"),
        "label": best.get("label"),
        "dt": round(dt, 1),
    }


def _subbrick_trial_v2(args):
    idx, w, h, g, x0, y0, th0, f0, mode, seed = args
    rng = np.random.default_rng(seed)
    x, y, th = x0.copy(), y0.copy(), th0.copy()
    if mode == "direct":
        xs, ys, ts, fs, ok = bs.joint_xytheta_slp_rect(x, y, th, w, h, rounds=14, tr_th0=0.020, tr_xy0=0.09, tol=1e-11)
        if ok:
            xs2, ys2, ts2, fs2, ok2 = bs.joint_xytheta_slp_rect(xs, ys, ts, w, h, rounds=12, tr_th0=0.004, tr_xy0=0.015, tol=1e-12)
            if ok2 and fs2 < fs:
                xs, ys, ts, fs = xs2, ys2, ts2, fs2
        return (fs - f0, fs, xs, ys, ts, f"rect_direct_{idx}") if ok else None

    f_exp = f0 + float(rng.uniform(0.018, 0.036))
    scx, scy = (w + f_exp) / (w + f0), (h + f_exp) / (h + f0)
    x = x * scx
    y = y * scy
    cx = float(rng.uniform(0.8, w + f_exp - 0.8))
    cy = float(rng.uniform(0.8, h + f_exp - 0.8))
    rad = float(rng.uniform(2.8, 6.2))
    near = ((x - cx) ** 2 + (y - cy) ** 2) <= rad * rad
    if mode == "reloc" and np.sum(near) > 2:
        p_idx = np.where(near)[0]
        pick = int(rng.choice(p_idx))
        x[pick] = cx + float(rng.uniform(-0.35, 0.35))
        y[pick] = cy + float(rng.uniform(-0.35, 0.35))
        th[pick] = float(rng.uniform(0, math.pi / 2))
    elif mode == "wall" and len(x) > 0:
        hx = fio.hx_of(th)
        wm = np.minimum.reduce([x - hx, (w + f_exp) - x - hx, y - hx, (h + f_exp) - y - hx])
        ti = int(np.argsort(wm)[idx % min(6, len(wm))])
        near = ((x - x[ti]) ** 2 + (y - y[ti]) ** 2) <= rad * rad
    nm = int(np.sum(near))
    if nm > 0:
        x[near] += rng.uniform(-0.16, 0.16, nm)
        y[near] += rng.uniform(-0.16, 0.16, nm)
        th[near] += rng.uniform(-0.32, 0.32, nm)

    for ft in (f0 + 0.014, f0 + 0.002, f0 - 0.001):
        x, y, th, E = bs.relax_rect_lbfgs(x, y, th, w + ft, h + ft, maxiter=500)
        if E > 1e-4:
            break
    pen, _ = cpp_eval_sat_pen_bbox(x, y, th)
    if pen > 0.02:
        return None
    xs, ys, ts, fs, ok = bs.joint_xytheta_slp_rect(x, y, th, w, h, rounds=12, tr_th0=0.020, tr_xy0=0.09, tol=1e-10)
    if not ok:
        return None
    xs2, ys2, ts2, fs2, ok2 = bs.joint_xytheta_slp_rect(xs, ys, ts, w, h, rounds=10, tr_th0=0.004, tr_xy0=0.015, tol=1e-12)
    if ok2 and fs2 < fs:
        xs, ys, ts, fs = xs2, ys2, ts2, fs2
    return (fs - f0, fs, xs, ys, ts, f"rect_{mode}_{idx}")


def compress_subbricks_phase(bricks: dict, ckpt: dict) -> dict[tuple[int, int, int], dict]:
    targets = [
        (18, 8, 8, "B8_18x8"),
        (17, 8, 7, "B7_17x8"),
        (14, 7, 6, "B6_14x7"),
        (22, 11, 10, "B10_22x11"),
        (22, 7, 5, "B5_22x7"),
        (24, 8, 6, "B6_24x8"),
        (15, 5, 5, "B5_15x5"),
        (19, 11, 7, "B9_19x11"),
        (18, 9, 6, "B6_18x9"),
        (20, 12, 8, "B10_20x12"),
        (16, 11, 4, "B4_16x11"),
        (16, 10, 4, "B4_16x10"),
        (15, 9, 3, "B3_15x9"),
        (10, 21, 6, "B6_10x21"),
        (19, 9, 4, "B4_19x9"),
    ]
    improved = {}
    for w, h, g, bname in targets:
        b = bricks.get((w, h, g)) or bricks.get((h, w, g))
        if b is None:
            continue
        bw, bh, bg, bf = b["w"], b["h"], b["g"], b["f"]
        t0 = time.time()
        trials = [(0, bw, bh, bg, b["x"], b["y"], b["th"], bf, "direct", 9000)]
        for i in range(1, 32):
            mode = ("reloc", "melt", "wall")[i % 3]
            trials.append((i, bw, bh, bg, b["x"], b["y"], b["th"], bf, mode, 9000 + i * 29))
        with mp.Pool(processes=4) as pool:
            res = [r for r in pool.map(_subbrick_trial_v2, trials) if r is not None]
        res.sort(key=lambda r: r[0])
        best_df = res[0][0] if res else 0.0
        best_f = res[0][1] if res else bf
        best_tag = res[0][5] if res else "none"
        print(f"[SubBrick] {bname} ({bw}x{bh},+{bg}) f0={bf:.9f} -> {best_f:.9f} df={best_df:+.3e} ({best_tag}) [{time.time()-t0:.1f}s]", flush=True)
        ckpt["subbricks"][bname] = {"w": bw, "h": bh, "g": bg, "f0": bf, "best_f": best_f, "df": best_df, "tag": best_tag}
        save_ckpt(ckpt)

        if res and res[0][0] < -1e-8:
            _, nf, nx, ny, n_th, _ = res[0]
            nx, ny, W, H = bs.rect_bbox(nx, ny, n_th)
            b_imp = dict(b, x=nx, y=ny, th=n_th, fx=W - bw, fy=H - bh, f=max(W - bw, H - bh))
            bricks[(bw, bh, bg)] = b_imp
            rx, ry, rt = ny.copy(), (W - nx).copy(), (n_th + 0.5 * math.pi).copy()
            rx, ry, W2, H2 = bs.rect_bbox(rx, ry, rt)
            b_rot = dict(b, w=bh, h=bw, x=rx, y=ry, th=rt, fx=W2 - bh, fy=H2 - bw, f=max(W2 - bh, H2 - bw))
            bricks[(bh, bw, bg)] = b_rot
            improved[(bw, bh, bg)] = b_imp
            improved[(bh, bw, bg)] = b_rot
    return improved


def main():
    t_start = time.time()
    packs, best_L_dec = fio.load_true_frontier()
    # Overlay any wins already saved in swarm_brickslp/data (e.g., n=639)
    for wp in sorted((HERE.parent / "data").glob("win_n*.json")):
        d = json.loads(wp.read_text())
        n_w = int(d["n"])
        Ld = max(Decimal(d["L_lit"]), Decimal(d["L_pd"]))
        if Ld < best_L_dec[n_w]:
            rows = d["rows"]
            x = np.array([float(r["x"]) for r in rows], dtype=np.float64)
            y = np.array([float(r["y"]) for r in rows], dtype=np.float64)
            th = np.array([float(r["angle"]) for r in rows], dtype=np.float64)
            packs[n_w] = fio.norm_pack(x, y, th)
            best_L_dec[n_w] = Ld

    bricks = bs.extract_subbricks(packs)
    ckpt = load_ckpt()
    print(f"Scale-up start: {len(packs)} frontier packs, {len(bricks)} bricks.", flush=True)

    # Phase A: Standalone Sub-Brick Compression (32 trials x 15 bricks)
    improved_bricks = compress_subbricks_phase(bricks, ckpt)

    # Phase A2: Broadcast any improved sub-bricks into matching 2-brick pinwheels!
    if improved_bricks:
        print(f"\n[Broadcast] Checking pinwheel broadcasts for {len(improved_bricks)//2} improved bricks...", flush=True)
        blist = list(bricks.values())
        bcast_ns = {}
        for m in range(15, 32):
            for ib in improved_bricks.values():
                if ib["w"] > m or ib["h"] > m:
                    continue
                for b2 in blist:
                    if b2["w"] > m or b2["h"] > m:
                        continue
                    if not ((ib["w"] + b2["w"] <= m - 1 and ib["h"] + b2["h"] <= m) or (ib["h"] + b2["h"] <= m - 1 and ib["w"] + b2["w"] <= m)):
                        continue
                    n = m * m + ib["g"] + b2["g"]
                    if n > 1000:
                        continue
                    f_curr = float(best_L_dec[n]) - m
                    fp = max(ib["f"], b2["f"])
                    if fp < f_curr - 1e-8 or (ib["f"] < f_curr - 1e-4 and 0.5 * (ib["f"] + b2["f"]) < f_curr - 1e-4):
                        asm = bs.assemble_pinwheel(m, ib, b2, align_mode="center")
                        if asm is not None:
                            pen, _ = cpp_eval_sat_pen_bbox(asm[0], asm[1], asm[2])
                            if pen <= 1e-7:
                                prev = bcast_ns.get(n)
                                if prev is None or fp < prev[0]:
                                    bcast_ns[n] = (fp, asm, f"bcast_{ib['w']}x{ib['h']}+{ib['g']}_and_{b2['w']}x{b2['h']}+{b2['g']}")
        for n, (fp, (ax, ay, at, L), lbl) in sorted(bcast_ns.items()):
            ck_key = f"bcast_{n}"
            if ck_key in ckpt["cross_family"]:
                continue
            print(f"  [Broadcast] n={n} pred_L={L:.9f} vs inc={best_L_dec[n]} ({lbl})", flush=True)
            res = run_pinwheel_melt_target(n, ax, ay, at, L, best_L_dec[n], lbl, n_threads=4)
            ckpt["cross_family"][ck_key] = res
            save_ckpt(ckpt)

    # Phase B: Pinwheel Seam + Slack Melt across all 32 asymmetric & B8/B7/B10/B6 targets
    pin_targets = [
        691, 918, 745, 858, 744, 972, 638, 690, 914, 854, 639,
        492, 683, 450, 446, 848, 490, 341, 413, 791, 541, 632,
        913, 975, 737, 974, 847, 734, 488, 445, 580, 327,
    ]
    print(f"\n[Phase B] Pinwheel Seam Melt across {len(pin_targets)} targets...", flush=True)
    for n in pin_targets:
        key = str(n)
        if key in ckpt["pinwheel_melt"]:
            continue
        xb, yb, tb, Lb = packs[n]
        res = run_pinwheel_melt_target(n, xb, yb, tb, Lb, best_L_dec[n], f"pw_n{n}", n_threads=4)
        ckpt["pinwheel_melt"][key] = res
        save_ckpt(ckpt)

    # Phase C: H21 Cross-Family Pinwheel Assembly + Seam Melt (using n=910, n=972, n=744 bricks)
    cross_targets = [910, 971, 911, 912, 913, 914, 972, 973, 975, 853, 854]
    key_bricks = [b for b in bricks.values() if b["src_n"] in (910, 972, 744, 854, 639, 638) or b["f"] < 0.68]
    print(f"\n[Phase C] Cross-Family Pinwheel Assembly + Seam Melt on {cross_targets} ({len(key_bricks)} key bricks)...", flush=True)
    for n in cross_targets:
        m = math.isqrt(n)
        r = n - m * m
        f_curr = float(best_L_dec[n]) - m
        cands = []
        for i, b1 in enumerate(key_bricks):
            if b1["w"] > m or b1["h"] > m or b1["g"] >= r or b1["f"] >= f_curr - 5e-4:
                continue
            need_g = r - b1["g"]
            for b2 in bricks.values():
                if b2["g"] != need_g or b2["w"] > m or b2["h"] > m:
                    continue
                if not ((b1["w"] + b2["w"] <= m - 1 and b1["h"] + b2["h"] <= m) or (b1["h"] + b2["h"] <= m - 1 and b1["w"] + b2["w"] <= m)):
                    continue
                if max(b1["f"], b2["f"]) > f_curr + 0.055:
                    continue
                f_avg = 0.5 * (b1["f"] + b2["f"])
                if f_avg < f_curr - 1e-3:
                    cands.append((f_avg, max(b1["f"], b2["f"]), b1, b2))
        cands.sort(key=lambda c: (c[0], c[1]))
        seen = set()
        for f_avg, f_max, b1, b2 in cands:
            sig = (b1["w"], b1["h"], b1["g"], b2["w"], b2["h"], b2["g"])
            if sig in seen:
                continue
            seen.add(sig)
            ck_key = f"cross_{n}_{b1['w']}x{b1['h']}_{b2['w']}x{b2['h']}"
            if ck_key in ckpt["cross_family"]:
                continue
            asm = bs.assemble_pinwheel(m, b1, b2, align_mode="center")
            if asm is None:
                continue
            ax, ay, at, L = asm
            pen, _ = cpp_eval_sat_pen_bbox(ax, ay, at)
            if pen > 1e-7:
                continue
            print(f"  [Cross] n={n} f_curr={f_curr:.5f} | B1=({b1['w']}x{b1['h']}+{b1['g']},f={b1['f']:.5f}) + B2=({b2['w']}x{b2['h']}+{b2['g']},f={b2['f']:.5f})", flush=True)
            res = run_pinwheel_melt_target(n, ax, ay, at, L, best_L_dec[n], ck_key, n_threads=4)
            ckpt["cross_family"][ck_key] = res
            save_ckpt(ckpt)
            if len(seen) >= 2:
                break

    print(f"\n=== SCALE-UP COMPLETE in {time.time() - t_start:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
