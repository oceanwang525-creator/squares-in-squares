#!/usr/bin/env python3
"""15-Min Falsification Gate for H17 (Pinwheel Sub-Brick Joint SLP + Broadcast) & H21 (Asymmetric Slack Sharing)."""
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


def run_track1_full_pinwheels(packs, best_L_dec, targets: list[int], n_threads: int = 4) -> list[dict]:
    """Run direct joint SLP + slack-shift seeds + pinwheel seam masked melt on existing targets."""
    results = []
    for n in targets:
        t0 = time.time()
        xb, yb, tb, Lb = packs[n]
        L_inc_dec = best_L_dec[n]
        print(f"\n[Track1] n={n} L_inc={L_inc_dec} (float={Lb:.9f})", flush=True)

        # 1. Direct tight joint SLP
        c_dir = polish_and_certify_full(n, xb, yb, tb, L_inc_dec, f"direct_slp_n{n}")
        dL_dir = c_dir.get("dL", None)
        print(f"  direct_slp: ok={c_dir.get('ok')} dL={dL_dir}", flush=True)

        # 2. Pinwheel seam masked melt + Idea 6 pockets
        rng = np.random.default_rng(17000 + n)
        q_list, m_list, ls_list, labels = bs.build_pinwheel_seam_seeds(n, xb, yb, tb, Lb, rng, L_expand=0.025)
        qb, qf, lf, _, po, lb = run_cpp_masked_melt(
            n, 1, q_list, m_list, ls_list, [5, 8, 12, 16],
            a_start=0.80, n_melt_stages=4, masked_iters=180, full_iters=180,
            n_threads=n_threads, base_seed=31000 + n,
        )
        order = sorted(range(len(labels)), key=lambda i: (0 if lf[i] < 1e9 else 1, float(lf[i]) if lf[i] < 1e9 else float(po[i])))
        tasks = []
        for i in order[:6]:
            use_f = lf[i] < 1e9
            L_ref = float(lf[i]) if use_f else ls_list[i][-1]
            x, y, th = cpp_expand_sym(n, 1, L_ref, qf[i] if use_f else qb[i])
            tasks.append((n, x, y, th, str(L_inc_dec), labels[i]))

        with mp.Pool(processes=min(4, len(tasks))) as pool:
            pol = pool.map(_polish_task, tasks)

        all_c = [c for c in ([c_dir] + pol) if c.get("ok")]
        best = min(all_c, key=lambda c: c["dL"]) if all_c else {"ok": False, "n": n}
        dt = time.time() - t0
        status = "WIN" if best.get("is_real_win") else ("OK" if best.get("ok") else "FAIL")
        print(
            f"  [Track1 DONE] n={n} status={status} best_dL={best.get('dL')} via {best.get('label')} ({dt:.1f}s)",
            flush=True,
        )
        results.append({
            "n": n,
            "status": status,
            "dL": best.get("dL"),
            "L_lit": best.get("L_lit"),
            "label": best.get("label"),
            "dt": round(dt, 1),
        })
    return results


def _subbrick_trial(args):
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

    # Masked pocket perturbation inside (w + f0 + 0.025) x (h + f0 + 0.025) box
    f_exp = f0 + 0.024
    scx, scy = (w + f_exp) / (w + f0), (h + f_exp) / (h + f0)
    x = x * scx
    y = y * scy
    n = len(x)
    # Pick either tightest wall/interior square or random center
    cx = float(rng.uniform(0.8, w + f_exp - 0.8))
    cy = float(rng.uniform(0.8, h + f_exp - 0.8))
    rad = float(rng.uniform(3.2, 5.5))
    near = ((x - cx) ** 2 + (y - cy) ** 2) <= rad * rad
    if mode == "reloc" and np.sum(near) > 2:
        # Teleport 1 square inside pocket to largest local gap in pocket
        p_idx = np.where(near)[0]
        pick = int(rng.choice(p_idx))
        x[pick] = cx + float(rng.uniform(-0.3, 0.3))
        y[pick] = cy + float(rng.uniform(-0.3, 0.3))
        th[pick] = float(rng.uniform(0, math.pi / 2))
    nm = int(np.sum(near))
    if nm > 0:
        x[near] += rng.uniform(-0.15, 0.15, nm)
        y[near] += rng.uniform(-0.15, 0.15, nm)
        th[near] += rng.uniform(-0.30, 0.30, nm)

    # Multi-stage shrink via L-BFGS in rectangular box
    for ft in (f0 + 0.012, f0 + 0.002, f0 - 0.0008):
        x, y, th, E = bs.relax_rect_lbfgs(x, y, th, w + ft, h + ft, maxiter=450)
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


def run_track2_subbricks(bricks: dict, packs, best_L_dec) -> list[dict]:
    """Optimize standalone rectangular sub-bricks and broadcast any df < -1e-8."""
    # Canonical sub-bricks from brief (in either wxh or hxw orientation)
    targets = [
        (18, 8, 8, "B8_18x8"),
        (17, 8, 7, "B7_17x8"),
        (14, 7, 6, "B6_14x7"),
        (22, 11, 10, "B10_22x11"),
        (22, 7, 5, "B5_22x7"),
        (24, 8, 6, "B6_24x8"),
    ]
    out = []
    for w, h, g, bname in targets:
        b = bricks.get((w, h, g)) or bricks.get((h, w, g))
        if b is None:
            print(f"[Track2] {bname} ({w}x{h},+{g}) NOT FOUND", flush=True)
            continue
        bw, bh, bg, bf = b["w"], b["h"], b["g"], b["f"]
        t0 = time.time()
        trials = [(0, bw, bh, bg, b["x"], b["y"], b["th"], bf, "direct", 5000)]
        for i in range(1, 12):
            mode = "reloc" if i % 2 == 1 else "melt"
            trials.append((i, bw, bh, bg, b["x"], b["y"], b["th"], bf, mode, 5000 + i * 17))
        with mp.Pool(processes=4) as pool:
            res = [r for r in pool.map(_subbrick_trial, trials) if r is not None]
        res.sort(key=lambda r: r[0])
        best_df = res[0][0] if res else None
        best_f = res[0][1] if res else bf
        best_tag = res[0][5] if res else "none"
        print(f"[Track2] {bname} ({bw}x{bh},+{bg} src=n{b['src_n']}) f0={bf:.9f} -> best_f={best_f:.9f} df={best_df:+.3e} ({best_tag}) [{time.time()-t0:.1f}s]", flush=True)
        out.append({"brick": bname, "w": bw, "h": bh, "g": bg, "f0": bf, "best_f": best_f, "df": best_df, "tag": best_tag})
        if res and res[0][0] < -1e-8:
            # Update brick and broadcast!
            _, nf, nx, ny, n_th, _ = res[0]
            nx, ny, W, H = bs.rect_bbox(nx, ny, n_th)
            improved_brick = dict(b, x=nx, y=ny, th=n_th, fx=W - bw, fy=H - bh, f=max(W - bw, H - bh))
            bricks[(bw, bh, bg)] = improved_brick
            # Also rotated version
            rx, ry, rt = ny.copy(), (W - nx).copy(), (n_th + 0.5 * math.pi).copy()
            rx, ry, W2, H2 = bs.rect_bbox(rx, ry, rt)
            bricks[(bh, bw, bg)] = dict(b, w=bh, h=bw, x=rx, y=ry, th=rt, fx=W2 - bh, fy=H2 - bw, f=max(W2 - bh, H2 - bw))
    return out


def run_track3_asymmetric_pinwheels(bricks: dict, packs, best_L_dec) -> list[dict]:
    """Assemble asymmetric cross-family pinwheels and run joint SLP slack sharing."""
    blist = list(bricks.values())
    target_ns = [911, 912, 913, 914, 972, 973, 975, 853, 854, 791, 492, 683, 446, 848, 490]
    tasks = []
    for n in target_ns:
        m = math.isqrt(n)
        r = n - m * m
        if r <= 0:
            continue
        f_curr = float(best_L_dec[n]) - m
        cands = []
        for i, b1 in enumerate(blist):
            if b1["w"] > m or b1["h"] > m or b1["g"] >= r:
                continue
            if b1["f"] >= f_curr - 2e-4:
                continue  # want at least one brick strictly tighter than frontier!
            need_g = r - b1["g"]
            for j, b2 in enumerate(blist):
                if b2["g"] != need_g or b2["w"] > m or b2["h"] > m:
                    continue
                if not ((b1["w"] + b2["w"] <= m - 1 and b1["h"] + b2["h"] <= m) or (b1["h"] + b2["h"] <= m - 1 and b1["w"] + b2["w"] <= m)):
                    continue
                if max(b1["f"], b2["f"]) > f_curr + 0.045:
                    continue
                # Score by average of f1 and f2 (expected slack-shared f)
                f_avg = 0.5 * (b1["f"] + b2["f"])
                if f_avg < f_curr - 1e-4:
                    cands.append((f_avg, max(b1["f"], b2["f"]), b1, b2))
        cands.sort(key=lambda c: (c[0], c[1]))
        seen_sig = set()
        picked = 0
        for f_avg, f_max, b1, b2 in cands:
            sig = (b1["w"], b1["h"], b1["g"], b1["src_n"], b2["w"], b2["h"], b2["g"], b2["src_n"])
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            asm = bs.assemble_pinwheel(m, b1, b2, align_mode="center")
            if asm is None:
                continue
            ax, ay, at, L = asm
            pen, _ = cpp_eval_sat_pen_bbox(ax, ay, at)
            if pen > 1e-7:
                continue
            lbl = f"asym_n{n}_({b1['w']}x{b1['h']}+{b1['g']}n{b1['src_n']})+({b2['w']}x{b2['h']}+{b2['g']}n{b2['src_n']})"
            tasks.append((n, ax, ay, at, str(best_L_dec[n]), lbl))
            picked += 1
            if picked >= 4:
                break

    print(f"\n[Track3] Launching {len(tasks)} asymmetric pinwheel slack-sharing runs...", flush=True)
    results = []
    if tasks:
        with mp.Pool(processes=4) as pool:
            res = pool.map(_polish_task, tasks)
        for r in res:
            st = "WIN" if r.get("is_real_win") else ("OK" if r.get("ok") else "FAIL")
            if r.get("ok"):
                print(f"  [Track3] n={r['n']} {st} dL={r['dL']:+.3e} | {r['label']}", flush=True)
            results.append({k: v for k, v in r.items() if k != "rows"})
    return results


def main():
    t_start = time.time()
    packs, best_L_dec = fio.load_true_frontier()
    bricks = bs.extract_subbricks(packs)
    print(f"Loaded {len(packs)} frontier packings and {len(bricks)} oriented bricks.", flush=True)

    # Track 2: Standalone sub-brick compression (fast, ~150-250 squares!)
    t2 = run_track2_subbricks(bricks, packs, best_L_dec)

    # Track 1: 15-min gate full pinwheels (639, 638, 691, 745, 858, 918, 972, 293, 300, 301)
    gate_ns = [639, 638, 293, 300, 301, 918, 972, 691]
    t1 = run_track1_full_pinwheels(packs, best_L_dec, gate_ns, n_threads=4)

    # Track 3: Asymmetric cross-family pinwheel slack sharing (H21)
    t3 = run_track3_asymmetric_pinwheels(bricks, packs, best_L_dec)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    report = {"dt": round(time.time() - t_start, 1), "track1": t1, "track2": t2, "track3": t3}
    (DOCS_DIR / "gate_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n=== GATE COMPLETE in {report['dt']}s ===", flush=True)


if __name__ == "__main__":
    main()
