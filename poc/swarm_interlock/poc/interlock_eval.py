"""Shared candidate-generation, masked-melt evaluation, and win-certification helpers for H35."""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from decimal import Decimal

import numpy as np

import interlock_lib as il
import brick_slp
import frontier_io
import utils as u
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, run_cpp_masked_melt


def save_win_interlock(n: int, cert: dict, label: str) -> bool:
    """Save strict REAL WIN (dL <= -1e-9) and rebuild poc/swarm_interlock/final.csv."""
    if not cert.get("is_real_win"):
        return False
    il.DATA_DIR.mkdir(parents=True, exist_ok=True)
    wp = il.DATA_DIR / f"win_n{n}.json"
    if wp.exists():
        import json

        prev = json.loads(wp.read_text())
        if max(Decimal(prev["L_lit"]), Decimal(prev["L_pd"])) <= max(Decimal(cert["L_lit"]), Decimal(cert["L_pd"])):
            return False
    import json

    payload = {
        "n": int(n),
        "label": label,
        "L_lit": cert["L_lit"],
        "L_pd": cert["L_pd"],
        "L_float": cert["L_float"],
        "dL": cert["dL"],
        "rows": cert["rows"],
    }
    frontier_io.auto_union.atomic(wp, json.dumps(payload, indent=2).encode())
    os.chmod(wp, 0o644)
    groups = {}
    for p in sorted(il.DATA_DIR.glob("win_n*.json")):
        d = json.loads(p.read_text())
        groups[int(d["n"])] = d["rows"]
    text = frontier_io.auto_union.final_text(groups)
    frontier_io.auto_union.atomic(il.FINAL_CSV, text.encode())
    os.chmod(il.FINAL_CSV, 0o644)
    return True


def _polish_one(args):
    n, x, y, th, L_inc_str, label = args
    L_inc_dec = Decimal(L_inc_str)
    hx = frontier_io.hx_of(th)
    x = x - float((x - hx).min())
    y = y - float((y - hx).min())
    pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
    if 1e-7 < pen < 0.12:
        x, y, th, _ = u.relax_lbfgs(x, y, th, bb + max(0.004, 2.0 * pen), maxiter=400, w_wall=4.0, w_pair=1.0)
        pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
        if 1e-6 < pen < 0.05:
            x, y, th, _ = u.relax_lbfgs(x, y, th, bb + max(0.008, 2.5 * pen), maxiter=350, w_wall=4.0, w_pair=1.0)
            pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
    if pen > 0.025:
        return {"ok": False, "n": n, "label": label, "pen": float(pen), "bb": float(bb)}

    xs, ys, ts, Ls, ok = brick_slp.joint_xytheta_slp_tight(x, y, th)
    if not ok:
        return {"ok": False, "n": n, "label": label, "pen": float(pen), "bb": float(bb)}

    best_cert = None
    best_dec = None
    for mlp in (1e-7, 2e-8, 5e-9):
        xf, yf, _, okf = u.fixed_angle_lp_tighten(xs, ys, ts, rounds=5, margin=mlp)
        cx, cy = (xf, yf) if okf else (xs, ys)
        cert = frontier_io.certify_against_frontier(n, cx, cy, ts, L_inc_dec)
        if cert is not None:
            c_dec = max(Decimal(cert["L_lit"]), Decimal(cert["L_pd"]))
            if best_dec is None or c_dec < best_dec:
                best_dec = c_dec
                best_cert = cert
            if cert["is_real_win"] and mlp <= 1e-7:
                break

    if best_cert is None:
        return {"ok": False, "n": n, "label": label, "pen": float(pen), "bb": float(bb)}
    best_cert["label"] = label
    return best_cert


def find_candidates_for_n(n: int, bricks: dict, L_inc: float, max_pairs: int = 8) -> list[dict]:
    """Find best interlocking (and tight seam) brick pairs for target n."""
    m = int(math.isqrt(n))
    while m * m > n:
        m -= 1
    r = n - m * m
    f_inc = L_inc - m
    if r <= 1:
        return []

    keys = sorted(bricks.keys())
    pair_cands = []
    seen = set()
    for k1 in keys:
        b1 = bricks[k1]
        if b1["f"] >= f_inc - 1e-3:
            continue
        if b1["w"] >= m - 1 or b1["h"] >= m - 1:
            continue
        for k2 in keys:
            b2 = bricks[k2]
            if b1["g"] + b2["g"] != r:
                continue
            if b2["f"] >= f_inc - 1e-3:
                continue
            if b2["w"] >= m - 1 or b2["h"] >= m - 1:
                continue
            kx = b1["w"] + b2["w"] - m
            ky = b1["h"] + b2["h"] - m
            if not (0 <= kx <= 5 and 0 <= ky <= 5 and (kx > 0 or ky > 0)):
                continue
            canon = tuple(sorted([(b1["w"], b1["h"], b1["g"]), (b2["w"], b2["h"], b2["g"])]))
            canon_t = tuple(sorted([(b1["h"], b1["w"], b1["g"]), (b2["h"], b2["w"], b2["g"])]))
            if canon in seen or canon_t in seen:
                continue
            seen.add(canon)
            seen.add(canon_t)
            f_max = max(b1["f"], b2["f"])
            pair_cands.append((f_max - f_inc + 0.0002 * (kx * ky), kx, ky, b1, b2))

    pair_cands.sort(key=lambda z: z[0])
    results = []
    for _, kx, ky, b1, b2 in pair_cands[: max_pairs * 2]:
        best_asm = None
        drop_modes = ["regrid", "b1", "b2"] if (kx > 0 and ky > 0) else ["b1"]
        for s1 in range(4):
            for s2 in range(4):
                for df in drop_modes:
                    for al in ("outer", "center"):
                        delta = 0.5 * abs(b2["f"] - b1["f"])
                        asm = il.assemble_interlock(m, b1, b2, s1, s2, drop_from=df, align=al, shift_delta=delta)
                        if asm is None:
                            continue
                        ax, ay, at, aL, I_box = asm
                        pen, _ = cpp_eval_sat_pen_bbox(ax, ay, at)
                        ntilt = il.count_tilted_in_box(ax, ay, at, I_box)
                        key_rank = (ntilt, pen)
                        if best_asm is None or key_rank < best_asm[0]:
                            best_asm = (key_rank, pen, ax, ay, at, aL, I_box, f"s{s1}{s2}_{df}_{al}_t{ntilt}")
        if best_asm is not None:
            _, pen, ax, ay, at, aL, I_box, cfg = best_asm
            results.append({
                "n": n, "m": m, "r": r, "kx": kx, "ky": ky,
                "b1": b1, "b2": b2, "pen0": pen,
                "x": ax, "y": ay, "th": at, "L": aL, "I_box": I_box,
                "tag": f"B{b1['g']}_{b1['w']}x{b1['h']}+B{b2['g']}_{b2['w']}x{b2['h']}_{cfg}",
            })
        if len(results) >= max_pairs:
            break
    return results


def eval_target(n: int, cands: list[dict], L_inc_dec: Decimal, n_threads: int = 3) -> dict:
    t0 = time.time()
    best_res = None
    for c_idx, cand in enumerate(cands):
        tc = time.time()
        ax, ay, at, aL, I_box, tag = cand["x"], cand["y"], cand["th"], cand["L"], cand["I_box"], cand["tag"]
        print(
            f"  [cand {c_idx+1}/{len(cands)}] {tag} (kx={cand['kx']}, ky={cand['ky']}, L0={aL:.6f}, inc={float(L_inc_dec):.6f}, pen0={cand['pen0']:.4f})",
            flush=True,
        )

        rng = np.random.default_rng(71000 + n * 13 + c_idx)
        q_list, m_list, ls_list, labels = il.build_interlock_melt_seeds(
            n, ax, ay, at, aL, I_box, rng, L_expand=0.045, L_inc=float(L_inc_dec)
        )
        qb, qf, lf, _, po, _ = run_cpp_masked_melt(
            n, 1, q_list, m_list, ls_list, [6, 10, 14, 18],
            a_start=0.82, n_melt_stages=4, masked_iters=220, full_iters=220,
            n_threads=n_threads, base_seed=81000 + n * 19 + c_idx * 100,
        )
        n_feas = int(np.sum(lf < 1e9))
        min_po = float(np.min(po))
        order = sorted(
            range(len(labels)),
            key=lambda i: (0 if lf[i] < 1e9 else 1, float(lf[i]) if lf[i] < 1e9 else float(po[i])),
        )[:6]

        tasks = []
        for i in order:
            use_f = lf[i] < 1e9
            L_ref = float(lf[i]) if use_f else ls_list[i][-1]
            x, y, th = cpp_expand_sym(n, 1, L_ref, qf[i] if use_f else qb[i])
            tasks.append((n, x, y, th, str(L_inc_dec), f"{tag}_{labels[i]}"))

        with mp.Pool(processes=min(3, len(tasks))) as pool:
            pol = pool.map(_polish_one, tasks)

        ok_list = [r for r in pol if r.get("ok")]
        fail_pens = [round(r.get("pen", 99.0), 4) for r in pol if not r.get("ok")]
        if ok_list:
            b = min(ok_list, key=lambda r: r["dL"])
            print(
                f"    -> feas={n_feas}/{len(labels)} min_po={min_po:.4e} best_dL={b['dL']:+.6e} L={b['L_lit'][:14]} ({time.time()-tc:.1f}s)",
                flush=True,
            )
            if b["is_real_win"]:
                save_win_interlock(n, b, b["label"])
                L_inc_dec = min(L_inc_dec, Decimal(b["L_lit"]))
            if best_res is None or b["dL"] < best_res["dL"]:
                best_res = b
        else:
            print(
                f"    -> feas={n_feas}/{len(labels)} min_po={min_po:.4e} ALL FAILED SLP (pens={fail_pens}, {time.time()-tc:.1f}s)",
                flush=True,
            )

    dt = time.time() - t0
    if best_res is None:
        return {"n": n, "status": "FAIL", "dt": round(dt, 1)}
    return {
        "n": n,
        "status": "WIN" if best_res["is_real_win"] else "OK",
        "dL": best_res["dL"],
        "L_lit": best_res["L_lit"],
        "label": best_res["label"],
        "dt": round(dt, 1),
    }
