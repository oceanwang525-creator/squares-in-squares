"""Hypotheses H50 & H51: Sub-brick harvest from mechwalk b4 (909, 972) + main (970, 793) and broadcast."""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import pathlib
import sys
import time
from decimal import Decimal

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
for p in (HERE, HERE.parent.parent / "swarm_interlock" / "poc"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import b5_lib as bl
import brick_slp
import frontier_io
import interlock_lib as il
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, full_to_free_sym, run_cpp_masked_melt


def _load_mw_rad_json(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = sorted(d["rows"], key=lambda r: int(r["square_id"]))
    x = np.array([float(r["x"]) for r in rows], dtype=np.float64)
    y = np.array([float(r["y"]) for r in rows], dtype=np.float64)
    th_raw = np.array([float(r["angle"]) for r in rows], dtype=np.float64)
    # Detect radians vs degrees via norm_pack L against L_lit
    L_ref = float(d["L_lit"])
    xr, yr, tr, Lr = frontier_io.norm_pack(x, y, th_raw)
    xd, yd, td, Ld = frontier_io.norm_pack(x, y, np.radians(th_raw))
    return (xr, yr, tr, Lr) if abs(Lr - L_ref) <= abs(Ld - L_ref) else (xd, yd, td, Ld)


def build_h50_bank(packs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]]) -> list[dict]:
    """Harvest compressed bricks from b4_win_n909, b4_win_n972, main_n970, main_n793, and donor_win_n850."""
    bank: list[dict] = []
    seen: set[tuple] = set()

    def _add(b: dict | None) -> None:
        if b is None:
            return
        for cand in (b, bl.transpose_brick(b)):
            key = (cand["tag"], cand["w"], cand["h"], cand["g"])
            if key not in seen:
                seen.add(key)
                bank.append(cand)

    # 1. mechwalk b4_win_n909 (L = 30.640084506)
    p909 = bl.DATA_DIR / "donor_b4_win_n909.json"
    if p909.exists():
        x, y, th, _ = _load_mw_rad_json(p909)
        hx = frontier_io.hx_of(th)
        m1 = (x + hx <= 10.69) & (y - hx >= 9.99)
        if m1.sum() == 205:
            b5_b4 = bl.make_brick(10, 20, 5, x[m1], y[m1], th[m1], 909, "B5_10x20_b4")
            _add(b5_b4)
            _add(bl.compress_brick_rect(b5_b4, n_trials=16, seed=909))
        m2 = (x - hx >= 13.99) & (y + hx <= 10.65)
        if m2.sum() == 164:
            _add(bl.make_brick(16, 10, 4, x[m2], y[m2], th[m2], 909, "B4_16x10_b4"))

    # 2. mechwalk b4_win_n972 (L = 31.656533614)
    p972 = bl.DATA_DIR / "donor_b4_win_n972.json"
    if p972.exists():
        x, y, th, _ = _load_mw_rad_json(p972)
        hx = frontier_io.hx_of(th)
        m1 = (x - hx >= 20.99) & (y + hx <= 21.66)
        if m1.sum() == 216:
            _add(bl.make_brick(10, 21, 6, x[m1], y[m1], th[m1], 972, "B6_10x21_b4"))
        m2 = (x + hx <= 20.66) & (y - hx >= 20.99)
        if m2.sum() == 205:
            _add(bl.make_brick(20, 10, 5, x[m2], y[m2], th[m2], 972, "B5_20x10_b4"))

    # 3. origin/main n=970 (L = 31.638981798)
    if 970 in packs:
        x, y, th, _ = packs[970]
        hx = frontier_io.hx_of(th)
        m1 = (x + hx <= 10.65) & (y - hx >= 14.99)
        if m1.sum() == 164:
            _add(bl.make_brick(10, 16, 4, x[m1], y[m1], th[m1], 970, "B4_10x16_n970"))
        m2 = (x - hx >= 10.99) & (y + hx <= 10.65)
        if m2.sum() == 204:
            _add(bl.make_brick(20, 10, 4, x[m2], y[m2], th[m2], 970, "B4_20x10_n970"))

    # 4. origin/main n=793 (L = 28.659397487)
    if 793 in packs:
        x, y, th, _ = packs[793]
        hx = frontier_io.hx_of(th)
        m1 = (x + hx <= 10.67) & (y - hx >= 8.99)
        if m1.sum() == 195:
            _add(bl.make_brick(10, 19, 5, x[m1], y[m1], th[m1], 793, "B5_10x19_n793"))
        m2 = (x - hx >= 10.99) & (y + hx <= 9.67)
        if m2.sum() == 157:
            _add(bl.make_brick(17, 9, 4, x[m2], y[m2], th[m2], 793, "B4_17x9_n793"))
        m3 = (x - hx >= 9.99) & (y + hx <= 9.67)
        if m3.sum() == 166:
            _add(bl.make_brick(18, 9, 4, x[m3], y[m3], th[m3], 793, "B4_18x9_n793"))

    # 5. Merge existing n850/w971/w910 bank
    for b in bl.load_b5_bank(packs, compress_b5_b4=False):
        _add(b)

    return bank


def find_h50_candidates(n: int, bank: list[dict], L_inc: float, max_cands: int = 5) -> list[dict]:
    """Find best (b1, b2) assemblies for n across all harvested compressed bricks."""
    m = int(math.isqrt(n))
    r = n - m * m
    f_inc = L_inc - m
    if r <= 1:
        return []

    good = [b for b in bank if min(b["fx"], b["fy"]) < f_inc + 0.025 and b["w"] < m and b["h"] < m]
    pair_scores = []
    seen_pairs = set()

    for b1 in good:
        for b2 in good:
            if b1["g"] + b2["g"] != r:
                continue
            kx = b1["w"] + b2["w"] - m
            ky = b1["h"] + b2["h"] - m
            # Keep zero-overlap (kx<=0, ky<=0), 1D-touch (kx<=1, ky<=0 or kx<=0, ky<=1),
            # and tight 2D interlock (kx>=0, ky>=0, kx+ky<=2), or narrow 1D gap when |kx| > ky
            if kx > 2 or ky > 2:
                continue
            if (kx > 0 and ky > 0 and kx + ky > 2):
                continue
            if (kx < 0 and ky > 0 and -kx <= ky) or (ky < 0 and kx > 0 and -ky <= kx):
                continue

            f_pair_tight = max(b1["fx"], b2["fy"])
            if f_pair_tight >= f_inc + 0.015:
                continue
            # Effective slack needed by the non-tight axes (only matters if adjacent to opposite brick)
            fx_need = b1["fx"] + b2["fx"] if kx == 0 else max(b1["fx"], b2["fx"])
            fy_need = b1["fy"] + b2["fy"] if ky == 0 else max(b1["fy"], b2["fy"])
            eff_x = fx_need if kx >= 0 else f_pair_tight
            eff_y = fy_need if ky >= 0 else f_pair_tight
            # For kx<0, ky<0, check total width w1+w2+fx1+fx2 <= m + f_pair_tight
            if kx < 0 and (b1["w"] + b2["w"] + b1["fx"] + b2["fx"] > m + f_pair_tight):
                eff_x = b1["w"] + b2["w"] + b1["fx"] + b2["fx"] - m
            if ky < 0 and (b1["h"] + b2["h"] + b1["fy"] + b2["fy"] > m + f_pair_tight):
                eff_y = b1["h"] + b2["h"] + b1["fy"] + b2["fy"] - m

            s1_tag = b1["tag"].replace("_T", "")
            s2_tag = b2["tag"].replace("_T", "")
            canon = ((s1_tag, b1["w"], b1["h"], b1["g"]), (s2_tag, b2["w"], b2["h"], b2["g"]))
            canon_t = ((s2_tag, b2["h"], b2["w"], b2["g"]), (s1_tag, b1["h"], b1["w"], b1["g"]))
            if canon in seen_pairs or canon_t in seen_pairs:
                continue
            seen_pairs.add(canon)
            seen_pairs.add(canon_t)

            overlap_pen = (
                max(0, kx) * 0.005
                + max(0, ky) * 0.005
                + max(0, kx) * max(0, ky) * 0.010
                + (0.012 if (kx < 0 and ky > 0) or (ky < 0 and kx > 0) else 0.0)
            )
            iso_pen = 0.03 * max(0.0, max(eff_x, eff_y) - f_pair_tight)
            b4_bonus = -0.0005 * (("b4" in b1["tag"]) + ("b4" in b2["tag"]) + ("n970" in b1["tag"]) + ("n970" in b2["tag"]))
            score = f_pair_tight + iso_pen + overlap_pen + b4_bonus
            pair_scores.append((score, b1, b2))

    pair_scores.sort(key=lambda z: z[0])

    cands = []
    for _, b1, b2 in pair_scores[: max_cands * 4]:
        f_full = max(b1["f"], b2["f"])
        f_tight = max(b1["fx"], b2["fy"])
        f_modes = ["tight", "full"] if (f_full - f_tight >= 0.008 and f_full <= f_inc + 0.020) else ["tight"]
        for fm in f_modes:
            best_for_fm = None
            kx = b1["w"] + b2["w"] - m
            ky = b1["h"] + b2["h"] - m
            for s1 in range(4):
                for s2 in range(4):
                    dfs = ["b1", "b2", "regrid"] if (kx > 0 and ky > 0) else ["b1"]
                    for df in dfs:
                        if df == "regrid":
                            sb1, sb2 = bl.sym_brick(b1, s1), bl.sym_brick(b2, s2)
                            asm = il.assemble_interlock(m, sb1, sb2, 0, 0, drop_from="regrid")
                        else:
                            asm = bl.assemble_b5_pair(m, r, b1, b2, s1, s2, f_mode=fm, drop_from=df)
                        if asm is None or len(asm[0]) != m * m + r:
                            continue
                        ax, ay, at, aL, sbox = asm
                        pen, _ = cpp_eval_sat_pen_bbox(ax, ay, at)
                        if best_for_fm is None or pen < best_for_fm[0]:
                            best_for_fm = (pen, ax, ay, at, aL, sbox, f"{fm}_s{s1}{s2}_{df}")
            if best_for_fm is not None:
                pen, ax, ay, at, aL, sbox, cfg = best_for_fm
                cands.append({
                    "n": n, "m": m, "r": r,
                    "b1": b1, "b2": b2,
                    "pen0": float(pen), "x": ax, "y": ay, "th": at, "L": aL, "sbox": sbox,
                    "tag": f"{b1['tag']}+{b2['tag']}_{cfg}",
                })
                if len(cands) >= max_cands * 2:
                    break
        if len(cands) >= max_cands * 2:
            break

    cands.sort(key=lambda c: (0 if (c["pen0"] < 0.10 and c["L"] <= L_inc + 0.005) else 1, c["L"] + 0.025 * c["pen0"]))
    return cands[:max_cands]


def build_h50_melt_seeds(
    n: int,
    xb: np.ndarray,
    yb: np.ndarray,
    tb: np.ndarray,
    Lb: float,
    seam_box: tuple[float, float, float, float],
    rng: np.random.Generator,
    L_inc: float,
    pen0: float,
    n_seeds: int = 24,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[float]], list[str]]:
    """Build C++ masked-melt seeds with realistic L_inc-aware annealing schedules."""
    L_expand = 0.075 if pen0 > 0.1 else 0.050
    L0 = max(Lb, L_inc) + L_expand
    sc = L0 / Lb
    x0, y0, t0 = xb * sc, yb * sc, tb.copy()
    sx0, sx1, sy0, sy1 = [v * sc for v in seam_box]
    scx, scy = 0.5 * (sx0 + sx1), 0.5 * (sy0 + sy1)

    L_mid = 0.5 * (Lb + L_inc)
    L_t_inc = L_inc - 0.003
    L_t_mid = max(Lb + 0.001, min(L_mid, L_inc - 0.005))
    L_t_lo = max(Lb + 0.0005, L_inc - 0.015) if pen0 < 0.1 else max(Lb + 0.003, L_t_mid - 0.004)
    scheds = [
        [L0, L_inc + 0.018, L_inc + 0.002, L_t_inc],
        [L0, 0.5 * (L0 + L_t_mid), L_t_mid + 0.004, L_t_mid],
        [L0, L_t_mid + 0.012, L_t_lo + 0.005, L_t_lo],
    ]

    centers = [
        (scx, scy, "seam_ctr"),
        (sx1, sy0, "b2_top_left"),
        (sx0, sy1, "b1_bot_right"),
        (0.5 * sx1, sy1, "b1_bot_edge"),
        (sx0, 0.5 * sy1, "b2_left_edge"),
        (0.5 * sx0, 0.5 * sy0, "bl_quad"),
        (sx0, sy0, "bl_corner"),
        (sx1, sy1, "tr_corner"),
    ]
    t_deg = np.degrees(np.mod(t0, 0.5 * math.pi))
    is_unit = np.minimum(t_deg, 90.0 - t_deg) < 2.0

    q_list, m_list, ls_list, labels = [], [], [], []
    for s_idx in range(n_seeds):
        x, y, th = x0.copy(), y0.copy(), t0.copy()
        cx, cy, cname = centers[s_idx % len(centers)]
        rad = float(rng.uniform(4.8, 9.5))
        near = ((x - cx) ** 2 + (y - cy) ** 2) <= rad * rad
        in_seam = (x >= sx0 - 1.2) & (x <= sx1 + 1.2) & (y >= sy0 - 1.2) & (y <= sy1 + 1.2)
        if s_idx % 3 == 0:
            mask = near | in_seam
        elif s_idx % 3 == 1:
            mask = (near & is_unit) | in_seam
        else:
            in_bl = (x <= sx1 + 0.5) & (y <= sy1 + 0.5)
            mask = (in_bl & is_unit) | in_seam | near
        nm = int(mask.sum())
        if nm > 0:
            x[mask] += rng.uniform(-0.18, 0.18, nm)
            y[mask] += rng.uniform(-0.18, 0.18, nm)
            th[mask] += rng.uniform(-0.28, 0.28, nm)
        q_list.append(full_to_free_sym(x, y, th, L0))
        m_list.append(np.concatenate([mask.astype(np.uint8)] * 3))
        ls_list.append(scheds[s_idx % len(scheds)])
        labels.append(f"h50_{cname}_{s_idx}")
    return q_list, m_list, ls_list, labels


def eval_target_h50(n: int, cands: list[dict], L_inc_dec: Decimal, n_threads: int = 4) -> dict:
    t0 = time.time()
    best_res = None
    for c_idx, cand in enumerate(cands):
        if best_res is not None and cand["L"] > float(L_inc_dec) + 0.003:
            print(f"  [cand {c_idx+1}/{len(cands)}] SKIP {cand['tag']} (L0={cand['L']:.6f} > cur={float(L_inc_dec):.6f})", flush=True)
            continue

        tc = time.time()
        ax, ay, at, aL, sbox, tag = cand["x"], cand["y"], cand["th"], cand["L"], cand["sbox"], cand["tag"]
        print(
            f"  [cand {c_idx+1}/{len(cands)}] {tag} (L0={aL:.6f}, inc={float(L_inc_dec):.6f}, pen0={cand['pen0']:.4f})",
            flush=True,
        )

        if cand["pen0"] <= 0.06:
            direct = bl.polish_and_certify(n, ax, ay, at, L_inc_dec, f"{tag}_direct", do_mechwalk=False)
            if direct.get("ok"):
                print(
                    f"    -> direct ok dL={direct['dL']:+.6e} L={direct['L_lit'][:14]} ({time.time()-tc:.1f}s)",
                    flush=True,
                )
                if direct["is_real_win"]:
                    bl.save_win_b5(n, direct, direct["label"])
                    L_inc_dec = min(L_inc_dec, Decimal(direct["L_lit"]))
                if best_res is None or Decimal(direct["L_lit"]) < Decimal(best_res["L_lit"]):
                    best_res = direct

        rng = np.random.default_rng(70000 + n * 19 + c_idx)
        q_list, m_list, ls_list, labels = build_h50_melt_seeds(
            n, ax, ay, at, aL, sbox, rng, L_inc=float(L_inc_dec), pen0=cand["pen0"], n_seeds=24
        )
        qb, qf, lf, _, po, _ = run_cpp_masked_melt(
            n, 1, q_list, m_list, ls_list, [5, 8, 12, 16],
            a_start=0.82, n_melt_stages=4, masked_iters=220, full_iters=220,
            n_threads=n_threads, base_seed=75000 + n * 29 + c_idx * 100,
        )
        n_feas = int(np.sum(lf < 1e9))
        min_po = float(np.min(po))
        order = sorted(
            range(len(labels)),
            key=lambda i: (0 if lf[i] < 1e9 else 1, float(lf[i]) if lf[i] < 1e9 else float(po[i])),
        )[:4]

        tasks = []
        for i in order:
            use_f = lf[i] < 1e9
            L_ref = float(lf[i]) if use_f else ls_list[i][-1]
            x, y, th = cpp_expand_sym(n, 1, L_ref, qf[i] if use_f else qb[i])
            tasks.append((n, x, y, th, str(L_inc_dec), f"{tag}_{labels[i]}", False))

        with mp.Pool(processes=min(4, len(tasks))) as pool:
            pol = pool.map(bl._polish_worker_mp, tasks)

        ok_list = [r for r in pol if r.get("ok")]
        fail_pens = [round(r.get("pen", 99.0), 4) for r in pol if not r.get("ok")]
        if ok_list:
            b = min(ok_list, key=lambda r: Decimal(r["L_lit"]))
            print(
                f"    -> feas={n_feas}/{len(labels)} min_po={min_po:.4e} best_dL={b['dL']:+.6e} L={b['L_lit'][:14]} ({time.time()-tc:.1f}s)",
                flush=True,
            )
            if b["is_real_win"]:
                bl.save_win_b5(n, b, b["label"])
                L_inc_dec = min(L_inc_dec, Decimal(b["L_lit"]))
            if best_res is None or Decimal(b["L_lit"]) < Decimal(best_res["L_lit"]):
                best_res = b
        else:
            print(
                f"    -> feas={n_feas}/{len(labels)} min_po={min_po:.4e} ALL FAILED SLP (pens={fail_pens}, {time.time()-tc:.1f}s)",
                flush=True,
            )

    # One final mechwalk kick on the single best packing for n if competitive
    if best_res is not None and best_res.get("rows") and Decimal(best_res["L_lit"]) <= L_inc_dec + Decimal("0.002"):
        rows = sorted(best_res["rows"], key=lambda r: int(r["square_id"]))
        mx = np.array([float(r["x"]) for r in rows], dtype=np.float64)
        my = np.array([float(r["y"]) for r in rows], dtype=np.float64)
        mt = np.radians(np.array([float(r["angle"]) for r in rows], dtype=np.float64))
        mw = bl.polish_and_certify(n, mx, my, mt, L_inc_dec, f"{best_res['label']}_mw", do_mechwalk=True)
        if mw.get("ok") and Decimal(mw["L_lit"]) < Decimal(best_res["L_lit"]):
            if mw["is_real_win"]:
                bl.save_win_b5(n, mw, mw["label"])
                L_inc_dec = min(L_inc_dec, Decimal(mw["L_lit"]))
            best_res = mw

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


def main() -> None:
    targets = [
        971, 850, 973, 911, 851, 738, 684, 794, 685, 686,
        912, 852, 795, 739, 740, 687, 634, 635,
    ]
    all_ns = sorted(set(targets + [970, 969, 910, 793]))
    packs, best_L_dec = bl.fast_load_frontier(target_ns=all_ns)

    # Also update best_L_dec if we already have win_n850.json locally
    w850 = bl.DATA_DIR / "win_n850.json"
    if w850.exists():
        d850 = json.loads(w850.read_text(encoding="utf-8"))
        best_L_dec[850] = min(best_L_dec[850], Decimal(d850["L_lit"]))

    bank = build_h50_bank(packs)
    print(f"H50/H51 bank loaded: {len(bank)} oriented bricks", flush=True)
    for b in bank:
        if ("b4" in b["tag"] or "n970" in b["tag"] or "n793" in b["tag"]) and not b["tag"].endswith("_T"):
            print(f"  brick {b['tag']}: {b['w']}x{b['h']} g={b['g']} fx={b['fx']:.6f} fy={b['fy']:.6f}", flush=True)

    ckpt_path = bl.DOCS_DIR / "h50_ckpt.json"
    ckpt = json.loads(ckpt_path.read_text()) if ckpt_path.exists() else {}
    wins = []
    t0 = time.time()

    for idx, n in enumerate(targets, 1):
        if str(n) in ckpt:
            if ckpt[str(n)].get("status") == "WIN":
                wins.append(n)
            print(f"\n[H50 {idx}/{len(targets)}] n={n} already in ckpt ({ckpt[str(n)].get('status')}), skipping", flush=True)
            continue
        m = int(math.isqrt(n))
        r = n - m * m
        L_inc_dec = best_L_dec[n]
        cands = find_h50_candidates(n, bank, float(L_inc_dec), max_cands=4)
        print(f"\n[H50 {idx}/{len(targets)}] n={n} (m={m}, r=+{r}), L_inc={L_inc_dec}, cands={len(cands)}", flush=True)
        if not cands:
            ckpt[str(n)] = {"n": n, "status": "NO_CAND"}
            ckpt_path.write_text(json.dumps(ckpt, indent=2))
            continue
        res = eval_target_h50(n, cands, L_inc_dec, n_threads=4)
        print(f"  [Result n={n}] -> {res}", flush=True)
        if res.get("status") == "WIN":
            wins.append(n)
        ckpt[str(n)] = res
        ckpt_path.write_text(json.dumps(ckpt, indent=2))

    print(f"\n=== H50/H51 Complete in {time.time()-t0:.1f}s | Wins ({len(wins)}): {wins} ===", flush=True)


if __name__ == "__main__":
    main()
