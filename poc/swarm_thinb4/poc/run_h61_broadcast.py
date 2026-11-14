#!/usr/bin/env python3
"""Hypothesis H61: Sub-brick extraction from newly ratcheted wins + SLP compression + zero-overlap/tight pinwheel broadcast."""
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
THIN_DIR = HERE.parent
REPO = THIN_DIR.parent.parent
sys.path.insert(0, str(REPO / "poc" / "swarm_b5xplant" / "poc"))
sys.path.insert(0, str(REPO / "poc" / "swarm_brickslp" / "poc"))
sys.path.insert(0, str(HERE))

import b5_lib as bl  # noqa: E402
import brick_slp  # noqa: E402
import frontier_io  # noqa: E402
import thinb4_lib as tl  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox  # noqa: E402

TARGETS = [909, 850, 971, 910, 972, 911, 852, 795, 687, 688, 796, 912, 632, 683, 849]

H61_CACHE = tl.DATA_DIR / "h61_bricks_cache.json"
H61_CKPT = tl.DOCS_DIR / "h61_ckpt.json"

SWARM_WORKTREES_ROOT = pathlib.Path("/home/lucas/.gemini/antigravity-cli/worktrees/a4fb0b93-b63b-426d-aa12-67454e99bc08")

SOURCE_SPECS = [
    (
        "mw970",
        SWARM_WORKTREES_ROOT / "swarm-mechwalk-8abd0e01/poc/swarm_mechwalk/data/b6s6_win_n970.json",
        [
            ("B4_11x16_mw970", "TL", 11, 16, 4),
            ("B5_20x11_mw970", "BR", 20, 11, 5),
        ],
    ),
    (
        "b8_909",
        SWARM_WORKTREES_ROOT / "swarm-b8xplant-7f6c04d2/poc/swarm_b8xplant/data/win_n909.json",
        [
            ("B4_10x16_b8_909", "TL", 10, 16, 4),
            ("B5_20x11_b8_909", "BR", 20, 11, 5),
        ],
    ),
    (
        "mw741",
        SWARM_WORKTREES_ROOT / "swarm-mechwalk-8abd0e01/poc/swarm_mechwalk/data/win_n741.json",
        [
            ("B6_9x18_mw741", "BR", 9, 18, 6),
            ("B6_18x9_mw741", "TL", 18, 9, 6),
        ],
    ),
    (
        "n971",
        SWARM_WORKTREES_ROOT / "swarm-b5xplant-d7159aed/poc/swarm_b5xplant/data/h53_win_n971.json",
        [
            ("B5_10x20_n971", "TL", 10, 20, 5),
            ("B5_11x20_n971", "TL", 11, 20, 5),
            ("B5_20x11_n971", "BR", 20, 11, 5),
        ],
    ),
    (
        "n970",
        SWARM_WORKTREES_ROOT / "swarm-b5xplant-d7159aed/poc/swarm_b5xplant/data/h53_win_n970.json",
        [
            ("B4_11x16_n970", "TL", 11, 16, 4),
            ("B5_20x11_n970", "BR", 20, 11, 5),
        ],
    ),
    (
        "n737",
        SWARM_WORKTREES_ROOT / "swarm-b5xplant-d7159aed/poc/swarm_b5xplant/data/h53_win_n737.json",
        [
            ("B4_11x16_n737", "TL", 11, 16, 4),
            ("B4_16x11_n737", "BR", 16, 11, 4),
            ("B4_10x16_n737_h53", "TL", 10, 16, 4),
        ],
    ),
    (
        "mw632",
        SWARM_WORKTREES_ROOT / "swarm-mechwalk-8abd0e01/poc/swarm_mechwalk/data/win_n632.json",
        [
            ("B3_14x9_mw632", "TL", 14, 9, 3),
            ("B4_10x16_mw632", "BR", 10, 16, 4),
        ],
    ),
]


def _load_rad_json(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = sorted(d["rows"], key=lambda r: int(r["square_id"]))
    x = np.array([float(r["x"]) for r in rows], dtype=np.float64)
    y = np.array([float(r["y"]) for r in rows], dtype=np.float64)
    th = np.array([float(r["angle"]) for r in rows], dtype=np.float64)
    L = float(d.get("L_float") or d["L_lit"])
    return x, y, th, L, len(rows)


def _extract_corner_cut_h61(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L: float,
    corner: str,
    w: int,
    h: int,
    g_exp: int,
    src_n: int,
    tag: str,
    margin: float = 0.75,
) -> dict | None:
    hx = frontier_io.hx_of(th)
    if corner == "TL":
        m = (x + hx <= w + margin) & (y - hx >= L - (h + margin))
    elif corner == "TR":
        m = (x - hx >= L - (w + margin)) & (y - hx >= L - (h + margin))
    elif corner == "BL":
        m = (x + hx <= w + margin) & (y + hx <= h + margin)
    else:
        m = (x - hx >= L - (w + margin)) & (y + hx <= h + margin)
    if int(m.sum()) != w * h + g_exp:
        return None
    return bl.make_brick(w, h, g_exp, x[m], y[m], th[m], src_n, tag)


def _ser_brick(b: dict) -> dict:
    return {
        "w": int(b["w"]),
        "h": int(b["h"]),
        "g": int(b["g"]),
        "src_n": int(b["src_n"]),
        "tag": str(b["tag"]),
        "x": b["x"].tolist(),
        "y": b["y"].tolist(),
        "th": b["th"].tolist(),
    }


def _deser_brick(item: dict) -> dict | None:
    return bl.make_brick(
        item["w"],
        item["h"],
        item["g"],
        np.array(item["x"], dtype=np.float64),
        np.array(item["y"], dtype=np.float64),
        np.array(item["th"], dtype=np.float64),
        item["src_n"],
        item["tag"],
    )


def _compress_brick_slp_worker(args: tuple) -> tuple[str, dict | None]:
    tag, b_ser, n_trials, seed = args
    b = _deser_brick(b_ser)
    w, h, g = b["w"], b["h"], b["g"]
    x0, y0, t0 = b["x"], b["y"], b["th"]
    W0, H0 = w + b["fx"], h + b["fy"]
    f0 = b["f"]
    fmin = min(b["fx"], b["fy"])
    rng = np.random.default_rng(seed)
    best = b
    tilt = np.minimum(np.mod(t0, 0.5 * math.pi), 0.5 * math.pi - np.mod(t0, 0.5 * math.pi)) > math.radians(2.0)

    if f0 > 0.67:
        ft_sched = (f0 - 0.004, max(0.665, f0 - 0.025), max(0.650, f0 - 0.045))
    else:
        ft_sched = (f0, max(0.635, fmin + 0.002), max(0.634, fmin - 0.004))

    for trial in range(n_trials):
        cx, cy, ct = x0.copy(), y0.copy(), t0.copy()
        if trial > 0:
            if trial % 3 == 1:
                mask = ~tilt
            elif trial % 3 == 2:
                mask = (cy < 2.2) | (cy > H0 - 2.2) | (cx < 2.2) | (cx > W0 - 2.2)
            else:
                mask = np.ones(len(cx), dtype=bool)
            nm = int(mask.sum())
            if nm > 0:
                amp = 0.055 if trial % 3 != 0 else 0.025
                cx[mask] += rng.uniform(-amp, amp, nm)
                cy[mask] += rng.uniform(-amp, amp, nm)
                ct[mask] += rng.uniform(-1.5 * amp, 1.5 * amp, nm)
            for ft in ft_sched:
                cx, cy, ct, _ = brick_slp.relax_rect_lbfgs(cx, cy, ct, w + ft, h + ft, maxiter=240)
        sx, sy, st, _, ok = brick_slp.joint_xytheta_slp_rect(
            cx, cy, ct, w, h, rounds=14, tr_th0=0.022, tr_xy0=0.09, tol=1e-11
        )
        if ok:
            cb = bl.make_brick(w, h, g, sx, sy, st, b["src_n"], b["tag"] + "_slp")
            if cb is not None and cb["f"] < best["f"] - 1e-8:
                best = cb
    return tag, _ser_brick(best)


def build_h61_banks(packs: dict) -> tuple[dict[str, dict], dict[str, dict]]:
    """Return (h61_new_bank, full_bank) with cached SLP compression."""
    cache = {}
    if H61_CACHE.exists():
        cache = json.loads(H61_CACHE.read_text(encoding="utf-8"))

    raw_new: list[dict] = []
    for src_label, jpath, specs in SOURCE_SPECS:
        if not jpath.exists():
            print(f"[H61 extract] missing {jpath}, skipping {src_label}", flush=True)
            continue
        x, y, th, L, n = _load_rad_json(jpath)
        for tag, corner, w, h, g in specs:
            b = _extract_corner_cut_h61(x, y, th, L, corner, w, h, g, n, tag)
            if b is None:
                print(f"[H61 extract] FAILED {tag} ({corner} {w}x{h} g=+{g} from {src_label})", flush=True)
                continue
            print(
                f"[H61 extract] {tag}: {w}x{h} g=+{g} fx={b['fx']:.6f} fy={b['fy']:.6f} f={b['f']:.6f}",
                flush=True,
            )
            raw_new.append(b)

    to_compress = []
    h61_bricks: dict[str, dict] = {}
    for idx, b in enumerate(raw_new):
        tag = b["tag"]
        h61_bricks[tag] = b
        slp_tag = f"{tag}_slp"
        if slp_tag in cache:
            cb = _deser_brick(cache[slp_tag])
            if cb is not None and cb["f"] < b["f"] - 1e-8:
                h61_bricks[slp_tag] = cb
                print(f"[H61 cache] {slp_tag}: f={cb['f']:.6f} (df={cb['f'] - b['f']:+.3e})", flush=True)
        else:
            to_compress.append((slp_tag, _ser_brick(b), 12, 61000 + idx * 37))

    if to_compress:
        print(f"[H61 compress] Running standalone SLP compression on {len(to_compress)} bricks...", flush=True)
        with mp.Pool(processes=min(8, len(to_compress))) as pool:
            res = pool.map(_compress_brick_slp_worker, to_compress)
        for slp_tag, cb_ser in res:
            if cb_ser is None:
                continue
            cb = _deser_brick(cb_ser)
            orig_tag = slp_tag[:-4]
            f0 = h61_bricks[orig_tag]["f"]
            cache[slp_tag] = cb_ser
            if cb["f"] < f0 - 1e-8:
                h61_bricks[slp_tag] = cb
                print(
                    f"[H61 compress] {slp_tag}: {f0:.6f} -> {cb['f']:.6f} (df={cb['f'] - f0:+.3e})",
                    flush=True,
                )
            else:
                print(f"[H61 compress] {orig_tag}: already minimal at f={f0:.6f}", flush=True)
        H61_CACHE.write_text(json.dumps(cache), encoding="utf-8")

    # Add transposes for h61 bricks
    h61_with_t: dict[str, dict] = {}
    for k, b in h61_bricks.items():
        h61_with_t[k] = b
        if b["w"] != b["h"]:
            bt = bl.transpose_brick(b)
            h61_with_t[bt["tag"]] = bt

    # Base library from thinb4 compressed cache (fast load via compress=True when cache exists)
    base_bank = tl.build_h52_bank(packs, compress=True)
    for k, b in h61_with_t.items():
        base_bank[k] = b
    return h61_with_t, base_bank


def find_h61_candidates(
    n: int,
    h61_keys: set[str],
    bank: dict[str, dict],
    L_inc: float,
    max_cands: int = 6,
) -> list[dict]:
    """Generate zero-overlap (z) and tight (t) pinwheel candidates involving >=1 H61 brick."""
    m = int(math.floor(math.sqrt(n)))
    r = n - m * m
    all_keys = sorted(bank.keys())
    best_by_pair: dict[tuple[str, str, str], dict] = {}

    for k1 in all_keys:
        b1 = bank[k1]
        for k2 in all_keys:
            if k1 not in h61_keys and k2 not in h61_keys:
                continue
            b2 = bank[k2]
            if b1["g"] + b2["g"] != r:
                continue

            # 1) Zero-overlap (z) pre-shifted across sf in (0.0, 0.5, 0.8)
            if b1["w"] + b2["w"] <= m and b1["h"] + b2["h"] <= m and (b1["w"] + b2["w"] < m or b1["h"] + b2["h"] < m):
                for sf in (0.0, 0.5, 0.8):
                    asms = tl.assemble_pair_best(m, r, b1, b2, shift_fracs=(sf,))
                    for cand in asms:
                        key = (k1, k2, f"z_sf{sf:.1f}")
                        prev = best_by_pair.get(key)
                        if prev is None or (cand["pen0"], cand["L"]) < (prev["pen0"], prev["L"]):
                            cand["tag"] = f"{k1}+{k2}_z_sf{sf:.1f}"
                            cand["kind"] = "z"
                            best_by_pair[key] = cand

            # 2) Tight (t) 1D or shallow 2D overlap (max(kx,ky)<=2, kx+ky<=2)
            kx = b1["w"] + b2["w"] - m
            ky = b1["h"] + b2["h"] - m
            if max(kx, ky) <= 2 and kx + ky <= 2 and min(kx, ky) <= 1 and (kx > 0 or ky > 0 or (kx == 0 and ky == 0)):
                best_t = None
                for f_mode in ("tight", "mid"):
                    for s1 in (0, 2, 4, 6):
                        for s2 in (0, 2, 4, 6):
                            out = bl.assemble_b5_pair(m, r, b1, b2, s1, s2, f_mode=f_mode)
                            if out is None:
                                continue
                            ax, ay, at, aL, sbox = out
                            if aL > L_inc + 0.015:
                                continue
                            pen0, _ = cpp_eval_sat_pen_bbox(ax, ay, at)
                            if pen0 > 2.0:
                                continue
                            cand = {
                                "x": ax, "y": ay, "th": at, "L": aL, "sbox": sbox,
                                "pen0": float(pen0), "tag": f"{k1}+{k2}_t_{f_mode}_s{s1}{s2}",
                                "kind": "t",
                            }
                            if best_t is None or (cand["pen0"] + max(0.0, cand["L"] - L_inc) * 5.0) < (
                                best_t["pen0"] + max(0.0, best_t["L"] - L_inc) * 5.0
                            ):
                                best_t = cand
                if best_t is not None:
                    best_by_pair[(k1, k2, "t")] = best_t

    cands = list(best_by_pair.values())

    def _score(c: dict) -> tuple:
        immediate = 0 if (c["pen0"] <= 1e-4 and c["L"] < L_inc - 1e-8) else 1
        dL_pos = max(0.0, c["L"] - L_inc)
        return (immediate, 0 if c["pen0"] <= 1e-4 else 1, c["pen0"] + 15.0 * dL_pos, c["L"])

    cands.sort(key=_score)
    # Deduplicate by (base pair family, kind) to keep diversity
    picked: list[dict] = []
    seen_fam: dict[tuple[str, str], int] = {}
    for c in cands:
        parts = c["tag"].split("+")
        p1 = parts[0].replace("_slp", "").replace("_T", "")
        p2 = parts[1].split("_")[0]
        fam = (p1, p2)
        cnt = seen_fam.get(fam, 0)
        if cnt >= 2:
            continue
        seen_fam[fam] = cnt + 1
        picked.append(c)
        if len(picked) >= max_cands:
            break
    return picked


def save_h61_win(n: int, cert: dict, label: str) -> bool:
    wp = tl.DATA_DIR / f"h61_win_n{n}.json"
    if wp.exists():
        old = json.loads(wp.read_text(encoding="utf-8"))
        if max(Decimal(old["L_lit"]), Decimal(old["L_pd"])) <= max(Decimal(cert["L_lit"]), Decimal(cert["L_pd"])):
            return False
    payload = {
        "n": int(n),
        "label": label,
        "L_lit": cert["L_lit"],
        "L_pd": cert["L_pd"],
        "L_float": float(cert["L_lit"]),
        "dL": float(cert["dL"]),
        "rows": cert["rows"],
    }
    wp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return True


def main() -> None:
    t0 = time.time()
    all_seed_ns = [684, 737, 848, 849, 908, 969, 632, 736, 793, 850, 909, 972, 631]
    packs, inc_decs = tl.fast_load_frontier(target_ns=sorted(set(TARGETS + all_seed_ns)))
    # Also check poc/contact_mechanism_walk/final.csv on origin/main so inc_decs is exact
    cmw_csv = REPO / "poc" / "contact_mechanism_walk" / "final.csv"
    if cmw_csv.exists():
        import union as auto_union
        cmw_groups = auto_union.parse_text(cmw_csv.read_text("utf-8"))
        for tn in TARGETS:
            if tn in cmw_groups:
                Ld = Decimal(auto_union.certify(tn, cmw_groups[tn]))
                if tn not in inc_decs or Ld < inc_decs[tn]:
                    inc_decs[tn] = Ld
    h61_with_t, full_bank = build_h61_banks(packs)
    h61_keys = set(h61_with_t.keys())
    print(f"[H61] H61 bricks (+T): {len(h61_keys)}, full bank: {len(full_bank)}", flush=True)

    # Monkey-patch save_win_thinb4 so eval_target writes h61_win_n*.json cleanly without clobbering PR #47 win_n*.json
    orig_save = tl.save_win_thinb4
    tl.save_win_thinb4 = save_h61_win
    try:
        ckpt: dict = {}
        if H61_CKPT.exists():
            ckpt = json.loads(H61_CKPT.read_text(encoding="utf-8"))

        for idx, n in enumerate(TARGETS, 1):
            sn = str(n)
            if sn in ckpt and ckpt[sn].get("status") in ("WIN", "OK"):
                print(f"[H61 {idx}/{len(TARGETS)}] n={n} already in ckpt ({ckpt[sn]['status']}), skip", flush=True)
                continue
            m = int(math.floor(math.sqrt(n)))
            r = n - m * m
            L_inc_dec = inc_decs[n]
            L_inc = float(L_inc_dec)
            print(f"\n[H61 {idx}/{len(TARGETS)}] n={n} (m={m}, r=+{r}), L_inc={L_inc:.9f}", flush=True)
            cands = find_h61_candidates(n, h61_keys, full_bank, L_inc, max_cands=4)
            if not cands:
                print(f"  [n={n}] 0 candidates matched", flush=True)
                ckpt[sn] = {"n": n, "status": "NO_CAND"}
            else:
                res = tl.eval_target(n, cands, L_inc_dec, n_threads=8, n_seeds=16, do_mechwalk_on_win=False)
                print(f"  [Result n={n}] -> {res}", flush=True)
                ckpt[sn] = res
            H61_CKPT.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")

        wins = [k for k, v in ckpt.items() if v.get("status") == "WIN"]
        print(f"\n[H61 DONE in {time.time() - t0:.1f}s] wins={len(wins)}: {wins}", flush=True)
    finally:
        tl.save_win_thinb4 = orig_save


if __name__ == "__main__":
    main()
