"""H25 + H28: B8_10x18_NEW sub-brick transplant and localized-brick contact-Jacobian SVD walk."""
from __future__ import annotations

import fcntl
import io
import json
import math
import os
import subprocess
import sys
import warnings
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds

REPO_ROOT = Path(__file__).resolve().parents[3]
AUTO_UNION_POC = REPO_ROOT / "poc" / "auto_union" / "poc"
HIGHD_POC = REPO_ROOT / "poc" / "swarm_highd" / "poc"
NEW159_POC = REPO_ROOT / "poc" / "swarm_new159" / "poc"
CASCADES_POC = REPO_ROOT / "poc" / "swarm_cascades" / "poc"
GAD_POC = REPO_ROOT / "poc" / "rectangle_gadgets" / "poc"
BASELINE_SRC = REPO_ROOT / "poc" / "baseline" / "poc" / "src"

for _p in (AUTO_UNION_POC, HIGHD_POC, NEW159_POC, CASCADES_POC, GAD_POC, BASELINE_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gadlib  # noqa: E402
import slp_utils as su  # noqa: E402
import union as auto_union  # noqa: E402
import utils as hu  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, full_to_free_sym, run_cpp_masked_melt  # noqa: E402
from joint_slp import joint_xytheta_slp  # noqa: E402

B8_DIR = REPO_ROOT / "poc" / "swarm_b8xplant"
DATA_DIR = B8_DIR / "data"
DOCS_DIR = B8_DIR / "docs"
FINAL_CSV = B8_DIR / "final.csv"
MERGED_CSV = REPO_ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
B8_JSON = DATA_DIR / "B8_10x18_NEW.json"

MECHWALK_FINAL = REPO_ROOT / "poc" / "swarm_mechwalk" / "final.csv"


def half_extent(theta: np.ndarray) -> np.ndarray:
    return 0.5 * (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))


def half_extent_derivative(theta: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return 0.5 * (-np.sign(c) * s + np.sign(s) * c)


def normalize(
    x: np.ndarray, y: np.ndarray, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64).copy()
    y = np.asarray(y, dtype=np.float64).copy()
    theta = np.asarray(theta, dtype=np.float64).copy()
    h = half_extent(theta)
    x -= float(np.min(x - h))
    y -= float(np.min(y - h))
    side = float(max(np.max(x + h), np.max(y + h)))
    return x, y, theta, side


def rect_bbox(
    x: np.ndarray, y: np.ndarray, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    h = half_extent(theta)
    nx = np.asarray(x, dtype=np.float64) - float(np.min(x - h))
    ny = np.asarray(y, dtype=np.float64) - float(np.min(y - h))
    W = float(np.max(nx + h))
    H = float(np.max(ny + h))
    return nx, ny, W, H


def ensure_b8_brick_json() -> dict:
    """Load or extract B8_10x18_NEW (188 squares from mechwalk n=639) into data/B8_10x18_NEW.json."""
    if B8_JSON.exists():
        d = json.loads(B8_JSON.read_text(encoding="utf-8"))
        return {
            "w": int(d["w"]),
            "h": int(d["h"]),
            "g": int(d["g"]),
            "x": np.array(d["x"], dtype=np.float64),
            "y": np.array(d["y"], dtype=np.float64),
            "th": np.array(d["th"], dtype=np.float64),
            "fx": float(d["fx"]),
            "fy": float(d["fy"]),
            "f": float(d["f"]),
        }
    mw = auto_union.parse_text(MECHWALK_FINAL.read_text(encoding="utf-8"))
    rows = sorted(mw[639], key=lambda r: int(r["square_id"]))
    x = np.array([float(r["x"]) for r in rows], dtype=np.float64)
    y = np.array([float(r["y"]) for r in rows], dtype=np.float64)
    th = np.array([float(r["angle"]) for r in rows], dtype=np.float64)
    x, y, th, _ = normalize(x, y, th)
    mask = (x >= 14.99) & (y <= 18.78)
    assert int(mask.sum()) == 188, f"Expected 188 squares in B8_10x18_NEW, got {int(mask.sum())}"
    bx, by, W, H = rect_bbox(x[mask], y[mask], th[mask])
    bt = th[mask].copy()
    fx, fy = W - 10.0, H - 18.0
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "w": 10,
        "h": 18,
        "g": 8,
        "fx": fx,
        "fy": fy,
        "f": max(fx, fy),
        "x": bx.tolist(),
        "y": by.tolist(),
        "th": bt.tolist(),
    }
    B8_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "w": 10,
        "h": 18,
        "g": 8,
        "x": bx,
        "y": by,
        "th": bt,
        "fx": fx,
        "fy": fy,
        "f": max(fx, fy),
    }


def transform_brick(brick: dict, mode: str) -> dict:
    """Return brick transformed by one of {'id', 'T', 'flipx', 'flipy', 'rot90', 'rot180', 'rot270'}."""
    x = brick["x"].copy()
    y = brick["y"].copy()
    th = brick["th"].copy()
    w, h = brick["w"], brick["h"]
    _, _, W, H = rect_bbox(x, y, th)
    if mode == "id":
        nx, ny, nt, nw, nh = x, y, th, w, h
    elif mode == "T":
        nx, ny, nt, nw, nh = y, x, 0.5 * math.pi - th, h, w
    elif mode == "flipx":
        nx, ny, nt, nw, nh = W - x, y, -th, w, h
    elif mode == "flipy":
        nx, ny, nt, nw, nh = x, H - y, -th, w, h
    elif mode == "rot180":
        nx, ny, nt, nw, nh = W - x, H - y, th, w, h
    elif mode == "rot90":
        nx, ny, nt, nw, nh = H - y, x, th + 0.5 * math.pi, h, w
    elif mode == "rot270":
        nx, ny, nt, nw, nh = y, W - x, th - 0.5 * math.pi, h, w
    else:
        raise ValueError(f"Unknown mode {mode}")
    nx, ny, W2, H2 = rect_bbox(nx, ny, nt)
    return {
        "w": nw,
        "h": nh,
        "g": brick["g"],
        "x": nx,
        "y": ny,
        "th": nt,
        "fx": W2 - nw,
        "fy": H2 - nh,
        "f": max(W2 - nw, H2 - nh),
    }


def load_true_frontier_targets(target_ns: list[int] | set[int]) -> dict[int, dict]:
    """Load TRUE frontier (merged.csv + origin/main poc/*/final.csv + PR #30 + mechwalk + brickslp) for target_ns."""
    need = {int(n) for n in target_ns}
    best: dict[int, dict] = {}

    df = pd.read_csv(MERGED_CSV, usecols=["n", "square_id", "x", "y", "angle"])
    sub_df = df[df["n"].isin(need)]
    for n_val, grp in sub_df.groupby("n"):
        n = int(n_val)
        grp_s = grp.sort_values("square_id")
        rows = [
            {
                "square_id": int(r.square_id),
                "x": repr(float(r.x)),
                "y": repr(float(r.y)),
                "angle": repr(float(r.angle)),
            }
            for r in grp_s.itertuples(index=False)
        ]
        L_dec = Decimal(auto_union.certify(n, rows))
        x = grp_s["x"].to_numpy(dtype=np.float64, copy=True)
        y = grp_s["y"].to_numpy(dtype=np.float64, copy=True)
        th = grp_s["angle"].to_numpy(dtype=np.float64, copy=True)
        x, y, th, L_f = normalize(x, y, th)
        best[n] = {"x": x, "y": y, "th": th, "L": L_f, "L_dec": L_dec, "source": "merged.csv"}

    # Check all poc/*/final.csv in the repo (includes contact_mechanism_walk, swarm_cleanpeel, swarm_mechwalk, etc.)
    extra_paths = list(sorted((REPO_ROOT / "poc").glob("*/final.csv")))

    def _consider_groups(groups_lit: dict, groups_pd: dict | None, label: str) -> None:
        for n in need:
            if n not in groups_lit or len(groups_lit[n]) != n:
                continue
            rows_lit = sorted(groups_lit[n], key=lambda r: int(r["square_id"]))
            if groups_pd is not None and n in groups_pd:
                res = auto_union.check((n, rows_lit, groups_pd[n]))
                if res is None:
                    continue
                L_dec = max(Decimal(res[0]), Decimal(res[1]))
            else:
                L_dec = Decimal(auto_union.certify(n, rows_lit))
            if n not in best or L_dec < best[n]["L_dec"]:
                x = np.array([float(r["x"]) for r in rows_lit], dtype=np.float64)
                y = np.array([float(r["y"]) for r in rows_lit], dtype=np.float64)
                th = np.array([float(r["angle"]) for r in rows_lit], dtype=np.float64)
                x, y, th, L_f = normalize(x, y, th)
                best[n] = {"x": x, "y": y, "th": th, "L": L_f, "L_dec": L_dec, "source": label}

    for p in extra_paths:
        if p == FINAL_CSV or not p.exists() or p.stat().st_size <= 20:
            continue
        try:
            lit, pd_t = auto_union.load(p)
        except Exception:
            continue
        _consider_groups(lit, pd_t, str(p))

    # Also include PR #30 (origin/swarm-cleanpeel-pr)
    try:
        pr30_text = subprocess.check_output(
            ["git", "show", "origin/swarm-cleanpeel-pr:poc/swarm_cleanpeel/final.csv"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8")
        pr30_lit = auto_union.parse_text(pr30_text)
        pr30_pd = auto_union.parse_text(pd.read_csv(io.StringIO(pr30_text)).to_csv(index=False))
        _consider_groups(pr30_lit, pr30_pd, "PR30")
    except Exception:
        pass

    return best


def certify_canonical(n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray) -> dict:
    """Certify candidate via canonical final_text in BOTH literal and pandas transport modes."""
    hx = half_extent(th)
    sx = x - float(np.min(x - hx)) + 1e-12
    sy = y - float(np.min(y - hx)) + 1e-12
    rows = [
        {
            "square_id": i,
            "x": repr(float(sx[i])),
            "y": repr(float(sy[i])),
            "angle": repr(float(th[i])),
        }
        for i in range(n)
    ]
    text = auto_union.final_text({n: rows})
    lit = auto_union.parse_text(text)
    pd_t = auto_union.parse_text(pd.read_csv(io.StringIO(text)).to_csv(index=False))
    res = auto_union.check((n, lit[n], pd_t[n]))
    if res is None:
        return {"ok": False, "L_lit": None, "L_pandas": None}
    L_lit, L_pd = res[0], res[1]
    L_worst = max(Decimal(L_lit), Decimal(L_pd))
    return {
        "ok": True,
        "L_lit": L_lit,
        "L_pandas": L_pd,
        "L_dec_str": str(L_worst),
        "L_float": float(L_worst),
        "rows": rows,
    }


def polish_and_certify(
    n: int,
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    inc_dec: Decimal,
    slp_rounds: int = 16,
) -> dict | None:
    """Run joint_xytheta_slp + multi-margin fixed_angle_lp_tighten + dual certification."""
    x, y, th, _ = normalize(x, y, th)
    sx, sy, st, _, ok = joint_xytheta_slp(
        x, y, th, rounds=slp_rounds, tr_th0=0.025, tr_xy0=0.10, margin=1e-8, tol=1e-10, patience=3
    )
    if not ok:
        pen0, _ = cpp_eval_sat_pen_bbox(x, y, th)
        if pen0 > 1e-10:
            return None
        sx, sy, st = x, y, th

    cands = []
    for margin in (1e-7, 2e-8, 5e-9):
        fx, fy, _, fixed_ok = hu.fixed_angle_lp_tighten(sx, sy, st, rounds=6, margin=margin)
        if fixed_ok:
            cert = certify_canonical(n, fx, fy, st)
            if cert["ok"]:
                cands.append(cert)
    cert_raw = certify_canonical(n, sx, sy, st)
    if cert_raw["ok"]:
        cands.append(cert_raw)
    if not cands:
        # Micro-expansion fallback if floating rounding leaves 1e-15 overlap
        for eps in (1e-10, 1e-9):
            cert_sc = certify_canonical(n, sx * (1.0 + eps), sy * (1.0 + eps), st)
            if cert_sc["ok"]:
                cands.append(cert_sc)
                break
    if not cands:
        return None
    best = min(cands, key=lambda c: max(Decimal(c["L_lit"]), Decimal(c["L_pandas"])))
    L_worst = max(Decimal(best["L_lit"]), Decimal(best["L_pandas"]))
    dL = L_worst - inc_dec
    best["n"] = int(n)
    best["dL"] = float(dL)
    best["is_real_win"] = bool(dL <= Decimal("-1e-9"))
    return best


def write_final_csv(wins_by_n: dict[int, dict], out_path: Path = FINAL_CSV) -> None:
    groups = {}
    for n, win in sorted(wins_by_n.items()):
        groups[int(n)] = [
            {
                "square_id": int(r["square_id"]),
                "x": str(r["x"]),
                "y": str(r["y"]),
                "angle": str(r["angle"]),
            }
            for r in win["rows"]
        ]
    text = auto_union.final_text(groups)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".csv.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, out_path)
    lit, pd_t = auto_union.load(out_path)
    for n in groups:
        if auto_union.check((n, lit[n], pd_t[n])) is None:
            raise ValueError(f"Exported final.csv failed dual certification for n={n}")


def save_win_locked(win_entry: dict, data_dir: Path = DATA_DIR, final_path: Path = FINAL_CSV) -> bool:
    if not win_entry.get("is_real_win"):
        return False
    data_dir.mkdir(parents=True, exist_ok=True)
    lock_path = data_dir / ".final.lock"
    with lock_path.open("w", encoding="utf-8") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        n = int(win_entry["n"])
        wp = data_dir / f"win_n{n}.json"
        new_dec = max(Decimal(win_entry["L_lit"]), Decimal(win_entry["L_pandas"]))
        if wp.exists():
            prev = json.loads(wp.read_text(encoding="utf-8"))
            prev_dec = max(Decimal(prev["L_lit"]), Decimal(prev["L_pandas"]))
            if new_dec >= prev_dec:
                return False
        wp.write_text(json.dumps(win_entry, indent=2), encoding="utf-8")
        wins_by_n: dict[int, dict] = {}
        for p in sorted(data_dir.glob("win_n*.json")):
            w = json.loads(p.read_text(encoding="utf-8"))
            wins_by_n[int(w["n"])] = w
        write_final_csv(wins_by_n, final_path)
    return True


def localized_contact_jacobian(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    free_mask: np.ndarray,
    contact_tol: float = 1e-4,
    include_walls: bool = True,
) -> tuple[csr_matrix, np.ndarray, dict]:
    """Build exact sparse contact Jacobian restricted to DOFs of `free_mask` squares.

    All constraints involving at least one square in `free_mask` are retained,
    freezing the exterior squares (`~free_mask`) to zero displacement.
    """
    n = len(x)
    free_idx = np.flatnonzero(free_mask)
    nf = len(free_idx)
    if nf == 0:
        return csr_matrix((0, 0), dtype=np.float64), free_idx, {"constraints": 0}

    local_pos = np.full(n, -1, dtype=np.int32)
    local_pos[free_idx] = np.arange(nf, dtype=np.int32)

    c = np.cos(theta)
    s = np.sin(theta)
    h = 0.5 * (np.abs(c) + np.abs(s))
    dh = 0.5 * (-np.sign(c) * s + np.sign(s) * c)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    row_offset = 0

    dx_mat = x[None, :] - x[:, None]
    dy_mat = y[None, :] - y[:, None]
    upper = np.arange(n)[:, None] < np.arange(n)[None, :]
    in_free_pair = free_mask[:, None] | free_mask[None, :]
    ni, nj = np.where((np.abs(dx_mat) <= 1.55) & (np.abs(dy_mat) <= 1.55) & upper & in_free_pair)

    pair_contacts = 0
    if len(ni) > 0:
        dx = dx_mat[ni, nj]
        dy = dy_mat[ni, nj]
        ci, si = c[ni], s[ni]
        cj, sj = c[nj], s[nj]
        ax_all = np.stack([ci, -si, cj, -sj], axis=1)
        ay_all = np.stack([si, ci, sj, cj], axis=1)
        proj_all = dx[:, None] * ax_all + dy[:, None] * ay_all
        cth = ci * cj + si * sj
        sth = ci * sj - si * cj
        radius = 0.5 * (1.0 + np.abs(cth) + np.abs(sth))
        gaps_all = np.abs(proj_all) - radius[:, None]
        best_k = np.argmax(gaps_all, axis=1)
        idx_m = np.arange(len(ni))
        gap = gaps_all[idx_m, best_k]
        active = gap <= contact_tol
        if np.any(active):
            ai, aj = ni[active], nj[active]
            adx, ady = dx[active], dy[active]
            ak = best_k[active]
            aax, aay = ax_all[active, ak], ay_all[active, ak]
            aproj = proj_all[active, ak]
            acth, asth = cth[active], sth[active]
            m = len(ai)
            pair_contacts = m

            sgn = np.where(aproj >= 0.0, 1.0, -1.0)
            d_rad_dphi = 0.5 * (-np.sign(acth) * asth + np.sign(asth) * acth)
            proj_rot = np.where(np.abs(aproj) > 1e-15, sgn * (-adx * aay + ady * aax), 0.0)
            i_owns = ak < 2
            d_thi = np.where(i_owns, proj_rot, 0.0) + d_rad_dphi
            d_thj = np.where(~i_owns, proj_rot, 0.0) - d_rad_dphi

            r_ids = np.arange(row_offset, row_offset + m, dtype=np.int32)
            row_offset += m

            li = local_pos[ai]
            mi = li >= 0
            if np.any(mi):
                rows.extend([r_ids[mi], r_ids[mi], r_ids[mi]])
                cols.extend([li[mi], nf + li[mi], 2 * nf + li[mi]])
                vals.extend([-sgn[mi] * aax[mi], -sgn[mi] * aay[mi], d_thi[mi]])

            lj = local_pos[aj]
            mj = lj >= 0
            if np.any(mj):
                rows.extend([r_ids[mj], r_ids[mj], r_ids[mj]])
                cols.extend([lj[mj], nf + lj[mj], 2 * nf + lj[mj]])
                vals.extend([sgn[mj] * aax[mj], sgn[mj] * aay[mj], d_thj[mj]])

    wall_contacts = 0
    if include_walls:
        side = float(max(np.max(x + h), np.max(y + h)))
        min_x = float(np.min(x - h))
        min_y = float(np.min(y - h))
        for axis, coord, origin in ((0, x, min_x), (1, y, min_y)):
            wgap = (coord - origin) - h
            widx = np.flatnonzero((wgap <= contact_tol) & free_mask)
            if len(widx) > 0:
                mw = len(widx)
                lw = local_pos[widx]
                r_ids = np.arange(row_offset, row_offset + mw, dtype=np.int32)
                row_offset += mw
                wall_contacts += mw
                rows.extend([r_ids, r_ids])
                cols.extend([axis * nf + lw, 2 * nf + lw])
                vals.extend([np.ones(mw, dtype=np.float64), -dh[widx]])

        for axis, coord in ((0, x), (1, y)):
            wgap = side - coord - h
            widx = np.flatnonzero((wgap <= contact_tol) & free_mask)
            if len(widx) > 0:
                mw = len(widx)
                lw = local_pos[widx]
                r_ids = np.arange(row_offset, row_offset + mw, dtype=np.int32)
                row_offset += mw
                wall_contacts += mw
                rows.extend([r_ids, r_ids])
                cols.extend([axis * nf + lw, 2 * nf + lw])
                vals.extend([-np.ones(mw, dtype=np.float64), -dh[widx]])

    if row_offset == 0:
        return csr_matrix((0, 3 * nf), dtype=np.float64), free_idx, {"constraints": 0, "n_free": nf}

    jac = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(row_offset, 3 * nf),
    ).tocsr()
    return jac, free_idx, {
        "constraints": int(row_offset),
        "pair_contacts": int(pair_contacts),
        "wall_contacts": int(wall_contacts),
        "n_free": int(nf),
    }


def localized_soft_modes(jac: csr_matrix, k: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Extract smallest right singular vectors of localized contact Jacobian."""
    if jac.shape[0] == 0 or jac.shape[1] == 0:
        return np.array([], dtype=np.float64), np.zeros((0, jac.shape[1]), dtype=np.float64)
    if jac.shape[1] <= 840 and jac.shape[0] <= 2400:
        # Exact dense SVD for n_free <= 280 avoids LOBPCG convergence misses
        _, singular, vt = np.linalg.svd(jac.toarray(), full_matrices=True)
        order = np.argsort(singular)
        k_eff = min(k, len(order))
        return singular[order[:k_eff]], vt[order[:k_eff]]
    k_eff = max(1, min(k, min(jac.shape) - 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, singular, vt = svds(jac, k=k_eff, which="SM", solver="lobpcg", maxiter=5000)
    order = np.argsort(singular)
    return singular[order], vt[order]


def find_tilted_brick_masks(
    x: np.ndarray, y: np.ndarray, th: np.ndarray, collar: float = 2.15
) -> list[tuple[str, np.ndarray]]:
    """Identify each tilted cluster + 2-layer unit-square collar as a localized free_mask."""
    hx = half_extent(th)
    a = np.column_stack([x, y, th])
    deg = np.degrees(np.mod(th, 0.5 * math.pi))
    tilt = np.minimum(deg, 90.0 - deg) > 1.5
    if not np.any(tilt):
        return []
    clusters = gadlib.clusters(a, tilt, link=1.42)
    masks: list[tuple[str, np.ndarray]] = []
    for ci, cl in enumerate(clusters):
        x0 = float((a[cl, 0] - hx[cl]).min())
        x1 = float((a[cl, 0] + hx[cl]).max())
        y0 = float((a[cl, 1] - hx[cl]).min())
        y1 = float((a[cl, 1] + hx[cl]).max())
        mask = (
            (x + hx >= x0 - collar)
            & (x - hx <= x1 + collar)
            & (y + hx >= y0 - collar)
            & (y - hx <= y1 + collar)
        )
        masks.append((f"cl{ci}_{len(cl)}t_{int(mask.sum())}f", mask))
    return masks


def run_localized_mechwalk(
    n: int,
    x0: np.ndarray,
    y0: np.ndarray,
    th0: np.ndarray,
    free_mask: np.ndarray,
    inc_dec: Decimal,
    tag: str = "loc",
    contact_tols: tuple[float, ...] = (1e-5, 4e-4, 2e-3),
    amplitudes: tuple[float, ...] = (0.008, 0.025, 0.08, 0.20, 0.45),
    n_modes: int = 12,
    top_k_slp: int = 6,
    slp_rounds: int = 14,
    threads: int = 4,
    seed_base: int = 51000,
) -> tuple[dict | None, np.ndarray, np.ndarray, np.ndarray, float]:
    """Run localized SVD soft-mode walk on `free_mask` + shake-free C++ L-BFGS re-jam + full joint SLP."""
    x0, y0, th0, L0 = normalize(x0, y0, th0)
    free_idx = np.flatnonzero(free_mask)
    nf = len(free_idx)
    if nf < 8:
        return None, x0, y0, th0, L0

    seeds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    seam_masks: list[np.ndarray] = []
    schedules: list[list[float]] = []
    labels: list[str] = []
    mask_u8 = free_mask.astype(np.uint8)

    for ctol in contact_tols:
        jac, fidx, _ = localized_contact_jacobian(x0, y0, th0, free_mask, contact_tol=ctol)
        if jac.shape[0] == 0:
            continue
        _, modes = localized_soft_modes(jac, k=n_modes)
        kept = []
        for m_i, mode in enumerate(modes):
            v = mode.astype(np.float64).copy()
            norm = float(np.linalg.norm(v))
            if norm < 1e-10:
                continue
            v /= norm
            if any(abs(float(np.dot(v, prev))) > 0.92 for prev in kept):
                continue
            kept.append(v)
            for amp in amplitudes:
                for sgn, slb in ((1.0, "p"), (-1.0, "m")):
                    cx, cy, cth = x0.copy(), y0.copy(), th0.copy()
                    cx[fidx] += sgn * amp * v[:nf]
                    cy[fidx] += sgn * amp * v[nf : 2 * nf]
                    cth[fidx] += sgn * amp * v[2 * nf :]
                    cx, cy, cth, _ = normalize(cx, cy, cth)
                    seeds.append((cx, cy, cth))
                    seam_masks.append(mask_u8)
                    if (len(seeds) % 2) == 0:
                        schedules.append([L0 + 0.012, L0 + 0.003, L0 - 2e-4])
                    else:
                        schedules.append([L0 + 0.025, L0 + 0.006, L0 - 5e-4])
                    labels.append(f"{tag}_t{ctol:.0e}_m{m_i}_{slb}_{amp:.3g}")
            if len(kept) >= 8:
                break

    if not seeds:
        return None, x0, y0, th0, L0

    best_cert: dict | None = None
    best_state = (x0, y0, th0, L0)
    cand_pool: list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray, str]] = []

    for start in range(0, len(seeds), 64):
        end = min(len(seeds), start + 64)
        relaxed = su.cpp_batch_melt(
            seeds[start:end],
            schedules[start:end],
            sym_mode=1,
            a_start=0.965,
            n_melt_stages=3,
            seam_masks=seam_masks[start:end],
            seam_iters=140,
            full_iters=260,
            n_ruin_iters=0,
            n_threads=min(4, max(1, threads)),
            base_seed=seed_base + n * 17 + start,
        )
        for bi, row in enumerate(relaxed):
            nx, ny, nth, L_meas = normalize(row["x"], row["y"], row["th"])
            if row["best_L_feas"] is not None:
                cand_pool.append((0.0, L_meas, nx, ny, nth, labels[start + bi]))
            elif row["pen"] < 0.02:
                cand_pool.append((1.0 + float(row["pen"]), L_meas, nx, ny, nth, labels[start + bi]))

    cand_pool.sort(key=lambda t: (t[0], t[1]))
    for _, _, cx, cy, cth, lbl in cand_pool[:top_k_slp]:
        cert = polish_and_certify(n, cx, cy, cth, inc_dec, slp_rounds=slp_rounds)
        if cert is not None:
            cert["label"] = lbl
            if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                best_cert = cert
            if cert["L_float"] < best_state[3]:
                best_state = (cx, cy, cth, cert["L_float"])

    return best_cert, best_state[0], best_state[1], best_state[2], best_state[3]
