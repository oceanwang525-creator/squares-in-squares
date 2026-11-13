"""H22: Iterated Sparse Contact-Jacobian SVD Soft-Mode Mechanism Walk + C++ Masked Melt."""
from __future__ import annotations

import importlib.util
import io
import os
import sys
import tempfile
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
HIGHD_POC = REPO_ROOT / "poc" / "swarm_highd" / "poc"
CASCADES_POC = REPO_ROOT / "poc" / "swarm_cascades" / "poc"
UNION_POC = REPO_ROOT / "poc" / "auto_union" / "poc"
BASELINE_SRC = REPO_ROOT / "poc" / "baseline" / "poc" / "src"

for _p in (HIGHD_POC, CASCADES_POC, UNION_POC, BASELINE_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import union as auto_union  # noqa: E402
import utils as hu  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, full_to_free_sym, run_cpp_masked_melt  # noqa: E402
from joint_slp import joint_xytheta_slp  # noqa: E402

MERGED_CSV = REPO_ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
MECHWALK_DIR = REPO_ROOT / "poc" / "swarm_mechwalk"
DATA_DIR = MECHWALK_DIR / "data"
DOCS_DIR = MECHWALK_DIR / "docs"
FINAL_CSV = MECHWALK_DIR / "final.csv"


def half_extent(theta: np.ndarray) -> np.ndarray:
    """Half axis-aligned bounding extent of unit squares rotated by theta."""
    return 0.5 * (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))


def half_extent_derivative(theta: np.ndarray) -> np.ndarray:
    """Derivative of half_extent(theta) w.r.t. theta."""
    c, s = np.cos(theta), np.sin(theta)
    return 0.5 * (-np.sign(c) * s + np.sign(s) * c)


def normalize(
    x: np.ndarray, y: np.ndarray, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Translate packing to [0, L]^2 and return (x, y, theta, L)."""
    x = np.asarray(x, dtype=np.float64).copy()
    y = np.asarray(y, dtype=np.float64).copy()
    theta = np.asarray(theta, dtype=np.float64).copy()
    h = half_extent(theta)
    x -= float(np.min(x - h))
    y -= float(np.min(y - h))
    side = float(max(np.max(x + h), np.max(y + h)))
    return x, y, theta, side


def contact_jacobian(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    contact_tol: float = 1e-5,
) -> tuple[csr_matrix, dict]:
    """Build exact sparse contact Jacobian d(gap)/d(x, y, theta) for near-active pairs and walls."""
    n = len(x)
    c = np.cos(theta)
    s = np.sin(theta)
    h = 0.5 * (np.abs(c) + np.abs(s))
    dh = 0.5 * (-np.sign(c) * s + np.sign(s) * c)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    row_offset = 0

    # Vectorized near-pair discovery
    dx_mat = x[None, :] - x[:, None]
    dy_mat = y[None, :] - y[:, None]
    upper = np.arange(n)[:, None] < np.arange(n)[None, :]
    ni, nj = np.where((np.abs(dx_mat) <= 1.55) & (np.abs(dy_mat) <= 1.55) & upper)

    pair_contacts = 0
    max_pair_gap: float | None = None
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
            ai = ni[active]
            aj = nj[active]
            adx = dx[active]
            ady = dy[active]
            ak = best_k[active]
            aax = ax_all[active, ak]
            aay = ay_all[active, ak]
            aproj = proj_all[active, ak]
            acth = cth[active]
            asth = sth[active]
            agap = gap[active]

            m = len(ai)
            pair_contacts = m
            max_pair_gap = float(np.max(agap))

            sgn = np.where(aproj >= 0.0, 1.0, -1.0)
            # radius = 0.5 * (1 + |cos(th_j - th_i)| + |sin(th_j - th_i)|)
            d_rad_dphi = 0.5 * (-np.sign(acth) * asth + np.sign(asth) * acth)
            proj_rot = np.where(np.abs(aproj) > 1e-15, sgn * (-adx * aay + ady * aax), 0.0)
            i_owns = ak < 2

            d_thi = np.where(i_owns, proj_rot, 0.0) + d_rad_dphi
            d_thj = np.where(~i_owns, proj_rot, 0.0) - d_rad_dphi

            r_ids = np.arange(row_offset, row_offset + m, dtype=np.int32)
            row_offset += m

            rows.extend([r_ids, r_ids, r_ids, r_ids, r_ids, r_ids])
            cols.extend([ai, n + ai, 2 * n + ai, aj, n + aj, 2 * n + aj])
            vals.extend([-sgn * aax, -sgn * aay, d_thi, sgn * aax, sgn * aay, d_thj])

    side = float(max(np.max(x + h), np.max(y + h)))
    min_x = float(np.min(x - h))
    min_y = float(np.min(y - h))
    wall_gaps: list[float] = []

    # Lower walls: gap = coord - min_coord - h(theta), d(gap)/d(coord) = +1, d(gap)/d(th) = -h'(th)
    for axis, coord, origin in ((0, x, min_x), (1, y, min_y)):
        wgap = (coord - origin) - h
        widx = np.flatnonzero(wgap <= contact_tol)
        if len(widx) > 0:
            mw = len(widx)
            r_ids = np.arange(row_offset, row_offset + mw, dtype=np.int32)
            row_offset += mw
            rows.extend([r_ids, r_ids])
            cols.extend([axis * n + widx, 2 * n + widx])
            vals.extend([np.ones(mw, dtype=np.float64), -dh[widx]])
            wall_gaps.extend(wgap[widx].tolist())

    # Upper walls: gap = side - coord - h(theta), d(gap)/d(coord) = -1, d(gap)/d(th) = -h'(th)
    for axis, coord in ((0, x), (1, y)):
        wgap = side - coord - h
        widx = np.flatnonzero(wgap <= contact_tol)
        if len(widx) > 0:
            mw = len(widx)
            r_ids = np.arange(row_offset, row_offset + mw, dtype=np.int32)
            row_offset += mw
            rows.extend([r_ids, r_ids])
            cols.extend([axis * n + widx, 2 * n + widx])
            vals.extend([-np.ones(mw, dtype=np.float64), -dh[widx]])
            wall_gaps.extend(wgap[widx].tolist())

    if row_offset == 0:
        jac = csr_matrix((0, 3 * n), dtype=np.float64)
    else:
        jac = coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(row_offset, 3 * n),
        ).tocsr()

    return jac, {
        "constraints": int(row_offset),
        "pair_contacts": int(pair_contacts),
        "wall_contacts": int(len(wall_gaps)),
        "max_pair_gap": max_pair_gap,
        "max_wall_gap": max(wall_gaps) if wall_gaps else None,
        "side": side,
    }


def soft_modes(jac: csr_matrix, k: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Return smallest right singular values and vectors of the contact Jacobian."""
    if jac.shape[0] == 0:
        return np.array([], dtype=np.float64), np.zeros((0, jac.shape[1]), dtype=np.float64)
    k_eff = max(1, min(k, min(jac.shape) - 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, singular, vt = svds(jac, k=k_eff, which="SM", solver="lobpcg", maxiter=5000)
    order = np.argsort(singular)
    return singular[order], vt[order]


def remove_rigid_motion(mode: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Project out rigid translation and whole-layout rotation from a mode."""
    n = len(x)
    projected = np.asarray(mode, dtype=np.float64).copy()
    projected[:n] -= np.mean(projected[:n])
    projected[n : 2 * n] -= np.mean(projected[n : 2 * n])

    cx, cy = x - np.mean(x), y - np.mean(y)
    rigid = np.concatenate((-cy, cx, np.ones(n, dtype=np.float64)))
    rn = float(np.linalg.norm(rigid))
    if rn > 1e-12:
        rigid /= rn
        projected -= rigid * float(np.dot(projected, rigid))
    return projected


def mode_participation(mode: np.ndarray, n: int) -> tuple[np.ndarray, float]:
    """Return per-square displacement magnitude and effective participating square count."""
    dx = mode[:n]
    dy = mode[n : 2 * n]
    dth = mode[2 * n :]
    disp = np.sqrt(dx * dx + dy * dy + 0.25 * dth * dth)
    s2 = float(np.sum(disp * disp))
    s4 = float(np.sum(disp ** 4))
    n_eff = (s2 * s2 / s4) if s4 > 1e-24 else 0.0
    return disp, n_eff


def select_diverse_modes(
    modes: np.ndarray,
    singular: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    max_keep: int = 8,
    min_variation: float = 1e-4,
    cos_dedup: float = 0.92,
) -> list[tuple[int, float, np.ndarray, float]]:
    """Project out rigid motions, deduplicate, and keep both softest and collective modes."""
    n = len(x)
    cleaned: list[tuple[int, float, np.ndarray, float]] = []
    for m_idx, mode in enumerate(modes):
        proj = remove_rigid_motion(mode, x, y)
        norm = float(np.linalg.norm(proj))
        if norm < 1e-10:
            continue
        proj /= norm
        var = float(np.sqrt(np.var(proj[:n]) + np.var(proj[n : 2 * n])))
        if var < min_variation:
            continue
        pivot = int(np.argmax(np.abs(proj)))
        if proj[pivot] < 0.0:
            proj = -proj
        if any(abs(float(np.dot(proj, prev[2]))) > cos_dedup for prev in cleaned):
            continue
        _, n_eff = mode_participation(proj, n)
        sv = float(singular[m_idx]) if m_idx < len(singular) else 0.0
        cleaned.append((m_idx, sv, proj, n_eff))

    if len(cleaned) <= max_keep:
        return cleaned

    # Keep top half by smallest singular value, and fill remainder with highest collective n_eff
    half = max(1, max_keep // 2)
    selected = list(cleaned[:half])
    used = {item[0] for item in selected}
    by_coll = sorted(cleaned[half:], key=lambda item: -item[3])
    for item in by_coll:
        if len(selected) >= max_keep:
            break
        if item[0] not in used:
            selected.append(item)
            used.add(item[0])
    return selected


def load_origin_main_portfolio(target_ns: list[int] | set[int]) -> dict[int, dict]:
    """Load exact origin/main best-per-n across merged.csv (rad) and all poc/*/final.csv (deg)."""
    need = set(int(n) for n in target_ns)
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
        # Certify exact merged.csv text via squares.core (read raw rows for exact Decimal)
        L_dec = Decimal(auto_union.certify(n, rows))
        x = grp_s["x"].to_numpy(dtype=np.float64, copy=True)
        y = grp_s["y"].to_numpy(dtype=np.float64, copy=True)
        th = grp_s["angle"].to_numpy(dtype=np.float64, copy=True)
        x, y, th, L_f = normalize(x, y, th)
        best[n] = {"x": x, "y": y, "th": th, "L": L_f, "L_dec": L_dec, "source": "merged.csv"}

    # Exact raw merged.csv check in case repr(float) differs in the 17th digit
    raw_merged = auto_union.parse_text(MERGED_CSV.read_text(encoding="utf-8"))
    for n in need:
        if n in raw_merged:
            L_raw = Decimal(auto_union.certify(n, raw_merged[n]))
            if L_raw < best[n]["L_dec"]:
                best[n]["L_dec"] = L_raw

    for p in sorted((REPO_ROOT / "poc").glob("*/final.csv")):
        if p == FINAL_CSV or p.stat().st_size <= 20:
            continue
        try:
            lit, pd_t = auto_union.load(p)
        except Exception:
            continue
        for n in need:
            if n not in lit or n not in pd_t or len(lit[n]) != n:
                continue
            res = auto_union.check((n, lit[n], pd_t[n]))
            if res is None:
                continue
            L_dec = max(Decimal(res[0]), Decimal(res[1]))
            if n not in best or L_dec < best[n]["L_dec"]:
                rows = sorted(lit[n], key=lambda r: int(r["square_id"]))
                x = np.array([float(r["x"]) for r in rows], dtype=np.float64)
                y = np.array([float(r["y"]) for r in rows], dtype=np.float64)
                th = np.array([float(r["angle"]) for r in rows], dtype=np.float64)
                x, y, th, L_f = normalize(x, y, th)
                best[n] = {
                    "x": x,
                    "y": y,
                    "th": th,
                    "L": L_f,
                    "L_dec": L_dec,
                    "source": str(p.relative_to(REPO_ROOT)),
                }

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


def write_final_csv(wins_by_n: dict[int, dict], out_path: Path = FINAL_CSV) -> None:
    """Write certified REAL WINs to final.csv (or header-only if empty) and verify dual certification."""
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


def save_win_locked(win_entry: dict, data_dir: Path = DATA_DIR, final_path: Path = FINAL_CSV) -> None:
    """Atomically save per-n win JSON under flock and rebuild final.csv from all win_n*.json."""
    import fcntl
    import json

    data_dir.mkdir(parents=True, exist_ok=True)
    lock_path = data_dir / ".final.lock"
    with lock_path.open("w", encoding="utf-8") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        n = int(win_entry["n"])
        wp = data_dir / f"win_n{n}.json"
        if wp.exists():
            prev = json.loads(wp.read_text(encoding="utf-8"))
            prev_dec = max(Decimal(prev["L_lit"]), Decimal(prev["L_pandas"]))
            new_dec = max(Decimal(win_entry["L_lit"]), Decimal(win_entry["L_pandas"]))
            if new_dec >= prev_dec:
                return
        wp.write_text(json.dumps(win_entry, indent=2), encoding="utf-8")
        wins_by_n: dict[int, dict] = {}
        for p in sorted(data_dir.glob("win_n*.json")):
            w = json.loads(p.read_text(encoding="utf-8"))
            wins_by_n[int(w["n"])] = w
        write_final_csv(wins_by_n, final_path)
