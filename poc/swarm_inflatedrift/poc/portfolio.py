"""Live-portfolio loader and dual (literal + pandas) certifier for swarm_inflatedrift."""
from __future__ import annotations

import csv
import fcntl
import io
import json
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
POC_DIR = Path(__file__).resolve().parent
MECHWALK_POC = REPO_ROOT / "poc" / "swarm_mechwalk" / "poc"
HIGHD_POC = REPO_ROOT / "poc" / "swarm_highd" / "poc"
LPSLACK_POC = REPO_ROOT / "poc" / "swarm_lpslack" / "poc"
UNION_POC = REPO_ROOT / "poc" / "auto_union" / "poc"
BASELINE_SRC = REPO_ROOT / "poc" / "baseline" / "poc" / "src"

for _p in (POC_DIR, MECHWALK_POC, HIGHD_POC, LPSLACK_POC, UNION_POC, BASELINE_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import lpslack  # noqa: E402
import mechwalk_lib as ml  # noqa: E402
import pslp_4c as pslp  # noqa: E402
import union as auto_union  # noqa: E402
import utils as hu  # noqa: E402

MERGED_CSV = REPO_ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
SWARM_DIR = REPO_ROOT / "poc" / "swarm_inflatedrift"
DATA_DIR = SWARM_DIR / "data"
DOCS_DIR = SWARM_DIR / "docs"
FINAL_CSV = SWARM_DIR / "final.csv"


def normalize(x: np.ndarray, y: np.ndarray, th: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    return ml.normalize(x, y, th)


def load_live_portfolio(target_ns: list[int] | set[int]) -> dict[int, dict]:
    """Load exact best-per-n across merged.csv and all other poc/*/final.csv."""
    need = {int(n) for n in target_ns}
    best: dict[int, dict] = {}

    raw_groups: dict[int, list[dict]] = {}
    with MERGED_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row["n"])
            if n in need:
                raw_groups.setdefault(n, []).append(
                    {
                        "square_id": int(row["square_id"]),
                        "x": row["x"],
                        "y": row["y"],
                        "angle": row["angle"],
                    }
                )
    for n, rows in raw_groups.items():
        rows.sort(key=lambda r: r["square_id"])
        L_dec = Decimal(auto_union.certify(n, rows))
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
            "L_lit": L_dec,
            "L_pandas": L_dec,
            "source": "merged.csv",
        }

    prefixes = {f"{n:03}_": n for n in need}
    for p in sorted((REPO_ROOT / "poc").glob("*/final.csv")):
        if p == FINAL_CSV or p.stat().st_size <= 20:
            continue
        matched: dict[int, list[str]] = {}
        with p.open("r", encoding="utf-8") as f:
            header = f.readline()
            if header.strip() != "id,x,y,deg":
                continue
            for line in f:
                idx_us = line.find("_")
                if idx_us <= 0:
                    continue
                n = prefixes.get(line[: idx_us + 1])
                if n is not None:
                    matched.setdefault(n, []).append(line)
        if not matched:
            continue
        src_label = str(p.relative_to(REPO_ROOT))
        for n, lines in matched.items():
            if len(lines) != n:
                continue
            sub_text = "id,x,y,deg\n" + "".join(lines)
            try:
                lit = auto_union.parse_text(sub_text)
                pd_text = pd.read_csv(io.StringIO(sub_text)).to_csv(index=False)
                pd_t = auto_union.parse_text(pd_text)
                res = auto_union.check((n, lit[n], pd_t[n]))
            except Exception:
                continue
            if res is None:
                continue
            l_lit, l_pd = Decimal(res[0]), Decimal(res[1])
            l_worst = max(l_lit, l_pd)
            if n not in best or l_worst < best[n]["L_dec"]:
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
                    "L_dec": l_worst,
                    "L_lit": l_lit,
                    "L_pandas": l_pd,
                    "source": src_label,
                }
    return best


def certify_state(n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray) -> dict:
    """Dual-certify (x, y, th), applying uniform shrink / tiny LP margin if needed."""
    x_n, y_n, th_n, L_n = normalize(x, y, th)
    centers = np.column_stack([x_n, y_n])

    # First try lpslack's 80-digit uniform separating-axis shrink if strictly feasible
    mp, mw = pslp.feas(x_n, y_n, th_n, L_n, L_n)
    if min(mp, mw) > 1e-12:
        shr = lpslack.try_shrink_and_certify(n, centers, th_n, Decimal("1e9"))
        if shr is not None:
            lit, pan, rows = shr
            L_w = max(lit, pan)
            # Re-verify via canonical final_text roundtrip
            text = auto_union.final_text({n: rows})
            lit_g = auto_union.parse_text(text)
            pd_g = auto_union.parse_text(pd.read_csv(io.StringIO(text)).to_csv(index=False))
            res = auto_union.check((n, lit_g[n], pd_g[n]))
            if res is not None:
                return {
                    "ok": True,
                    "L_lit": res[0],
                    "L_pandas": res[1],
                    "L_dec_str": str(max(Decimal(res[0]), Decimal(res[1]))),
                    "L_float": float(max(Decimal(res[0]), Decimal(res[1]))),
                    "rows": rows,
                    "x": x_n,
                    "y": y_n,
                    "th": th_n,
                }

    if mp > 1e-13 and mw > -1e-12:
        ver = ml.certify_canonical(n, x_n, y_n, th_n)
        if ver.get("ok"):
            ver["x"], ver["y"], ver["th"] = x_n, y_n, th_n
            return ver

    for m_lp in (5e-13, 2e-12, 1e-11, 1e-10):
        xf, yf, _, ok_lp = hu.fixed_angle_lp_tighten(x_n, y_n, th_n, rounds=3, margin=m_lp)
        if ok_lp:
            xf_n, yf_n, th_f, L_f = normalize(xf, yf, th_n)
            mp2, mw2 = pslp.feas(xf_n, yf_n, th_f, L_f, L_f)
            if mp2 > 1e-13 and mw2 > -1e-12:
                ver = ml.certify_canonical(n, xf_n, yf_n, th_f)
                if ver.get("ok"):
                    ver["x"], ver["y"], ver["th"] = xf_n, yf_n, th_f
                    return ver

    cx, cy = 0.5 * L_n, 0.5 * L_n
    for eps in (2e-13, 1e-12, 5e-12, 2e-11, 1e-10):
        xs = cx + (x_n - cx) * (1.0 + eps)
        ys = cy + (y_n - cy) * (1.0 + eps)
        xs_n, ys_n, th_e, L_e = normalize(xs, ys, th_n)
        mp3, mw3 = pslp.feas(xs_n, ys_n, th_e, L_e, L_e)
        if mp3 > 1e-13 and mw3 > -1e-12:
            ver = ml.certify_canonical(n, xs_n, ys_n, th_e)
            if ver.get("ok"):
                ver["x"], ver["y"], ver["th"] = xs_n, ys_n, th_e
                return ver

    return {"ok": False, "L_lit": None, "L_pandas": None}


def save_win_locked(win_entry: dict, data_dir: Path = DATA_DIR, final_path: Path = FINAL_CSV) -> bool:
    """Save certified win JSON and rebuild final.csv under flock. Returns True if newly improved."""
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
        ml.write_final_csv(wins_by_n, final_path)
        return True


def rebuild_final_csv(data_dir: Path = DATA_DIR, final_path: Path = FINAL_CSV) -> int:
    """Rebuild final.csv from all win_n*.json (or header-only if none)."""
    wins_by_n: dict[int, dict] = {}
    if data_dir.exists():
        for p in sorted(data_dir.glob("win_n*.json")):
            w = json.loads(p.read_text(encoding="utf-8"))
            wins_by_n[int(w["n"])] = w
    ml.write_final_csv(wins_by_n, final_path)
    return len(wins_by_n)
