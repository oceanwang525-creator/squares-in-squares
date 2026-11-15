"""Fast live-portfolio loader and dual (literal + pandas) certifier for H125 (armslack2)."""
from __future__ import annotations

import csv
import fcntl
import io
import json
import math
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
MECHWALK_POC = REPO_ROOT / "poc" / "swarm_mechwalk" / "poc"
HIGHD_POC = REPO_ROOT / "poc" / "swarm_highd" / "poc"
UNION_POC = REPO_ROOT / "poc" / "auto_union" / "poc"
BASELINE_SRC = REPO_ROOT / "poc" / "baseline" / "poc" / "src"

for _p in (MECHWALK_POC, HIGHD_POC, UNION_POC, BASELINE_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mechwalk_lib as ml  # noqa: E402
import union as auto_union  # noqa: E402
import utils as hu  # noqa: E402

MERGED_CSV = REPO_ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
SWARM_DIR = REPO_ROOT / "poc" / "swarm_armslack2"
DATA_DIR = SWARM_DIR / "data"
DOCS_DIR = SWARM_DIR / "docs"
FINAL_CSV = SWARM_DIR / "final.csv"


def _float_L_rad(rows: list[dict]) -> float:
    x = np.array([float(r["x"]) for r in rows], dtype=np.float64)
    y = np.array([float(r["y"]) for r in rows], dtype=np.float64)
    th = np.array([float(r["angle"]) for r in rows], dtype=np.float64)
    e = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    return float(max((x + e).max() - (x - e).min(), (y + e).max() - (y - e).min()))


def _float_L_deg_lines(lines: list[str]) -> float:
    xs, ys, ds = [], [], []
    for line in lines:
        _, sx, sy, sd = line.strip().split(",")
        xs.append(float(sx[1:] if sx.startswith("s") else sx))
        ys.append(float(sy[1:] if sy.startswith("s") else sy))
        ds.append(math.radians(float(sd[1:] if sd.startswith("s") else sd)))
    x = np.array(xs, dtype=np.float64)
    y = np.array(ys, dtype=np.float64)
    th = np.array(ds, dtype=np.float64)
    e = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    return float(max((x + e).max() - (x - e).min(), (y + e).max() - (y - e).min()))


def load_live_portfolio(target_ns: list[int] | set[int]) -> dict[int, dict]:
    """Load exact best-per-n across merged.csv and all other poc/*/final.csv (float-prefiltered)."""
    need = {int(n) for n in target_ns}
    cands: dict[int, list[tuple[float, str, object]]] = {n: [] for n in need}

    raw_groups: dict[int, list[dict]] = {}
    with MERGED_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
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
        cands[n].append((_float_L_rad(rows), "merged.csv", rows))

    prefixes = {f"{n:03}_": n for n in need}
    for p in sorted((REPO_ROOT / "poc").glob("*/final.csv")):
        if p == FINAL_CSV or p.stat().st_size <= 20:
            continue
        matched: dict[int, list[str]] = {}
        with p.open("r", encoding="utf-8") as f:
            if f.readline().strip() != "id,x,y,deg":
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
            if len(lines) == n:
                cands[n].append((_float_L_deg_lines(lines), src_label, lines))

    best: dict[int, dict] = {}
    for n in sorted(need):
        # Certify in ascending float-L order and stop once strictly below next float-L
        sorted_cands = sorted(cands[n], key=lambda t: t[0])
        for fl, src_label, payload in sorted_cands:
            if n in best and fl > float(best[n]["L_dec"]) + 1e-10:
                break
            if src_label == "merged.csv":
                rows = payload
                L_dec = Decimal(auto_union.certify(n, rows))
                L_lit = L_pd = L_dec
            else:
                sub_text = "id,x,y,deg\n" + "".join(payload)
                try:
                    lit = auto_union.parse_text(sub_text)
                    pdt = auto_union.parse_text(pd.read_csv(io.StringIO(sub_text)).to_csv(index=False))
                    res = auto_union.check((n, lit[n], pdt[n]))
                except Exception:
                    continue
                if res is None:
                    continue
                L_lit, L_pd = Decimal(res[0]), Decimal(res[1])
                L_dec = max(L_lit, L_pd)
                rows = sorted(lit[n], key=lambda r: int(r["square_id"]))
            if n not in best or L_dec < best[n]["L_dec"]:
                x = np.array([float(r["x"]) for r in rows], dtype=np.float64)
                y = np.array([float(r["y"]) for r in rows], dtype=np.float64)
                th = np.array([float(r["angle"]) for r in rows], dtype=np.float64)
                x, y, th, L_f = ml.normalize(x, y, th)
                best[n] = {
                    "x": x,
                    "y": y,
                    "th": th,
                    "L": L_f,
                    "L_dec": L_dec,
                    "L_lit": L_lit,
                    "L_pandas": L_pd,
                    "source": src_label,
                }
    return best


def certify_state(n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray) -> dict:
    """Certify (x, y, th) in canonical s-prefixed format (both literal and pandas roundtrip)."""
    x, y, th, _ = ml.normalize(x, y, th)
    ver = ml.certify_canonical(n, x, y, th)
    if ver.get("ok"):
        ver["x"] = x.copy()
        ver["y"] = y.copy()
        ver["th"] = th.copy()
        return ver

    for m_lp in (1e-9, 1e-8, 5e-8, 2e-7, 1e-6):
        xf, yf, _, ok_lp = hu.fixed_angle_lp_tighten(x, y, th, rounds=4, margin=m_lp)
        if not ok_lp:
            continue
        xf, yf, th_c, _ = ml.normalize(xf, yf, th)
        cand = ml.certify_canonical(n, xf, yf, th_c)
        if cand.get("ok"):
            cand["x"] = xf.copy()
            cand["y"] = yf.copy()
            cand["th"] = th_c.copy()
            return cand
    return {"ok": False}


def save_win_locked(win_entry: dict) -> Path:
    """Atomically save a certified win to poc/swarm_armslack2/data/win_n{n}.json."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    n = int(win_entry["n"])
    out_p = DATA_DIR / f"win_n{n}.json"
    lock_p = DATA_DIR / ".win.lock"
    with lock_p.open("w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        if out_p.exists():
            try:
                old = json.loads(out_p.read_text(encoding="utf-8"))
                old_w = max(Decimal(old["L_lit"]), Decimal(old["L_pandas"]))
                new_w = max(Decimal(win_entry["L_lit"]), Decimal(win_entry["L_pandas"]))
                if old_w <= new_w:
                    return out_p
            except Exception:
                pass
        tmp_p = DATA_DIR / f".win_n{n}.json.tmp"
        tmp_p.write_text(json.dumps(win_entry, indent=1), encoding="utf-8")
        tmp_p.replace(out_p)
    return out_p


def export_final_csv(min_dl: Decimal = Decimal("1e-9")) -> tuple[int, float]:
    """Export all strict certified wins from data/win_n*.json into final.csv vs live portfolio."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    win_files = sorted(DATA_DIR.glob("win_n*.json"))
    if not win_files:
        FINAL_CSV.write_text("id,x,y,deg\n", encoding="utf-8")
        return 0, 0.0

    cands: dict[int, dict] = {}
    for wf in win_files:
        try:
            d = json.loads(wf.read_text(encoding="utf-8"))
            cands[int(d["n"])] = d
        except Exception:
            continue

    if not cands:
        FINAL_CSV.write_text("id,x,y,deg\n", encoding="utf-8")
        return 0, 0.0

    port = load_live_portfolio(list(cands.keys()))
    groups: dict[int, list[dict]] = {}
    tot_ds = 0.0
    for n, d in sorted(cands.items()):
        rows = [
            {
                "square_id": str(r["square_id"]),
                "x": str(r["x"]),
                "y": str(r["y"]),
                "angle": str(r["angle"]),
            }
            for r in sorted(d["rows"], key=lambda r: int(r["square_id"]))
        ]
        txt = auto_union.final_text({n: rows})
        lit = auto_union.parse_text(txt)
        pdt = auto_union.parse_text(pd.read_csv(io.StringIO(txt)).to_csv(index=False))
        res = auto_union.check((n, lit[n], pdt[n]))
        if res is None:
            continue
        L_lit, L_pd = Decimal(res[0]), Decimal(res[1])
        L_w = max(L_lit, L_pd)
        inc = port[n]
        if inc["L_dec"] - L_w >= min_dl and L_lit < inc["L_lit"] and L_pd < inc["L_pandas"]:
            groups[n] = rows
            tot_ds += float((L_w ** 2 - inc["L_dec"] ** 2) / n)

    if not groups:
        FINAL_CSV.write_text("id,x,y,deg\n", encoding="utf-8")
        return 0, 0.0

    FINAL_CSV.write_text(auto_union.final_text(groups), encoding="utf-8")
    return len(groups), tot_ds
