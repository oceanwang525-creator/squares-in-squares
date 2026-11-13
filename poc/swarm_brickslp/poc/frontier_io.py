#!/usr/bin/env python3
"""True frontier loader (merged.csv + all poc/*/final.csv) and literal+pandas certifier."""
from __future__ import annotations

import fcntl
import io
import json
import os
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
UNION_POC = REPO / "poc" / "auto_union" / "poc"
HOMO_POC = REPO / "poc" / "swarm_homology" / "poc"
for p in (UNION_POC, HOMO_POC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import union as auto_union  # noqa: E402

BRICK_DIR = REPO / "poc" / "swarm_brickslp"
DATA_DIR = BRICK_DIR / "data"
DOCS_DIR = BRICK_DIR / "docs"
FINAL_CSV = BRICK_DIR / "final.csv"
MERGED_CSV = REPO / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"


def hx_of(th: np.ndarray) -> np.ndarray:
    return 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))


def norm_pack(x: np.ndarray, y: np.ndarray, th: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64).copy()
    y = np.asarray(y, dtype=np.float64).copy()
    th = np.asarray(th, dtype=np.float64).copy()
    h = hx_of(th)
    x -= float((x - h).min())
    y -= float((y - h).min())
    L = float(max((x + h).max(), (y + h).max()))
    return x, y, th, L


def load_true_frontier(check_finals: bool = True) -> tuple[dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]], dict[int, Decimal]]:
    """Load minimum-L packing per n across merged.csv (radians) and all poc/*/final.csv (degrees)."""
    packs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    best_L_dec: dict[int, Decimal] = {}

    df = pd.read_csv(MERGED_CSV)
    for n_val, sub in df.groupby("n"):
        n = int(n_val)
        sub = sub.sort_values("square_id")
        x, y, th, L = norm_pack(sub["x"].to_numpy(), sub["y"].to_numpy(), sub["angle"].to_numpy())
        packs[n] = (x, y, th, L)
        best_L_dec[n] = Decimal(f"{L:.17g}")

    # Refine exact Decimal L on merged.csv via union.json if present
    union_json = REPO / "poc" / "auto_union" / "docs" / "union.json"
    if union_json.exists():
        try:
            rep = json.loads(union_json.read_text())
            for r in rep.get("reports", [{}])[0].get("results", []):
                if r.get("L") is not None:
                    best_L_dec[int(r["n"])] = Decimal(str(r["L"]))
        except Exception:
            pass

    if check_finals:
        for final_path in sorted((REPO / "poc").glob("*/final.csv")):
            if final_path.parent.name == "swarm_brickslp":
                continue
            if final_path.stat().st_size <= 20:
                continue
            try:
                groups = auto_union.parse_text(final_path.read_text())
            except Exception:
                continue
            for n, rows in groups.items():
                x = np.array([float(r["x"]) for r in rows], dtype=np.float64)
                y = np.array([float(r["y"]) for r in rows], dtype=np.float64)
                th = np.array([float(r["angle"]) for r in rows], dtype=np.float64)
                x, y, th, L = norm_pack(x, y, th)
                if L < float(best_L_dec[n]) + 1e-9:
                    Ld = Decimal(auto_union.certify(n, rows))
                    if Ld < best_L_dec[n]:
                        packs[n] = (x, y, th, L)
                        best_L_dec[n] = Ld

    return packs, best_L_dec


def exact_official_L(n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray) -> tuple[Decimal | None, Decimal | None, list[dict] | None]:
    """Certify using union.check in BOTH literal and pandas transport modes via final_text."""
    hx = hx_of(th)
    min_x = float((x - hx).min())
    min_y = float((y - hx).min())
    sx = x - min_x + 1e-12
    sy = y - min_y + 1e-12
    rows = [
        {
            "square_id": str(i),
            "x": repr(float(sx[i])),
            "y": repr(float(sy[i])),
            "angle": repr(float(th[i])),
        }
        for i in range(n)
    ]
    text = auto_union.final_text({n: rows})
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cand.csv"
        p.write_text(text, encoding="utf-8")
        try:
            lit_groups, pd_groups = auto_union.load(p)
            res = auto_union.check((n, lit_groups[n], pd_groups[n]))
        except Exception:
            return None, None, None
    if res is None:
        return None, None, None
    return Decimal(res[0]), Decimal(res[1]), rows


def certify_against_frontier(n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray, L_inc_dec: Decimal) -> dict | None:
    """Certify candidate against L_inc_dec in BOTH literal and pandas modes."""
    L_lit, L_pd, rows = exact_official_L(n, x, y, th)
    if L_lit is None or L_pd is None:
        # Retry with tiny 1e-10 expansion if floating-point rounding left ~1e-15 overlap
        for eps in (1e-10, 1e-9):
            sc = 1.0 + eps
            L_lit, L_pd, rows = exact_official_L(n, x * sc, y * sc, th)
            if L_lit is not None and L_pd is not None:
                break
    if L_lit is None or L_pd is None:
        return None
    L_worst = max(L_lit, L_pd)
    dL = L_worst - L_inc_dec
    return {
        "ok": True,
        "n": int(n),
        "L_lit": str(L_lit),
        "L_pd": str(L_pd),
        "L_float": float(L_worst),
        "dL": float(dL),
        "is_real_win": bool(dL <= Decimal("-1e-9")),
        "rows": rows,
    }


def save_win_and_rebuild_final(n: int, cert: dict, label: str) -> bool:
    """Save certified REAL WIN (dL <= -1e-9) atomically and rebuild poc/swarm_brickslp/final.csv."""
    if not cert.get("is_real_win"):
        return False
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = DATA_DIR / ".win_lock"
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        win_path = DATA_DIR / f"win_n{n}.json"
        if win_path.exists():
            prev = json.loads(win_path.read_text())
            if max(Decimal(prev["L_lit"]), Decimal(prev["L_pd"])) <= max(Decimal(cert["L_lit"]), Decimal(cert["L_pd"])):
                return False
        payload = {
            "n": int(n),
            "label": label,
            "L_lit": cert["L_lit"],
            "L_pd": cert["L_pd"],
            "L_float": cert["L_float"],
            "dL": cert["dL"],
            "rows": cert["rows"],
        }
        auto_union.atomic(win_path, json.dumps(payload, indent=2).encode())
        os.chmod(win_path, 0o644)

        groups = {}
        for p in sorted(DATA_DIR.glob("win_n*.json")):
            d = json.loads(p.read_text())
            groups[int(d["n"])] = d["rows"]
        text = auto_union.final_text(groups)
        auto_union.atomic(FINAL_CSV, text.encode())
        os.chmod(FINAL_CSV, 0o644)
    return True
