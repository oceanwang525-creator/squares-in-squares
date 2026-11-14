#!/usr/bin/env python3
"""Verify poc/swarm_mechwalk/final.csv against full origin/main portfolio and generate README.md."""
from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

POC_DIR = Path(__file__).resolve().parent
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import mechwalk_lib as ml  # noqa: E402
import union as auto_union  # noqa: E402


def main() -> None:
    final_path = ml.FINAL_CSV
    lit, pd_t = auto_union.load(final_path)
    win_ns = sorted(lit.keys())

    portfolio = ml.load_origin_main_portfolio(win_ns) if win_ns else {}

    verified_wins: list[dict] = []
    tot_ds = Decimal(0)
    for n in win_ns:
        res = auto_union.check((n, lit[n], pd_t[n]))
        if res is None:
            raise ValueError(f"Dual certification failed for n={n}")
        l_lit, l_pd = Decimal(res[0]), Decimal(res[1])
        inc_dec = portfolio[n]["L_dec"]
        l_worst = max(l_lit, l_pd)
        dL = l_worst - inc_dec
        if dL > Decimal("-1e-9"):
            raise ValueError(f"Non-real win in final.csv for n={n}: dL={float(dL)}")
        ds = dL / Decimal(n).sqrt()
        tot_ds += ds
        wp = ml.DATA_DIR / f"win_n{n}.json"
        label = json.loads(wp.read_text(encoding="utf-8")).get("label", "unknown") if wp.exists() else "unknown"
        verified_wins.append({
            "n": n,
            "L_lit": str(l_lit),
            "L_pd": str(l_pd),
            "lit_eq_pd": l_lit == l_pd,
            "L_inc": str(inc_dec),
            "inc_src": portfolio[n]["source"],
            "dL": float(dL),
            "ds": float(ds),
            "label": label,
        })

    def _load_runs(files: tuple[str, ...]) -> dict[int, dict]:
        runs: dict[int, dict] = {}
        for name in files:
            p = ml.DOCS_DIR / name
            if not p.exists():
                continue
            for row in json.loads(p.read_text(encoding="utf-8")):
                row["run_file"] = name
                runs[int(row["n"])] = row
        return runs

    def _append_run_table(title: str, runs: dict[int, dict]) -> None:
        wins_r = [runs[n] for n in sorted(runs) if runs[n]["status"] == "REAL_WIN"]
        misses_r = [runs[n] for n in sorted(runs) if runs[n]["status"] != "REAL_WIN"]
        lines.extend([
            "",
            f"## {title} ({len(wins_r)} REAL WINs, {len(misses_r)} Ties/Misses across {len(runs)} `n`s)",
            "",
            "| `n` | Run | Status | Best `dL` (vs seed) | Time (s) | Best Label |",
            "|---:|---|---|---:|---:|---|",
        ])
        for n in sorted(runs):
            r = runs[n]
            dL_s = f"`{r['best_dL']:+.6e}`" if r.get("best_dL") is not None else "N/A"
            lines.append(
                f"| {n} | `{r['run_file']}` | **{r['status']}** | {dL_s} | {r['dt_s']} | `{r.get('best_label')}` |"
            )

    batch7_runs = _load_runs(("b6_s3_results.json", "b6_s2_results.json", "b6_s4_results.json"))

    lines = [
        "# H22 / H31 / H39 / H45 / H56: Iterated Sparse Contact-Jacobian SVD Soft-Mode Mechanism Walk + C++ Masked Melt",
        "",
        "## Method",
        "",
        "- **Exact Sparse Contact Jacobian (`mechwalk_lib.py`)**: Vectorized near-active square-square separating axes and 4-wall contacts across multiple slack thresholds (`contact_tol in {2e-5, 5e-5, 2e-4, 4e-4, 1e-3, 2e-3}`), with exact analytical rotation/owner derivatives verified against finite differences (`tests/test_mechwalk.py`).",
        "- **Diverse Soft-Mode Extraction & `L_inf` Stepping**: Extracts `k=14..16` smallest right singular modes (`scipy.sparse.linalg.svds`), projects out whole-layout rigid translation and rotation, deduplicates by cosine similarity (`< 0.90`), keeps both softest (`min sigma`) and collective (`max n_eff`) modes, and walks `(+1, -1)` across `L_inf`-normalized steps `(0.008, 0.025, 0.075, 0.18)`.",
        "- **Dual Repair + Iterated Beam (4 rounds, beam=2, top-k=8)**:",
        "  - **`rejam`**: Full-box soft-core SAT L-BFGS re-jam (`run_cpp_masked_melt` with all-1s mask, no ILS) + 2-stage `joint_xytheta_slp` + `fixed_angle_lp_tighten`.",
        "  - **`mmelt` (Hybrid)**: Mode-activity pocket mask (top moving squares + spatial neighbors within `R=2.4`, capped at `<= 50%` of `n`) -> `run_cpp_masked_melt` (`ils=[0, 5, 10]`, `a_start=0.82`) -> 2-stage `joint_xytheta_slp` + `fixed_angle_lp_tighten`.",
        "- **Certification**: Every exported group in `final.csv` is certified in BOTH `literal` and `pandas` roundtrip modes via `poc/auto_union/poc/union.py` against the full `origin/main` portfolio (`merged.csv` + all `poc/*/final.csv`), keeping only strict REAL WINs (`dL <= -1e-9`).",
        "",
        f"## Batch 7 Summary (H56 Stages 2-4: {len(verified_wins)} Certified REAL WINs in `final.csv`, total `ds = {float(tot_ds):+.6e}`)",
        "",
        "| `n` | `dL` | `ds = dL / sqrt(n)` | Certified `L` (`literal == pandas`) | Incumbent `L` (`origin/main`) | Method Label |",
        "|---:|---:|---:|---|---|---|",
    ]
    for w in verified_wins:
        lines.append(
            f"| {w['n']} | `{w['dL']:+.6e}` | `{w['ds']:+.6e}` | `{w['L_lit'][:22]}` | `{w['L_inc'][:22]}` ({w['inc_src']}) | `{w['label']}` |"
        )

    _append_run_table("Batch 7 Run Log", batch7_runs)
    for title, files in (
        ("Batch 6 Archive (Merged in PR #45 / `e6910c3`)", ("b6_s1_results.json",)),
        ("Batch 5 Archive (Merged in PR #43 / `985a1e2`)", ("b5_s1_results.json",)),
        ("Batch 4 Archive (Merged in PR #42 / `e241efd`)", ("b4_s1_results.json",)),
        ("Batch 3 Archive (Merged in PR #40 / `37be2a8`)", ("b3_s1_results.json", "b3_s2_results.json")),
        ("Batch 2 Archive (Merged in PR #35 / `34fd019`)", ("scale_b2_s1_results.json", "scale_b2_s2_results.json")),
        ("Batch 1 Archive (Merged in PR #32 / `e5751c5`)", ("gate_results.json", "scale_s1_results.json", "scale_s2_results.json")),
    ):
        _append_run_table(title, _load_runs(files))

    lines.append("")
    (ml.MECHWALK_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"VERIFIED {len(verified_wins)} REAL WINs, total_ds={float(tot_ds):+.6e}")
    for w in verified_wins:
        print(f"  n={w['n']:4d} lit==pd:{w['lit_eq_pd']} dL={w['dL']:+.6e} ds={w['ds']:+.6e} L={w['L_lit'][:22]} ({w['inc_src']})")


if __name__ == "__main__":
    main()
