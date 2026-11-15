#!/usr/bin/env python3
"""H125 Gate & Scaling Runner with checkpointing (--resume) and CPU fairness (max 2 workers)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

POC_DIR = Path(__file__).resolve().parent
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import armslack2_lib as a2  # noqa: E402
import portfolio as pf  # noqa: E402

GATE_TARGETS = [974, 973, 969, 934, 914, 913, 908, 857, 856, 791, 687, 640, 634]


def _worker_solve(task: tuple[int, dict, int, bool, bool, bool, tuple[float, ...], Path]) -> dict:
    n, inc, n_threads, enable_mw, tier1_only, deep_melt, reaches, ckpt_dir = task
    summary, _ = a2.solve_target(
        n,
        inc,
        n_threads=n_threads,
        enable_mechwalk=enable_mw,
        tier1_only=tier1_only,
        deep_melt=deep_melt,
        reaches=reaches,
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp_p = ckpt_dir / f".n_{n}.json.tmp"
    out_p = ckpt_dir / f"n_{n}.json"
    tmp_p.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    tmp_p.replace(out_p)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", default=GATE_TARGETS)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=1, help="C++ threads per worker (workers*threads <= 2)")
    ap.add_argument("--no-mechwalk", action="store_true")
    ap.add_argument("--tier1-only", action="store_true")
    ap.add_argument("--deep-melt", action="store_true")
    ap.add_argument("--reaches", type=float, nargs="+", default=[3.5, 2.6])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag", default="gate")
    args = ap.parse_args()

    workers = max(1, min(2, args.workers))
    n_threads = max(1, min(2, args.threads))
    if workers * n_threads > 2:
        n_threads = 1

    # Compile C++ library once before forking workers
    a2.cpp_wrap.get_cpp_lib()

    ckpt_dir = pf.DOCS_DIR / f"ckpt_{args.tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    todo_ns = []
    results = []
    for n in args.targets:
        cp = ckpt_dir / f"n_{n}.json"
        if args.resume and cp.exists():
            results.append(json.loads(cp.read_text(encoding="utf-8")))
        else:
            todo_ns.append(n)

    reaches = tuple(float(r) for r in args.reaches)
    if todo_ns:
        port = pf.load_live_portfolio(todo_ns)
        tasks = [
            (n, port[n], n_threads, not args.no_mechwalk, args.tier1_only, args.deep_melt, reaches, ckpt_dir)
            for n in todo_ns
            if n in port
        ]
        if workers == 1:
            for t in tasks:
                results.append(_worker_solve(t))
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_worker_solve, t) for t in tasks]
                for fut in as_completed(futs):
                    results.append(fut.result())

    results.sort(key=lambda r: r["n"])
    summary_path = pf.DOCS_DIR / f"{args.tag}_summary.json"
    summary_path.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")

    n_wins, tot_ds = pf.export_final_csv(min_dl=Decimal("1e-9"))
    real_wins = [r for r in results if r["status"] == "REAL_WIN"]
    print(
        f"\n=== H125 ({args.tag}) DONE: {len(real_wins)}/{len(results)} REAL_WINs in run | "
        f"exported final.csv={n_wins} wins, total_ds_L2n={tot_ds:+.6e} ===",
        flush=True,
    )


if __name__ == "__main__":
    main()
