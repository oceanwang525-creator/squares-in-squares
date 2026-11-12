"""Orchestrator for from-scratch GPU inflation -> LP/SLP polish -> dual certification.

Runs under /mnt/e/squares-in-squares/.venv/bin/python, invokes gpu_worker.py under
CUDA-enabled PyTorch Python, then runs SciPy LP/SLP polish and squares.core certification.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

POC_DIR = Path(__file__).resolve().parent
SWARM_DIR = POC_DIR.parent
REPO = SWARM_DIR.parents[1]
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import polish_and_certify as pc  # noqa: E402

CUDA_PYTHON = "/home/lucas/miniconda3/envs/vllm-gemma4/bin/python"


def get_target_bounds(n: int) -> tuple[float, float]:
    """Read target box L_ours and L_frontier only (never reads incumbent coordinates)."""
    gap_df = pd.read_csv(REPO / "poc/frontier_nov12/docs/per_n_gap.csv").set_index("n")
    L_front = float(gap_df.loc[n, "L_frontier"])
    inc_L = float(pc.load_incumbent_L(n))
    return inc_L, L_front


def run_for_n(
    n: int,
    batch: int,
    top_k: int,
    seed: int,
    preset: str,
    cuda_python: str,
    tmp_dir: Path,
) -> dict:
    t0 = time.time()
    L_ours, L_front = get_target_bounds(n)
    gap = max(1e-4, L_ours - L_front)
    # Probe 4 L_target levels spanning frontier -> near-incumbent
    L_targets = [
        L_front + 0.001,
        L_front + 0.35 * gap,
        L_front + 0.70 * gap,
        L_ours - min(0.002, 0.15 * gap),
    ]
    L_str = ",".join(f"{x:.8f}" for x in L_targets)
    npz_path = tmp_dir / f"cand_n{n}_s{seed}.npz"
    gpu_sum_path = SWARM_DIR / "docs" / f"gpu_n{n}_s{seed}.json"

    cmd = [
        cuda_python,
        str(POC_DIR / "gpu_worker.py"),
        "--n", str(n),
        "--L-targets", L_str,
        "--batch", str(batch),
        "--top-k", str(top_k),
        "--seed", str(seed),
        "--preset", preset,
        "--out-npz", str(npz_path),
        "--summary-json", str(gpu_sum_path),
    ]
    proc = subprocess.run(cmd, check=True, text=True)
    _ = proc

    data = np.load(npz_path, allow_pickle=True)
    pos = np.asarray(data["pos"], dtype=np.float64)
    th = np.asarray(data["th"], dtype=np.float64)
    meta = [str(x) for x in data["meta"]]

    win_csv = SWARM_DIR / "data" / f"win_n{n}.csv"
    res = pc.polish_candidates(
        n=n,
        pos_batch=pos,
        th_batch=th,
        meta=meta,
        out_win_csv=win_csv,
        lp_rounds=8,
        slp_rounds=8,
    )
    gpu_sum = json.loads(gpu_sum_path.read_text(encoding="utf-8"))
    res["gpu_summary"] = gpu_sum
    res["L_frontier"] = L_front
    res["total_seconds"] = round(time.time() - t0, 2)

    out_json = SWARM_DIR / "docs" / f"run_n{n}_s{seed}.json"
    out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")

    # Remove transient >1MB .npz if any
    npz_path.unlink(missing_ok=True)
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-list", default="367,910,971")
    parser.add_argument("--batch", type=int, default=160)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preset", choices=["fast", "deep"], default="fast")
    parser.add_argument("--cuda-python", default=CUDA_PYTHON)
    args = parser.parse_args()

    tmp_dir = Path("/tmp/swarm_inflation")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ns = [int(x) for x in args.n_list.split(",") if x.strip()]
    all_res = []
    for n in ns:
        print(f"=== Running inflation for n={n} ===", flush=True)
        res = run_for_n(
            n=n,
            batch=args.batch,
            top_k=args.top_k,
            seed=args.seed,
            preset=args.preset,
            cuda_python=args.cuda_python,
            tmp_dir=tmp_dir,
        )
        all_res.append(res)
        print(
            f"=== Finished n={n} in {res['total_seconds']}s | inc={res['inc_L'][:14]} "
            f"best={str(res['best_cert_L'])[:14]} win={res['win']} ===",
            flush=True,
        )

    summary_path = SWARM_DIR / "docs" / f"batch_summary_s{args.seed}.json"
    summary_path.write_text(
        json.dumps(
            [
                {
                    "n": r["n"],
                    "inc_L": r["inc_L"],
                    "L_frontier": r["L_frontier"],
                    "best_cert_L": r["best_cert_L"],
                    "win": r["win"],
                    "gpu_best_L_cert": r["gpu_summary"]["best_L_cert"],
                    "gpu_best_lam": r["gpu_summary"]["best_lam"],
                    "gpu_best_meta": r["gpu_summary"]["best_meta"],
                    "total_seconds": r["total_seconds"],
                }
                for r in all_res
            ],
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
