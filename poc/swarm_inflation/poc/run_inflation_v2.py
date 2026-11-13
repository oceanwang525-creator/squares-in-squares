"""Idea 2 orchestrator: exact-capacity seeds_v2 -> nvcc sm_89 CUDA inflation/L-min -> LP/SLP certify."""
from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_var] = "1"

import argparse
import csv
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

POC_DIR = Path(__file__).resolve().parent
SWARM_DIR = POC_DIR.parent
REPO = SWARM_DIR.parents[1]
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import polish_and_certify as pc  # noqa: E402
import seeds_v2  # noqa: E402

CUDA_SRC = POC_DIR / "sat_lmin.cu"
CUDA_BIN = Path("/tmp/sat_lmin_cuda")


def ensure_cuda_binary() -> Path:
    """Compile sat_lmin.cu to /tmp/sat_lmin_cuda if missing or stale."""
    if not CUDA_BIN.exists() or CUDA_SRC.stat().st_mtime > CUDA_BIN.stat().st_mtime:
        cmd = [
            "/usr/local/cuda/bin/nvcc",
            "-O3",
            "-arch=sm_89",
            str(CUDA_SRC),
            "-o",
            str(CUDA_BIN),
        ]
        print("[cuda] compiling:", " ".join(cmd), flush=True)
        subprocess.check_call(cmd)
    return CUDA_BIN


def load_targets(n_list: list[int]) -> dict[int, tuple[float, float]]:
    """Return {n: (L_frontier, L_incumbent)} from per_n_gap.csv & merged.csv."""
    gap_csv = REPO / "poc" / "frontier_nov12" / "docs" / "per_n_gap.csv"
    frontier: dict[int, float] = {}
    with open(gap_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ni = int(row["n"])
            if ni in n_list:
                frontier[ni] = float(row["L_frontier"])
    out: dict[int, tuple[float, float]] = {}
    for n in n_list:
        inc_L = float(pc.load_incumbent_L(n))
        f_L = frontier.get(n, inc_L - 0.02)
        out[n] = (f_L, inc_L)
    return out


def run_cuda_batch(
    pos_init: np.ndarray,
    th_init: np.ndarray,
    box_L: np.ndarray,
    use_elastic_L: int = 1,
    fast_mode: int = 1,
    seed: int = 1,
) -> dict[str, np.ndarray]:
    """Run compiled shared-memory CUDA solver on batch (B, N)."""
    bin_path = ensure_cuda_binary()
    B, N, _ = pos_init.shape
    in_bin = Path(f"/tmp/sat_v2_in_{os.getpid()}.bin")
    out_bin = Path(f"/tmp/sat_v2_out_{os.getpid()}.bin")
    try:
        with open(in_bin, "wb") as f:
            f.write(struct.pack("<iiiiI", int(B), int(N), int(use_elastic_L), int(fast_mode), int(seed) & 0xFFFFFFFF))
            f.write(np.ascontiguousarray(pos_init[:, :, 0], dtype=np.float32).tobytes())
            f.write(np.ascontiguousarray(pos_init[:, :, 1], dtype=np.float32).tobytes())
            f.write(np.ascontiguousarray(th_init, dtype=np.float32).tobytes())
            f.write(np.ascontiguousarray(box_L, dtype=np.float32).tobytes())

        subprocess.check_call([str(bin_path), "--in", str(in_bin), "--out", str(out_bin)])
        raw = np.fromfile(out_bin, dtype=np.float64)
        bn = B * N
        ox = raw[0:bn].reshape(B, N)
        oy = raw[bn : 2 * bn].reshape(B, N)
        oth = raw[2 * bn : 3 * bn].reshape(B, N)
        L_cert = raw[3 * bn : 3 * bn + B]
        lam = raw[3 * bn + B : 3 * bn + 2 * B]
        pen = raw[3 * bn + 2 * B : 3 * bn + 3 * B]
        return {
            "pos": np.stack([ox, oy], axis=-1),
            "th": oth,
            "L_cert": L_cert,
            "lam": lam,
            "pen": pen,
        }
    finally:
        if in_bin.exists():
            in_bin.unlink()
        if out_bin.exists():
            out_bin.unlink()


def select_diverse_top_indices(L_cert: np.ndarray, meta: list[str], top_k: int) -> list[int]:
    """Select overall best candidates plus top representatives per seed family."""
    order = np.argsort(L_cert)
    chosen: list[int] = []
    seen: set[int] = set()
    # First guarantee top-2 from each distinct family
    per_fam: dict[str, int] = {}
    for idx in order:
        i = int(idx)
        fam = meta[i].split("(")[0] if i < len(meta) else "unknown"
        if per_fam.get(fam, 0) < 2:
            chosen.append(i)
            seen.add(i)
            per_fam[fam] = per_fam.get(fam, 0) + 1
    # Fill remainder by global L_cert order
    for idx in order:
        if len(chosen) >= top_k:
            break
        i = int(idx)
        if i not in seen:
            chosen.append(i)
            seen.add(i)
    return chosen[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-list", required=True, help="comma-separated n values")
    parser.add_argument("--batch", type=int, default=336)
    parser.add_argument("--top-k", type=int, default=28)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--fast-mode", type=int, default=1)
    parser.add_argument("--elastic-L", type=int, default=1)
    args = parser.parse_args()

    n_list = [int(x) for x in args.n_list.split(",") if x.strip()]
    targets = load_targets(n_list)
    docs_dir = SWARM_DIR / "docs"
    data_dir = SWARM_DIR / "data"
    docs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    ensure_cuda_binary()
    batch_results = []

    for n in n_list:
        t0 = time.time()
        f_L, inc_L = targets[n]
        L_targets = [
            round(f_L + 0.001, 8),
            round(f_L + 0.35 * (inc_L - f_L), 8),
            round(f_L + 0.75 * (inc_L - f_L), 8),
            round(inc_L + 0.012, 8),
        ]
        print(f"=== Idea 2 n={n} batch={args.batch} f_L={f_L:.6f} inc_L={inc_L:.6f} ===", flush=True)

        t_s0 = time.time()
        pos_init, th_init, box_L, meta = seeds_v2.build_seed_batch_v2(
            n=n, L_targets=L_targets, batch_size=args.batch, rng_seed=args.seed * 1009 + n
        )
        t_seed = time.time() - t_s0

        t_g0 = time.time()
        cuda_out = run_cuda_batch(
            pos_init, th_init, box_L,
            use_elastic_L=args.elastic_L, fast_mode=args.fast_mode, seed=args.seed * 1009 + n,
        )
        t_gpu = time.time() - t_g0

        L_cert = cuda_out["L_cert"]
        lam = cuda_out["lam"]
        pen = cuda_out["pen"]
        order = np.argsort(L_cert)

        fam_stats: dict[str, dict[str, float | int]] = {}
        for idx, m in enumerate(meta):
            fam = m.split("(")[0]
            st = fam_stats.setdefault(fam, {"count": 0, "best_L_cert": 1e9, "best_lam": 1e9, "best_pen": 1e9})
            st["count"] = int(st["count"]) + 1
            if float(L_cert[idx]) < float(st["best_L_cert"]):
                st["best_L_cert"] = float(L_cert[idx])
                st["best_lam"] = float(lam[idx])
                st["best_pen"] = float(pen[idx])

        bi = int(order[0])
        print(
            f"[cuda_v2] n={n} seed={t_seed:.2f}s gpu={t_gpu:.2f}s | "
            f"best_L_cert={L_cert[bi]:.6f} lam={lam[bi]:.6f} pen={pen[bi]:.2e} | {meta[bi]}",
            flush=True,
        )
        for fam, st in sorted(fam_stats.items(), key=lambda kv: kv[1]["best_L_cert"]):
            print(f"    fam {fam:20s}: L_cert={st['best_L_cert']:.6f} lam={st['best_lam']:.5f} pen={st['best_pen']:.2e}", flush=True)

        gpu_json = {
            "n": n,
            "batch": args.batch,
            "fast_mode": args.fast_mode,
            "elastic_L": args.elastic_L,
            "L_targets": L_targets,
            "seed_seconds": round(t_seed, 3),
            "gpu_seconds": round(t_gpu, 3),
            "best_L_cert": float(L_cert[bi]),
            "best_lam": float(lam[bi]),
            "best_pen": float(pen[bi]),
            "best_meta": meta[bi],
            "family_stats": fam_stats,
            "top_L_certs": [float(L_cert[i]) for i in order[:10]],
        }
        (docs_dir / f"gpu_v2_n{n}_s{args.seed}.json").write_text(json.dumps(gpu_json, indent=2), encoding="utf-8")

        sel = select_diverse_top_indices(L_cert, meta, args.top_k)
        win_csv = data_dir / f"win_n{n}.csv"
        res = pc.polish_candidates(
            n=n,
            pos_batch=cuda_out["pos"][sel],
            th_batch=cuda_out["th"][sel],
            meta=[meta[i] for i in sel],
            out_win_csv=win_csv,
            lp_rounds=10,
            slp_rounds=10,
            slp_top_k=8,
        )
        elapsed = time.time() - t0
        res["elapsed_seconds"] = round(elapsed, 2)
        res["seed_seconds"] = round(t_seed, 2)
        res["gpu_seconds"] = round(t_gpu, 2)
        (docs_dir / f"run_v2_n{n}_s{args.seed}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(
            f"=== Finished v2 n={n} in {elapsed:.2f}s | inc={res['inc_L'][:16]} "
            f"best={str(res['best_cert_L'])[:16]} win={res['win']} ===",
            flush=True,
        )
        batch_results.append(
            {
                "n": n,
                "inc_L": res["inc_L"],
                "best_cert_L": res["best_cert_L"],
                "win": res["win"],
                "gpu_best_L_cert": float(L_cert[bi]),
                "elapsed_seconds": round(elapsed, 2),
            }
        )

    (docs_dir / f"batch_summary_v2_s{args.seed}.json").write_text(
        json.dumps(batch_results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
