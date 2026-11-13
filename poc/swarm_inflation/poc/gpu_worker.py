"""GPU inflation worker executed under CUDA-enabled PyTorch Python."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

POC_DIR = Path(__file__).resolve().parent
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import sat_cuda  # noqa: E402
import seeds  # noqa: E402


STAGES_PRESETS = {
    "fast": [
        (0.880, 100, 0.018),
        (0.905, 100, 0.015),
        (0.928, 120, 0.013),
        (0.945, 120, 0.011),
        (0.960, 140, 0.008),
        (0.972, 150, 0.006),
        (0.982, 160, 0.004),
        (0.990, 180, 0.0022),
        (0.995, 200, 0.0012),
        (0.998, 200, 0.0006),
        (0.9995, 200, 0.0003),
        (1.000, 260, 0.00015),
    ],
    "deep": [
        (0.880, 140, 0.018),
        (0.895, 140, 0.016),
        (0.910, 140, 0.015),
        (0.925, 160, 0.013),
        (0.940, 160, 0.011),
        (0.952, 180, 0.009),
        (0.962, 180, 0.0075),
        (0.971, 200, 0.0055),
        (0.979, 200, 0.0040),
        (0.986, 220, 0.0028),
        (0.991, 220, 0.0018),
        (0.995, 240, 0.0011),
        (0.9975, 240, 0.0006),
        (0.999, 260, 0.0003),
        (0.9997, 260, 0.00018),
        (1.000, 320, 0.0001),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--L-targets", required=True, help="comma-separated L_target values")
    parser.add_argument("--batch", type=int, default=160)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preset", choices=list(STAGES_PRESETS), default="fast")
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--summary-json", default="")
    args = parser.parse_args()

    t0 = time.time()
    assert torch.cuda.is_available(), "CUDA required for gpu_worker.py"
    device = torch.device("cuda")
    free_b, total_b = torch.cuda.mem_get_info()
    print(
        f"[gpu_worker] n={args.n} batch={args.batch} preset={args.preset} "
        f"GPU={torch.cuda.get_device_name(0)} free={free_b / 1e9:.2f}GB",
        flush=True,
    )

    L_targets = [float(x) for x in args.L_targets.split(",") if x.strip()]
    pos_init, th_init, box_L, meta = seeds.build_seed_batch(
        n=args.n, L_targets=L_targets, batch_size=args.batch, rng_seed=args.seed
    )
    t_seed = time.time() - t0

    t1 = time.time()
    out = sat_cuda.inflate_batch(
        init_pos=pos_init,
        init_th=th_init,
        box_L=box_L,
        stages=STAGES_PRESETS[args.preset],
        seed=args.seed,
        device=device,
        verbose=True,
    )
    torch.cuda.synchronize()
    t_gpu = time.time() - t1

    L_cert = out["L_cert"]
    lam = out["lam"]
    pen = out["pen_exact"]

    # Summarize best per seed family
    fam_stats: dict[str, dict] = {}
    for b, m in enumerate(meta):
        fam = m.split("(")[0]
        rec = fam_stats.setdefault(fam, {"count": 0, "best_L_cert": 1e9, "best_lam": 1e9, "best_pen": 1e9})
        rec["count"] += 1
        rec["best_L_cert"] = min(rec["best_L_cert"], float(L_cert[b]))
        rec["best_lam"] = min(rec["best_lam"], float(lam[b]))
        rec["best_pen"] = min(rec["best_pen"], float(pen[b]))

    order = np.argsort(L_cert)[: min(args.top_k, len(L_cert))]
    top_pos = out["pos"][order]
    top_th = out["th"][order]
    top_meta = [meta[int(i)] for i in order]

    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_npz,
        n=args.n,
        pos=top_pos,
        th=top_th,
        meta=np.array(top_meta, dtype=object),
        L_cert=L_cert[order],
        lam=lam[order],
        pen=pen[order],
    )

    summary = {
        "n": args.n,
        "batch": args.batch,
        "preset": args.preset,
        "L_targets": L_targets,
        "seed_seconds": round(t_seed, 2),
        "gpu_seconds": round(t_gpu, 2),
        "best_L_cert": float(L_cert[order[0]]),
        "best_lam": float(lam[order[0]]),
        "best_pen": float(pen[order[0]]),
        "best_meta": top_meta[0],
        "family_stats": fam_stats,
        "top_L_certs": [float(L_cert[i]) for i in order[:8]],
        "top_lams": [float(lam[i]) for i in order[:8]],
    }
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    print(
        f"[gpu_worker] done n={args.n} in {t_gpu:.2f}s | best_L_cert={summary['best_L_cert']:.6f} "
        f"(lam={summary['best_lam']:.6f}, pen={summary['best_pen']:.3e}) | {summary['best_meta']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
