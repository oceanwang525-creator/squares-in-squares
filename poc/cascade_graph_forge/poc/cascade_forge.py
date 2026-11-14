#!/usr/bin/env python3
"""Graph-wide L-border cascade runner using proven H78 solvers."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import sys
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

REPO = Path(__file__).resolve().parents[3]
HOMO_POC = REPO / "poc" / "swarm_homocasc2" / "poc"
if str(HOMO_POC) not in sys.path:
    sys.path.insert(0, str(HOMO_POC))

import cascade_engine2 as ce  # noqa: E402
import homocasc2_lib as hc  # noqa: E402

HERE = Path(__file__).resolve().parent
POC_DIR = HERE.parent
DATA_DIR = POC_DIR / "data"
DOCS_DIR = POC_DIR / "docs"
FINAL_CSV = POC_DIR / "final.csv"
MERGED_CSV = REPO / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Multiple graph edges run concurrently and may rebuild final.csv together.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def rebuild_final() -> None:
    groups = {}
    for p in sorted((POC_DIR / "data").glob("win_n*.json")):
        d = json.loads(p.read_text())
        groups[int(d["n"])] = d["rows"]
    text = hc.auto_union.final_text(groups)
    _atomic(FINAL_CSV, text.encode())
    os.chmod(FINAL_CSV, 0o644)


def save_win(n: int, cert: dict, label: str) -> bool:
    if not cert.get("is_real_win"):
        return False
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lock = DATA_DIR / ".win_lock"
    with lock.open("w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        out = DATA_DIR / f"win_n{n}.json"
        if out.exists():
            old = json.loads(out.read_text())
            old_l = max(Decimal(old["L_lit"]), Decimal(old["L_pd"]))
            new_l = max(Decimal(cert["L_lit"]), Decimal(cert["L_pd"]))
            if old_l <= new_l:
                return False
        payload = {
            "n": n,
            "label": label,
            "L_lit": cert["L_lit"],
            "L_pd": cert["L_pd"],
            "L_kr": cert.get("L_kr", cert["L_pd"]),
            "L_float": cert["L_float"],
            "dL": cert["dL"],
            "rows": cert["rows"],
        }
        _atomic(out, json.dumps(payload, indent=2).encode())
        os.chmod(out, 0o644)
        rebuild_final()
        return True


def prioritize(seeds: list, max_keep: int) -> list:
    def score(seed):
        label, x, y, th, _, _ = seed
        pen, _ = hc.cpp_eval_sat_pen_bbox(x, y, th)
        tier = (
            0 if "clean" in label else
            1 if "ext" in label else
            2 if "relocRep" in label else
            3 if "xcross" in label else
            4 if "seamSurg" in label else
            5 if "reloc" in label else
            6
        )
        return tier, pen

    return sorted(seeds, key=score)[:max_keep]


def load_entry(n: int) -> dict:
    """Load the current canonical union group and build float arrays."""
    rows = []
    with MERGED_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if int(row["n"]) == n:
                rows.append(
                    {
                        "square_id": row["square_id"],
                        "x": hc._strip_s(row["x"]),
                        "y": hc._strip_s(row["y"]),
                        "angle": hc._strip_s(row["angle"]),
                    }
                )
    if len(rows) != n:
        raise ValueError(f"canonical group n={n} has {len(rows)} rows")
    cert = hc.certify_rows_against_frontier(n, rows, Decimal("999.0"))
    if cert is None:
        raise ValueError(f"canonical group n={n} failed dual certification")
    x, y, th, L = hc._rows_to_arrays(rows)
    return {
        "x": x,
        "y": y,
        "th": th,
        "L": L,
        "L_dec": max(Decimal(cert["L_lit"]), Decimal(cert["L_pd"])),
        "source": "merged.csv",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--donor", type=int, required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-seeds", type=int, default=24)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261114)
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    rebuild_final()

    # solve_target calls homocasc2_lib's hard-coded saver.  Redirect it into
    # this POC so no other POC is modified.
    hc.save_win_and_rebuild_final = save_win

    target = load_entry(args.target)
    donor = load_entry(args.donor)
    rng = np.random.default_rng(args.seed + 1000 * args.target + args.donor)

    seeds = ce.generate_peel_seeds(
        args.target,
        args.donor,
        donor["x"],
        donor["y"],
        donor["th"],
        donor["L"],
        target["L"],
        rng,
        layers=args.layers,
        max_prot=18,
        max_rem_tilt=8,
    )
    if args.layers == 1:
        seeds += ce.generate_interior_cross_seeds(
            args.target,
            args.donor,
            donor["x"],
            donor["y"],
            donor["th"],
            donor["L"],
            target["L"],
            max_seeds=8,
        )
    seeds = prioritize(seeds, args.max_seeds)
    print(
        json.dumps(
            {
                "target": args.target,
                "donor": args.donor,
                "target_L": str(target["L_dec"]),
                "target_source": target["source"],
                "donor_L": str(donor["L_dec"]),
                "donor_source": donor["source"],
                "seeds": len(seeds),
            },
            indent=2,
        ),
        flush=True,
    )

    result = ce.solve_target(
        args.target,
        seeds,
        target["L_dec"],
        threads=args.threads,
        rng=rng,
        n_seeds_per_spec=4,
        masked_iters=180,
        full_iters=220,
        top_polish=10,
        run_mechwalk=True,
    )
    result["donor"] = args.donor
    result["target_source"] = target["source"]
    result["donor_source"] = donor["source"]
    result.pop("best_pack", None)
    out = DOCS_DIR / f"edge_{args.donor}_to_{args.target}.json"
    _atomic(out, json.dumps(result, indent=2).encode())
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
