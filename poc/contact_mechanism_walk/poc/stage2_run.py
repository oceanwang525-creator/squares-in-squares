#!/usr/bin/env python3
"""Run the iterated mechanism gate from a certified group in a final CSV."""
from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MECHWALK_POC = ROOT / "poc" / "swarm_mechwalk" / "poc"
UNION_POC = ROOT / "poc" / "auto_union" / "poc"
for path in (MECHWALK_POC, UNION_POC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mechwalk_lib as ml  # noqa: E402
import run_gate  # noqa: E402
import union as auto_union  # noqa: E402


def load_group(path: Path, n: int) -> dict:
    literal, transported = auto_union.load(path)
    if n not in literal or n not in transported or len(literal[n]) != n:
        raise ValueError(f"{path} has no complete group n={n}")
    certified = auto_union.check((n, literal[n], transported[n]))
    if certified is None:
        raise ValueError(f"{path} group n={n} is not certified in both transports")
    rows = sorted(literal[n], key=lambda row: int(row["square_id"]))
    x = np.array([float(row["x"]) for row in rows], dtype=np.float64)
    y = np.array([float(row["y"]) for row in rows], dtype=np.float64)
    theta = np.array([float(row["angle"]) for row in rows], dtype=np.float64)
    x, y, theta, side = ml.normalize(x, y, theta)
    return {
        "x": x,
        "y": y,
        "th": theta,
        "L": side,
        "L_dec": max(Decimal(certified[0]), Decimal(certified[1])),
        "source": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--source-final", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--beam", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    incumbent = load_group(args.source_final, args.n)
    print(
        f"loaded n={args.n} L={incumbent['L_dec']} source={args.source_final}",
        flush=True,
    )
    summary, win = run_gate.solve_target(
        args.n,
        incumbent,
        n_rounds=args.rounds,
        beam_width=args.beam,
        top_k_polish=args.top_k,
        n_threads=min(4, max(1, args.threads)),
    )
    payload = {"summary": summary, "win": win}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
