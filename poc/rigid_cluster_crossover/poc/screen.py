#!/usr/bin/env python3
"""Screen parent packings for replaceable contact-connected clusters."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "poc/auto_union/poc"))
import union  # noqa: E402

TARGETS = (803, 791, 908)
ALTERNATES = {
    803: ["poc/external_union_nov14/final.csv"],
    791: ["poc/public_watch_nov14/final.csv"],
    908: ["poc/swarm_watchxplant/final.csv"],
}
CONTACT_GAP = 0.05
MIN_LARGE_COMPONENT = 20


def coordinates(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [[float(row["x"]), float(row["y"]), float(row["angle"])] for row in rows],
        dtype=np.float64,
    )


def pair_gaps(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = data[:, :2]
    theta = data[:, 2]
    pairs = cKDTree(xy).query_pairs(1.55, output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    dx = xy[j, 0] - xy[i, 0]
    dy = xy[j, 1] - xy[i, 1]
    ci, si = np.cos(theta[i]), np.sin(theta[i])
    cj, sj = np.cos(theta[j]), np.sin(theta[j])
    gaps = []
    for ax in ((ci, -si, cj, -sj), (si, ci, sj, cj)):
        projection = dx * ax[0] + dy * ax[1] + dx * ax[2] + dy * ax[3]
        cth = ci * cj + si * sj
        sth = ci * sj - si * cj
        radius = 0.5 * (1.0 + np.abs(cth) + np.abs(sth))
        gaps.append(np.abs(projection) - radius)
    return pairs, np.max(np.asarray(gaps), axis=0)


def components(rows: list[dict]) -> list[dict]:
    data = coordinates(rows)
    n = len(data)
    pairs, gaps = pair_gaps(data)
    keep = gaps <= CONTACT_GAP
    adjacency = csr_matrix(
        (np.ones(int(keep.sum())), (pairs[keep, 0], pairs[keep, 1])),
        shape=(n, n),
    )
    adjacency = adjacency + adjacency.T
    _, labels = connected_components(adjacency, directed=False)
    result = []
    half = 0.5 * (np.abs(np.cos(data[:, 2])) + np.abs(np.sin(data[:, 2])))
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        result.append(
            {
                "size": int(len(idx)),
                "bbox": [
                    float(np.min(data[idx, 0] - half[idx])),
                    float(np.min(data[idx, 1] - half[idx])),
                    float(np.max(data[idx, 0] + half[idx])),
                    float(np.max(data[idx, 1] + half[idx])),
                ],
            }
        )
    return sorted(result, key=lambda item: (-item["size"], item["bbox"]))


def relationship(candidate: list[dict], incumbent: list[dict]) -> str:
    """Classify whether two groups are effectively the same coordinate parent."""
    if len(candidate) != len(incumbent):
        return "not_comparable"
    a = coordinates(candidate)
    b = coordinates(incumbent)
    return "identical_rows" if np.allclose(a, b, atol=1e-12, rtol=0) else "distinct"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "poc/rigid_cluster_crossover/docs/screen.json",
    )
    args = parser.parse_args()

    incumbent, _ = union.load(union.BASE)
    report = {"contact_gap": CONTACT_GAP, "targets": {}}
    for n in TARGETS:
        row = {"incumbent": components(incumbent[n]), "donors": {}}
        row["has_large_cluster"] = row["incumbent"][0]["size"] >= MIN_LARGE_COMPONENT
        for donor_n in (n - 1, n + 1):
            comps = components(incumbent[donor_n])
            row["donors"][str(donor_n)] = {
                "source": "canonical-adjacent",
                "largest": comps[0]["size"],
                "components": comps[:12],
            }
        for relpath in ALTERNATES[n]:
            candidate, _ = union.load(ROOT / relpath)
            comps = components(candidate[n])
            row["donors"][relpath] = {
                "source": "same-n-final",
                "relationship": relationship(candidate[n], incumbent[n]),
                "largest": comps[0]["size"],
                "components": comps[:12],
            }
        report["targets"][str(n)] = row

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(
        {
            n: {
                "incumbent_largest": row["incumbent"][0]["size"],
                "donor_largest": max(d["largest"] for d in row["donors"].values()),
                "has_large_cluster": row["has_large_cluster"],
            }
            for n, row in report["targets"].items()
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
