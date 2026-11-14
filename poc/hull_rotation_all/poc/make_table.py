"""Merge the audit and the geometry diagnostics into one compact per-n table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audit", type=Path)
    ap.add_argument("tightness", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))["rows"]
    tight = json.loads(args.tightness.read_text(encoding="utf-8"))["rows"]
    fields = ["n", "L_official", "width_axis", "height_axis", "side_min",
              "theta_deg", "reduction", "reduction_hp", "kink_min",
              "kink_certified", "hull_vertices", "corner_gap"]
    rows = []
    for n in sorted(audit, key=int):
        a, t = audit[n], tight[n]
        rows.append(dict(
            n=n,
            L_official=f"{a['L_official']:.15f}",
            width_axis=f"{a['width_axis']:.15f}",
            height_axis=f"{a['height_axis']:.15f}",
            side_min=f"{a.get('hp', a)['side_min']:.15f}",
            theta_deg=f"{a.get('hp', a)['theta'] * 180 / 3.141592653589793:.6f}",
            reduction=f"{a['reduction']:.3e}",
            reduction_hp=("" if "hp" not in a else f"{a['hp']['reduction']:.3e}"),
            kink_min=f"{min(t['kink']):.6f}",
            kink_certified=int(bool(t["kink_certified"])),
            hull_vertices=a["hull_vertices"],
            corner_gap=f"{t['gap']:.6f}",
        ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
