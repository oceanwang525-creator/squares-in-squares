#!/usr/bin/env python3
"""Extract gadlib clean-cut sub-rectangles as pocket records (with x0,x1,y0,y1)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc" / "faultline_dp2" / "poc"))
import gadlib as gl  # noqa: E402


def main() -> None:
    src_csv = sys.argv[1]
    out_json = sys.argv[2]
    D = gl.load_all(src_csv)
    out = []
    for n in sorted(D):
        a, e, L = D[n]
        K = int(L + 1e-9)
        t = np.degrees(np.mod(a[:, 2], np.pi / 2))
        tilt = np.minimum(t, 90 - t) > 0.5
        if tilt.sum() == 0:
            continue
        cls = gl.clusters(a, tilt)
        seeds = [c for c in cls]
        for i in range(len(cls)):
            for j in range(i + 1, len(cls)):
                seeds.append(np.concatenate([cls[i], cls[j]]))
        seen = set()
        for cl in seeds:
            x0 = float((a[cl, 0] - e[cl]).min())
            x1 = float((a[cl, 0] + e[cl]).max())
            y0 = float((a[cl, 1] - e[cl]).min())
            y1 = float((a[cl, 1] + e[cl]).max())
            X0, X1, Y0, Y1, idx = gl.clean_rect(a, e, x0, x1, y0, y1)
            W, H = float(X1 - X0), float(Y1 - Y0)
            w, h = int(W + 1e-7), int(H + 1e-7)
            key = (round(float(X0), 4), round(float(Y0), 4), round(W, 4), round(H, 4))
            if key in seen:
                continue
            seen.add(key)
            g = int(len(idx) - w * h)
            if g <= 0 or w < 1 or h < 1:
                continue
            f = max(W - w, H - h)
            if f < 1.15:
                out.append({
                    "src": n,
                    "w": w,
                    "h": h,
                    "g": g,
                    "f": f,
                    "cnt": int(len(idx)),
                    "x0": float(X0),
                    "x1": float(X1),
                    "y0": float(Y0),
                    "y1": float(Y1),
                })
        if n > K * K and L - K < 1 - 1e-9:
            out.append({
                "src": n,
                "w": K,
                "h": K,
                "g": int(n - K * K),
                "f": float(L - K),
                "cnt": int(n),
                "x0": 0.0,
                "x1": float((a[:, 0] + e).max()),
                "y0": 0.0,
                "y1": float((a[:, 1] + e).max()),
            })
    Path(out_json).write_text(json.dumps(out, indent=0), encoding="utf-8")
    print(f"Wrote {len(out)} gadlib pocket records to {out_json}")


if __name__ == "__main__":
    main()
