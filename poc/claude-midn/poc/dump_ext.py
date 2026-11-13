#!/usr/bin/env python3
"""Dump per-square axis extents of a full submission as raw float64 for crop_scan.

usage: dump_ext.py SUBMISSION.csv OUT.bin
layout: for n=1..1000, n records of (xmin, xmax, ymin, ymax), translated so the
packing's bounding box starts at (0, 0).
"""
import sys
import numpy as np
import pandas as pd

d = pd.read_csv(sys.argv[1])
c, s = np.cos(d.angle.values), np.sin(d.angle.values)
e = 0.5 * (np.abs(c) + np.abs(s))
out = []
for n, g in d.groupby("n"):
    idx = g.index.values
    x, y, ee = d.x.values[idx], d.y.values[idx], e[idx]
    x0, y0 = (x - ee).min(), (y - ee).min()
    out.append(np.stack([x - ee - x0, x + ee - x0, y - ee - y0, y + ee - y0], 1))
np.concatenate(out).astype("<f8").tofile(sys.argv[2])
