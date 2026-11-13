#!/usr/bin/env python3
"""Dependency-free PNG rendering of one packing (numpy + zlib only).

usage: render.py SUBMISSION.csv N OUT.png [px_per_unit]
upright squares are light grey, tilted ones are coloured by angle mod 90.
"""
import struct
import sys
import zlib

import numpy as np
import pandas as pd


def write_png(path, img):
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def render(x, y, a, out, s=36):
    c, sn = np.cos(a), np.sin(a)
    e = 0.5 * (np.abs(c) + np.abs(sn))
    x0, y0 = (x - e).min(), (y - e).min()
    L = max((x + e).max() - x0, (y + e).max() - y0)
    W = int(np.ceil(L * s)) + 2
    img = np.full((W, W, 3), 255, np.uint8)
    for i in range(len(x)):
        cx, cy = (x[i] - x0) * s, (y[i] - y0) * s
        r = int(np.ceil(0.72 * s))
        xs = np.arange(max(0, int(cx) - r), min(W, int(cx) + r + 1))
        ys = np.arange(max(0, int(cy) - r), min(W, int(cy) + r + 1))
        X, Y = np.meshgrid(xs + 0.5 - cx, ys + 0.5 - cy)
        u = (X * c[i] + Y * sn[i]) / s
        v = (-X * sn[i] + Y * c[i]) / s
        m = np.maximum(np.abs(u), np.abs(v))
        t = np.degrees(a[i]) % 90
        t = min(t, 90 - t)
        if t < 0.01:
            col = (215, 215, 215)
        else:
            f = t / 45
            col = (int(255 * f), int(120 * (1 - f) + 60), int(255 * (1 - f)))
        sub = img[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]
        sub[m <= 0.5] = col
        sub[(m <= 0.5) & (m > 0.5 - 1.2 / s)] = (40, 40, 40)
    write_png(out, img[::-1].copy())
    return L


if __name__ == "__main__":
    d = pd.read_csv(sys.argv[1])
    g = d[d.n == int(sys.argv[2])]
    s = int(sys.argv[4]) if len(sys.argv) > 4 else 36
    print(render(g.x.values, g.y.values, g.angle.values, sys.argv[3], s))
