"""Official-metric check of one saved packing: python certify.py n dL"""
import sys, numpy as np
from decimal import Decimal
from squares import core
n, dL = int(sys.argv[1]), sys.argv[2]
sel = np.load(f"/Users/lucaszhang/squares-in-squares/poc/multi_angle_band/docs/best_n{n}_dL{dL}.npy")[:n]
assert len(sel) == n
x0, y0 = sel[:,0].min(), sel[:,1].min()
F = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0000002  # centre dilation opens float contact gaps
rows = [dict(square_id=str(i), x=repr(float(x*F)), y=repr(float(y*F)), angle=repr(float(a))) for i,(x,y,a) in enumerate(sel)]
L = core.certify(n, rows)
rt = core.certify(n, core.kaggle_roundtrip({n: rows})[n])
print(n, "dL", dL, "official L literal", L[:14], "pandas", rt[:14])
