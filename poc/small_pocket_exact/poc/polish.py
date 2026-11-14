"""Exact polish + independent validation of a candidate packing.

* `pslp_np.squeeze` (the repo's sequential-LP squeeze, `poc/gadget_squeeze/poc/`)
  shrinks f in a trust region with exact SAT linearisation;
* `satb.report` re-checks the polished state with the *full* pair list and an
  independent min-over-4-axes depth (no branch sharing with the optimiser).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'gadget_squeeze', 'poc'))
sys.path.insert(0, HERE)

import pslp_np                      # noqa: E402
import satb                         # noqa: E402


def validate(q, a, b):
    q2, W, Hh = satb.tighten(q)
    f = max(W - a, Hh - b)
    dep, wall = satb.report(q2, a + f + 1e-12, b + f + 1e-12)
    return f, dep, wall, q2


def polish(q, a, b, max_it=300, f_extra=1e-9):
    """SLP-squeeze one candidate, then validate.  Returns dict."""
    q2, W, Hh = satb.tighten(q)
    f0 = max(W - a, Hh - b) + f_extra
    X = q2[:, 0].copy(); Y = q2[:, 1].copy(); T = q2[:, 2].copy()
    f, X, Y, T, v, it = pslp_np.squeeze(X, Y, T, float(a), float(b), f0, max_it=max_it)
    qp = np.c_[X, Y, T]
    fp, dep, wall, qt = validate(qp, a, b)
    f0p, dep0, wall0, _ = validate(q2, a, b)
    if fp > f0p:                      # never return worse than the input
        return dict(f=f0p, q=q2, violation=float(max(dep0, wall0)), slp_f=f, it=it)
    return dict(f=fp, q=qt, violation=float(max(dep, wall)), slp_f=f, it=it)
