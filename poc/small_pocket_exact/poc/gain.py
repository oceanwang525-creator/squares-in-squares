"""Evaluate the staircase-DP gain of a gadget at a *specific* realised f.

`payline.py` bisects for the threshold; this asks the concrete question
"is my packing of (w,h,+g) with realised f good enough to improve any tracked
row?" -- the only thing that matters once a real packing exists.

  gain.py 3,2,1:0.7071067811865476 4,7,3:0.79
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import payline                                      # noqa: E402
import dpx                                          # noqa: E402


def main():
    items, L, base_sides = payline.load()
    for spec in sys.argv[1:]:
        sh, fs = spec.split(':')
        w, h, g = (int(v) for v in sh.split(','))
        f = float(fs)
        tot, hits = payline.gain_at(items, L, base_sides, w, h, g, f)
        print('(%d,%d,+%d) at f=%.12f: total gain %.6e over %d n' % (w, h, g, f, tot, len(hits)))
        for sg, n, K, ff, d in hits[:12]:
            print('   n=%-4d K=%d j=%-2d ours %.9f -> %.9f  (dL %+.3e, gain %.3e)'
                  % (n, K, n - K * K, L[n], K + ff, -d, sg))
    tot, hits = 0, []


if __name__ == '__main__':
    main()
