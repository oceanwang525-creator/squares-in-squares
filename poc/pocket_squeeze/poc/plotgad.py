import json, sys, math, collections
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
lib = json.load(open('gadlib.json'))
want = [tuple(int(v) for v in s.split(',')) for s in sys.argv[2:]]
best = {}
for i, g in enumerate(lib):
    if g['whole']: continue
    for (w, h) in ((g['w'], g['h']),):
        k = (min(w, h), max(w, h), g['g']); f = max(g['fx'], g['fy'])
        if k not in best or f < best[k][0]: best[k] = (f, i)
fig, axes = plt.subplots(1, len(want), figsize=(7 * len(want), 8))
if len(want) == 1: axes = [axes]
for ax, k in zip(axes, want):
    f, i = best[k]; g = lib[i]; a = np.array(g['sq'])
    W = g['w'] + g['fx']; H = g['h'] + g['fy']
    t = np.degrees(np.mod(a[:, 2], np.pi / 2)); fold = np.minimum(t, 90 - t)
    for (x, y, th), d in zip(a, fold):
        c, s = math.cos(th), math.sin(th)
        P = [(x + sx * c - sy * s, y + sx * s + sy * c) for sx, sy in [(-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)]]
        col = 'lightsteelblue' if d < 1e-4 else plt.cm.autumn(min(d, 45) / 45)
        ax.add_patch(Polygon(P, closed=True, fc=col, ec='k', lw=.5))
        if d >= 1e-4: ax.text(x, y, f'{t[np.argmin(np.abs(a[:,0]-x)+np.abs(a[:,1]-y))]:.1f}', ha='center', va='center', fontsize=5)
    ax.set_xlim(-0.1, W + 0.1); ax.set_ylim(-0.1, H + 0.1); ax.set_aspect('equal')
    ax.set_title(f'(w,h,g)={k} f={f:.6f} from n={g["src"]}: {W:.4f} x {H:.4f}, {len(a)} squares, {int((fold>=1e-4).sum())} tilted')
    ax.set_xticks(range(0, int(W) + 2)); ax.set_yticks(range(0, int(H) + 2)); ax.grid(alpha=.3)
    c = collections.Counter(np.round(t[fold >= 1e-4], 2))
    print(k, 'f=%.6f' % f, 'src', g['src'], 'tilted angle counts:', sorted(c.items(), key=lambda kv: -kv[1])[:12])
fig.savefig(sys.argv[1], dpi=75, bbox_inches='tight')
