"""Certify candidate rows with the team gate: cert_one.py cand.json [more.json]; each JSON = {n, L, sq: [[x,y,theta],..]}.
Writes work/cand_final.csv (team format) and prints literal/pandas L against the incumbent sides."""
import json, sys
from decimal import Decimal
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'repo/poc/auto_union/poc'))
import union as u
EPS = 2e-12
if __name__ == '__main__':
    sides = {int(k): tuple(map(Decimal, v)) for k, v in json.loads((W / 'work/gate/sides.json').read_text())['sides'].items()}
    groups = {}
    for p in sys.argv[1:]:
        d = json.loads(Path(p).read_text())
        d = d if 'sq' in d else next(iter(d.values()))
        a = np.array(d['sq']); n = len(a)
        e = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
        a[:, 0] -= (a[:, 0] - e).min(); a[:, 1] -= (a[:, 1] - e).min()
        groups[n] = [dict(square_id=str(i), x=repr(float(x) * (1 + EPS)), y=repr(float(y) * (1 + EPS)), angle=repr(float(t))) for i, (x, y, t) in enumerate(a)]
    out = W / 'work/cand_final.csv'; out.write_text(u.final_text(groups))
    a, b = u.load(out)
    for n in sorted(a):
        r = u.check((n, a[n], b[n]))
        if r is None: print(f'n={n}: INVALID'); continue
        lit, pd_ = map(Decimal, r)
        print(f'n={n}: literal {str(lit)[:20]} pandas {str(pd_)[:20]} | incumbent {str(sides[n][0])[:20]} / {str(sides[n][1])[:20]} | dL {float(max(sides[n]) - max(lit, pd_)):.3e} | strict win: {lit < sides[n][0] and pd_ < sides[n][1]}')
