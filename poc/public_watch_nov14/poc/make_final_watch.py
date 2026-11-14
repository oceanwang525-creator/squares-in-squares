"""Export watcher rows that beat the incumbent into poc/<folder>/final.csv with the team converter.
usage: make_final_watch.py folder [exclude n,n,...]"""
import json, sys
from decimal import Decimal
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'repo/poc/auto_union/poc'))
import union as u
folder = sys.argv[1]; excl = {int(v) for v in sys.argv[2].split(',')} if len(sys.argv) > 2 and sys.argv[2] else set()
sides = {int(k): tuple(map(Decimal, v)) for k, v in json.loads((W / 'work/gate/sides.json').read_text())['sides'].items()}
best = json.loads((W / 'work/state/best.json').read_text())
groups = {}
for k, v in best.items():
    n = int(k)
    if n not in excl and Decimal(v['L']) < min(sides[n]):
        groups[n] = [dict(square_id=r['square_id'], x=r['x'].strip(), y=r['y'].strip(), angle=r['angle'].strip()) for r in v['rows']]
out = W / 'repo/poc' / folder / 'final.csv'; out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(u.final_text(groups)); print(folder, len(groups), 'groups', out.stat().st_size, 'bytes')
