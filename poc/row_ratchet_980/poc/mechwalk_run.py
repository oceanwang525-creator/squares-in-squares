"""Run the team's mechanism walk (poc/swarm_mechwalk) on one n from a given start, writing only to out_dir.
usage: mechwalk_run.py out_dir n [start.json with sq]"""
import json, os, sys, time
import multiprocessing as mp
from decimal import Decimal
from pathlib import Path
import numpy as np
ROOT = Path(os.environ['REPO_ROOT'])
for p in (ROOT / 'poc/swarm_mechwalk/poc', ROOT / 'poc/swarm_highd/poc', ROOT / 'poc/auto_union/poc', ROOT / 'poc/baseline/poc/src'):
    sys.path.insert(0, str(p))
import cpp_wrap                                    # noqa: E402
cpp_wrap.CPP_LIB = Path(os.environ['CPP_LIB'])
import mechwalk_lib as ml                          # noqa: E402
import run_gate                                    # noqa: E402
import union as auto_union                         # noqa: E402

if __name__ == '__main__':
    mp.set_start_method('fork')
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); n = int(sys.argv[2])
    cpp_wrap.get_cpp_lib()
    portfolio = ml.load_origin_main_portfolio([n])
    print('portfolio start', n, portfolio[n]['L'], portfolio[n].get('source'), flush=True)
    if len(sys.argv) > 3:
        d = json.loads(Path(sys.argv[3]).read_text()); d = d if 'sq' in d else next(iter(d.values()))
        a = np.array(d['sq']); e = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
        a[:, 0] -= (a[:, 0] - e).min(); a[:, 1] -= (a[:, 1] - e).min()
        rows = [dict(square_id=i, x=repr(float(x) * (1 + 2e-12)), y=repr(float(y) * (1 + 2e-12)), angle=repr(float(t))) for i, (x, y, t) in enumerate(a)]
        L_dec = Decimal(auto_union.certify(n, rows))
        if L_dec < portfolio[n]['L_dec']:
            x = np.array([float(r['x']) for r in rows]); y = np.array([float(r['y']) for r in rows]); th = np.array([float(r['angle']) for r in rows])
            x, y, th, Lf = ml.normalize(x, y, th)
            portfolio[n] = dict(x=x, y=y, th=th, L=Lf, L_dec=L_dec, source=sys.argv[3])
            print('using external start', L_dec, flush=True)
    t = time.time()
    summary, win = run_gate.solve_target(n, portfolio[n], n_rounds=int(os.environ.get('ROUNDS', '4')), beam_width=2, top_k_polish=8, n_threads=int(os.environ.get('THREADS', '4')))
    (out / f'summary_{n}.json').write_text(json.dumps(summary, indent=1, default=str))
    if win is not None:
        (out / f'win_{n}.json').write_text(json.dumps(win, default=str))
    print(f'n={n}: win={win is not None} in {time.time()-t:.0f}s; summary keys {list(summary)[:12]}', flush=True)
    for k in ('status', 'best_dL', 'dL', 'L_best', 'L_new'):
        if k in summary: print(k, summary[k])
