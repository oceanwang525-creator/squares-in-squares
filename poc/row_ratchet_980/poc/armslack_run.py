"""Run the team's H12 arm-slack gate (poc/swarm_armslack) on chosen n without writing into their folder.
usage: armslack_run.py out_dir n [n ...]
"""
import json, os, sys, time
import multiprocessing as mp
from pathlib import Path

ROOT = Path(os.environ['REPO_ROOT'])
for p in (ROOT / 'poc/swarm_armslack/poc', ROOT / 'poc/swarm_highd/poc', ROOT / 'poc/baseline/poc/src'):
    sys.path.insert(0, str(p))
import cpp_wrap                                    # noqa: E402
cpp_wrap.CPP_LIB = Path(os.environ['CPP_LIB'])     # build the helper library in a scratch directory
import numpy as np, pandas as pd                   # noqa: E402
import run_gate as rg                              # noqa: E402
from squares.core import read_submission           # noqa: E402

if __name__ == '__main__':
    mp.set_start_method('fork')
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    cpp_wrap.get_cpp_lib()
    merged_df = pd.read_csv(rg.MERGED_CSV); base = read_submission(rg.MERGED_CSV)
    for n in map(int, sys.argv[2:]):
        t = time.time()
        x, y, th, L, Llit, Lpd = rg.load_base_puzzle(n, merged_df, base)
        res = rg.solve_target(n, x, y, th, L, Llit, Lpd, n_threads=int(os.environ.get('THREADS', '4')), full_iters=180)
        win = res.pop('win')
        (out / f'res_{n}.json').write_text(json.dumps(res, indent=1, default=str))
        if win is not None:
            (out / f'win_{n}.json').write_text(json.dumps(win, default=str))
        print(f'n={n}: status {res["status"]} best_dL {res["best_dL"]} in {time.time()-t:.0f}s', flush=True)
