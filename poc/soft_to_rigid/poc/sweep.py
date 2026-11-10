import sys, json, time, numpy as np, run
n = int(sys.argv[1]); budget = float(sys.argv[2]); out = f'../data/sweep_{n}.json'
cfgs = []
for s in range(100):
    cfgs += [('random', s, dict(stages=40, shrink=0.0)),
             ('perturbed', s, dict(sigma=0.05, shrink=0.0)),
             ('perturbed', s, dict(sigma=0.15, shrink=0.03)),
             ('perturbed', s, dict(sigma=0.30, shrink=0.0))]
res = []; t0 = time.time(); best = None
for mode, seed, kw in cfgs:
    if time.time()-t0 > budget: break
    r = run.one(n, mode, seed, **kw); r['cfg'] = kw
    rows = r.pop('rows', None)
    r = {k: (float(v) if isinstance(v, np.floating) else v) for k, v in r.items()}
    res.append(r); print(r, flush=True)
    if r.get('L_cert') and r['reason'] == 'ok' and (best is None or r['L_cert'] < best):
        best = r['L_cert']
        json.dump(rows, open(f'../data/best_{n}.json', 'w'))
    json.dump(res, open(out, 'w'))
