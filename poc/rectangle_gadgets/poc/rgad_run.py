import sys, json, time
from multiprocessing import Pool
from rgad import search
if __name__ == '__main__':
    secs = float(sys.argv[1]); out = sys.argv[2]
    specs = [tuple(int(v) for v in s.split(',')) for s in sys.argv[3:]]
    jobs = [(a, b, g, secs, 1000 * a + 37 * b + g * 7 + r) for (a, b, g) in specs for r in range(int(8 // max(1, len(specs) // 8 or 1)) if len(specs) < 8 else 1)]
    res = {}
    with Pool(8) as pool:
        for a, b, g, f, X, runs in pool.imap_unordered(search, jobs):
            k = f'{a},{b},{g}'
            if k not in res or f < res[k]['f']:
                res[k] = dict(f=f, X=X, runs=res.get(k, {}).get('runs', 0) + runs)
            else:
                res[k]['runs'] += runs
    json.dump(res, open(out, 'w'))
    for k in sorted(res, key=lambda k: tuple(map(int, k.split(',')))):
        print(f'(a,b,g)=({k})  n={int(k.split(",")[0])*int(k.split(",")[1])+int(k.split(",")[2])}  best f={res[k]["f"]:.6f}  runs={res[k]["runs"]}', flush=True)
