"""Structured basin hopping around an extracted pocket, with the SLP squeeze as the polish.

For every pocket we already have a *realised* packing at its incumbent f0, which is a strong
local optimum (the repo's SLP / LP work says the tracked rows have no free-angle slack).  The
only way an isolated-pocket search can pay is therefore to change the contact topology: these
shakes move a *localised part* of the packing (a row, a block, a corner, a random subset)
while keeping the engineered meshed structure, and each trial is then repaired by the SAT
penalty relaxation and re-squeezed by the SLP (side continuation inside pslp.squeeze).

Shakes
  scale   : affine shrink of all centres about the box centre + isotropic jitter
  row     : shift the squares in a narrow band along a random direction
  block   : translate the squares inside a random disc
  corner  : translate / rotate the squares in one corner quadrant
  subset  : re-randomise a random 5-25 % of the squares (jitter 0.05-0.25) and repair
  rot     : rotate a random localised group by +-0.03..0.4 rad

usage: hop.py --shape 9,15,3 --rounds 30 --chains 24 [--out ../data/hopped.json]
"""
import argparse, json, math, os, sys, time
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import squeeze as S        # noqa: E402

HP = math.pi / 2


def shake(q, w, h, f, rng):
    q = q.copy()
    n = len(q)
    cx, cy = (w + f) / 2, (h + f) / 2
    kind = rng.choice(['scale', 'row', 'block', 'corner', 'subset', 'rot', 'scale', 'subset'])
    if kind == 'scale':
        sc = 1.0 + rng.normal(0, rng.choice([2e-4, 1e-3, 5e-3, 2e-2]))
        sig = rng.choice([1e-4, 1e-3, 5e-3])
        q[:, 0] = cx + (q[:, 0] - cx) * sc + rng.normal(0, sig, n)
        q[:, 1] = cy + (q[:, 1] - cy) * sc + rng.normal(0, sig, n)
        q[:, 2] += rng.normal(0, sig, n)
    elif kind == 'row':
        th = rng.uniform(0, math.pi)
        proj = q[:, 0] * math.cos(th) + q[:, 1] * math.sin(th)
        lo = rng.uniform(proj.min(), proj.max() - 0.5)
        band = (proj > lo) & (proj < lo + rng.uniform(0.6, 1.8))
        if band.sum() == 0 or band.sum() == n:
            return None
        d = rng.normal(0, 1, 2); d /= np.linalg.norm(d)
        q[band, 0] += d[0] * rng.choice([0.05, 0.15, 0.35])
        q[band, 1] += d[1] * rng.choice([0.05, 0.15, 0.35])
    elif kind == 'block':
        c = np.array([rng.uniform(0, w + f), rng.uniform(0, h + f)])
        r = rng.uniform(0.8, 3.0)
        m = np.hypot(q[:, 0] - c[0], q[:, 1] - c[1]) < r
        if m.sum() == 0 or m.sum() == n:
            return None
        d = rng.normal(0, 1, 2); d /= np.linalg.norm(d)
        q[m, 0] += d[0] * rng.uniform(0.02, 0.3)
        q[m, 1] += d[1] * rng.uniform(0.02, 0.3)
    elif kind == 'corner':
        sx = rng.random() < 0.5; sy = rng.random() < 0.5
        mx = w + f if sx else 0.0
        my = h + f if sy else 0.0
        r = rng.uniform(1.0, 4.0)
        m = np.hypot(q[:, 0] - mx, q[:, 1] - my) < r
        if m.sum() == 0 or m.sum() == n:
            return None
        q[m, 0] += rng.normal(0, 0.06, m.sum())
        q[m, 1] += rng.normal(0, 0.06, m.sum())
        q[m, 2] += rng.normal(0, 0.05, m.sum())
    elif kind == 'subset':
        frac = rng.uniform(0.05, 0.25)
        m = rng.random(n) < frac
        if m.sum() == 0:
            return None
        sig = rng.uniform(0.02, 0.25)
        q[m, 0] += rng.normal(0, sig, m.sum())
        q[m, 1] += rng.normal(0, sig, m.sum())
        q[m, 2] += rng.normal(0, 0.15, m.sum())
    elif kind == 'rot':
        c = np.array([rng.uniform(0, w + f), rng.uniform(0, h + f)])
        r = rng.uniform(0.7, 2.5)
        m = np.hypot(q[:, 0] - c[0], q[:, 1] - c[1]) < r
        if m.sum() == 0 or m.sum() == n:
            return None
        a = rng.choice([-1, 1]) * rng.uniform(0.03, 0.4)
        ca, sa = math.cos(a), math.sin(a)
        dx = q[m, 0] - c[0]; dy = q[m, 1] - c[1]
        q[m, 0] = c[0] + ca * dx - sa * dy
        q[m, 1] = c[1] + sa * dx + ca * dy
        q[m, 2] += a
    return q


def chain(job):
    key, w, h, f0, q0, rounds, seed, iters, polish_it = job
    rng = np.random.default_rng(seed)
    W, H = w + f0, h + f0
    best = (f0, np.asarray(q0, float).copy(), 'pocket')
    trace = []
    for r in range(rounds):
        q = shake(best[1], w, h, best[0], rng)
        if q is None:
            continue
        # repair inside the *incumbent* box, then ask the SLP for a smaller one
        q, E = S.relax(q, w + best[0], h + best[0], iters=iters)
        if not np.isfinite(E) or E > 1e-9:
            trace.append((r, None, E))
            continue
        f2, q2, _ = S.polish(q, w, h, max_it=polish_it)
        ok, v = S.certified(q2, w, h, f2)
        trace.append((r, f2 if ok else None, v))
        if ok and f2 < best[0] - 1e-9:
            best = (float(f2), q2, f'hop{r}')
        elif ok and f2 < best[0] + 1e-12:
            best = (best[0], q2, best[2])            # same f, fresh basin
    return key, best, trace


def hop(w, h, g, pocket, f0, rounds=30, chains=24, procs=24, iters=150, polish_it=120, seed=0,
        verbose=True):
    jobs = [(f'{w}x{h}g{g}', w, h, f0, pocket, rounds, seed * 7919 + c, iters, polish_it)
            for c in range(chains)]
    best = (f0, np.asarray(pocket, float), 'pocket')
    improved = []
    with Pool(procs) as pool:
        for key, b, tr in pool.imap_unordered(chain, jobs):
            nf = sum(1 for r, f, _ in tr if f is not None)
            if verbose:
                print(f'  chain seed: {nf}/{len(tr)} feasible, best {b[0]:.9f} ({b[2]})', flush=True)
            if b[0] < best[0] - 1e-9:
                best = b
            improved += [(r, f) for r, f, _ in tr if f is not None and f < f0 - 1e-9]
    if verbose:
        print(f'({w},{h},+{g}): {f0:.9f} -> {best[0]:.9f} '
              f'({"IMPROVED" if best[0] < f0 - 1e-9 else "no improvement"})', flush=True)
    return best


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--shape', required=True)
    ap.add_argument('--rounds', type=int, default=30)
    ap.add_argument('--chains', type=int, default=24)
    ap.add_argument('--procs', type=int, default=24)
    ap.add_argument('--iters', type=int, default=150)
    ap.add_argument('--polish-it', type=int, default=120)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--seed-file', default='../data/targets.json')
    ap.add_argument('--out', default='../data/hopped.json')
    a = ap.parse_args()
    w, h, g = (int(x) for x in a.shape.split(','))
    d = json.load(open(a.seed_file))
    ent = None
    for k, v in d.items():
        if (v['w'], v['h'], v['g']) == (w, h, g) or (v['h'], v['w'], v['g']) == (w, h, g):
            ent = v
            break
    if ent is None:
        sys.exit(f'no pocket for {a.shape} in {a.seed_file}')
    q = np.asarray(ent['sq'], float)
    if ent['w'] != w:
        q = np.column_stack([q[:, 1], q[:, 0], HP - q[:, 2]])
    t0 = time.time()
    f, qb, prov = hop(w, h, g, q, float(ent['f']), rounds=a.rounds, chains=a.chains,
                      procs=a.procs, iters=a.iters, polish_it=a.polish_it, seed=a.seed)
    print(f'wall {time.time()-t0:.0f}s')
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    key = f'{w}x{h}g{g}'
    if f < ent['f'] - 1e-9:
        out[key] = dict(w=w, h=h, g=g, f=float(f), sq=qb.tolist(), src=prov, prev=float(ent['f']))
        json.dump(out, open(a.out, 'w'))
    print(f'{key}: {ent["f"]:.9f} -> {f:.9f}')
