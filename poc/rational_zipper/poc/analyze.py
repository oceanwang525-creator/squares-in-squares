"""Geometry analysis helpers: exact bbox of a submission group (accounts for rotation)."""
import sys, math, collections

sys.path.insert(0, 'poc/baseline/poc/src')


def ext(rows, ps=0):
    """bbox side and extents of a group of rows (dicts with x,y,angle strings)."""
    xs, ys = [], []
    for r in rows:
        x, y, a = float(r['x']), float(r['y']), float(r['angle'])
        c, s = math.cos(a), math.sin(a)
        # half-extent along x and y of a unit square rotated by a:
        hx = 0.5 * (abs(c) + abs(s))
        hy = hx
        xs += [x - hx, x + hx]
        ys += [y - hy, y + hy]
    return max(xs) - min(xs), max(ys) - min(ys)


def main():
    import squares.core as core
    P = core.read_submission('poc/pell_corrugation/data/border-scan/merged.csv')
    for n in (int(a) for a in sys.argv[1:]):
        rows = P[n]
        W, H = ext(rows)
        K = math.isqrt(n)
        print(f'n={n:4d} K={K:2d} D={n-K*K:2d}  W={W:.9f} H={H:.9f} '
              f'L={max(W,H):.9f}  c={max(W,H)-K:.9f}')
        c = collections.Counter()
        for r in rows:
            c[round(math.degrees(float(r['angle'])) % 90, 4)] += 1
        print('   angles deg%90:', sorted(c.items(), key=lambda t: -t[1])[:10])


if __name__ == '__main__':
    main()
