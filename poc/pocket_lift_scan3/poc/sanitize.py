"""Sanitise a pocket library: make every record's (cnt, w, h, g, f, box) agree.

The scanned-pocket libraries are assembled by several tools (scan -> polish ->
merge) and a record can end up with a `box` that is *not* the tight bounding box
of `cnt` squares.  This bit E1: C1's `pockets_pareto.json` carries

    src=852 (7,19,+5) f=0.699128038  cnt=138  box=(0, 7.6991280, 10, 29.6991280)

whose box actually holds only **125** squares (125 < 7*19, so the shape is not a
pocket at all).  `dp.py` picked it because its `f` is the smallest of any
(7,19,5), and `build_cert.py` then built 899 of the 912 squares of the biggest
E1 win.  The record survived every Pareto merge because the merges only compare
`f` -- they never look at the donor row again.

This tool re-derives every record from its donor row: select the squares inside
the recorded box with a 1e-9 tolerance, take their exact tight bbox, set
`w = floor(W)`, `h = floor(H)`, `cnt` = the number selected, `g = cnt - w*h`,
`f = max(W-w, H-h)`, and drop the record if `g < 1`.  Output records are
self-consistent by construction.

usage: sanitize.py donor.csv pockets.json out.json
"""
import json
import sys

sys.path.insert(0, 'poc/pocket_lift_scan3/poc')
from scan_pockets import Row, load_rows

TOL = 1e-9


def main():
    csv, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = load_rows(csv)
    lib = json.load(open(src))
    cache = {}
    out, dropped, fixed, skipped = [], [], [], []
    for g in lib:
        n = g['src']
        if n not in rows:            # donor rows that are not in this CSV
            skipped.append(g)        # (the D1 iteration donors 2000..) -- keep as is
            out.append(g)
            continue
        if n not in cache:
            cache[n] = Row(n, rows[n])
        row = cache[n]
        m = ((row.lx >= g['x0'] - TOL) & (row.hx <= g['x1'] + TOL) &
             (row.ly >= g['y0'] - TOL) & (row.hy <= g['y1'] + TOL))
        k = int(m.sum())
        if k < 2:
            dropped.append(g)
            continue
        X0, X1 = float(row.lx[m].min()), float(row.hx[m].max())
        Y0, Y1 = float(row.ly[m].min()), float(row.hy[m].max())
        W, H = X1 - X0, Y1 - Y0
        w = int(W + 1e-6)
        h = int(H + 1e-6)
        gg = k - w * h
        if gg < 1 or w < 1 or h < 1:
            dropped.append(g)
            continue
        f = max(W - w, H - h)
        rec = dict(g)
        rec.update(w=w, h=h, g=gg, f=f, cnt=k, x0=X0, x1=X1, y0=Y0, y1=Y1)
        if gg != g['g'] or abs(f - g['f']) > 1e-12:
            fixed.append((g, rec))
        out.append(rec)
    print(f'{len(lib)} records -> {len(out)} kept, {len(dropped)} dropped, '
          f'{len(fixed)} re-measured, {len(skipped)} unvalidated (donor not in CSV)')
    for g in dropped[:20]:
        print(f"  DROP  src={g['src']} ({g['w']},{g['h']},+{g['g']}) f={g['f']:.9f} "
              f"cnt={g['cnt']}")
    for a, b in fixed[:20]:
        print(f"  FIX   src={a['src']} ({a['w']},{a['h']},+{a['g']}) f={a['f']:.9f} "
              f"cnt={a['cnt']} -> ({b['w']},{b['h']},+{b['g']}) f={b['f']:.9f} cnt={b['cnt']}")
    json.dump(out, open(dst, 'w'), indent=0)
    print('wrote', dst)


if __name__ == '__main__':
    main()
