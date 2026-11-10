# Construction notes

## 1. Frame

Work in the unit-square plane. An upright square is a cell `[i,i+1] x [j,j+1]`
with centre `(i+1/2, j+1/2)`. A 45-degree square is a unit square rotated by
`pi/4`; its axis-aligned half-extent is `h = (|cos t| + |sin t|)/2 = 1/sqrt(2)`.

For the 45-degree lattice use coordinates

```
u = (x + y)/sqrt(2),   v = (x - y)/sqrt(2)
x = (u + v)/sqrt(2),   y = (u - v)/sqrt(2)
```

A unit square at integer `(u, v)` has neighbours at `u+-1` or `v+-1`, so integer
`(u,v)` cells tile the plane edge-to-edge at 45 degrees.

## 2. Monolithic four-staircase band

```
        NW(a2)                        NE(b2)
   +--------------+             +--------------+
   |\             |             |             /|
   | \  upright   |    band     |  upright   / |
   |  \ staircase |  (l x w)    | staircase /  |
   |   \          |  45 deg     |          /   |
   +----\---------+-------------+---------/----+
   |     \        |             |        /     |
   |      \       |             |       /      |
   +--------------+-------------+--------------+
        SW(b1)                        SE(a1)
```

Four upright staircase triangles sit in the corners. SE has `a1` rows, NW `a2`,
SW `b1`, NE `b2`; triangle `a` holds `tri(a) = a(a+1)/2` cells. The central
45-degree band is `l` columns by `w` rows, `w*l` cells. Hence

```
n = tri(a1) + tri(a2) + tri(b1) + tri(b2) + w*l.
```

The side is fixed by the band and flanks:

```
L = (a1 + a2)/2 + 1 + w/sqrt(2).
```

### Wall consistency

For the placed point set to be a real packing in `[0,L]^2`:

* adjacent corner staircases must not collide: `max(a1,a2) + max(b1,b2) <= L`;
* the band must fit the `u` channel between SW and NE:
  `l - 1 <= u_hi - u_lo`, with
  `u_lo = max((b1+1)/sqrt2 + 1/2, u_wall)`,
  `u_hi = sqrt2 L - max((b2+1)/sqrt2 + 1/2, u_wall)`,
  `u_wall = (w-1)/2 + |a1-a2|/(2 sqrt2) + 1/sqrt2`;
* the band must sit inside the box. With
  `v_start = -(w-1)/2 - (a1-a2)/(2 sqrt2)` the four containment checks are
  `u_lo + v_start >= 1`, `u_hi + v_start + w - 1 <= sqrt2 L - 1`,
  `u_lo - v_start - (w-1) >= 1`, `u_hi - v_start <= sqrt2 L - 1`.

Two corner cells are dropped when a staircase would cross a wall
(`a1+b1 > L`, `a1+b2 > L`, `a2+b1 > L`, `a2+b2 > L`); the search rejects any
configuration with a drop so the capacity stays exactly `w*l + sum tri`.

### Worked example (`n = 739`, the Pell-17 row)

`w=13, l=25, a=(18,17), b=(9,9)`:

```
tri(18)+tri(17)+tri(9)+tri(9) = 171+153+45+45 = 414
w*l = 325
n = 739
L = (18+17)/2 + 1 + 13/sqrt2 = 27.692388...
```

This matches the recorded Pell-17 core, which validates the placement rule.

## 3. Bent two-block variant

`bent_two_block(L, t1, t2, w1, w2, l1, l2)`:

* block A is a `l1 x w1` lattice block grown from `(0,0)` along `t1`;
* block B is a `l2 x w2` block grown from `(L,L)` backwards along `t2`;
* cells outside `[0,L]^2` are clipped;
* A/B overlap is removed from B by an exact separating-axis test;
* the four corners are filled with upright half-grid cells (four corner
  phases, added greedily, each phase internally disjoint).

```
  B (t2)  . . . . . . . . . . . .
        .                         .
      .      corner upright fill   .
    .                               .
   +---------------------------------+   <- L
   |  . . . . . . . . . . . . . . .  |
   |  .        block B (t2)         . |
   |  . . . . . . . . . . . . . . .  |
   |                                 |
   |   block A (t1)                  |
   |  . . . . . . . . . . . .        |
   |  . . . . . . . . . . . .        |
   +---------------------------------+
   (0,0)
```

The seam between A and B is where the two lattices meet. Because a square
lattice rotated by `t1` and one rotated by `t2` share a direction only when
`t1 = t2` (mod 90 degrees), there is no exact integer staircase that belongs to
both lattices. The builder therefore removes the small A/B overlap and lets the
upright fill cover the rest; that is a *bounded approximation* of the proposed
seam, not an exact tiling.

## 4. Clearance and certification

Exact contact leaves zero gaps, which the official metric rejects at floating
precision. `to_rows` normalises the point set, then dilates centres about the
box centre by `1.00000002` and prints 17 decimals. This adds positive clearance
of order `1e-6` at `L ~ 30` (cost `< 1e-6` in `L`, far below any reported
delta). Certification is run twice: on the literal CSV, and on
`pandas.read_csv(literal).to_csv(index=False)`.

## 5. Search

`search.best_exact(n)` enumerates `w`, `a1`, `a2`, `b1`, `b2`, derives
`l = (n - sum tri)/w`, applies the wall-consistency checks, and keeps the
smallest `L` whose builder returns exactly `n` points with no drops.
`search.max_capacity_at(L)` enumerates the same family with `L' <= L` and
returns the largest capacity, giving the shortfall at the incumbent side.
Both are bounded (a few thousand candidate tuples) and documented as search
results, not proofs of optimality.
