# Bent band joining two tilted blocks for `n = k^2 + D`

A bounded CPU test of the claim that a *bent* band — two tilted parallelogram
blocks joined along an integer staircase seam, with integer corner fills —
reaches about `k + 0.79` for `n = k*k + D` with `D` near `k`, i.e. a side-length
gain of about `-0.07` versus the monolithic tilted band at `k + 0.86..0.93`.

**Result: the claim is not reproduced.** The exact monolithic 45-degree
staircase family reaches at most 4–5 squares short of each target at the
incumbent side, and its best exact-capacity side lengths are *larger* than the
incumbents. The bent two-block variant built here is worse still. No target
certifies below its incumbent; the reachable total score change is `+0.04524`
(a loss). Details, arithmetic and measured strings are below.

The case page `https://squaremath.org/cases/303.html` was unreachable
(`curl` timed out after 20 s), so the geometry here follows the written brief.

## Files

```
poc/bent_band.py    exact monolithic 45-degree band + bent two-block builder,
                    contact helpers, clearance dilation, CSV/certify helpers
poc/search.py       bounded discrete capacity search (exact n, max capacity at L)
poc/run.py          build, certify (literal + pandas round-trip), report
docs/construction.md  geometry, capacity arithmetic, seam notes
docs/failures.md      measured shortfalls and why the exact seam is not realisable
```

Run from the repository root:

```sh
python3 poc/bent_band_join/poc/run.py --out-dir /tmp/bent_band_out
```

The default output directory is `poc/bent_band_join/data/`; point `--out-dir`
elsewhere to avoid writing into the repository.

## Construction and capacity arithmetic

`monolithic_45(w, l, a1, a2, b1, b2)` places four upright corner staircases
(SE `a1`, NW `a2`, SW `b1`, NE `b2`) and one central 45-degree band of `l`
columns by `w` rows. With `tri(x) = x(x+1)/2`:

```
L    = (a1 + a2)/2 + 1 + w/sqrt(2)
n    = tri(a1) + tri(a2) + tri(b1) + tri(b2) + w*l      (minus 2 wall drops)
```

The band columns are placed on the wall-consistent `u` interval
`[max((b1+1)/sqrt2 + 1/2, u_wall), sqrt2 L - max((b2+1)/sqrt2 + 1/2, u_wall)]`
with `u_wall = (w-1)/2 + |a1-a2|/(2 sqrt2) + 1/sqrt2`. A candidate is accepted
only if every band cell lies inside `[0,L]^2` and the four staircases stay
clear of each other (`max(a1,a2) + max(b1,b2) <= L`). Capacity and side length
are therefore exact integers/rationals, and the builder is re-checked against
`len(points)` and the actual corner extent.

`bent_two_block(L, t1, t2, w1, w2, l1, l2)` grows block A from `(0,0)` along
`t1` and block B from `(L,L)` backwards along `t2`, removes any A/B overlap
exactly, then fills the four corners with upright half-grid squares. Its
capacity is `len(points)`. Angles tested: `45`, `arctan(1/2)`, `arctan(2/3)`,
`arctan(3/2)`, `30` degrees.

## Evidence (measured, certified)

All four packings pass the unchanged official metric (`squares.core.certify`)
on the literal CSV and on a plain `pandas.read_csv(path).to_csv(index=False)`
round-trip under pandas 3.1.0. Dilation is `1.00000002` (positive clearance),
17-decimal literals.

| n | params `(w,l,a1,a2,b1,b2)` | literal L | pandas L | incumbent L | `dL` | per-n score `dL/sqrt(n)` |
|---|---|---|---|---|---|---|
| 645 | `(13,21,16,16,9,10)` | `26.19238865927288273000000000000000000000000000000000` | `26.19238865927288264000000000000000000000000000000000` | `25.87031812251975364740446552983030849778887781162520` | `+0.322070` | `+0.012682` |
| 751 | `(15,22,16,17,11,11)` | `28.10660225993025385000000000000000000000000000000000` | `28.10660225993025340000000000000000000000000000000000` | `27.86424008616910855961199900976410503972147125750780` | `+0.242363` | `+0.008844` |
| 925 | `(19,23,17,17,13,13)` | `31.43502945124498283000000000000000000000000000000000` | `31.43502945124498275000000000000000000000000000000000` | `30.86321837316950898175795253708386667312879581004381` | `+0.571811` | `+0.018801` |
| 987 | `(17,25,19,19,13,13)` | `32.02081590058761406000000000000000000000000000000000` | `32.02081590058761430000000000000000000000000000000000` | `31.86630526508366147711534673847566523363639134913683` | `+0.154511` | `+0.004918` |

Both certification outcomes are **pass**; every candidate is *worse* than its
incumbent (`dL > 0`). Total over the four groups: `+0.0452445965` score, i.e. a
loss, not a gain.

## Shortfall at the incumbent side (the bent claim's real hurdle)

The monolithic family cannot reach `n` at the incumbent `L`. The largest
capacity found by the bounded search with `L <= L_incumbent` is:

| n | max capacity at `L_inc` | shortfall | best params `(w,A,a1,a2,b1,b2,l)` | `L` used |
|---|---|---|---|---|
| 645 | 640 | 5 | `(4,44,22,22,2,2,32)` | 25.828427 |
| 751 | 746 | 5 | `(4,48,24,24,2,2,35)` | 27.828427 |
| 925 | 921 | 4 | `(21,30,15,15,15,15,21)` | 30.849242 |
| 987 | 982 | 5 | `(9,49,24,25,6,6,35)` | 31.863961 |

So a bent band would have to add **4–5 squares** at the incumbent side before it
can even tie. The bent two-block builder reaches only:

| n | bent capacity at `L_inc` | below `n` by |
|---|---|---|
| 645 | 606 | 39 |
| 751 | 710 | 41 |
| 925 | 881 | 44 |
| 987 | 942 | 45 |

The best bent configurations degenerate to `t1 = t2 = 45` with small blocks and
a large upright corner fill, so the measured "bent" number is an upper bound on
what this particular builder can do, not on every conceivable bent band.

## Claim check

For `n = 645`, `k + 0.79 = 25.79`; the incumbent is `25.870318`, so the claim
predicts `dL ≈ -0.080` and a per-n score gain of `0.080/sqrt(645) ≈ +0.00315`.
Measured: `dL = +0.32207`, score `-0.01268`. For all four groups the claim's
sign is wrong. The `k + 0.79` target is **not** claimed.

## Limits and main risk

* The monolithic family is the exact four-staircase/45-degree band; it may not
  be the same object the brief calls "one monolithic tilted band", which the
  brief places at `k + 0.86..0.93` (our exact-capacity sides are higher).
* The max-capacity-at-`L` table is a bounded search result, not a proof of
  optimality; the true shortfall could be larger, not smaller.
* The main structural risk is in the brief itself: two square lattices rotated
  by different angles share no lattice direction, so an **exact** integer
  staircase seam between two differently tilted blocks does not exist. A bent
  band must either use an upright (approximate) seam with small gaps, or leave
  a wedge. That is why the numeric bent builder loses capacity and why the
  earlier `poc/hinged_band` experiment found zero wins.
* All numbers are CPU, single-machine, bounded search; no GPU or annealing.
