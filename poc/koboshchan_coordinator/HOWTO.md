# Subagent HOWTO — squares-in-squares

Repo root: `/mnt/e/squares-in-squares-commandcode`
Python (+ installed editable `squares` package): `.venv/bin/python`

## Ground truth

- Incumbent: `poc/pell_corrugation/data/border-scan/merged.csv`
  (copy: `poc/koboshchan_coordinator/data/incumbent-22.15325720501.csv`),
  SHA256 `61004c6fbb45d64d1d4bdc4de4672012a18b4d5e23514e96abab88f8d8d3f7ac`,
  official score **22.153257205012313**, 1000/1000 accepted.
- Submission schema (this file): `id,n,square_id,x,y,angle`, ids `0..N-1`,
  `n` = group, `x,y` = **centre** of each unit square, `angle` in **radians**.
  Score for a group = `L/sqrt(n) - 1`, `L = max(width,height)` of the bounding box
  (translation invariant). Lower is better. Total = sum over n=1..1000.

## Ranking data

`poc/koboshchan_coordinator/data/gap_vs_frontier.csv`:
`n,K,D,ours,frontier,leader,dL,gap_score,gap_leader`
(`K=floor(sqrt(n))`, `D=n-K²`, `gap_score=(ours-frontier)/sqrt(n)`).
Target: bring the total from 22.153257 to **≤ 22.0999**.

## Certify a candidate (BOTH modes required)

```python
import sys; sys.path.insert(0, 'poc/baseline/poc/src')
from squares.core import certify, kaggle_roundtrip, score

# rows = [dict(square_id="0", x="...", y="...", angle="0.123..."), ...]  (strings, radians)
L_literal = certify(n, rows)                 # 50-digit L, raises if invalid
p = {n: rows}
L_pandas  = certify(n, kaggle_roundtrip(p)[n])  # actual pandas read_csv/to_csv transport
```
A claimed win needs `Decimal(L) < Decimal(incumbent_L_for_n)` in BOTH modes.
Only strict wins, `dL <= -1e-9` (ideally `<= -1e-5`), count.

## final.csv format (REQUIRED for every touched POC folder)

Header exactly `id,x,y,deg` (`deg` = degrees, not radians). ids `{n:03}_{sid}`,
complete groups only (all `n` squares, sids `0..n-1`). Prefix decimals with `s`
to survive pandas. Header-only file if nothing qualifies.

```python
import csv, mpmath as mp
def write_final(path, groups):            # groups: {n: [dict(square_id,x,y,angle),...]}
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["id","x","y","deg"])
        with mp.workdps(95):
            for n, rows in sorted(groups.items()):
                for r in sorted(rows, key=lambda r: int(r["square_id"])):
                    deg = mp.nstr(mp.mpf(r["angle"])*180/mp.pi, 75)
                    w.writerow([f"{n:03}_{r['square_id']}", "s"+r["x"], "s"+r["y"], "s"+deg])
```

## Free win: L-border propagation

In the incumbent, **470 pairs** satisfy `L(n') = L(n)+1` with
`n' = n + 2*K + 1`, `K = floor(sqrt(n))`. Mechanism: if a packing of `n`
squares has side `L` and `ceil(L) = K+1`, an axis-aligned L-border of
`2*ceil(L)-1 = 2K+1` unit squares (a full new column at `x = L+0.5`, plus the
matching new row minus the corner) gives `n' = n+2K+1` squares at side `L+1`.
So **whenever you improve a base `n`, also emit the border-extended
`n' = n+2*K+1` group** (same `dL`, and it cascades). Certify the extended group
in both modes; the border is exact geometry (axis-aligned, outside the old
bbox). Do this before declaring the win.

## Rules

- Work ONLY inside your own `poc/{idea}/` folder. Do not modify other folders.
- Do NOT run `git` commands — the coordinator handles commits/PRs.
- No harvesting public notebooks/datasets; construct yourself. No Kaggle submission.
- Read `poc/loop_log.md` (loops 1-40) and relevant POC READMEs first, so you do
  not repeat a known failure. The portfolio is already a local optimum under
  fixed-angle LP, Lipschitz free-angle SLP, contact-topology search, SA/PT, and
  MILP set-packing — do not just re-run those.
- Start with 1-3 `n` (~15 min). No certified strict win → write the failure in
  your README, leave header-only `final.csv`, stop.
- Real win → scale up (longer runs, neighbouring `n`, detached with logs and
  checkpoints). Report exact per-`n` `dL` (literal + pandas) and file paths.
