# Full-portfolio fixed-angle LP

**Idea.** Every repair in this repository re-solved candidates inside a small
trust region around the source coordinates. For a packing with fixed square
angles, the container side is a linear program in the centres once one
separating axis per nearby pair is fixed. This POC solves that LP with **no
bounds** on the centres, then rebuilds the pair set and active axis from the LP
solution and re-solves (3 SLP rounds). Any LP side strictly below the incumbent
is emitted and gated by the unchanged exact certifier (literal decimal strings
and actual pandas 3.1.0 transport).

This is a rigorous test for translational slack: if a group can be compressed in
the box in any direction that its fixed contact pattern leaves free, this LP
finds it.

## Measured result: 2 strict wins

Full 1000-group sweep against the `22.17650298137176` incumbent
(130 s, one LP per `n`, 3 SLP rounds each):

| n | score gain | old L | new certified L |
|---:|---:|---:|---:|
| 595 | `1.2442839765e-8` | `24.86192565354330775433594822850320341001258076580554` | `24.86192535001240769152874135100165402121368801394200` |
| 875 | `8.545911458e-10` | `29.98873321773301106923132166975700539764409810823621` | `29.98873319247497383157810537769367948888875651734886` |

Both certify identically on literal decimal strings and after actual pandas
3.1.0 `read_csv`/`to_csv` transport; no dilation was needed.

The merged portfolio scores **22.176502968074328** on both transports, with
**1000/1000 accepted, 0 missing, 0 rejected**. Previous incumbent:
`22.17650298137176`. Certified reduction: `1.3297431220727618e-8`.

Status counts over the 1000 groups: 995 `no-gain`, 3 `certify-fail` (sub-margin
LP solutions correctly rejected by the exact gate), 2 `WIN`.

## Why the other 995 are saturated

For a locally optimised packing, the LP side comes back a constant margin
*above* the incumbent: the 1e-9 safety added to every wall and pair constraint
(`2*1e-9` from the walls plus pair margins). That is the signature of an active
constraint set. The smallest `LP side - base` over the portfolio is `-3.3e-8`
(width of the safety slack), and only n=595 and n=875 had enough genuine slack
to beat it. So local polishing of the tracked rows is otherwise exhausted.

Combined with the earlier POCs (`leader_poc_k2p4` finds the incumbent is a
strict free-angle local optimum at n=328; `free_angle_repolish` recovers only
gallery rounding), the remaining headroom to the leaders is a **structural**
gap, not a numerical one.

## Reproduce

```sh
. .venv/bin/activate
python poc/alln_fixed_lp/poc/alln_fixed_lp.py \
  --incumbent poc/pell_corrugation/data/border-scan/merged.csv \
  --report poc/pell_corrugation/data/border-scan/merged-literal-official.json \
  --output /tmp/lp_all.json --wins /tmp/lp_all_wins.json
# probe a subset: add --n-list 971,911,910,972,793,634,791,908,969,803
```

Scripts: `poc/alln_fixed_lp.py`. Requires `numpy`, `scipy` (HiGHS) and the
shared official metric.
