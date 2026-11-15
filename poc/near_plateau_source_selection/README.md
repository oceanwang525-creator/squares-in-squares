# Near-plateau source selection

## Idea

The verified optimal-transport probes use donors near the target by puzzle
index. A better source may instead be a row whose side length is close, whose
wall contacts have similar counts, and whose orientation/contact profile
matches the target.

This probe ranks candidate donors by:

- absolute side-length difference, constrained to at most `0.02`;
- orientation-class histogram distance;
- unit-contact density;
- left/right/bottom/top wall-attachment counts.

It then feeds the top donors into the unchanged multi-donor Sinkhorn,
fixed-side repair, minimum dilation, fixed-angle LP, and dual certification
stack from `ot_multidonor_barycenter`.

## Run

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
PYTHONPATH=poc/baseline/poc/src $PY \
  poc/near_plateau_source_selection/poc/probe.py \
  --targets 803,328,791 --seconds-per-target 120 \
  --donors 8 --repair-iterations 6000 --top-starts 24 \
  --out poc/near_plateau_source_selection/docs/results.json
```

No public notebook/dataset coordinates are used. A candidate is written to
`final.csv` only after the unchanged official metric accepts a complete
strictly shorter group in literal-decimal and actual pandas
`read_csv`/`to_csv` transports.

## Results

**Positive at `n=791`.** The near-side selector chose donors `790` and `792`
(offsets `-1,+1`) rather than the global near-weight donor ordering used by
the H16 extension. The best map used:

- Sinkhorn `epsilon=0.0185`;
- donor weights `1:0.85` for `790:792`;
- interpolation `alpha=0.05`;
- requested side reduction `1e-4`;
- 6,000 repair iterations.

After repair the maximum penetration was `6.0885e-8`; minimum safe dilation
produced `28.626425643480978`, then fixed-angle LP recovered
`28.626405023313754301...`.

| transport | incumbent H16 side | candidate side |
|---|---:|---:|
| literal | `28.626523948339571162...` | `28.626405023313754301...` |
| pandas `read_csv` / `to_csv` | `28.626523948339571162...` | `28.626405023313754301...` |

This is `dL=-1.18925025816e-4` against the already improved H16 row, and
`dL=-1.230735217796e-3` against the original upstream row. The score gain is
`-4.37599066093e-5` against upstream.

The control targets in the first wave did not win:

| n | selected donors | result |
|---:|---|---|
| 803 | `802,804` | best safe side above incumbent |
| 328 | none within `0.02` | no transport map |

### Extended wave

After the real win, eight neighboring or high-gap rows were tested with ten
selected donors and 8,000 repair iterations:

`790,792,803,908,969,861,365,367`.

None produced a second certified win. Rows 790/792 selected their immediate
neighbors and came no closer than the incumbent; 365 again had no donor inside
the `0.02` side window. This is an extended tie-only wave after the win.

The final file contains exactly one complete 791-square group. Its SHA256 is
`9c3e146bc7aa60c3443eaed8ee196bfd47689ab3e10e167efddefdf95471638a`.
