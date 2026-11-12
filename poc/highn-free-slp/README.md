# High-n kink-free free-angle SLP

## Idea

Run the collaborator's kink-free corner SLP (`poc/claude-midn/poc/free_slp.py`)
on the high-`n` range that Codex owns: `n=600..1000`. Unlike the conservative
Lipschitz-safe model, this formulation can represent a rigid joint rotation of
an aligned block. Fixed-angle and old LIP-SLP had already reported zero wins on
these same rows.

## Method

The runner invokes the collaborator's SLP with:

```text
n=600..1000
32 workers
30 iterations
initial trust 0.002
```

The SLP's own official group gate requires both literal and pandas-roundtrip
`L` to be strictly lower. The emitted rows are then spliced into the full
portfolio and rescored from scratch.

## Measured result

- 55 strict per-`n` wins against the latest upstream incumbent
- 13 passes, stopped after the final pass produced a group gain below `1e-9`
- full literal score: `22.173715215321167`
- full pandas score: `22.173715215321167`
- both transports: `1000/1000` accepted

The largest improvements are:

| n | delta `L` |
| ---: | ---: |
| 862 | `-4.089e-6` |
| 970 | `-3.964e-7` |
| 912 | `-3.584e-7` |
| 936 | `-2.958e-7` |
| 997 | `-2.919e-7` |
| 694 | `-1.684e-7` |

## Files

- `poc/run.py` - reproducible runner; requires the collaborator POC at
  `poc/claude-midn/poc/free_slp.py`.
- `data/multi-round-gate.json` - exact old/new per-n values for all 55 wins.
- `data/multi-round-rows.csv` - the 55 improved puzzle groups only.
- `data/round-summary.json` - per-pass win count and group gain.
- `docs/NOTES.md` - execution and certification details.
- `docs/MULTI_ROUND_WINS.md` - human-readable per-n comparison.

No complete 500,500-row file is committed.
