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

- 47 strict per-`n` wins in the assigned range
- score gain at group level: `1.3364579325249855e-8`
- full literal score: `22.173715436104136`
- full pandas score: `22.173715436104136`
- both transports: `1000/1000` accepted

The largest improvements are:

| n | delta `L` |
| ---: | ---: |
| 936 | `-1.537e-7` |
| 862 | `-1.214e-7` |
| 738 | `-2.405e-8` |
| 739 | `-2.262e-8` |
| 997 | `-1.039e-8` |
| 694 | `-1.077e-8` |

## Files

- `poc/run.py` - reproducible runner; requires the collaborator POC at
  `poc/claude-midn/poc/free_slp.py`.
- `data/wins.json` - exact old/new per-n values.
- `data/changed-rows.csv` - the 47 improved puzzle groups only.
- `docs/NOTES.md` - execution and certification details.

No complete 500,500-row file is committed.
