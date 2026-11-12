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
coarse pass: 30 iterations, trust 0.002
fine pass:   60 iterations, trust 5e-4 -> 1e-5
```

The SLP's own official group gate requires both literal and pandas-roundtrip
`L` to be strictly lower. The emitted rows are then spliced into the full
portfolio and rescored from scratch.

## Measured result

- 55 strict per-`n` wins from the 13-pass coarse schedule
- 16 additional strict per-`n` wins from the 7-pass fine schedule
- full literal score: `22.173715210474928`
- full pandas score: `22.173715210474928`
- both transports: `1000/1000` accepted

The largest improvements are:

| n | delta `L` |
| ---: | ---: |
| 862 | `-4.089e-6` |
| 970 | `-4.292e-7` |
| 912 | `-3.726e-7` |
| 936 | `-2.969e-7` |
| 997 | `-2.919e-7` |
| 694 | `-1.684e-7` |

## Files

- `poc/run.py` - reproducible runner; requires the collaborator POC at
  `poc/claude-midn/poc/free_slp.py`.
- `data/multi-round-gate.json` - exact old/new per-n values for all 55 coarse wins.
- `data/multi-round-rows.csv` - the 55 improved puzzle groups only.
- `data/round-summary.json` - per-pass win count and group gain.
- `data/fine-round-gate.json` - exact old/new per-n values for the 16 fine wins.
- `data/fine-round-rows.csv` - the 16 fine-round puzzle groups only.
- `data/fine-round-summary.json` - fine-pass win count and group gain.
- `docs/NOTES.md` - execution and certification details.
- `docs/MULTI_ROUND_WINS.md` - human-readable per-n comparison.
- `docs/FINE_ROUND_WINS.md` - human-readable fine-round comparison.

No complete 500,500-row file is committed.
