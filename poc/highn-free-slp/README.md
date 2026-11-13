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
fine pass:   60 iterations, trust 5e-4 -> 2e-6
escape pass: 100 iterations, trust 0.002
wide pass:   100 iterations, trust 0.004 -> 0.006
long pass:   100 iterations, trust 0.008
long pass:   100 iterations, trust 0.010
long pass:   100 iterations, trust 0.012
long pass:   100 iterations, trust 0.014
long pass:   100 iterations, trust 0.016
long pass:   100 iterations, trust 0.018
long pass:   100 iterations, trust 0.020
long pass:   100 iterations, trust 0.022
```

The SLP's own official group gate requires both literal and pandas-roundtrip
`L` to be strictly lower. The emitted rows are then spliced into the full
portfolio and rescored from scratch.

## Measured result

- 55 strict per-`n` wins from the 13-pass coarse schedule
- 16 additional strict per-`n` wins from the 9-pass fine schedule
- 5 additional strict per-`n` wins from a 100-iteration trust-`0.002` escape pass
- 6 additional strict per-`n` wins from the 3-pass wide escape schedule,
  including the first real jump at `n=645`
- 5 additional strict per-`n` wins from the trust-`0.008` long pass
- 5 additional strict per-`n` wins from the trust-`0.010` long pass
- 6 additional strict per-`n` wins from the trust-`0.012` long pass
- 5 additional strict per-`n` wins from the trust-`0.014` long pass
- 4 additional strict per-`n` wins from the trust-`0.016` long pass
- 5 additional strict per-`n` wins from the trust-`0.018` long pass
- 3 additional strict per-`n` wins from the trust-`0.020` long pass
- 4 additional strict per-`n` wins from the trust-`0.022` long pass
- standalone R30 full literal score: `22.173714763006313`
- standalone R30 full pandas score: `22.173714763006313`
- `final.csv` union with current merged after R32: `22.173714779017388`
- current upstream merged baseline: `22.17371545003316`
- R34 certified full union: `22.173714777045685`
- R40 certified full union: `22.173714776460457`
- both transports: `1000/1000` accepted, `gate=58/58`

The largest improvements are:

| n | delta `L` |
| ---: | ---: |
| 862 | `-4.089e-6` |
| 970 | `-4.292e-7` |
| 912 | `-3.726e-7` |
| 936 | `-2.969e-7` |
| 997 | `-2.919e-7` |
| 694 | `-1.684e-7` |

The later long passes additionally improved `747`, `791`, `816`, and `868`
against the current upstream incumbent.

R32 (`trust=0.018`, 100 iterations) added `791`, `868`, `908`, `910`, and
`936`; all five passed the per-group official gate (`5/5`).

R33 (`trust=0.020`, 100 iterations) added `791`, `868`, and `936`; all three
passed the per-group official gate (`3/3`).

R34 (`trust=0.022`, 100 iterations) added `695`, `791`, `868`, and `936`;
all four passed the per-group official gate (`4/4`). The certified union of
`final.csv` with the current incumbent scores `22.173714777045685` in both
literal and pandas transports, with all 58 final groups strictly better.

R35-R39 continued the same search at trusts `0.024`, `0.026`, `0.028`,
`0.032`, and `0.036`. Each pass found 56 strict wins, mostly on the same
groups. The rows were merged per-n only when the new literal and pandas scores
were both lower than the current `final.csv` group. R39 replaced ten groups
(`604, 687, 703, 748, 791, 818, 914, 971, 981, 990`) and certified the full
union at `22.173714776462209` in both transports, with `gate=58/58`.

The shared `791 -> 908` regression remains exact. Their incumbent sides differ
by `2.000000004`, and both receive the same `-5.68090286e-9` change from the
certified rows, so neither improvement is counted as an independent structural
jump.

R40 at trust `0.040` found 58 strict wins and replaced six R39 groups
(`644, 689, 695, 740, 791, 869`). The full union moved from
`22.173714776462209` to `22.173714776460457`, only `1.752e-12` lower. This is
the numerical floor of the current deterministic SLP basin, not a topology
change.

## Files

- `poc/run.py` - reproducible runner; requires the collaborator POC at
  `poc/claude-midn/poc/free_slp.py`.
- `poc/certify_final.py` - rebuilds the full union, checks every final group is
  a strict non-fallback win, and runs both official transports.
- `data/multi-round-gate.json` - exact old/new per-n values for all 55 coarse wins.
- `data/multi-round-rows.csv` - the 55 improved puzzle groups only.
- `data/round-summary.json` - per-pass win count and group gain.
- `data/fine-round-gate.json` - exact old/new per-n values for the 16 fine wins.
- `data/fine-round-rows.csv` - the 16 fine-round puzzle groups only.
- `data/fine-round-summary.json` - fine-pass win count and group gain.
- `data/escape-round-gate.json` - exact old/new per-n values for the 5 escape wins.
- `data/escape-round-rows.csv` - the 5 escape-round puzzle groups only.
- `data/escape-round-summary.json` - escape-pass win count and group gain.
- `data/wide-escape-round-gate.json` - exact old/new per-n values for the 6 wide wins.
- `data/wide-escape-round-rows.csv` - the 6 wide-escape puzzle groups only.
- `data/wide-escape-round-summary.json` - wide-pass win count and group gain.
- `data/long-escape-r27-report.json` - exact SLP report for the trust-`0.008` pass.
- `data/long-escape-r27-rows.csv` - the 5 trust-`0.008` puzzle groups only.
- `data/long-escape-r28-report.json` - exact SLP report for the trust-`0.010` pass.
- `data/long-escape-r28-rows.csv` - the 5 trust-`0.010` puzzle groups only.
- `data/long-escape-r28-gate.csv` - incumbent-gated rows for the final pass.
- `data/long-escape-r29-report.json` - exact SLP report for the trust-`0.012` pass.
- `data/long-escape-r29-rows.csv` - the 6 trust-`0.012` puzzle groups only.
- `data/long-escape-r29-gate.csv` - incumbent-gated rows for the final pass.
- `data/long-escape-r30-report.json` - exact SLP report for the trust-`0.014` pass.
- `data/long-escape-r30-rows.csv` - the 5 trust-`0.014` puzzle groups only.
- `data/long-escape-r30-gate.csv` - incumbent-gated rows for the final pass.
- `data/long-escape-r31-report.json` - exact SLP report for the trust-`0.016` pass.
- `data/long-escape-r31-rows.csv` - the 4 trust-`0.016` puzzle groups only.
- `data/long-escape-r31-gate.csv` - incumbent-gated rows for the final pass.
- `data/free_slp_600_1000_r32_t1p8e-2_i100_report.json` - R32 SLP report.
- `data/free_slp_600_1000_r32_t1p8e-2_i100_rows.csv` - R32 strict-win groups.
- `data/free_slp_600_1000_r32_t1p8e-2_i100_gate.csv` - official-gated R32 rows.
- `data/free_slp_600_1000_r33_t2p0e-2_i100_report.json` - R33 SLP report.
- `data/free_slp_600_1000_r33_t2p0e-2_i100_rows.csv` - R33 strict-win groups.
- `data/free_slp_600_1000_r33_t2p0e-2_i100_gate.csv` - official-gated R33 rows.
- `data/free_slp_600_1000_r34_t2p2e-2_i100_report.json` - R34 SLP report.
- `data/free_slp_600_1000_r34_t2p2e-2_i100_rows.csv` - R34 strict-win groups.
- `data/free_slp_600_1000_r34_t2p2e-2_i100_gate.csv` - official-gated R34 rows.
- `data/highn-free-slp-r39-report.json` - exact R39 SLP report.
- `data/highn-free-slp-r40-report.json` - exact R40 SLP report.
- `final.csv` - the best certified rows for all 58 current strict-win n groups.
- `docs/FINAL_GATE.json` - per-n literal and pandas comparison for `final.csv`.
- `data/long-escape-round-summary.json` - later long-pass win count and group gain.
- `docs/NOTES.md` - execution and certification details.
- `docs/MULTI_ROUND_WINS.md` - human-readable per-n comparison.
- `docs/FINE_ROUND_WINS.md` - human-readable fine-round comparison.
- `docs/ESCAPE_ROUND_WINS.md` - human-readable escape-round comparison.
- `docs/WIDE_ESCAPE_ROUND_WINS.md` - human-readable wide-escape comparison.
- `docs/LONG_ESCAPE_ROUND_WINS.md` - human-readable later long-pass comparison.

No complete 500,500-row file is committed.
