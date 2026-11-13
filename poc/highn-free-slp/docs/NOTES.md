# Notes

## Why this range

The shared split assigns Claude `n=200..599` and Codex `n=1..199` plus
`n=600..1000`. This POC only touches the Codex range.

## Search history

The following had already failed on the current high-n rows:

- fixed-angle LP, 12 rounds: 0 wins;
- conservative LIP-SLP, 8 rounds: 0 wins;
- support/corner contact SLP: 0 wins;
- fault-slab peel: 0 wins;
- exact upright refill: 5 one-square near misses, no strict win.

The kink-free corner formulation found 47 wins because it allows a rigid joint
rotation and then removes dilation slack with a margin-tight fixed-angle LP.

After merging Claude's mid-range work and syncing upstream, the same runner was
restarted from the current best rows and iterated until convergence. There were
13 passes in total. The last pass produced 11 accepted groups with a group-level
gain of `8.073e-10`, so the loop was stopped at the `1e-9` threshold. The
largest changes came from repeated rotational escapes on `862`, `970`, `936`,
`912`, and `997`.

A second fine-radius schedule then restarted from that file and used
`5e-4 -> 2e-6` with 60 iterations per pass. It produced 7-14 strict wins per
pass, 16 distinct `n` in total, and a group gain of `1.662e-7`. The last pass
still had 8 wins, so it is a useful
post-convergence polish rather than a new topology search. A `1..199` rerun
with the same kink-free model produced no strict win.

A final 100-iteration pass at the original `0.002` trust then found 5 more
strict wins, showing that the fine-radius convergence had left a small
longer-range escape available.

A wider schedule at `0.004` and then `0.006` found the first real jump:
`n=645` moved by `1.117e-5`. Across three wide passes, six distinct `n`
improved and the total group gain was `1.123e-5`.

After that, trust-`0.008` found 5 strict wins, trust-`0.010` found 5,
trust-`0.012` found 6, and trust-`0.014` found 5. Across the four later passes,
8 distinct `n` improved:
`747, 791, 816, 868, 908, 910, 912, 936`. The final R29 rows were gated against
the then-current upstream incumbent and accepted `6/6`; R30 was then gated
against the newer PR #7 incumbent and accepted `5/5`.

## Latest incumbent

The file was re-merged against the upstream PR4/PR5 union. The new upstream
incumbent is:

```text
literal 22.173715450068453
pandas  22.173715450068453
accepted 1000/1000
```

The per-puzzle union keeps 55 coarse plus 16 fine high-n strict wins and has no
regression. The final file is:

```text
literal 22.173714766133950
pandas  22.173714766133950
accepted 1000/1000
```

The later long-escape file is:

```text
literal 22.173714763006313
pandas  22.173714763006313
accepted 1000/1000
```

`final.csv` contains the best currently certified rows for all 58 strict wins
owned by this POC. Its per-n gate accepted all 58 groups in both transports
against the current `merged.csv`. Unioning `final.csv` into the current
incumbent gives:

```text
literal 22.173714780634086
pandas  22.173714780634086
accepted 1000/1000
```

The exact per-n values and the pandas-round-trip checks are in
`data/multi-round-gate.json` and `docs/MULTI_ROUND_WINS.md`.
The fine-round values are in `data/fine-round-gate.json` and
`docs/FINE_ROUND_WINS.md`.

## Verification

The full merged file was scored with the unchanged official metric on the
literal strings and after an actual `pandas.read_csv(...).to_csv(index=False)`
round trip:

```text
literal 22.173714780634086 1000 0 0
pandas  22.173714780634086 1000 0 0
```

## Provenance

Search code is from the collaborator POC `poc/claude-midn`; this folder owns
only the high-range runner, exact result rows and measured summary.
