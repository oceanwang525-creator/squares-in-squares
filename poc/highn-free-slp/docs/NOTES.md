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

R31 moved the trust radius to `0.016` for 100 iterations. It found four strict
wins (`791, 868, 910, 936`), all accepted `4/4` against the R30 union. The
R30/R31 progress shows that the larger trust radius is still productive after
the fine passes have converged.

R32 moved the trust radius to `0.018` for 100 iterations from the complete
`merged.csv` plus `final.csv` union. It found five strict wins (`791, 868,
908, 910, 936`), accepted `5/5` against the upstream incumbent. The `908`
move is small (`1.077e-12` in `L`) but is the first strict SLP improvement at
that count in this POC.

R33 moved the trust radius to `0.020` for 100 iterations from the R32 union.
It found three strict wins (`791, 868, 936`), accepted `3/3` against the
upstream incumbent. The internal group gain was `9.577504056198076e-10`; `868`
supplied most of it with `dL = 2.798e-8`.

R34 moved the trust radius to `0.022` for 100 iterations and found four strict
wins (`695`, `791`, `868`, `936`), accepted `4/4`. The internal group gain was
`1.0139526745110658e-9`. The per-group literal changes were:

```text
695  -1.8768938915684757e-12
791  -5.653345294808802e-9
868  -3.1058414782230857e-7
936  -2.9987539359681513e-7
```

This is still numerical polishing, not a change of topology. It is recorded
because the row gates and full-file certification are exact.

R35 through R39 raised the trust radius from `0.024` to `0.036` in steps of
`0.002`, `0.004`, and `0.004`. Every pass again reported 56 strict wins, but
most were replacements on a heavily overlapping set of groups. A per-n update
gate now compares both official transports before replacing an existing
`final.csv` group.

The overlap replacement counts were:

```text
R36  18
R37   5
R38   3
R39  10
```

R39 replaced `604, 687, 703, 748, 791, 818, 914, 971, 981, 990`. It remains
numerical polishing, but `n=971` improved by `4.034e-9`, an unusually large
change for this late stage.

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

`final.csv` contains the best currently certified rows for all 58 strict-win
`n` groups owned by this POC. Its per-n gate accepted all 58 groups in both transports
against the current `merged.csv`. Unioning `final.csv` into the current
incumbent gives:

```text
literal 22.173714779017388
pandas  22.173714779017388
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
literal 22.173714779017388 1000 0 0
pandas  22.173714779017388 1000 0 0
```

R33 and R34 were then folded into `final.csv` and re-certified from the latest
upstream incumbent:

```text
gate=58/58 literal=22.173714777045685 1000/1000 pandas=22.173714777045685 1000/1000
```

The final R39 union was certified from the unchanged incumbent with:

```text
gate=58/58 literal=22.173714776462209 1000/1000 pandas=22.173714776462209 1000/1000
```

R40 moved the trust radius to `0.040`. It found 58 strict wins against the
incumbent and replaced six existing final groups (`644, 689, 695, 740, 791,
869`). The certified result is:

```text
gate=58/58 literal=22.173714776460457 1000/1000 pandas=22.173714776460457 1000/1000
```

The net change from R39 is only `1.752e-12`. At this point the deterministic
trust sweep is acting as numerical polishing. Any material gain has to come
from a different contact topology.

R41 moved the trust radius to `0.050`. It found 58 strict wins and replaced
six existing final groups (`748, 791, 818, 874, 934, 988`). The certified
union is:

```text
gate=58/58 literal=22.173714776457253 1000/1000 pandas=22.173714776457256 1000/1000
```

The net change from R40 is `3.204e-12`. This remains numerical polishing; the
larger radius gives no indication of a new contact topology.

R42 repeated trust `0.050`. It found 55 strict wins and replaced seven final
groups (`703, 738, 748, 791, 793, 908, 988`). The full certification is:

```text
gate=58/58 literal=22.173714776454094 1000/1000 pandas=22.173714776454094 1000/1000
```

The net change from R41 is `3.159e-12`. This is treated as the numerical floor
of the current contact graph; further trust sweeps are not a route to the
target score.

## Provenance

Search code is from the collaborator POC `poc/claude-midn`; this folder owns
only the high-range runner, exact result rows and measured summary.
