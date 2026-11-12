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

## Verification

The full merged file was scored with the unchanged official metric on the
literal strings and after an actual `pandas.read_csv(...).to_csv(index=False)`
round trip:

```text
literal 22.173715436104136 1000 0 0
pandas  22.173715436104136 1000 0 0
```

## Provenance

Search code is from the collaborator POC `poc/claude-midn`; this folder owns
only the high-range runner, exact result rows and measured summary.
