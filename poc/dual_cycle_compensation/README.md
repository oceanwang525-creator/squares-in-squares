# Dual-cycle compensation

## Idea

H4 found many contact-face flips that were valid only a few micrometres below
the incumbent and then relaxed back to the same side. This POC tests a
different move: apply two transforms to nearby, disjoint faces at the same
time and release their shared contact ring. Two individually weak ownership
changes may open and close complementary vacancies, producing a topology that
neither face can reach alone.

The candidate construction is:

1. build the contact graph of the in-repository canonical packing;
2. select short chordless faces and rank them by enclosed centre area;
3. pair spatially nearby faces whose release rings interact;
4. assign a deterministic set of rotations/reflections to both faces;
5. re-jam the union of their rings inside the same box;
6. export only a complete strict winner passing literal and pandas transport.

This is a paired discrete topology move, not a point relocation or a repeat of
the single-face search.

## Quick probe

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
$PY poc/dual_cycle_compensation/poc/probe.py \
  --targets 969,908 --seconds-per-target 300 --workers 2
```

The side schedule is
`incumbent + 0.008 -> incumbent + 0.001 -> incumbent - 2e-5`.
A useful pair must retain a candidate at the last side with no significant
overlap. Any export is also required to be at least `1e-9` shorter in both
official transports.

## Results

**Quick probe: negative.**

| n | chordless faces | nearby pairs | screened combinations | fully re-jammed | best penetration at `L_inc - 2e-5` |
|---:|---:|---:|---:|---:|---:|
| 969 | 2,328 | 40 | 480 | 64 | 1.16e-8 |
| 908 | 1,786 | 40 | 480 | 64 | 7.34e-9 |

Both searches exhausted the sampled pair pool in under 30 seconds. The best
combined flip remained only a rounding-scale distance from the incumbent; it
did not produce a lower measured side or a complete candidate passing the
strict-win gate. No pair showed the complementary-vacancy effect proposed by
the hypothesis.

Pairing therefore does not rescue the sampled elementary-face move. The current
graphs are densely triangulated and the robust-looking individual transforms
are local symmetries that return to the same jammed topology rather than
opening a compensating seam.

`final.csv` is header-only because no strict winner was found.

## Limits

The probe samples nearby disjoint chordless faces. Failure here closes the
sampled pairing family, not every conceivable simultaneous mechanism change.
No public coordinates, external packings, or Kaggle submission are used.
