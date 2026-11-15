# Transport sequence beam

## Idea

H7 and H16 showed that a Sinkhorn transport followed by a short fixed-side
repair can move a jammed packing into a new feasible basin. Earlier searches
used one transport map and one long repair.

This POC treats each short transport/repair pair as a beam-search state:

1. map the current geometry to one or more nearby donors;
2. repair briefly at the current side with a small expansion allowance;
3. transport the repaired geometry to a second donor;
4. repair at a target side below the incumbent;
5. dilate minimally, tighten with fixed-angle LP, and certify both transports.

The beam retains candidates by residual penetration and donor-path diversity.
The purpose is to test whether an intermediate geometry can cross a barrier
that neither a single transport nor ordinary local polish can cross.

## Run

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
PYTHONPATH=poc/baseline/poc/src $PY \
  poc/transport_sequence_beam/poc/probe.py \
  --targets 803,367,328 \
  --beam-width 8 --stage1-repair 500 --stage2-repair 2500 \
  --stage1-starts 8 --stage2-starts 20 --workers 3 \
  --out poc/transport_sequence_beam/docs/results.json
```

No public notebook or dataset coordinates are used. No Kaggle submission is
made.

## Results

**Quick probe: negative.** The two-stage beam ran on the three top-gap fast
targets and on the known 791/908 transport controls. The intermediate repair
was feasible in several cases, but the second transport consistently moved
the geometry back toward a more conflicting state. Its required safe dilation
then exceeded the incumbent side.

| n | incumbent L | stage-2 starts | best stage-2 violation | best safe side | result |
|---:|---:|---:|---:|---:|---|
| 803 | `28.81726587682309` | 20 | `9.01e-5` | `28.81967894110539` | above incumbent |
| 367 | `19.692388155425203` | 20 | `1.19e-5` | `19.69252169843161` | above incumbent |
| 328 | `18.614368938490983` | 20 | `1.34e-4` | `18.61678978373945` | above incumbent |
| 791 | `28.627434433035795` | 20 | `7.27e-6` | `28.62754207447928` | above incumbent |
| 908 | `30.62753521206918` | 20 | `2.72e-5` | `30.628239069190208` | above incumbent |

The first stage frequently found a nearly feasible intermediate layout, but
the second map destroyed the useful wall placement. For example, the best
791 intermediate had residual `1.35e-8`, while after the second transport and
repair it rose to `7.27e-6`. The controls confirm that the negative result is
not a failure to reproduce the one-stage basin: the comparison uses the
already-improved incumbent sides.

This closes the sampled 2-stage transport/repair beam. A future sequence would
need a mapping that updates only the local wall contacts rather than remapping
all square centers.

No complete strict candidate was produced, so `final.csv` remains header-only.
