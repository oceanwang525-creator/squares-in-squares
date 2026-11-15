# Localized donor weights on remaining top gaps

## Idea

H21 showed that a spatial field over the two-donor Sinkhorn barycenter can
improve an already certified optimal-transport winner. This POC applies the
same mechanism to the remaining high-gap rows instead of spending every wave
on `n=791`.

The effective incumbent is the best complete group for each `n` among the
canonical row and all existing `poc/*/final.csv` files. Each candidate is
still a complete `n`-square packing. A candidate is exported only after the
unchanged official metric accepts both its literal decimal strings and the
actual pandas `read_csv` / `to_csv` round trip at a strictly smaller side.

## Initial wave

The first wave screens:

`969,861,747,804,848`

It uses the proven H21 center-transport, localized two-donor mixture,
fixed-side repair, minimum safe dilation, and fixed-angle LP. The parameter
grid includes donor offsets `1,2,3,4`, several Sinkhorn temperatures,
interpolation strengths, spatial weight patterns, and small target-side
reductions.

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
PYTHONPATH=poc/baseline/poc/src $PY \
  poc/localized_topgaps/poc/probe.py \
  --targets 969,861,747,804,848 \
  --seconds-per-target 120 --repair-iterations 8000 \
  --top-starts 32 --workers 5
```

No public notebook or dataset coordinates are used.

## Results

**Initial five-row wave: negative.** Each target screened 27,000 localized
weight/transport starts and completed 4-5 long repairs. The residual conflict
was distributed over the whole jammed interface rather than a local endcap, so
minimum safe dilation erased the requested side reduction.

| n | incumbent L | best residual | best safe L | result |
|---:|---:|---:|---:|---|
| 969 | `31.62763575853136` | `3.50e-5` | `31.628711291542565` | above |
| 861 | `29.815764558940618` | `5.59e-5` | `29.817283459500146` | above |
| 747 | `27.810663458507218` | `5.76e-5` | `27.812061907137192` | above |
| 804 | `28.827155380556654` | `8.08e-5` | `28.82948602213017` | above |
| 848 | `29.6230652120617` | `5.59e-5` | `29.624721042794693` | above |

The nearest safe dilation overshoot is still `1.1e-3`, more than two orders of
magnitude beyond the useful sub-`1e-5` regime seen in the successful transport
rows. This implementation therefore stops at the quick-falsification gate.
No complete candidate was certifiable at a smaller side, so `final.csv`
contains only the required header.
