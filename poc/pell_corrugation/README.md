# Pell-17 Corrugated Bands

Out-of-phase pinch-column corrugated staircase bands with upright L-border wrapping (`D = 17, l = w + 12`) and exact transport certification. Produces a certified improvement at `n = 910` (`L = 30.690921074188` vs public/targeted incumbent `30.690960120646`, `delta_L = -0.00003905`) and beats the `rtx4070s-20261101` portfolio baseline on 10 of 13 target groups (`543-545`, `591-593`, `908-910`, `971`).

Run from the repository root after installing the shared package.

```sh
python3 poc/pell_corrugation/poc/pell_corrugation.py --help
```

[Measured report](docs/pell_corrugation.md). Saved CSVs and results are in `data/`.
