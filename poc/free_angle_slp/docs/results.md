# Measured results

## Scope

| check | result |
|---|---|
| groups scanned (all `n` = 1..1000) | 1000 |
| groups with an SLP side below the incumbent | **0** |
| strict per-`n` wins after exact literal + pandas certification | **0** |
| `merged.csv` changed | no |

The full run's per-`n` SLP side and status are in `docs/scan.json`
(1000 × `no-gain`). `docs/wins.json` is an empty object.

## Top-gap probe

Ten largest leader gaps, 9 seconds total:

```text
n=971 no-gain 31.664318325 vs 31.664318325
n=911 no-gain 30.695622171 vs 30.695622171
n=910 no-gain 30.679149465 vs 30.679149465
n=972 no-gain 31.683990215 vs 31.683990215
n=793 no-gain 28.677242856 vs 28.677242856
n=634 no-gain 25.707106781 vs 25.707106781
n=791 no-gain 28.634070753 vs 28.634070753
n=908 no-gain 30.634070758 vs 30.634070758
n=969 no-gain 31.633982502 vs 31.633982502
n=803 no-gain 28.819189735 vs 28.819189735
```

## Conclusion

The portfolio is a local optimum in the full `(x, y, θ)` space under this SLP.
The remaining leader gap is structural, not numerical. No metric, tolerance or
tracked file was changed.
