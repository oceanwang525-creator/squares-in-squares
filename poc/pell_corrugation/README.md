# Pell-17 Corrugated Bands

Out-of-phase pinch-column corrugated staircase bands with upright L-border wrapping (`D = 17, l = w + 12`) and exact transport certification. Produces a certified improvement at `n = 910` (`L = 30.690921074188` vs public/targeted incumbent `30.690960120646`, `delta_L = -0.00003905`) and beats the `rtx4070s-20261101` portfolio baseline on 10 of 13 target groups (`543-545`, `591-593`, `908-910`, `971`).

Run from the repository root after installing the shared package.

```sh
python3 poc/pell_corrugation/poc/pell_corrugation.py --help
```

[Measured report](docs/pell_corrugation.md). Saved CSVs and results are in `data/`.

The completed CPU border scan improves the public23 merge at n=739, 794, 851 and 910. Whole-portfolio official score is **22.254062083924335**, down from 22.25406709809781. All 1000 groups pass literal CSV and actual pandas 3.1.0 round-trip certification, independently checked by C++. See [border scan](docs/border_scan.md).

The current tracked portfolio is **22.17382428260707** after the November 12
public-output import unioned with team merge `e6f7b69`, the post-resweep
refresh (`n=487`, `n=733`), seven external witnesses
(`n=132,156,157,180,237,268,269`), one more external bank win at `n=269`
(`16.899469075855`), the full-portfolio fixed-angle LP (`n=595`, `n=875`), and
per-`n` unions with the team's `public_refresh_nov12` and
`certified_union_nov12` files (`n=971`, `n=687`, ...), and a midday kernel
refresh (`n=694`), the live tracker (`n=269`, `16.889730073596`), and the
morning kernel refresh (114 `n`), the tiny deletion/wrap pass (8 `n`), the sigmaborov contact polish (3 `n`), two fixed-angle LP re-runs (`n=740`, `n=595`), the jazivxt refresh
(17 `n`), and the corrected Lipschitz-safe free-angle SLP (`n=740`).
Both literal and pandas transport certify 1000/1000. See
[fresh public outputs](../fresh_public_sources_nov12/README.md),
[post-resweep refresh](../post_resweep_refresh/README.md),
[external witness refresh](../external_witness_refresh/README.md),
[external bank expansion](../external_bank_expansion/README.md),
[full-portfolio fixed-angle LP](../alln_fixed_lp/README.md),
[team union](../team_union_nov12/README.md),
[midday kernel refresh](../kernel_refresh_noon/README.md),
[live tracker watch](../live_tracker_watch/README.md),
[morning kernel refresh](../kernel_refresh_morning/README.md),
[tiny pass re-run](../tiny_pass_rerun/README.md),
[sigmaborov contact polish](../sigmaborov_contact_polish/README.md), and
[measured results](../sigmaborov_contact_polish/docs/results.md).
