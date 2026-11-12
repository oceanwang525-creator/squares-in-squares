# Cross-packing corner interlocks

The large-`n` arm decomposition in `arm_gadget` identified rotated squares entering vacancies in the upright corner lattice. This pilot tests whether an occupied interlock from another puzzle can replace a small patch at a corresponding corner of the recipient and permit a shorter container.

The generator detects a dense corner lattice and a nearby tilted square whose bounding box enters that lattice. It replaces 9 or 16 squares around the contact with the same number of squares from a different `n`, matched at an upright anchor. A reflection across the local diagonal is also tested. Candidates must differ from the recipient in position or angle after assignment. The donor patch and a small contact halo move continuously against the fixed outside squares, then the whole layout is relaxed with its original count in a container shorter by `0.001`. An incumbent-seeded control uses the same whole-layout relaxation.

The experiment keeps the actual outside squares at the interlock during the patch repair. The swapped patch uses observed occupied donor geometry. `boundary_zipper` tested finite moves across a much larger seam and tried to add a square. `dual-pocket` tested rigid pocket relocation and upright fill. Here the patch is small, the count stays fixed, and the patch and its contact halo can deform.

The pilot produced no wins in 24 transplants across six groups. The best attempt was `0.00195` above the incumbent at `n=969`. Fixed-angle LP polish on three groups also produced no wins.

Run from the repository root with NumPy, SciPy and the shared baseline package installed.

```sh
python poc/corner_interlock_patch/poc/patch_search.py
python poc/corner_interlock_patch/poc/patch_search.py --targets 908 969 971 --polish
```

The `--polish` run reuses the repository's fixed-angle LP on the donor-seeded contact states after the penalty repair. The JSON reports in `data/` record the baseline hash, control and transplant measurements. A candidate file is emitted only after a strict per-puzzle improvement passes both the bundled official evaluator on the literal CSV and the actual pandas read/write round trip. See [the measured results](docs/results.md).
