# Rounded gallery repair

Fixed-angle SAT-guided linear programming repair of the public Squares in Squares gallery. The gallery rounds coordinates to three decimal places and angles to two decimal degrees. Geometry is imported as data only, never executed.

The November 10 run repaired all 553 tilted layouts. Every repair passed official literal and pandas 3.1.0 checks, plus C++ literal and round-trip checks. Only 194 strictly improved the incumbent and entered the merged portfolio. Official merged score was 22.216763458415837; all 1,000 puzzles passed all six whole-file checks.

The solver minimizes box side under a fixed separating-axis topology and a +/-0.03 center trust region. This is not a global minimum guarantee. Restored coordinates are exact decimal expansions of floating-point proposals, with high-precision angles and safety dilation, not a claim of 65-digit numerical optimization. Relative to the gallery's rounded side labels, growth reached 0.000799221, above the initial 1e-6 target; no invalid packing was retained.

Source and credit: https://squares-in-squares-gallery.vercel.app . Original packing authors retain credit. Repair code is independent. Per-puzzle results and certificates are in `../pell_corrugation/data/border-scan/gallery-repair-provenance.json` and `gallery-repair-attempts.json`.

Run `python poc/gallery_repair/poc/repair_gallery.py`. Set `GALLERY_INPUT` to the directory containing `gallery/{n}.json`, `GALLERY_AUDIT` to the prior gallery audit directory, `SQUARES_CERTIFIER` to the compiled C++ certifier, and `GALLERY_REPAIR_OUTPUT` to a new output directory. A run requires the pre-repair incumbent and its literal official report. Repairs are resumable from per-n JSON files; use a fresh output directory when changing inputs or algorithm. The script never submits, commits, or pushes.
