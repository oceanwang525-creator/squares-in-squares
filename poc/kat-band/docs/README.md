# Thick tilted bands

Research prototype for equal unit squares; all angles are radians. This does not
claim global optimality, a competitive win, or permission to submit.

band_geometry.py independently constructs finite off-center bands. Width-2
Wainwright n19 and width-3 Stenlund n66 match source-authored SVG geometry.
The gate artifacts passed the original metric on literal decimals and an
ordinary pandas roundtrip. Widths, phases, angle, corner depths, layer shifts
and separate endpoint caps are explicit parameters. Geometry uses no atlas
coordinates. Corner triangles are intentionally incomplete outside this family.

band_relax.py uses float64 PyTorch, near-pair SAT penalties, independent angles,
collective strip moves and annealed perturb-and-shrink. Periodic all-pair checks
and best retention are screening only, never a certificate. run_relax.py refuses
CPU for the main search. Run on Colab A100, one runtime, then stop and verify it.

band_polish.py implements sparse HiGHS fixed-angle LP and free-angle trust-region
proposals followed by exact fixed-angle repair. Every accepted step must satisfy
all-pair SAT and nonworsening measured L. An external validator callback can gate
every update with the official interval metric.

Install numpy, scipy, mpmath and torch in an isolated environment. Run unittest
discovery here. The SVG-oracle test needs the external source SVG dataset; all
other unit tests are portable. Competitive outputs require independent C++
literal and roundtrip plus original-metric pandas verification.

Baseline control rows must preserve original decimal strings for final merges.
Do not mistake float64 screening, lower loss, or changed parser behavior for a
verified gain. Pin pandas to the successful reference version when reproducing
serialization checks; certification artifacts document the parser mismatch.

The generator now exposes augment_to_target. It greedily fills actual upright
vacancies first, then explicitly inserts low-conflict overlapping squares only
for bounded remaining deficits. requires_repair in metadata must be respected;
exact count is not a validity certificate. Independent n909 topology starts at
907 cells; its two endpoint insertions have depth about 0.271 before repair.
