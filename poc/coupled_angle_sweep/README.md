# Coupled angle sweep

This POC tests whether forcing a coherent rotation across a group of contacting tilted squares opens a smaller packing. It joins contacting squares of nearly equal angle magnitude, holds the upright grid angles fixed, and makes one signed angle variable per connected group. For each seed it forces a rotation of one of the largest groups, then re-solves the signed contact and wall constraints while releasing the forced step.

Unlike the unrestricted small-step refinement in [contact_slp_submission189](../contact_slp_submission189/README.md), the initial rotation is imposed even when the initial LP cannot find a descending step. The LP retains the four branches of the support function so coincident angles have the right directional constraints.

**Result:** zero lower-side packings in 298 starts across 26 targets. The positive control recovered a feasible packing after deliberate dilation. See [results and reproduction](docs/results.md) and [per-target data](data/results.json).
