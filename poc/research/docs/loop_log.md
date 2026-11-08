# Optimization Loop Log

Record of attempts and variants explored to beat incumbent packings in `poc/pell_corrugation/data/border-scan/merged.csv`.

## Baseline Reference
- Baseline Portfolio Score (`merged.csv`): `22.254062083924335`
- Evaluator: Official metric `certify` with exact 50-digit high-precision arithmetic under both literal CSV and pandas 3.1.0 transport round-trip.

---

## Log of Attempts

### Attempt 1: Asymmetric Argon Defect Propagation & Pocket Screening
- **Idea / Variant**: Perturb boundary defects and pocket shifts on Argon family configurations for $n \in [900, 999]$.
- **Why**: Argon family holds records for several near-1000 configurations (such as $n=910$); checking if corner lattice relaxation yields tighter envelopes.
- **Target $n$**: $910, 971, 972$
- **Result**:
  - $n=910$: Candidate $L = 30.690921074188$, Incumbent $L = 30.690921074188$ ($\Delta L = 0.0$, tie - not a win).
  - $n=971$: Candidate $L \ge$ incumbent $L$ (no strict improvement).
  - $n=972$: Candidate $L \ge$ incumbent $L$ (no strict improvement).

### Attempt 2: Dual Pocket & Variable Slope Band Search
- **Idea / Variant**: Variable slope band insertion at tilted boundaries ($w=9, 10, 12$) with dual-pocket corner packing.
- **Why**: Corner pockets often have sub-unit slack that can accommodate additional squares if tilt angles deviate slightly from $\arctan(1/\sqrt{2})$.
- **Target $n$**: $794, 800, 850$
- **Result**:
  - $n=794$: Candidate $L = 28.690921187280$, Incumbent $L = 28.690921187280$ ($\Delta L = 0.0$, tie).
  - $n=800$: Overlap violations during high-precision certification.
  - $n=850$: Extent exceeded incumbent bounding box.

### Attempt 3: Structured Grid-Border L-Wrap on Certified Pell-17 Core ($w=13$)
- **Idea / Variant**: Exact structured grid-border ($t=2$) wrapping on certified width-13 Pell-17 core ($n_0=739, L_{\text{core}} = 27.690921182142$) to replace asymmetric coordinate-sliced extreme-prefix truncation.
- **Why**:
  The incumbent $n=851$ in `merged.csv` was generated via `kind: "extreme-prefix", axis: 0, reverse: false`, which sorted squares along $x$ from an overfull wrapper and truncated the first 851. That truncation sheared off boundary squares unevenly, forcing a looser bounding box ($L = 29.690933698508$).
  In contrast, exactly partitioning $n=851 = 739 + 112$ into the $n_0=739$ core plus a structured $t=2$ L-border ($2 \times 29$ East strip + $27 \times 2$ North strip) places all 112 border squares symmetrically along the outer perimeter with zero interior disturbance, strictly preserving vertical and horizontal contact limits.
- **Target $n$**: **$851$**
- **Incumbent certified $L$**: `29.69093369850775100000000000000000000000000000000000`
- **Candidate certified $L$ (Literal)**: `29.69092199588171600961566084573689062356096442229735`
- **Candidate certified $L$ (Pandas 3.1.0)**: `29.69092199588171600961566084573689062356096442229735`
- **$\Delta L$**: **$-1.17026260362251833 \times 10^{-5}$** (STRICT WIN)
- **Portfolio Impact**:
  - Full 1000-puzzle portfolio score before: `22.254062083924335`
  - Full 1000-puzzle portfolio score after: `22.254061682763480`
  - Portfolio score $\Delta$: **$-4.01160855 \times 10^{-7}$**
  - Accepted: 1000 / 1000 on both literal CSV and pandas 3.1.0 round-trip.

---

## Winning Candidate Summary

| Target $n$ | Incumbent $L$ | New Certified $L$ | $\Delta L$ | Literal Cert | Pandas 3.1.0 Cert | Construction Idea |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **851** | `29.690933698507751...` | `29.690921995881716...` | **$-1.1702626 \times 10^{-5}$** | **PASSED** | **PASSED** | Exact structured grid-border $t=2$ on certified Pell-17 $w=13$ core ($n_0=739$), eliminating asymmetric coordinate-slice truncation |
