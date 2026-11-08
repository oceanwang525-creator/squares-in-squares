# Slope-Band Dual-Pocket Symmetric Composite Generator (POC 3)

Synthetic CPU construction for large-$n$ square packings via variable-slope dual-pocket composite geometry, targeting the high-deficit region $(k^2 - n)/k \in [1, 2]$ for $n = 908$ and $n = 968$.

Executes **Mechanism 1 (Variable-Slope Bands)** and **Rank 3 execution plan** from [`poc/research/docs/q5_beyond_atlas_and_plan.md`](../research/docs/q5_beyond_atlas_and_plan.md) and operationalizes the structural findings from [`poc/research/docs/q3_public_leaders_anatomy.md`](../research/docs/q3_public_leaders_anatomy.md).

---

## 1. Geometric Mechanism

Single-band $45^\circ$ constructions force a monolithic band across the full diagonal, dilating the container side by $w/\sqrt{2}$ and wasting space at the corners.

The dual-pocket symmetric composite eliminates this diagonal dilation constraint:
1. **Decoupled Pockets:** Tilted squares are isolated into two independent quadrant pockets (NW Pocket 1 and SE Pocket 2) with angles centered around rational lattice slope $\theta \approx \arctan(1/2) \approx 26.565^\circ$ ($\theta \in [24.1^\circ, 26.6^\circ]$).
   - Reduces transverse inflation penalty by $5.13\%$ relative to $45^\circ$ ($\cos\theta + \sin\theta = 3/\sqrt{5} \approx 1.3416$ vs $\sqrt{2} \approx 1.4142$).
   - 13-column profile: $(5, 7, 8, 8, 9, 8, 8, 9, 9, 8, 8, 8, 5)$ totaling 100 squares per pocket.
2. **Discrete Transposition Symmetry:** The two 100-square blocks are linked by exact transposition symmetry with integer offsets:
   $$\begin{pmatrix} x_2 \\ y_2 \\ \theta_2 \end{pmatrix} = \begin{pmatrix} y_1 - 3.0 \\ x_1 + 2.0 \\ -\theta_1 \end{pmatrix}$$
3. **Solid Central Diagonal Corridor:** Upright squares ($0^\circ$) run continuously along the main diagonal from SW to NE, achieving $100\%$ corridor density (708 upright squares for $n = 908$, container side $L = 30.637030$).
4. **Upright L-Border Composite Wrapping ($n = 968$):** Wrapping the dual-pocket core with 60 border squares (30 East and 30 North) yields $n = 968$ with container side:
   $$L_{968} = L_{908} + 1.0 + \text{margin} = 31.637031$$
   Beats the portfolio baseline ($L = 31.707107$) by **$\Delta L = -0.070076$** with zero boundary jamming.

---

## 2. Execution

Run from repository root with shared squares package installed:

```sh
# Run Fast Falsification Gate
python3 poc/slope_band/poc/slope_band.py --test-falsification

# Generate certified target suite and data artifacts
python3 poc/slope_band/poc/slope_band.py --generate

# Run unit tests
python3 -m unittest poc/slope_band/poc/test_slope_band.py
```

See [measured report](docs/slope_band.md).
