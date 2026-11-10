# k2p10_wildcard: free-slope strip with independent rows + anchored grids (negative)

Pick: generalisation of ideas (a)/(c)/(b): instead of testing single rational slopes by hand, count exactly how many squares ANY single tilted strip
(w rows of tilted unit squares, each row sliding independently, theta free in 0.20..1.37 rad, optional upright corner caps) can hold between two upright
grids (each grid: column phase 0 or frac(L), with one phase split, rows anchored bottom/top). Wedge fill is automatic (grid cells outside the strip).
It is a pure counting model (`poc/tiltband.py`, `poc/scan.py`); it reproduces the incumbent (686 at L=26.7071, 635 at 25.7071, via w=1, theta=45deg, SW/NE caps).

## Result: no packing, nothing certified
Exact-ish checks (c grid 1e-3), theta=45deg: 635 needs L>=25.7071 (630 at 25.706); 686 needs 26.7071 (681 at 26.706). The incumbent is the model optimum.
Screening scan (`docs/scan_log.txt`, grid slack 0.02 so optimistic) at leader L and at ours-0.001, best count with theta != 45deg:

| n (need) | K^2 | best non-45deg at leader L | at ours-0.001 |
|---|---|---|---|
| 635 | 625 | 629 | 630 |
| 686 | 676 | 679 | 680 |
| 910 | 900 | 900 | 900 |
| 911 | 900 | 900 | 904 |
| 971 | 961 | 961 | 962 |
| 972 | 961 | 962 | 962 |

Non-45deg strips give at most K^2+5, never K^2+10/11. For K=26 non-45deg first reaches 686 around L~26.9 (theta~0.32, `docs/nonfortyfive_L_sweep_K26.txt`), 0.2 above ours.
(The screen's 635 at 45deg at leader L is a grid-slack artefact; the 1e-3 check above shows 630.)

## Conclusion
No win, no CSV, merged.csv untouched. L measured vs ours vs target: none produced (no packing with n squares below ours exists in this family).
Consequence: tilting/tapering a single strip (with arbitrary slope, row slides, corner caps) cannot explain the leaders' L-K; theirs must use multiple tilted blocks
or interior vacancy relaxation (cf. q3: 908 uses two 100-square tilted pockets at ~26deg, 971 uses a relaxed 13-strip band). Not modelled: several disjoint strips/pockets,
non-uniform row angles, rows with gaps, more than one phase split per grid, falling strips. Those are the remaining open directions for a counting model.
