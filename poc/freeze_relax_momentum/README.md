# freeze_relax_momentum

Freeze the lattice/regular part of the incumbent, relax the rest with a penalty optimizer (heavy-ball momentum, beta 0.9) while hardness rises (container shrink 2e-5 per stage, penalty weight x1.05 per stage), then rigid repair (centre dilation 1+1e-9..1e-6) and certification with the official metric (`squares.core.evaluate`).

**Estimated score gain: 0. No win.**

Recipe caveat: the Kaggle discussion 756739 page and the Santa 2025 1st-place writeup could not be retrieved (page returned only its title; search found no writeup). This is a generic reconstruction of the described idea, not the authors' exact method.

## Evidence (measured side L, official metric; `docs/run_n*.log`, 150 s each)
| n | incumbent L | free / total squares | best repaired L | shrink reached with penalty E < 1e-3 |
|---|---|---|---|---|
| 634 | 25.70710678119 | 292/634 | 25.70710680589 (stage 0, no shrink) | ~1.2e-3 but E never ~0 after stage 0 |
| 738 | 27.67938124879 | 408/738 | 27.67938151668 | same |
| 685 | 26.68414360636 | 486/685 | 26.68414386420 | same |

Every "repaired" result is the unshrunk incumbent plus dilation, i.e. worse than the incumbent. For every shrink step >= 2e-5 the penalty energy stayed positive and grew with hardness, so no feasible shrunken packing was found; nothing reached repair. No CSV written; merged.csv untouched.

## Negatives / limits
- The incumbents at these n look rigid (jammed contact networks); a 2e-5 uniform shrink of the free region was already infeasible within the frozen environment.
- Only 150 s per n, one seed, three radius/band settings (one each). Stronger variants (random jiggle between hardening stages, multiple seeds, bigger free regions, swapping squares) not tried. Total CPU ~8 min.
- Note: `high_n_gap_ranking.md` lists L(738)=27.690919 but merged.csv measures 27.679381.

Run: `python poc/freeze_relax_momentum/poc/freeze_relax.py N radius band seconds` from repo root.
