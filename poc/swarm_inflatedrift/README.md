# `swarm_inflatedrift` — Hypothesis H124 (Coupled Over-Inflation / Wall-Drift C++ Melt + 4-Corner Reduced PSLP + `lpslack` + `mechwalk`)

## Overview
`poc/swarm_inflatedrift` couples:
1. **Stage 0 (`pslp_4c` + `lpslack`)**: 4-corner static-block-reduced arm-row free-angle kink-safe LP squeeze (`poc/pslp_4c.py` + `poc/arm_model.py`, preserving both `-theta` and `+theta` corners across reduced inner blocks and full blocks) followed by 2nd-order contact-slack LP (`poc/swarm_lpslack/poc/lpslack.py`) when gap < `5e-5`.
2. **Track 1 (`overinf_mpk`)**: C++ masked-pocket over-inflation (`a_start = 1.0015 .. 1.0030 -> 1.00`, implemented in `poc/inflatedrift_cpp.cpp`) with directional wall-drift (`kx, ky` boundary-anchored squeeze) followed by coupled `pslp_4c + lpslack` polish.
3. **Track 2 (`soft_melt`)**: C++ interior soft-`a` wall-drift thermal melt followed by coupled `pslp_4c + lpslack` polish.
4. **Track 3 (`mechwalk`)**: `mechwalk` SVD soft-mode perturbation masked melt + coupled `pslp_4c + lpslack` polish.

Every candidate is dual-certified (`literal == pandas == kaggle_roundtrip`) against the live `origin/main` portfolio (`merged.csv` + all `poc/*/final.csv`).

## PR1 Results (8-Target Gate vs `origin/main` `a2293f6`)
Tested 8 targets (`453, 459, 743, 914, 916, 974, 975, 976`) across two batches (`pilot2` and `gate6`), yielding **6 strict dual-certified wins**:

| `n` | `L_new` (`literal == pandas == kaggle_roundtrip`) | Incumbent `L` (`merged.csv`) | `dL` |
| --- | --- | --- | --- |
| 453 | `21.77809630301240608620557513361170572831685236486139` | `21.77809630303069377475828372552080942829402424466665` | `-1.828769e-11` |
| 459 | `21.89499783005804004511938527216958877886952251276065` | `21.89499828173165578516614629249828218007886507984216` | `-4.516736e-07` |
| 743 | `27.75218922201035772579939692647606697176363155670199` | `27.75219911014502018473946740730354496024073426404537` | `-9.888135e-06` |
| 914 | `30.72975275511344007576201553032648403027116648106630` | `30.72975429984269149809712368171184924240300138293158` | `-1.544729e-06` |
| 974 | `31.69257719959248780742352703848976174683444791284957` | `31.69258193382914020216233915998624843367026953288747` | `-4.734237e-06` |
| 975 | `31.71583985461358743658239303779597834193106625993044` | `31.71584144380572014378497894028910762718244091313506` | `-1.589192e-06` |

## Verification Commands
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -v poc/swarm_inflatedrift/tests/test_inflatedrift.py
python3 .github/scripts/check_final_csv.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 poc/swarm_inflatedrift/poc/verify_and_export.py
```

## Next Scale-Up Plan
Scale coupled `overinf_mpk` + `mechwalk` + `pslp_4c` + `lpslack` across the remaining `m = 18..31, r = +4..+24` neighbor family on `swarm-inflatedrift` (leaving `swarm-inflatedrift-pr1` frozen for verification).
