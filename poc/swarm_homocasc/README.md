# Hypothesis H30: Homology Cascade from New PR #32 `mechwalk` & `brickslp` Wins (`swarm_homocasc`)

## Summary
- **Branch**: `swarm-homocasc` (based on `origin/main` @ `b67cd44`)
- **Result**: **2 strict REAL WINs** (`n=850, 909`), **total `ds = -7.307473e-04`** (`literal == pandas` dual-certified against `origin/main` `b67cd44` portfolio).

| `n` | `L_inc` (`origin/main` `b67cd44`) | `L_win` (`literal == pandas`) | `dL` | `ds` | Method |
|---:|---:|---:|---:|---:|---|
| 850 | `29.648984277273773665621` | `29.64257301327960054430204358155568882541714595573275` | `-6.411264e-03` | `-4.472163e-04` | `peel1_909_BL_k0.990_reloc_v1_u0` + masked melt + `joint_xytheta_slp` + 2-round `mechwalk_compress` |
| 909 | `30.646784800324614321230` | `30.64257967119275752119589417028284162798463453944575` | `-4.205129e-03` | `-2.835310e-04` | Upward `1x1` L-border extension from new `win_n850` + masked melt + `joint_xytheta_slp` + 4-round `mechwalk_compress` |

## Mechanism & Gate / Scale-Up Findings
1. **15-Minute Falsification Gate (`docs/gate_results.json`)**:
   - **`n=850 <- 909` (`L_inc=29.648984`)**: Peeling the pure `59`-square `1x1` `BL` L-border from `n=909` (`30.646785`) left only 1 protruding tilted square (`idx=61`, `amt=0.10332`). Direct clamping + SLP yielded `29.646786` (`dL=-2.199e-03`), void relocation (`reloc_v1_u0`) + masked melt + SLP broke through to `29.644237` (`dL=-4.748e-03`), and `mechwalk_compress` further compressed `n=850` to **`29.642573013280` (`dL=-6.411264e-03`)**, opening a `-4.21e-03` gap *below* `n=909` and harvesting the `B5_10x19` (`f=0.642571`) and `B4_17x9` (`f=0.642571`) sub-bricks.
   - **`n=852 <- 911`**: `peel1_911_BL_k1.020_dropProt_clamp` + 2-round `mechwalk` reached `29.699760` (`dL=+6.296e-04`).
   - **`n=635 <- 686`**: Best `25.792855` (`dL=+8.575e-02`); `686` has `prot=4` (`max=0.437`) across `BL`, pinning `n=635` above `25+1/sqrt(2)`.
   - **`n=793 <- 909+850`**: Reached `28.665175` (`dL=+5.778e-03`).
2. **Scale-Up (`docs/scale_results.json`, `docs/scale.log`)**:
   - **`n=909 <- 850` (upward cascade)**: Extending `win_n850` (`29.642573`) upward by one `59`-square `1x1` L-border + masked melt + SLP + 4-round `mechwalk` landed **`30.642579671193` (`dL=-4.205129e-03`)**.
   - **Near-miss & negative pairs**: `586<-635` (`dL=+3.54e-06`), `740<-795` (`dL=+5.41e-04`), `852<-911` (`dL=+1.80e-03`), `851<-910` (`dL=+1.89e-03`), `970<-909` (`dL=+3.59e-03`), `972<-911` (`dL=+4.89e-03`), `793<-850` (`dL=+7.56e-03`), `975<-914` (`dL=+1.02e-02`), `855<-914` (`dL=+3.29e-02`), `910<-971` (`dL=+3.47e-02`), `794<-910` (`dL=+5.08e-02`), `795<-911` (`dL=+1.02e-01`), `794<-851` (exact tie `28.684143606359`).

## Layout & Verification
- `poc/homocasc_lib.py`: fast `origin/main` portfolio loader, dual `literal + pandas` certifier, atomic locked win storage, SLP + sparse contact-Jacobian SVD `mechwalk_compress`.
- `poc/cascade_engine.py`: multi-cut `BL/TL/BR/TR` L-border peeling, interior upright-cross surgery, upward `1x1` L-border extension, C++ masked melt + SLP + mechwalk pipeline.
- `poc/run_gate.py`, `poc/run_scale.py`, `poc/run_wave3.py`: gate, scale, and Wave 3 runners.
- `tests/test_homocasc.py`: unit & `final.csv` dual-certification tests.
