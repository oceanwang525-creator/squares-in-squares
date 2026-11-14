# H99 (`liftpolish3`): 2-Pocket Seam Polish + Step-`0.0625` Sub-Pocket Scan & DP Re-Lift

## Summary

Hypothesis H99 (`poc/swarm_liftpolish3/`) produced **15 dual-certified (`literal == pandas`) strict wins** against `origin/main` (`cf838aa`, `dL <= -1e-9`) exported to [`final.csv`](final.csv) and audited in [`docs/verification.json`](docs/verification.json):

- **Part B Headline Win (`n=852`)**: **`29.6885273895` -> `29.684143610175023`** (**`dL = -4.383779e-3`**), realized by combining `(11, 20, +6)` (`src=2007`, from `poc/swarm_gstep/data/brick_*.json`) with `(17, 9, +5)` (`n=852`) via `dp.py` + `build_cert.py` (`29.6862194285`) and polishing the seam with `lpslack` (`-2.0758e-3` additional drop).
- **Part A Headline Win (`n=799`)**: **`28.7553277454` -> `28.754294804672018`** (**`dL = -1.032941e-3`**), via `lpslack` (`-9.7605e-4`) + 4-round SVD `mechwalk_compress` (`-5.6888e-5`).
- **`scan4` Seam Polish + `border_prop` Extensions**: Additional strict wins on `n=988` (`-2.7209e-5`), `n=530` (`-2.3270e-5`), `n=577` (`-2.3269e-5`, `border_prop` of polished `n=530`), `n=485` (`-1.3744e-5`), `n=588` (`-2.7215e-6`), `n=987` (`-2.0429e-7`), `n=804` (`-1.3562e-8`), `n=538` (`-9.2370e-9`), `n=585` (`-9.2370e-9`, `border_prop` of polished `n=538`), `n=587` (`-5.7019e-9`), `n=803` (`-5.3415e-9`), `n=736` (`-4.6404e-9`), and `n=970` (`-2.6228e-9`).

## Pipeline Components

1. **Part A (`lpslack` + 4-Round SVD `mechwalk_compress`)**:
   - Polished unpolished 2-pocket builds `{538, 585, 588, 743, 799, 803}` (`poc/swarm_lpslack/poc/lpslack.py` + [`poc/run_mechwalk.py`](poc/run_mechwalk.py)) and 13 unpolished `poc/swarm_pocket_lift_scan4/final.csv` rows `{988, 987, 978, 970, 797, 804, 683, 736, 639, 587, 485, 530, 577}`.
2. **Part B (Step-`0.0625` Dual-Phase Scan + `gstep` / `gadlib` DP Re-Lift)**:
   - [`poc/prepare_partb_inputs.py`](poc/prepare_partb_inputs.py) overlays Part A wins and converts `poc/swarm_gstep/data/brick_*.json` bricks (`src >= 2000`, Row-normalized to `x0=0, y0=0`) into pocket records ([`data/gstep_pockets.json`](data/gstep_pockets.json)).
   - [`poc/gadlib_to_pockets.py`](poc/gadlib_to_pockets.py) extracts `gadlib` clean-cut pockets (`data/gadlib_pockets.json`).
   - [`poc/run_partb.sh`](poc/run_partb.sh) runs `scan_pockets.py` at `--step 0.0625 --fmax 1.15 --cap 3000 --procs 1` across phases `(0.015625, 0.015625)` and `(0.046875, 0.046875)`, merges with `poc/pocket_lift_scan3/data/pockets_pareto_e3_sane.json`, runs `sanitize.py` -> `dp.py` -> `build_cert.py`, and [`poc/polish_n852.py`](poc/polish_n852.py) runs `lpslack` on the realized `n=852` packing.
3. **Export & Certification**:
   - [`poc/export_final.py`](poc/export_final.py) dual-certifies all candidates in both `literal` and `pandas` modes via `poc/auto_union/poc/union.py` against `origin/main` (`cf838aa`) and exports all 15 strict wins (`dL <= -1e-9`).

## Certified Wins (`docs/verification.json`, vs `origin/main` `cf838aa`)

| `n` | Incumbent `L` | New `L` (`literal == pandas`) | `dL` | Source |
| --- | --- | --- | --- | --- |
| 485 | `22.525757568745916...` | `22.525743824399685...` | `-1.374435e-05` | `scan4_lpslack.json` |
| 530 | `23.525757568745916...` | `23.525734298491244...` | `-2.327025e-05` | `scan4_lpslack.json` |
| 538 | `23.705882368141986...` | `23.705882358905035...` | `-9.236951e-09` | `partA_lpslack.json` |
| 577 | `24.525757568745916...` | `24.525734299491244...` | `-2.326925e-05` | `wins/win_n577_ext.json` (`border_prop` of `n=530`) |
| 585 | `24.705882369141986...` | `24.705882359905035...` | `-9.236951e-09` | `wins/win_n585_ext.json` (`border_prop` of `n=538`) |
| 587 | `24.741844287669139...` | `24.741844281967282...` | `-5.701856e-09` | `scan4_lpslack.json` |
| 588 | `24.755329545414229...` | `24.755326823949174...` | `-2.721465e-06` | `partA_lpslack.json` |
| 736 | `27.634362450807068...` | `27.634362446166717...` | `-4.640351e-09` | `scan4_lpslack.json` |
| 799 | `28.755327745409380...` | `28.754294804672018...` | `-1.032941e-03` | `wins/win_n799.json` (`lpslack` + `mechwalk`) |
| 803 | `28.817284788017157...` | `28.817284782675622...` | `-5.341535e-09` | `wins/win_n803.json` (`lpslack` + `mechwalk`) |
| 804 | `28.827155394118940...` | `28.827155380556656...` | `-1.356228e-08` | `scan4_lpslack.json` |
| 852 | `29.688527389492102...` | `29.684143610175023...` | `-4.383779e-03` | `wins/win_n852.json` (Part B DP + `lpslack`) |
| 970 | `31.637686867994810...` | `31.637686865372015...` | `-2.622795e-09` | `scan4_lpslack.json` |
| 987 | `31.864244591109686...` | `31.864244386823066...` | `-2.042866e-07` | `scan4_lpslack.json` |
| 988 | `31.873077555721326...` | `31.873050346448629...` | `-2.720927e-05` | `scan4_lpslack.json` |

## Validation

```bash
BASE_SHA=$(git rev-parse origin/main) HEAD_SHA=$(git rev-parse HEAD) /tmp/highrpeel_venv/bin/python .github/scripts/check_final_csv.py
/tmp/highrpeel_venv/bin/python -m pytest -q poc/swarm_liftpolish3/tests/ poc/auto_union/poc/test_union.py .github/scripts/test_check_final_csv.py
```
