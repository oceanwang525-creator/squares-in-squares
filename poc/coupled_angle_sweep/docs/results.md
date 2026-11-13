# November 12 coupled angle sweep

The incumbent was the tracked `poc/pell_corrugation/data/border-scan/merged.csv` at `b57e78b89784a87904f0793ec388276741d9f8e8`, SHA256 `1f9c0fd3b7a88e5a0f6c44b567ff2ef354872812c6b0ad8e37807bd1a64e965c`.

| Outcome | Count |
| --- | ---: |
| Target puzzles | 26 |
| Unforced controls | 26 |
| Forced starts | 272 |
| Candidates with lower side after feasibility repair | 0 |
| Accepted replacements | 0 |

Targets included the larger documented leader gaps and several midrange tilted layouts. The trial signs were −0.018, −0.006, +0.006 and +0.018 radians on up to three groups per puzzle. A candidate's actual separating axes are recalculated before applying the minimum common dilation needed to restore non-overlap with a safety margin. The repaired candidate must beat the input before the official literal and pandas checks run.

The five-square diamond positive control, with centers dilated by 0.8%, compressed from `2.72076363543604` to `2.707108195403833` and passed the official literal evaluator. For the 26 production targets, no candidate reached the official acceptance stage. This measures the selected groups, rotation steps and repair path. The model does not test solutions that change the grouping or substantially alter the contact topology.

From the repository root with NumPy, SciPy and the baseline dependencies installed:

```text
python poc/coupled_angle_sweep/poc/sweep.py poc/pell_corrugation/data/border-scan/merged.csv coupled-angle-wins.csv coupled-angle-results.json 157 183 257 328 365 367 411 453 497 585 634 642 686 687 740 747 791 793 803 908 910 911 969 971 972 973 --workers 2
```

The results JSON is written even when no candidate CSV is emitted.
