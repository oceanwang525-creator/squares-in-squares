Boundary zipper POC finished on CPU. No verified win and no successful seam insertion. All files are confined to this directory on kaggle-squares. No git operation, GPU, submission, installation, shared-file edit or other host was used. No CLAUDE.md or AGENTS.md was found in repository or ancestors.

This is a new cardinality-augmentation experiment, not the previous frozen internal same-count LNS. For target n, the donor is n-1 at its existing side. All four perimeter cohorts, including every bounding extremizer, can move. A two-cell-wide alternating staircase connects the opposite boundary caps through the interior. Both tilted and upright squares in the evacuated region are optional finite-candidate states. Each gets independent displacement labels from half-unit, sqrt(2)-1 and the actual donor wall residue. Upright and diagonal wall-anchored cap chains run from opposing phases. The conflict MILP requires exactly evacuated+1 cells, rather than a same-count refill or a binary common translation.

Six final uncapped tests used donors907/909/971 and two staircase phases per donor. Removed populations were305,305,309,304,317,318. Candidate populations were3278,3234,3161,3039,3399,3368 with43674–54662 conflict edges. No candidate cap was reached. All original evacuated states were retained, and separate full-donor controls confirm no original-state conflict for every trial. HiGHS returned finite-model infeasible for all six +1 requests. This is only a failure within the enumerated displacement/rotation/cap library. It is not a continuous geometric impossibility certificate. run.log records every result.

The first exploratory run had a2400-candidate cap that truncated cap candidates. It was discarded, the cap was raised to12000, and all six final tests were regenerated without truncation. The retained final CPU run took52.06seconds including initial validation; search trials totaled roughly35.6seconds. No endless search or GPU escalation.

output.csv contains exactly908,910,972 squares, constructed afresh with core.grid as explicitly labeled constructive_grid_fallback. It is NOT an incumbent replay and NOT a successful zipper construction. Lengths are31,31,32. These all lose.

At n908 the incumbent is30.637029559612035 and fallback31, a side loss0.362970440387965.

At n910 the incumbent is30.69096012064609301099087714033620383408805471054880 and fallback31, a side loss0.30903987935390698900912285966379616591194528945120.

At n972 the incumbent is31.69238815542838555961566082107349804168366869287818 and fallback32, a side loss0.30761184457161444038433917892650195831633130712182.

Baseline rows come from /home/kit/out/public-20261104/merged-material-candidate.csv and retain original decimal strings in baseline-targets.csv. Final certification used /home/kit/band-certification/.venv/bin/python with pandas3.1.0 and mpmath1.4.1, not repository pandas3.0.6. The reference interpreter lacks SciPy, so it runs the independent certify_outputs.py after the CPU proposal generator. Reference validation took17.79seconds. All three incumbent and all three fallback packings passed unchanged squares.core.certify on literal parser output AND actual pd.read_csv(file).to_csv(index=False) parsed by squares._official_metric._parse_submission_csv. Actual transported files are saved as baseline-targets.pandas.csv and output.pandas.csv. results.json contains the full50-decimal lengths and exact Decimal comparisons under reference validation. No float residual or uncertified improvement claim. No C++ validation was run.

Four tests pass, covering SAT contacts/overlap, all extremizers movable, original-state conflict-free controls for all six full-donor regions, and exact fallback counts/IDs. tests.log contains evidence; all Python sources compile.

Reproduce the CPU generator from the repository with PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=poc/baseline/poc/src .venv/bin/python poc/boundary_zipper/poc/boundary_zipper.py --baseline YOUR_BASELINE.csv --outdir YOUR_DIRECTORY --seconds70 --solve-seconds6, using spaces after option names, i.e. --seconds 70 and --solve-seconds 6.

Then, from the repository, run PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=poc/baseline/poc/src /home/kit/band-certification/.venv/bin/python poc/boundary_zipper/poc/certify_outputs.py --outdir YOUR_DIRECTORY to obtain reference results. The certifier deliberately requires pandas3.1.0 and refuses another version. A fresh generator run writes provisional local-interpreter checks until this second command replaces them. The code also supports --targets and portable --baseline/--outdir flags.

The experiment removes the frozen-extrema objection but still fails to add one square. A useful next search would need richer continuously solved seam states or nonlocal row-population gadgets, not merely another tiny jitter in these six finite libraries.
