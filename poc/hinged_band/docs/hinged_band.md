Hinged finite bands, CPU POC

This is a newly generated split-angle construction, not an incumbent replay.
Two subbands follow independently angled lattice frames, with an actual
transverse slide, asymmetric explicit endpoint caps, finite junction deletions
and four independently corner-anchored upright fill phases. This is not the
earlier straight band library or a free-angle atlas relaxation.

The alternate fallback is explicitly a two-level corner grammar with sourced
Hamalainen17/Cantrell37 interlocking atoms at opposite corners, independent plus/minus
0.125 slides and four upright caps. It is constructed, not a source-row replay.
Neither fallback nor a screened count alone constitutes an improvement.

The small17 source atom is reconstructed analytically from
https://kingbird.myphotos.cc/packing/square-17_r2.svg
and https://kingbird.myphotos.cc/packing/square-37.svg, following prior
child_finite/finite.py. Cantrell37 is retained whole. This experiment does not
claim reproducing the better Bidwell17 benchmark. The stricter17/37 gate remains
unpassed, so these bounded large cases are exploratory only, not evidence to
scale this family.

Elapsed CPU wall time was 14.490 seconds.
Generated 120 finite configurations. Meaningful certified wins
0. Every failed target count is retained in
results.json search_attempts/search_failures. Official rejections are preserved
separately and never counted as wins.

n908 emitted constructed_hinged_band. Literal L 31.00000060000000300000000000000000000000000000000000. Actual pandas L 31.00000060000000300000000000000000000000000000000000. Incumbent literal L 30.63702955961203500000000000000000000000000000000000. Delta L {'literal': '0.3629710403879680000000000000', 'actual_pandas': '0.3629710403879680000000000000'}.

n910 emitted constructed_hinged_band. Literal L 31.00000060000000400000000000000000000000000000000000. Actual pandas L 31.00000060000000400000000000000000000000000000000000. Incumbent literal L 30.69096012064609301099087714033620383408805471054880. Delta L {'literal': '0.3090404793539109890091228597', 'actual_pandas': '0.3090404793539109890091228597'}.

n972 emitted valid_constructive_fallback. Literal L 32.00000062000000200000000000000000000000000000000000. Actual pandas L 32.00000062000000200000000000000000000000000000000000. Incumbent literal L 31.69238815542838555961566082107349804168366869287818. Delta L {'literal': '0.3076124645716164403843391789', 'actual_pandas': '0.3076124645716164403843391789'}.

Validation invokes unchanged squares.core.certify on the literal CSV
parsed with squares._official_metric._parse_submission_csv and on actual
pandas.read_csv(...).to_csv(index=False) text parsed by that same parser.
output-pandas.csv preserves the actual serialized bytes. No custom metric.
Lengths include the center-only 1.00000002 numerical-clearance dilation.
No win is claimed from float screening or numerical dust.

Reference pandas version is 3.1.0. The verified interpreter was
/home/kit/band-relax/.venv/bin/python. Run with PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=poc/baseline/poc/src and that interpreter, followed by
poc/hinged_band/poc/hinged_band.py --baseline YOUR_BASELINE.csv --seconds 30.
Outputs default to the script directory and --out overrides that. CPU only.
No GPU, git/index operations, shared edits, push or submission occurred.

Four local unittest cases passed in 1.784 seconds with reference pandas3.1.0. Both source atoms pass literal and actual-pandas certification. The17 atom is the alternate Hamalainen layout, not the tighter Bidwell record. No large incumbent coordinates are replayed. All Python sources compile.
