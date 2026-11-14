# New-gadget closure after the latest merge

## Idea

The hypothesis was that the newest merge had introduced enough reusable pockets
that the old composition-closure negative was stale. I therefore regenerated the
pocket library from the current incumbent, added clean cuts, then ran a fresh
staircase DP against per-row sides recomputed from the same incumbent. I did not
read coordinates from the frontier table; it was used only to rank side targets.

The incumbent is the artifact in Git commit `926ed95`, named
`merged-22.15074331648.csv`. Its nominal score is `22.15074331648`. The side
table confirms that the top stale gap rows were still relevant, but its `L`
column had been superseded on several rows, so all ranking used recomputed
current `L`.

## Result

`final.csv` contains one certified strict win:

| n | incumbent L | new L | dL (new minus incumbent) | composition |
|---:|---:|---:|---:|---|
| 974 | 31.705886745787048 | 31.70588236820239466613381477509345626574081318069471 | -4.3775846521043604e-6 | `(18,9,+6)T` from n=538 plus `(10,22,+7)T` from n=974 |

This is `K=31, f=0.7058823681409834`. The independent final-bytes check gives
identical accepted `L` for literal decimal transport and a real
`pandas.read_csv` / `to_csv` round trip:

```text
n=974: complete=True
literal L=31.70588236820239466613381477509345626574081318069471
pandas L=31.70588236820239466613381477509345626574081318069471
dL=-4.3775846521043604e-06 in both modes
total score gain=-1.402668598699e-07
```

The win is real but narrow: it lowers the nominal union score only from about
`22.15074331648` to about `22.1507431762`, nowhere near `22.0999`.

## Exact commands

This machine has no repository-root `.venv`; `python3` has the required
`numpy`, `pandas`, `mpmath`, and SciPy. Commands are from the repository root.

```sh
git show 926ed95:merged-22.15074331648.csv > /tmp/incumbent.csv

python3 poc/gadget_squeeze/poc/gadlib.py /tmp/incumbent.csv \
  poc/newgadget_closure/data/lib_current.json \
  | tee poc/newgadget_closure/docs/extract_current.txt

python3 - <<'PY'
import sys
sys.path.insert(0, 'poc/gadget_squeeze/poc')
import common
L = common.ours_L('/tmp/incumbent.csv',
                  cache='poc/newgadget_closure/data/oursL.json')
print(len(L), L[974])
PY

python3 poc/gadget_squeeze/poc/dp.py \
  poc/newgadget_closure/data/lib_current.json \
  poc/newgadget_closure/data/oursL.json \
  --out poc/newgadget_closure/data/pred_current.json \
  | tee poc/newgadget_closure/docs/dp_current.txt

python3 poc/newgadget_closure/poc/certify_wins.py \
  | tee poc/newgadget_closure/docs/certify_wins.txt
python3 poc/newgadget_closure/poc/verify_final.py \
  | tee poc/newgadget_closure/docs/final_verify.txt
```

The base library extracted from the current merge has 878 gadgets, 323 proper
sub-gadgets, and 127 distinct proper shape types. A clean-subrectangle pass
increased this to 1,449 items / 571 distinct oriented shape types, but did not
change the DP result:

```sh
python3 poc/gadget_squeeze/poc/cutlib.py \
  poc/newgadget_closure/data/lib_current.json \
  poc/newgadget_closure/data/lib_cut_current.json --nmax 8 \
  | tee poc/newgadget_closure/docs/cut_current.txt
python3 poc/gadget_squeeze/poc/dp.py \
  poc/newgadget_closure/data/lib_cut_current.json \
  poc/newgadget_closure/data/oursL.json \
  --out poc/newgadget_closure/data/pred_cut_current.json \
  | tee poc/newgadget_closure/docs/dp_cut_current.txt
```

## Extended scan and negative findings

After finding the win, I scaled the scan to the recomputed top 30 gap rows and
then the neighboring `n=960..978` range. These are the exact scanner commands;
the three steps are `0.25`, `0.125`, and `0.0625`.

```sh
python3 poc/pocket_lift_scan3/poc/scan_pockets.py /tmp/incumbent.csv \
  poc/newgadget_closure/data/scan_p0.json --step 0.25 --fmax 0.9 --nmax 31 \
  --cap 1200 --procs 8 \
  --rows 367,791,908,969,803,328,365,902,861,963,842,491,901,333,747,804,962,843,748,848,456,366,795,228,965,340,974,585,982,262 \
  --verbose | tee poc/newgadget_closure/docs/scan_p0.txt

python3 poc/pocket_lift_scan3/poc/scan_pockets.py /tmp/incumbent.csv \
  poc/newgadget_closure/data/scan_p0125.json --step 0.125 --fmax 0.9 --nmax 31 \
  --cap 2000 --procs 8 \
  --rows 367,791,908,969,803,328,365,902,861,963,842,491,901,333,747,804,962,843,748,848,456,366,795,228,965,340,974,585,982,262 \
  --verbose | tee poc/newgadget_closure/docs/scan_p0125.txt

python3 poc/pocket_lift_scan3/poc/scan_pockets.py /tmp/incumbent.csv \
  poc/newgadget_closure/data/scan_p00625.json --step 0.0625 --fmax 0.9 --nmax 31 \
  --cap 3000 --procs 8 \
  --rows 367,791,908,969,803,328,365,902,861,963,842,491,901,333,747,804,962,843,748,848,456,366,795,228,965,340,974,585,982,262,971,970,973,975,976,977,968,967,961,960 \
  --verbose | tee poc/newgadget_closure/docs/scan_p00625.txt

python3 poc/pocket_lift_scan3/poc/merge_pockets.py \
  --gadlib poc/newgadget_closure/data/lib_current.json \
  --out poc/newgadget_closure/data/pockets_fine.json \
  poc/newgadget_closure/data/scan_p0.json \
  poc/newgadget_closure/data/scan_p0125.json \
  poc/newgadget_closure/data/scan_p00625.json \
  | tee poc/newgadget_closure/docs/merge_fine.txt

python3 poc/pocket_lift_scan3/poc/dp.py extra \
  --lib poc/newgadget_closure/data/lib_current.json \
  --pockets poc/newgadget_closure/data/pockets_fine.json \
  --report poc/newgadget_closure/data/report_current.json \
  --out poc/newgadget_closure/data/pred_fine.json \
  | tee poc/newgadget_closure/docs/dp_fine.txt
```

The scanner produced 183,277 raw gadgets. Across the merged scan, 8,128
oriented shape types were new relative to the incumbent library, but 322
survived the Pareto filter and no scanned shape improved an existing shape's
`f`. The DP predicted 555 composition cells; only the `n=974` cell beat the
current incumbent. The extended run ended with one strict win and no ties
suggesting another payoff, so I stopped rather than leaving a speculative job
running.

The negative result is consistent with `poc/gadget_squeeze`: adding shape
supply is not enough. The new merge did add useful primitives, and the
`n=974` composition exploits one, but the remaining high-gap rows need either
better gain density at roughly `f=0.60-0.67`, better whole-row constructions,
or a different seam mechanism.

## Files

- `final.csv`: the 974-square complete group.
- `poc/certify_wins.py`: reproducible build + literal/pandas certification.
- `poc/verify_final.py`: independent final-bytes certification in both modes.
- `data/lib_current.json`: pocket library from the requested merge.
- `data/lib_cut_current.json`: clean-cut expansion used in the negative check.
- `data/pred_*.json` and `data/pockets_*.json`: DP predictions and scanner
  Pareto libraries.
- `data/scan_*.json`: raw scanner evidence for the three phases.
- `docs/*.txt` and `docs/final_verify.json`: run logs and final verification.

Nothing was submitted externally.
