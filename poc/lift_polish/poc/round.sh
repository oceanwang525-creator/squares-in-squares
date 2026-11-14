#!/usr/bin/env bash
# One donor-cascade round: polish output -> pockets -> DP -> build.
#
#   round.sh <rows.csv> <report.json> <tag> [n,n,...]
#
# 1. scan the given rows for sub-pockets (3 offset lattices, step 0.25)
# 2. merge them with the previous Pareto library
# 3. patch the full 1000-cell side table with the new per-n sides
# 4. staircase DP -> predictions (pred_<tag>.json)
# 5. donor table (main + these rows)
# 6. build + certify the DP's predictions (built_<tag>/)
set -euo pipefail
cd /mnt/e/squares-in-squares-commandcode

PY=.venv/bin/python
D=poc/lift_polish
ROWS=$1
REPORT=$2
TAG=$3
NLIST=${4:-442,447,485,530,577,798}
POCK_BASE=${5:-$D/data/pockets_pareto3.json}

for o in "0 0" "0.125 0" "0 0.125"; do
  set -- $o
  $PY poc/pocket_lift_scan/poc/scan_pockets.py "$ROWS" "$D/data/pockets_${TAG}_o$1_$2.json" \
      --step 0.25 --fmax 1.0 --cap 800 --procs 12 --offx "$1" --offy "$2" --rows "$NLIST"
done

$PY poc/pocket_lift_scan/poc/merge_pockets.py --gadlib poc/faultline_dp2/data/gadlib.json \
    --out "$D/data/pockets_pareto_${TAG}.json" "$POCK_BASE" \
    "$D/data/pockets_${TAG}_o0_0.json" "$D/data/pockets_${TAG}_o0.125_0.json" \
    "$D/data/pockets_${TAG}_o0_0.125.json"

$PY - "$REPORT" "$D/data/report_full_${TAG}.json" <<'EOF'
import json, sys
main = json.load(open('poc/pocket_lift_scan/data/report.json'))
new = {r['n']: r for r in json.load(open(sys.argv[1]))['results'] if r['L']}
for r in main['results']:
    if r['n'] in new:
        r['L'], r['score'] = new[r['n']]['L'], new[r['n']]['score']
json.dump(main, open(sys.argv[2], 'w'))
print('patched full report with', len(new), 'groups')
EOF

$PY poc/pocket_lift_scan/poc/dp.py extra --lib poc/faultline_dp2/data/gadlib.json \
    --pockets "$D/data/pockets_pareto_${TAG}.json" \
    --report "$D/data/report_full_${TAG}.json" --out "$D/data/pred_${TAG}.json"

$PY $D/poc/mk_donors.py "$ROWS" "$D/data/donors_${TAG}.csv"

$PY poc/pocket_lift_scan/poc/build_cert.py --pred "$D/data/pred_${TAG}.json" \
    --pockets "$D/data/pockets_pareto_${TAG}.json" \
    --report "$D/data/report_full_${TAG}.json" --csv "$D/data/donors_${TAG}.csv" \
    --outdir "$D/data/built_${TAG}" --final "$D/data/built_${TAG}.csv"
