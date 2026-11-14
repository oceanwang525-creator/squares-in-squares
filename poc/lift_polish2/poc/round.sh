#!/usr/bin/env bash
# One donor-cascade round: polished rows -> sub-pocket scan -> Pareto library ->
# staircase DP -> donor table -> build+certify.
#
#   round.sh ROWS.csv REPORT.json TAG "n,n,..." POCK_BASE.json
#
# Empty POCK_BASE.json is skipped (first round). The scan uses step 0.125 with
# fine phases; raw scan output and the donor CSV are deleted at the end (both
# are one command to regenerate and are large).
set -euo pipefail
cd /mnt/e/squares-in-squares-commandcode

PY=.venv/bin/python
D=poc/lift_polish2
ROWS=$1
REPORT=$2
TAG=$3
NLIST=$4
POCK_BASE=${5:-}
PROCS=${6:-8}

mkdir -p "$D/data/scan" "$D/logs"

for o in "0.0625 0.0625" "0.0625 0" "0 0.0625" "0.09375 0.0625"; do
  set -- $o
  $PY poc/pocket_lift_scan2/poc/scan_pockets.py "$ROWS" "$D/data/scan/o$1_$2.json" \
      --step 0.125 --fmax 1.10 --cap 2000 --procs "$PROCS" --offx "$1" --offy "$2" \
      --rows "$NLIST" >> "$D/logs/scan_${TAG}.log" 2>&1 &
done
wait

INPUTS="$D/data/scan/o0.0625_0.0625.json $D/data/scan/o0.0625_0.json $D/data/scan/o0_0.0625.json $D/data/scan/o0.09375_0.0625.json"
if [ -n "$POCK_BASE" ]; then INPUTS="$POCK_BASE $INPUTS"; fi

$PY poc/pocket_lift_scan2/poc/merge_pockets.py --gadlib poc/faultline_dp2/data/gadlib.json \
    --out "$D/data/pockets_${TAG}.json" $INPUTS

# full 1000-cell side table with our rows substituted
$PY - "$REPORT" "$D/data/report_full_${TAG}.json" <<'EOF'
import json, sys
main = json.load(open('poc/lift_polish2/data/main_report.json'))
new = {r['n']: r for r in json.load(open(sys.argv[1]))['results'] if r['L']}
for r in main['results']:
    if r['n'] in new:
        r['L'], r['score'] = new[r['n']]['L'], new[r['n']]['score']
json.dump(main, open(sys.argv[2], 'w'))
print('patched full report with', len(new), 'groups')
EOF

$PY poc/pocket_lift_scan2/poc/dp.py extra --lib poc/faultline_dp2/data/gadlib.json \
    --pockets "$D/data/pockets_${TAG}.json" \
    --report "$D/data/report_full_${TAG}.json" --out "$D/data/pred_${TAG}.json" \
    >> "$D/logs/dp_${TAG}.log" 2>&1
tail -40 "$D/logs/dp_${TAG}.log"

$PY poc/lift_polish/poc/mk_donors.py "$ROWS" "$D/data/donors_${TAG}.csv"

$PY poc/pocket_lift_scan2/poc/build_cert.py --pred "$D/data/pred_${TAG}.json" \
    --pockets "$D/data/pockets_${TAG}.json" \
    --report "$D/data/report_full_${TAG}.json" --csv "$D/data/donors_${TAG}.csv" \
    --outdir "$D/data/built_${TAG}" --final "$D/data/built_${TAG}.csv" \
    >> "$D/logs/build_${TAG}.log" 2>&1
tail -30 "$D/logs/build_${TAG}.log"

rm -rf "$D/data/scan" "$D/data/donors_${TAG}.csv"
echo "round $TAG done -> $D/data/built_${TAG}.csv"
