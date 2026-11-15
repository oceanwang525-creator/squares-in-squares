#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY=/tmp/highrpeel_venv/bin/python
D=poc/swarm_highdpocket/data
ROWS="10588,10748,10799,863,924,10975,10980,10988,10459,10550,10595,10599,10650,10706,10758,10761,10813,10818,10856,10872,10916,10977,10991,10270,10375,10379,10416,10419,10420,10421,10462,10499,10500"

$PY -u poc/pocket_lift_scan3/poc/scan_pockets.py \
  $D/donors_all.csv $D/scan_b1_p1.json \
  --step 0.0625 --fmax 1.01 --cap 1500 --nmax 31 --procs 1 \
  --offx 0.015625 --offy 0.015625 --rows "$ROWS" --verbose > $D/scan_b1_p1.log 2>&1 &
P1=$!

$PY -u poc/pocket_lift_scan3/poc/scan_pockets.py \
  $D/donors_all.csv $D/scan_b1_p2.json \
  --step 0.0625 --fmax 1.01 --cap 1500 --nmax 31 --procs 1 \
  --offx 0.046875 --offy 0.046875 --rows "$ROWS" --verbose > $D/scan_b1_p2.log 2>&1 &
P2=$!

wait $P1 $P2
echo "Batch 1 done."
