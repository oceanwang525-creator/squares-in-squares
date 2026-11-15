#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY=/tmp/highrpeel_venv/bin/python
D=poc/swarm_highdpocket/data

ROWS_A="270,305,419,501,503,504,506,551,593,597,598,604,639,640,642,645,646,649,651,654,691,695,701,702,703,706,707,743,744,745,746,750,756"
ROWS_B="757,758,798,802,803,804,812,813,814,858,861,863,867,868,871,872,873,874,875,914,915,917,918,924,926,927,929,932,933,934,978,987,989"

echo "=== Batch 2: step=0.0625 on 66 changed D=14..34 rows on cc58c7e ==="
$PY -u poc/pocket_lift_scan3/poc/scan_pockets.py \
  $D/donors_all.csv $D/scan_b2_a.json \
  --step 0.0625 --fmax 1.01 --cap 1500 --nmax 31 --procs 1 \
  --offx 0.015625 --offy 0.015625 --rows "$ROWS_A" > $D/scan_b2_a.log 2>&1 &
P1=$!

$PY -u poc/pocket_lift_scan3/poc/scan_pockets.py \
  $D/donors_all.csv $D/scan_b2_b.json \
  --step 0.0625 --fmax 1.01 --cap 1500 --nmax 31 --procs 1 \
  --offx 0.046875 --offy 0.046875 --rows "$ROWS_B" > $D/scan_b2_b.log 2>&1 &
P2=$!

wait $P1 $P2

echo "=== Fine step=0.05 scan on top D=15..28 champions ==="
ROWS_05_A="748,799,863,924,975,980,988"
ROWS_05_B="588,856,915,916,977,758,813"

$PY -u poc/pocket_lift_scan3/poc/scan_pockets.py \
  $D/donors_all.csv $D/scan_005_a.json \
  --step 0.05 --fmax 0.96 --cap 1500 --nmax 31 --procs 1 \
  --offx 0.0125 --offy 0.0125 --rows "$ROWS_05_A" > $D/scan_005_a.log 2>&1 &
P3=$!

$PY -u poc/pocket_lift_scan3/poc/scan_pockets.py \
  $D/donors_all.csv $D/scan_005_b.json \
  --step 0.05 --fmax 0.96 --cap 1500 --nmax 31 --procs 1 \
  --offx 0.0375 --offy 0.0375 --rows "$ROWS_05_B" > $D/scan_005_b.log 2>&1 &
P4=$!

wait $P3 $P4
echo "Batch 2 + step=0.05 done."
