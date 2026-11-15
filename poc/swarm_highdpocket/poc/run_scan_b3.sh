#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY=/tmp/highrpeel_venv/bin/python
D=poc/swarm_highdpocket/data

# 8 newly improved D=14..34 donors on origin/main (0e74cfc) at step=0.05 + key unscanned F3B_high_r rows
ROWS_A="743,748,863,914,916,924,934,975,543,591,641,690,692,747,749"
ROWS_B="751,800,801,805,806,855,857,864,865,872,919,920,921,976,979"

echo "=== Batch 3: step=0.05 scan on 30 improved/unscanned D=14..34 rows (0e74cfc) ==="
$PY -u poc/pocket_lift_scan3/poc/scan_pockets.py \
  $D/donors_all.csv $D/scan_b3_a.json \
  --step 0.05 --fmax 1.00 --cap 1500 --nmax 31 --procs 1 \
  --offx 0.0125 --offy 0.0125 --rows "$ROWS_A" > $D/scan_b3_a.log 2>&1 &
P1=$!

$PY -u poc/pocket_lift_scan3/poc/scan_pockets.py \
  $D/donors_all.csv $D/scan_b3_b.json \
  --step 0.05 --fmax 1.00 --cap 1500 --nmax 31 --procs 1 \
  --offx 0.0375 --offy 0.0375 --rows "$ROWS_B" > $D/scan_b3_b.log 2>&1 &
P2=$!

wait $P1 $P2
echo "Batch 3 scan done."
