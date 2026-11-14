#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY=/tmp/highrpeel_venv/bin/python
D=poc/swarm_liftpolish3/data
ROWS="442,485,492,530,538,577,585,588,634,695,743,799,803,852,861,911,914,926,974,975"

echo "=== 1. Prepare Part B inputs ==="
$PY -u poc/swarm_liftpolish3/poc/prepare_partb_inputs.py

echo "=== 2. Build report + gadlib ==="
$PY -u poc/faultline_dp2/poc/mkreport.py $D/partb_incumbent.csv $D/partb_report.json
$PY -u poc/faultline_dp2/poc/gadlib.py $D/partb_incumbent.csv $D/partb_gadlib.json

echo "=== 3. Run step-0.0625 pocket scan across two phases (2 procs total) ==="
$PY -u poc/pocket_lift_scan3/poc/scan_pockets.py \
  $D/partb_donors.csv $D/scan_00625_p1.json \
  --step 0.0625 --fmax 1.15 --cap 3000 --nmax 31 --procs 1 \
  --offx 0.015625 --offy 0.015625 --rows "$ROWS" --verbose &
PID1=$!

$PY -u poc/pocket_lift_scan3/poc/scan_pockets.py \
  $D/partb_donors.csv $D/scan_00625_p2.json \
  --step 0.0625 --fmax 1.15 --cap 3000 --nmax 31 --procs 1 \
  --offx 0.046875 --offy 0.046875 --rows "$ROWS" --verbose &
PID2=$!

wait $PID1 $PID2

echo "=== 4. Merge + Sanitize + DP ==="
$PY -u poc/pocket_lift_scan3/poc/merge_pockets.py \
  --gadlib $D/partb_gadlib.json \
  --out $D/pockets_partb_pareto.json \
  poc/pocket_lift_scan3/data/pockets_pareto_e3_sane.json \
  $D/gstep_pockets.json \
  $D/scan_00625_p1.json \
  $D/scan_00625_p2.json

$PY -u poc/pocket_lift_scan3/poc/sanitize.py \
  $D/partb_donors.csv $D/pockets_partb_pareto.json $D/pockets_partb_sane.json

$PY -u poc/pocket_lift_scan3/poc/dp.py extra \
  --lib $D/partb_gadlib.json \
  --pockets $D/pockets_partb_sane.json \
  --report $D/partb_report.json \
  --out $D/pred_partb.json

echo "=== 5. Build + Border Propagate ==="
mkdir -p $D/built_partb $D/built_partb_prop
$PY -u poc/pocket_lift_scan3/poc/build_cert.py \
  --pred $D/pred_partb.json \
  --pockets $D/pockets_partb_sane.json \
  --csv $D/partb_donors.csv \
  --report $D/partb_report.json \
  --outdir $D/built_partb

$PY -u poc/pocket_lift_scan3/poc/border_prop.py \
  --report $D/partb_report.json \
  --built $D/built_partb \
  --outdir $D/built_partb_prop

echo "=== Part B pipeline finished ==="
