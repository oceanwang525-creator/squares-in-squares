#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY=/tmp/highrpeel_venv/bin/python
D=poc/swarm_liftpolish3/data

echo "=== 1. Prepare Part C inputs ==="
$PY -u poc/swarm_liftpolish3/poc/prepare_partc_inputs.py
ROWS=$(cat $D/partc_scan_rows.txt)

echo "=== 2. Build report + gadlib ==="
$PY -u poc/faultline_dp2/poc/mkreport.py $D/partc_incumbent.csv $D/partc_report.json
$PY -u poc/faultline_dp2/poc/gadlib.py $D/partc_incumbent.csv $D/partc_gadlib.json

echo "=== 3. Run step-0.0625 pocket scan across two phases (2 procs total) ==="
$PY -u poc/pocket_lift_scan3/poc/scan_pockets.py \
  $D/partc_donors.csv $D/scan_partc_p1.json \
  --step 0.0625 --fmax 1.15 --cap 3000 --nmax 31 --procs 1 \
  --offx 0.015625 --offy 0.015625 --rows "$ROWS" --verbose &
PID1=$!

$PY -u poc/pocket_lift_scan3/poc/scan_pockets.py \
  $D/partc_donors.csv $D/scan_partc_p2.json \
  --step 0.0625 --fmax 1.15 --cap 3000 --nmax 31 --procs 1 \
  --offx 0.046875 --offy 0.046875 --rows "$ROWS" --verbose &
PID2=$!

wait $PID1 $PID2

echo "=== 4. Merge + Sanitize + DP + gadlib_to_pockets ==="
$PY -u poc/pocket_lift_scan3/poc/merge_pockets.py \
  --gadlib $D/partc_gadlib.json \
  --out $D/pockets_partc_pareto.json \
  poc/pocket_lift_scan3/data/pockets_pareto_e3_sane.json \
  $D/pockets_partb_sane.json \
  $D/partc_bricks.json \
  $D/scan_partc_p1.json \
  $D/scan_partc_p2.json

$PY -u poc/pocket_lift_scan3/poc/sanitize.py \
  $D/partc_donors.csv $D/pockets_partc_pareto.json $D/pockets_partc_sane.json

$PY -u poc/pocket_lift_scan3/poc/dp.py extra \
  --lib $D/partc_gadlib.json \
  --pockets $D/pockets_partc_sane.json \
  --report $D/partc_report.json \
  --out $D/pred_partc.json

$PY -u poc/swarm_liftpolish3/poc/gadlib_to_pockets.py \
  $D/partc_donors.csv \
  $D/gadlib_partc_pockets.json

$PY - <<'EOF'
import json
from pathlib import Path
d = Path("poc/swarm_liftpolish3/data")
a = json.loads((d / "pockets_partc_sane.json").read_text())
b = json.loads((d / "gadlib_partc_pockets.json").read_text())
(d / "pockets_partc_all.json").write_text(json.dumps(a + b, indent=0))
print(f"Combined {len(a)} sane + {len(b)} gadlib -> {len(a)+len(b)} pockets")
EOF

echo "=== 5. Build + Border Propagate ==="
mkdir -p $D/built_partc $D/built_partc_prop
$PY -u poc/pocket_lift_scan3/poc/build_cert.py \
  --pred $D/pred_partc.json \
  --pockets $D/pockets_partc_all.json \
  --csv $D/partc_donors.csv \
  --report $D/partc_report.json \
  --outdir $D/built_partc

$PY -u poc/pocket_lift_scan3/poc/border_prop.py \
  --report $D/partc_report.json \
  --built $D/built_partc \
  --outdir $D/built_partc_prop

echo "=== Part C pipeline finished ==="
