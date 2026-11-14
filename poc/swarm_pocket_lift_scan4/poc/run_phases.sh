#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 CSV PREFIX [ROWS] [LOGDIR]" >&2
  exit 2
fi

csv=$1
prefix=$2
rows=${3:-}
logdir=${4:-poc/swarm_pocket_lift_scan4/logs}
python_bin=${PYTHON:-python}
mkdir -p "$logdir"

phases=(
  "0 0"
  "0 0.03125"
  "0.03125 0"
  "0.03125 0.03125"
  "0.0078125 0.0078125"
  "0.015625 0.015625"
  "0.0234375 0.0234375"
  "0.0234375 0.0078125"
)

pids=()
for phase in "${phases[@]}"; do
  read -r offx offy <<<"$phase"
  tag="o${offx}_${offy}"
  args=(
    poc/swarm_pocket_lift_scan4/poc/scan_pockets.py
    "$csv"
    "${prefix}_${tag}.json"
    --step 0.03125
    --fmax 1.15
    --cap 3000
    --nmax 31
    --procs 1
    --offx "$offx"
    --offy "$offy"
    --verbose
  )
  if [[ -n "$rows" ]]; then
    args+=(--rows "$rows")
  fi
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    VECLIB_NUM_THREADS=1 KMP_NUM_THREADS=1 \
    "$python_bin" "${args[@]}" >"${logdir}/${tag}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
