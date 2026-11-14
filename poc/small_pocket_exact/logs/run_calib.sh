#!/bin/bash
cd "$(dirname "$0")/.."
PY=../../.venv/bin/python
run(){ $PY poc/search.py --shape $1 --fpay $2 --R $3 --sweeps $4 --iters $5 --seed $6 --tag $7 --out data/$7.json > logs/$7.log 2>&1; }
run 4,4,1  0.67553  1024 30 80 11 cal_4_4_1 &
run 4,5,1  0.621320 1024 30 80 12 cal_4_5_1 &
run 8,8,1  0.535534  512 30 80 13 cal_8_8_1 &
wait
echo CALIB_DONE
