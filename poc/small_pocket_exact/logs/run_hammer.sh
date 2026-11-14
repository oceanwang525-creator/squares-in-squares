#!/bin/bash
cd "$(dirname "$0")/.."
PY=../../.venv/bin/python
$PY poc/hammer.py --shape 2,2,1 --f0 0.85 --R 2048 --iters 400 --procs 12 --seed 1 --tag cal_2_2_1 > logs/hammer_2_2_1.log 2>&1 &
$PY poc/hammer.py --shape 4,4,1 --f0 0.95 --R 1024 --iters 500 --hit 0.70 --procs 16 --seed 1 --tag cal_4_4_1 > logs/hammer_4_4_1.log 2>&1 &
$PY poc/hammer.py --shape 3,2,1 --f0 0.80 --R 4096 --iters 400 --hit 0.7065 --procs 16 --seed 2 --tag q_3_2_1 > logs/hammer_3_2_1.log 2>&1 &
wait
echo HAMMER_DONE
