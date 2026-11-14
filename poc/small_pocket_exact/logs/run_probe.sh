#!/bin/bash
cd "$(dirname "$0")/.."
PY=../../.venv/bin/python
$PY poc/probe.py --shape 2,2,1 --f 0.72 0.71 0.7085 0.7072 0.7065 --R 4096 --iters 900 --seed 5 > logs/probe_2_2_1.log 2>&1 &
$PY poc/probe.py --shape 3,2,1 --f 0.72 0.712 0.7085 0.7072 0.7069 0.70625 0.70 --R 4096 --iters 900 --seed 6 > logs/probe_3_2_1.log 2>&1 &
wait
echo PROBE_DONE
