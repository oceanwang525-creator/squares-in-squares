#!/bin/sh
cd /Users/lucaszhang/squares-in-squares
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/Users/lucaszhang/miniforge3/bin/python
F=poc/k2p10_fill2/poc/b_fill.py
$PY $F 1 25.70 3 2
$PY $F 2 25.70 3 3
$PY $F 0 25.705 3 4
$PY $F 0 25.7071 2 5
