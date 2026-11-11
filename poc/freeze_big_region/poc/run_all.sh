#!/bin/bash
# sequential sweep; usage from repo root
PY=/Users/lucaszhang/miniforge3/bin/python
for n in 634 911 738; do
  for R in 3 6 10; do $PY poc/freeze_big_region/poc/big_free.py $n $R 0 8; done
  for s in 0 1 2 3 4 5 6 7; do $PY poc/freeze_big_region/poc/big_free.py $n 0 $s 22; done
done
