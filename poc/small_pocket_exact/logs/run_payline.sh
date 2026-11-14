#!/bin/bash
cd "$(dirname "$0")/.."
../../.venv/bin/python poc/payline.py --shapes 3,2,1 4,7,3 4,13,3 5,13,3 4,16,3 4,19,3 5,6,2 4,7,2 6,7,2 --n 6 --out data/paylines.json > logs/payline.log 2>&1
echo PAYLINE_DONE
