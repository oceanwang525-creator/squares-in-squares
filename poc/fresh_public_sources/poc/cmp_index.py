import sys,json
sys.path.insert(0,'/Users/lucaszhang/squares-in-squares/poc/baseline/poc/src')
from squares.core import read_submission,score
import inspect
import squares.core as c
print([n for n in dir(c) if not n.startswith('_')])
