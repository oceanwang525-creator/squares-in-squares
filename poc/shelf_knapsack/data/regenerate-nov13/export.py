import sys,json,pathlib,importlib.util
r=pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0,str(r/'poc/baseline/poc/src'))
sys.path.insert(0,str(r/'poc/shelf_knapsack/poc'))
import shelf_knapsack as m
results=m.evaluate_shelf_solver()
pathlib.Path(__file__).with_name('generated.json').write_text(json.dumps(results,indent=2))
