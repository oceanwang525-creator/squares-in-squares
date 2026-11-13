import sys,json,pathlib,importlib.util
r=pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0,str(r/'poc/baseline/poc/src'))
sys.path.insert(0,str(r/'poc/diamond_core/poc'))
import diamond_core as m
results=m.run_eval()
pathlib.Path(__file__).with_name('generated.json').write_text(json.dumps(results,indent=2))
