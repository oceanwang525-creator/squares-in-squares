"""dH gain of the dphi=0.5 corrugated cell as a function of slack per length (w=3), fine H step."""
import sys, json, time
sys.path.insert(0, '/Users/lucaszhang/squares-in-squares/poc/corrugated_flanks/poc')
from run_cell import search
from cell import R2
k, m, w, starts = map(int, sys.argv[1:5]); dphi = float(sys.argv[5]); step = float(sys.argv[6])
t = time.time()
c, H, z = search(k, m, w, starts, seed=k*100+m, step=step, dphi=dphi)
Hr = w + 2*R2
print(json.dumps(dict(k=k, m=m, w=w, dphi=dphi, H=round(H, 4), Hr=round(Hr, 4), dH=round(H-Hr, 4), slack_per_len=round((k*R2-m)/(k*R2), 5), sec=round(time.time()-t, 1))), flush=True)
