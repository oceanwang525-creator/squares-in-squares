import sys; sys.path.insert(0,"/mnt/e/squares-poc-regenerate-nov13/poc/k2p10_fill2/poc")
from a_relax import *
q45,L45=base633()
for L in (25.657, 25.705):
    q=q45.copy(); q[:,:2]*=L/L45
    c=free_candidates(q,L,step=0.05)
    print(L,"top clearances (neg = overlap depth of best unit square):",[ (round(float(a),4),round(float(b),2),round(float(d),2),round(e,2)) for a,b,d,e in c[:6]])
