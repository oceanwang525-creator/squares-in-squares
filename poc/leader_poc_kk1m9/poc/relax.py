"""Shrink incumbent n to L_target (centre scaling) and try to remove overlap with a SAT-penetration penalty + L-BFGS.
usage: relax.py n L_target seconds   -> docs/relax_n{n}.npy (x,y,angle), prints residual penalty."""
import sys, time, numpy as np, pandas as pd
from scipy.optimize import minimize
n, Lt, secs = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
d = pd.read_csv('/Users/lucaszhang/squares-in-squares/poc/pell_corrugation/data/border-scan/merged.csv')
g = d[d.n == n]; P = g[['x','y','angle']].to_numpy().copy()
def ext(P):
    h = .5*(np.abs(np.cos(P[:,2]))+np.abs(np.sin(P[:,2])))
    return P[:,0]-h, P[:,0]+h, P[:,1]-h, P[:,1]+h
x0,x1,y0,y1 = ext(P); L0 = max(x1.max()-x0.min(), y1.max()-y0.min())
cx, cy = (x0.min()+x1.max())/2, (y0.min()+y1.max())/2
P[:,0] = (P[:,0]-cx)*Lt/L0; P[:,1] = (P[:,1]-cy)*Lt/L0   # coordinates in [-Lt/2, Lt/2]
# pair list (fixed, generous radius)
D = np.hypot(P[:,None,0]-P[None,:,0], P[:,None,1]-P[None,:,1])
I, J = np.where(np.triu(D < 2.0, 1))
# colouring so FD can perturb many squares at once
nb = [set() for _ in range(n)]
for i,j in zip(I,J): nb[i].add(j); nb[j].add(i)
col = -np.ones(n, int)
for i in range(n):
    used = {col[j] for j in nb[i]|{k for j in nb[i] for k in nb[j]} if col[j] >= 0}
    c = 0
    while c in used: c += 1
    col[i] = c
ncol = col.max()+1
H = Lt/2
def energy_parts(P):
    xi,yi,ai = P[I,0],P[I,1],P[I,2]; xj,yj,aj = P[J,0],P[J,1],P[J,2]
    dx,dy = xj-xi, yj-yi
    pen = np.full(len(I), np.inf)
    for a_ in (ai, aj):
        ux,uy = np.cos(a_), np.sin(a_)
        for (px,py) in ((ux,uy),(-uy,ux)):
            proj = np.abs(dx*px+dy*py)
            def sup(t): return .5*(np.abs(np.cos(t)*px+np.sin(t)*py)+np.abs(-np.sin(t)*px+np.cos(t)*py))
            pen = np.minimum(pen, sup(ai)+sup(aj)-proj)
    pe = np.maximum(pen, 0)**2
    h = .5*(np.abs(np.cos(P[:,2]))+np.abs(np.sin(P[:,2])))
    w = (np.maximum(P[:,0]+h-H,0)**2+np.maximum(-H-(P[:,0]-h),0)**2+np.maximum(P[:,1]+h-H,0)**2+np.maximum(-H-(P[:,1]-h),0)**2)
    return pe, w
def f(v):
    pe,w = energy_parts(v.reshape(n,3)); return pe.sum()+w.sum()
def fg(v):
    P = v.reshape(n,3); e = f(v); g = np.zeros_like(P); eps = 1e-6
    for c in range(ncol):
        m = (col==c)
        for k in range(3):
            Pp = P.copy(); Pp[m,k]+=eps; Pm = P.copy(); Pm[m,k]-=eps
            # per-square energy of perturbed squares only: use full energy diff attributed via pair sums
            pep,wp = energy_parts(Pp); pem,wm = energy_parts(Pm)
            # attribute pair energies to the (unique, by colouring) perturbed member
            gi = np.zeros(n)
            dp = (pep-pem)/(2*eps)
            np.add.at(gi, I[m[I]], dp[m[I]]); np.add.at(gi, J[m[J]], dp[m[J]])
            gi[m] += ((wp-wm)/(2*eps))[m]
            g[m,k] = gi[m]
    return e, g.ravel()
t=time.time(); v=P.ravel().copy(); e0=f(v); print('start penalty',e0,'ncol',ncol,'pairs',len(I))
while time.time()-t < secs:
    r = minimize(fg, v, jac=True, method='L-BFGS-B', options=dict(maxiter=300, ftol=1e-30, gtol=1e-14)); v = r.x
    print(round(time.time()-t), r.fun, flush=True)
    if r.fun < 1e-22: break
P = v.reshape(n,3); P[:,0]+=Lt/2; P[:,1]+=Lt/2
np.save(f'/Users/lucaszhang/squares-in-squares/poc/leader_poc_kk1m9/docs/relax_n{n}.npy', P)
pe,w = energy_parts(v.reshape(n,3)); print('final pen',pe.sum()+w.sum(),'max pair pen',np.sqrt(pe.max()))
