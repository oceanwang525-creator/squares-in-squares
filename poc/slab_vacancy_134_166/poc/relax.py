import math

import numpy as np
from scipy.optimize import minimize
from sqk.geometry import neighbor_pairs

def relax(p, length, seed, displacement=0.13):
    rng = np.random.default_rng(seed)
    n = len(p)
    i, j = neighbor_pairs(p, math.sqrt(2) + 2*math.sqrt(2)*displacement + 1e-6).T
    idx = np.arange(len(i))
    def energy(values):
        x, y, a = values.reshape(3, n)
        c, s = np.cos(a), np.sin(a)
        h = (abs(c)+abs(s))/2
        dh = (np.sign(s)*c-np.sign(c)*s)/2
        wl, wr = np.maximum(h-x,0), np.maximum(x+h-length,0)
        wb, wt = np.maximum(h-y,0), np.maximum(y+h-length,0)
        dx,dy = x[i]-x[j], y[i]-y[j]
        ax = np.stack((c[i],-s[i],c[j],-s[j]),1)
        ay = np.stack((s[i],c[i],s[j],c[j]),1)
        pr = dx[:,None]*ax + dy[:,None]*ay
        winner = abs(pr).argmax(1)
        pr = pr[idx,winner]
        ax, ay = ax[idx,winner], ay[idx,winner]
        dp = np.sign(pr)
        d = a[i]-a[j]
        cd, sd = np.cos(d), np.sin(d)
        width = (1+abs(cd)+abs(sd))/2
        overlap = np.maximum(width-abs(pr),0)
        derivative = (np.sign(sd)*cd-np.sign(cd)*sd)/2
        theta = dp*(-dx*ay+dy*ax)
        scale = 2*overlap
        def pair_grad(vi,vj):
            return np.bincount(i,scale*vi,n)+np.bincount(j,scale*vj,n)
        g = np.concatenate((
            2*(wr-wl)+pair_grad(-dp*ax,dp*ax),
            2*(wt-wb)+pair_grad(-dp*ay,dp*ay),
            2*(wl+wr+wb+wt)*dh+pair_grad(derivative-theta*(winner<2),-derivative-theta*(winner>=2))
        ))
        e=wl@wl+wr@wr+wb@wb+wt@wt+overlap@overlap
        return e,g
    bounds = [(v-displacement,v+displacement) for v in p[:,:2].T.ravel()]
    bounds.extend((v-0.18,v+0.18) for v in p[:,2])
    start = p.copy()
    if seed:
        start += rng.normal(0, 0.004, p.shape)
    result = minimize(energy,start.T.ravel(),jac=True,method='L-BFGS-B',bounds=bounds,
                      options={'maxiter':220,'ftol':1e-14,'gtol':1e-9,'maxls':22})
    return result.x.reshape(3,n).T,result.fun
