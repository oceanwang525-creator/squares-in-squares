import math

import numpy as np
from numba import njit
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix, vstack
from scipy.spatial import ConvexHull, Voronoi, cKDTree
from sqk import lp
from sqk.geometry import corners, neighbor_pairs, side


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


@njit
def add_interval(tree, increments, start, end, value, leaves):
    a, b = start + leaves, end + leaves
    while a <= b:
        if a % 2 == 1:
            increments[a] += value
            tree[a] += value
            a += 1
        if b % 2 == 0:
            increments[b] += value
            tree[b] += value
            b -= 1
        a //= 2
        b //= 2
    a, b = (start + leaves) // 2, (end + leaves) // 2
    while a:
        tree[a] = increments[a] + max(tree[2*a], tree[2*a+1])
        tree[b] = increments[b] + max(tree[2*b], tree[2*b+1])
        a //= 2
        b //= 2


@njit
def max_window(lo, hi, xs, xe, ys, inv_y, width):
    size = len(xs)
    base = 1
    while base < size:
        base *= 2
    tree = np.zeros(base*2, np.int32)
    increment = np.zeros(base*2, np.int32)
    end = np.zeros(size, np.int32)
    j = 0
    for q in np.argsort(lo[:,1]):
        while j < size and hi[ys[j],1] - width <= lo[q,1] + 2e-10:
            j += 1
        end[q] = j - 1
    active = 0
    best, bx, by = 0, 0., 0.
    for q in xs:
        x = hi[q,0] - width
        while active < size and lo[xe[active],0] < x - 2e-10:
            r = xe[active]
            if end[r] >= inv_y[r]:
                add_interval(tree, increment, inv_y[r], end[r], -1, base)
            active += 1
        if end[q] >= inv_y[q]:
            add_interval(tree, increment, inv_y[q], end[q], 1, base)
        if tree[1] > best:
            best = tree[1]
            bx = x
            i = 1
            while i < base:
                i *= 2
                if tree[i+1] > tree[i]:
                    i += 1
            by = hi[ys[i-base],1] - width
    return best, bx, by


def orient(p):
    angles = p[:,2,None] + np.arange(4)[None,:]*np.pi/2 + np.pi/4
    all_vertices = (p[:,None,:2] + np.stack((np.cos(angles), np.sin(angles)), axis=-1)/np.sqrt(2)).reshape(-1,2)
    hull = all_vertices[ConvexHull(all_vertices).vertices]
    edges = np.roll(hull,-1,axis=0)-hull
    period = np.pi/2
    events = np.unique(np.concatenate(([0.,period], np.arctan2(edges[:,1],edges[:,0])%period)))
    mid = (events[:-1]+events[1:])/2
    c,s=np.cos(mid),np.sin(mid)
    u=hull@np.array([c,s])
    v=hull@np.array([-s,c])
    a=hull[u.argmax(axis=0)]-hull[u.argmin(axis=0)]
    b=hull[v.argmax(axis=0)]-hull[v.argmin(axis=0)]
    cross=np.arctan2(-a[:,0]+b[:,1],a[:,1]+b[:,0])%np.pi
    candidates=np.concatenate((events,cross[(cross>events[:-1])&(cross<events[1:])]))
    c,s=np.cos(candidates),np.sin(candidates)
    sizes=np.maximum(np.ptp(hull@np.array([c,s]),axis=0),np.ptp(hull@np.array([-s,c]),axis=0))
    idx=sizes.argmin()
    angle=candidates[idx]
    if sizes[idx] >= side(p) - 1e-8:
        return None
    return np.column_stack((p[:,:2]@np.array([[c[idx],-s[idx]],[s[idx],c[idx]]]),p[:,2]-angle))


def perp(v):
    return np.stack((-v[..., 1], v[..., 0]), axis=-1)


def sparse_rows(cols, vals, width):
    return coo_matrix((vals.ravel(), (np.repeat(np.arange(len(cols)), cols.shape[1]), cols.ravel())), shape=(len(cols),width)).tocsr()


def step(p, translation, rotation):
    n=len(p)
    width=3*n+1
    current=side(p)
    c=corners(p)
    edges=c-p[:,None,:2]
    pairs=lp.choose_axes(p,1.51)
    i,j=pairs.pairs.T
    owns_i=pairs.owner==0
    owner=np.where(owns_i,i,j)
    other=np.where(owns_i,j,i)
    axis=lp._axis(p[owner,2],pairs.k,pairs.sign*np.where(owns_i,1,-1))
    offset=p[other,:2]-p[owner,:2]
    vectors=offset[:,None,:]+edges[other]
    gap=np.einsum('md,mkd->mk',axis,vectors)-.5
    ids,k=np.nonzero(gap<.12)
    own,oth=owner[ids],other[ids]
    a=axis[ids]
    angle_own=np.einsum('md,md->m',perp(a),vectors[ids,k])
    angle_oth=np.einsum('md,md->m',a,perp(edges[oth,k]))
    pair_cols=np.column_stack((own,own+n,oth,oth+n,own+2*n,oth+2*n))
    pair_coeff=np.column_stack((a,-a,-angle_own,-angle_oth))
    parts=[sparse_rows(pair_cols,pair_coeff,width)]
    rhs=[gap[ids,k]-1e-10]
    for dim in (0,1):
        for sign in (-1,1):
            upper=sign==1
            slack=(current-c[:,:,dim]) if upper else c[:,:,dim]
            sq,corner=np.nonzero(slack<.12)
            col=np.column_stack((sq+n*dim,sq+2*n,np.full(len(sq),3*n)))
            coeff=np.column_stack((np.full(len(sq),sign),sign*perp(edges)[sq,corner,dim],np.full(len(sq),-int(upper))))
            parts.append(sparse_rows(col,coeff,width))
            rhs.append(slack[sq,corner])
    objective=np.zeros(width)
    objective[-1]=1
    result=linprog(objective,A_ub=vstack(parts).tocsr(),b_ub=np.concatenate(rhs),
        bounds=[(-translation,translation)]*(2*n)+[(-rotation,rotation)]*n+[(-.04,.02)],
        method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not result.success or result.x[-1]>-5e-10:
        return None
    return result.x[:-1].reshape(3,n).T


def seeds(p, L):
    h = (np.abs(np.cos(p[:,2]))+np.abs(np.sin(p[:,2])))/2
    span = (p[:,:2]+h[:,None]).max(0)
    p = p.copy()
    p[:,:2] += (L-span)/2
    xy=p[:,:2]
    centers=Voronoi(xy).vertices
    wall=np.linspace(.5,L-.5,max(4,int(L*8)))
    boundary=np.concatenate([np.column_stack((wall,wall*0+y)) for y in (.5,L-.5)])
    positions=np.unique(np.round(np.vstack((centers,boundary,boundary[:,::-1])),7),axis=0)
    positions=positions[(positions>=.45).all(1)&(positions<=L-.45).all(1)]
    _, ids=cKDTree(xy).query(positions,k=12)
    near=p[ids]
    displacement=positions[:,None,:]-near[:,:,:2]
    x,y=displacement[:,:,0],displacement[:,:,1]
    c,s=np.cos(near[:,:,2]),np.sin(near[:,:,2])
    directions=np.maximum(np.abs(x*c+y*s),np.abs(y*c-x*s))
    options=[]
    for a in (0, math.pi/12, -math.pi/12, math.pi/6, -math.pi/6, math.pi/4):
        c,s=math.cos(a),math.sin(a)
        ha=(abs(c)+abs(s))/2
        d=near[:,:,2]-a
        width=(1+np.abs(np.cos(d))+np.abs(np.sin(d)))/2
        projection=np.maximum(directions,np.maximum(np.abs(x*c+y*s),np.abs(y*c-x*s)))
        overlap=np.maximum(width-projection,0)
        wallcost=np.maximum(ha-positions,0)**2+np.maximum(positions+ha-L,0)**2
        loss=(overlap**2).sum(1)+wallcost.sum(1)
        for k in np.argsort(loss)[:7]:
            options.append((float(loss[k]),np.array([*positions[k],a])))
    options.sort(key=lambda item:item[0])
    result=[]
    for loss, q in options:
        if all(np.linalg.norm(q[:2]-r[:2])>0.7 for _,r in result):
            result.append((loss,q))
        if len(result)==2:
            break
    return [(loss,np.vstack((p,q))) for loss,q in result]
