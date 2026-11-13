import csv
import json
import math
import sys
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from numba import njit

@njit
def crop(lo, hi, length):
    best, bx, by = 0, 0., 0.
    for right in hi[:, 0]:
        x = right - length
        valid = (lo[:, 0] >= x-1e-12) & (hi[:, 0] <= right+1e-12)
        if valid.sum() <= best:
            continue
        ylo = np.sort(lo[valid, 1])
        yhi = np.sort(hi[valid, 1])
        left = 0
        for j in range(len(yhi)):
            y = yhi[j] - length
            while left < len(ylo) and ylo[left] < y-1e-12:
                left += 1
            count = j+1-left
            if count > best:
                best, bx, by = count, x, y
    return best, bx, by

def read(path):
    groups = defaultdict(list)
    with path.open(newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            n = int(r['n'])
            if 100 <= n <= 265:
                groups[n].append([float(r[k]) for k in ('x','y','angle')])
    return {n:np.array(p) for n,p in groups.items()}

def bounds(p):
    h = (abs(np.cos(p[:,2]))+abs(np.sin(p[:,2])))[:,None]/2
    return p[:,:2]-h, p[:,:2]+h

def rotated(p, theta):
    c, s = math.cos(theta), math.sin(theta)
    q = p.copy()
    q[:,:2] = p[:,:2] @ np.array([[c,s],[-s,c]])
    q[:,2] += theta
    return q

def orientations(p):
    a = p[:,2]
    c, s = np.cos(a)/2,np.sin(a)/2
    u,v = np.column_stack((c,s)),np.column_stack((-s,c))
    corners = np.concatenate([p[:,:2]+i*u+j*v for i in (-1,1) for j in (-1,1)])
    hull = corners[ConvexHull(corners).vertices]
    edges = np.roll(hull,-1,axis=0)-hull
    shifts = -np.arctan2(edges[:,1],edges[:,0])
    shifts = (shifts+math.pi/4)%(math.pi/2)-math.pi/4
    near = np.linspace(-.016,.016,17)
    return np.unique(np.round(np.concatenate([near,shifts[np.abs(shifts)<.09]]),10))

def main():
    source, report, output = map(Path,sys.argv[1:4])
    groups = read(source)
    sides = {}
    for n,p in groups.items():
        lo,hi=bounds(p)
        sides[n] = max(hi.max(0)-lo.min(0))
    trials, wins, nearest = 0, {}, {}
    for m,p in groups.items():
        targets = [n for n in range(max(100,m-26),min(m,200)+1)
                   if sides[n] > math.sqrt(n)+1e-6 and sides[m]-sides[n]<1.05]
        if not targets or not np.any(abs(np.sin(2*p[:,2]))>1e-8):
            continue
        for angle in orientations(p):
            if abs(angle)<1e-11:
                continue
            q=rotated(p,angle)
            lo,hi=bounds(q)
            for n in targets:
                target=sides[n]-1e-8
                count,x,y=crop(lo,hi,target)
                trials+=1
                nearest[n]=min(nearest.get(n,n),n-count)
                if count<n:
                    continue
                keep=np.where(((lo >= [x-1e-12,y-1e-12]) & (hi <= [x+target+1e-12,y+target+1e-12])).all(axis=1))[0][:n]
                if len(keep)!=n:
                    continue
                r=q[keep].copy()
                r[:,:2]-=lo[keep].min(axis=0)
                r[:,:2]*=1+5e-12
                l,h=bounds(r)
                length=max(h.max(0)-l.min(0))
                if length<sides[n]-1e-9 and (n not in wins or length<wins[n][0]):
                    wins[n]=(float(length),m,float(angle),r)
                    print('candidate',n,m,length,flush=True)
        if m%20==0:
            print(m,trials,flush=True)
    report.write_text(json.dumps(dict(sha256=sha256(source.read_bytes()).hexdigest(),trials=trials,nearest=nearest,candidates={n:w[:3] for n,w in wins.items()}),indent=2)+'\n')
    if wins:
        with output.open('w',newline='') as f:
            w=csv.writer(f)
            w.writerow(('id','n','square_id','x','y','angle'))
            for n,(_,_,_,r) in sorted(wins.items()):
                for j,row in enumerate(r):
                    w.writerow((n*(n-1)//2+j,n,j,*map(float,row)))
    print('done',trials,sorted(wins),flush=True)

if __name__=='__main__':
    main()
