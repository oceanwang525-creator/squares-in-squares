"""Proxy crop experiment: for donors n'>=900 in merged.csv, anchor windows at all
(xmin,ymin) of squares (x4 reflections); s(m)=m-th smallest Chebyshev extent."""
import sys, json, time
import numpy as np, pandas as pd
M='/Users/lucaszhang/squares-in-squares/poc/pell_corrugation/data/border-scan/merged.csv'
df=pd.read_csv(M,dtype=str)
df['n']=df['n'].astype(int)
lit=json.load(open('/Users/lucaszhang/squares-in-squares/poc/pell_corrugation/data/border-scan/merged-literal-official.json')) if False else None
def box(g):
    x=g.x.astype(float).values;y=g.y.astype(float).values;a=g.angle.astype(float).values
    h=(np.abs(np.cos(a))+np.abs(np.sin(a)))/2
    return x-h,x+h,y-h,y+h
def side(g):
    x0,x1,y0,y1=box(g);return max(x1.max()-x0.min(),y1.max()-y0.min())
donors=[int(a) for a in sys.argv[1].split(',') if a.strip()]
grp=dict(list(df.groupby('n')))
ours={n:side(grp[n]) for n in range(1,1001)}
best={}  # n -> (s,donor,flip,anchor)
t=time.time()
for d in donors:
    x0,x1,y0,y1=box(grp[d]); 
    for fx in (0,1):
      for fy in (0,1):
        X0,X1,Y0,Y1=x0,x1,y0,y1
        if fx: X0,X1=-x1,-x0
        if fy: Y0,Y1=-y1,-y0
        ax=np.unique(np.round(X0,7)); ay=np.unique(np.round(Y0,7))
        for a in ax:
            ok=X0>=a-1e-9
            for b in ay:
                k=ok&(Y0>=b-1e-9)
                if k.sum()<50: continue
                e=np.sort(np.maximum(X1[k]-a,Y1[k]-b))
                m=np.arange(1,len(e)+1)
                for mm in range(50,min(len(e),d-1)+1):
                    pass
                # vectorised update
                for mm in np.nonzero(e<np.array([best.get(i,(9e9,))[0] for i in m])-1e-12)[0] if False else []: pass
                cur=best_arr[m-1] if 'best_arr' in globals() else None
                if cur is None:
                    best_arr=np.full(1001,9e9); best_info=[None]*1001; cur=best_arr[m-1]
                imp=e+1e-9<cur
                if imp.any():
                    for i in np.nonzero(imp)[0]:
                        best_arr[m[i]-1]=e[i]; best_info[m[i]-1]=(d,fx,fy,float(a),float(b))
    print('donor',d,time.time()-t,flush=True)
out={}
for n in range(100,901):
    s=best_arr[n-1]; L=ours[n]
    out[n]=dict(s=float(s),ours=L,delta=float(s-L),info=best_info[n-1])
json.dump(out,open('/Users/lucaszhang/squares-in-squares/poc/bign_donors/docs/proxy_results.json','w'))
w=[n for n in out if out[n]['delta']<-1e-9]
print('wins',len(w),w[:20]); 
ds=np.array([out[n]['delta'] for n in out]); print('min delta',ds.min(),'median',np.median(ds))
for n in (100,200,300,500,645,700,800,900): print(n,out[n])
