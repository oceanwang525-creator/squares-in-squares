import sys, numpy as np, pandas as pd
def Ls(path):
    df=pd.read_csv(path, dtype=str)
    n=df['n'].astype(int).values
    x=df.x.str.lstrip('s').astype(float).values; y=df.y.str.lstrip('s').astype(float).values; a=df.angle.str.lstrip('s').astype(float).values
    c,s=np.cos(a),np.sin(a)
    ex=0.5*(np.abs(c)+np.abs(s))
    out={}
    d=pd.DataFrame(dict(n=n,xl=x-ex,xh=x+ex,yl=y-ex,yh=y+ex))
    g=d.groupby('n').agg(dict(xl='min',xh='max',yl='min',yh='max'))
    L=np.maximum(g.xh-g.xl,g.yh-g.yl)
    return L
if __name__=='__main__':
    L=Ls(sys.argv[1]); print('score', (L/np.sqrt(L.index)-1).sum())
