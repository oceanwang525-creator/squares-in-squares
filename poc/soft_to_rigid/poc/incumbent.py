import pandas as pd, numpy as np, math, sys
INC='/Users/lucaszhang/squares-in-squares/poc/pell_corrugation/data/border-scan/merged.csv'
def load():
    return pd.read_csv(INC)
def side(g):
    h=(np.abs(np.cos(g.angle))+np.abs(np.sin(g.angle)))/2
    return max((g.x+h).max()-(g.x-h).min(),(g.y+h).max()-(g.y-h).min())
def table():
    d=load(); L={n:side(g) for n,g in d.groupby('n')}; return d,L
if __name__=='__main__':
    d,L=table(); r={n:L[n]/math.sqrt(n)-1 for n in L}
    sq={k*k for k in range(1,40)}; rows=[]
    for n in range(20,201):
        if n in sq: continue
        m=[r[k] for k in range(n-5,n+6) if k!=n and k not in sq]
        rows.append((r[n]-np.median(m),n,L[n],math.ceil(math.sqrt(n))))
    rows.sort(reverse=True)
    for x in rows[:20]: print('gap %.5f n=%d L=%.6f ceil=%d'%x)
