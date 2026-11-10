import pandas as pd, math, sys
p=pd.read_csv('../leader_curve_analysis/data/perpuzzle_best.csv',dtype={'L':str}).set_index('n')
rs=[2,3,5,6,7,10,11,13,14,15,17,19,21,22,23,26,29,30,31,34,37,41,53,61,65,73,85,89,97,101,113,125,145,157,181,197]
def ident(L,tol=3e-9):
    out=[]
    for r in rs:
        s=math.sqrt(r)
        for c in range(1,61):
            for a in range(-12,13):
                if a==0: continue
                m=L*c-a*s
                if abs(m-round(m))<tol*c: out.append((int(round(m)),a,r,c))
    out.sort(key=lambda t:(t[3]+abs(t[1])+abs(t[0])//5)); return out[:3]
ns=[int(a) for a in sys.argv[1:]]
for n in ns:
    L=float(p.loc[n,'L']); print(n,L-int(L),ident(L))
