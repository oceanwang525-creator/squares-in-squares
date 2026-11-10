import numpy as np, tiltband as t, json
T={635:(25,25.69874),686:(26,26.68672),910:(30,30.646884),911:(30,30.660916),971:(31,31.636789),972:(31,31.651401)}
ours={635:25.7071067812,686:26.7071067812,910:30.679154,911:30.695622,971:31.676795,972:31.684061}
res={}
for n,(K,Lt) in T.items():
    row={}
    for tag,L in (("leaderL",Lt),("oursL-0.001",ours[n]-0.001)):
        out=[]
        for th in list(np.arange(0.20,1.37,0.01))+[np.pi/4]:
            out.append((t.best_count(L,th)[0],round(float(th),3)))
        out.sort(reverse=True)
        non45=[o for o in out if abs(o[1]-0.785)>0.02]
        row[tag]=dict(need=n,K2=K*K,best=out[:3],best_non45=non45[:3])
        print(n,tag,round(L,6),row[tag],flush=True)
    res[n]=row
json.dump(res,open('../docs/scan_counts.json','w'),indent=1,default=float)
