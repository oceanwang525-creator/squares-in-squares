import json, math
S2=math.sqrt(2); I2=1/S2
rows=json.load(open('/Users/lucaszhang/squares-in-squares/poc/wildcard_round3/docs/scan_phase.json'))
out=[]
for n,lo,lb,le,lod,o in rows:
    if o is None or lod>=lo+0.0151: continue
    L,w,a1,a2,b1,b2,t,c=o
    L0=.5*(a1+a2)+1+w*I2
    uw=.5*(w-1)+.5*abs(a1-a2)*I2+I2
    lo_=max((b1+1)*I2+.5,uw); hi=S2*L0-max((b2+1)*I2+.5,uw)
    span=hi-lo_; slack=span-math.floor(span+1e-9)
    out.append((round(lod-lo,5),n,w,round(slack,4),round(span,2),round(slack/span,5),round(lb-lo,5),c,(a1,a2,b1,b2,t)))
for r in sorted(out): print(r)
