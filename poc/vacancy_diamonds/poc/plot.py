import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, math, sys
q=np.load('data/q401.npy'); L=20.535534
fig,ax=plt.subplots(figsize=(11,11))
for x,y,t in q:
    c,s=math.cos(t),math.sin(t); P=[(x+(a*c-b*s),y+(a*s+b*c)) for a,b in [(-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5)]]
    ta=abs(((t%(math.pi/2))-math.pi/4))<0.05
    ax.add_patch(plt.Polygon(P,fc='tab:red' if ta else 'lightblue',ec='k',lw=.4))
ax.plot([-L/2,L/2,L/2,-L/2,-L/2],[-L/2,-L/2,L/2,L/2,-L/2],'k');ax.set_aspect('equal');ax.set_xlim(-L/2-.2,L/2+.2);ax.set_ylim(-L/2-.2,L/2+.2)
plt.savefig('docs/inc401.png',dpi=80)
