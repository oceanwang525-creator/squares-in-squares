import sys,glob,pickle,collections
sys.path.insert(0,'/Users/lucaszhang/squares-in-squares/poc/public_resweep_other/poc')
import scan
from scan import *
for f in sorted(glob.glob(F+'data/src/**/*.npz',recursive=True)+glob.glob(F+'data/ext/**/*.npz',recursive=True)):
    if 'Minuii' in f:continue
    rel=os.path.relpath(f,F+'data');z=np.load(f,allow_pickle=False)
    if 'ns' in z.files and 'coords' in z.files:
        off=0;co=z['coords']
        for n in z['ns']:
            n=int(n);c=co[off:off+n];off+=n
            offer(n,rel,c[:,0].tolist(),c[:,1].tolist(),c[:,2].tolist(),'auto')
        print(rel,'bank',off,len(co))
        continue
    for k in z.files:
        a=z[k]
        if a.ndim==2 and a.shape[1]==3 and a.shape[0]>=1:
            offer(int(a.shape[0]),rel+':'+k,a[:,0].tolist(),a[:,1].tolist(),a[:,2].tolist(),'auto')
agg=collections.defaultdict(lambda:[0,0,0])
for s,d in scan.stats.items():
    a=agg[s.split(':')[0]];a[0]+=d['examined'];a[1]+=d['unique'];a[2]+=d['pass_']
print(dict(agg))
pickle.dump(dict(cand=dict(scan.cand),agg={}),open(F+'data/cand_npz.pkl','wb'))
print(sum(len(v) for v in scan.cand.values()))
