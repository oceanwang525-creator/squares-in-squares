"""Harvest cert/yaml/json/npz packings from cloned catalog repos in data/ext (data only)."""
import sys,gzip,pickle,glob,os,re,json,collections
sys.path.insert(0,'/Users/lucaszhang/squares-in-squares/poc/public_resweep_other/poc')
import scan
from scan import *
from fractions import Fraction
import mpmath as mp
import yaml
mp.mp.dps=70
def fmt(v):return mp.nstr(v,65)
def rat(s):
    s=s.strip()
    if '/' in s:a,b=s.split('/');return Fraction(int(a),int(b))
    return Fraction(s)
def do_cert(text,src):
    lines=[l for l in text.splitlines() if l.strip() and not l.startswith('#')]
    h=lines[0].split()
    if len(h)!=2:return
    n=int(h[0])
    if not 1<=n<=1000 or len(lines)-1!=n:return
    R=[l.split() for l in lines[1:]]
    if any(len(r)!=3 for r in R):return
    Fx=[rat(r[0]) for r in R];Fy=[rat(r[1]) for r in R];Ft=[rat(r[2]) for r in R]
    xf=np.array([float(v) for v in Fx]);yf=np.array([float(v) for v in Fy])
    af=np.array([2*np.arctan(float(t)) for t in Ft])
    def lazy():
        return ([fmt(mp.mpf(v.numerator)/v.denominator) for v in Fx],[fmt(mp.mpf(v.numerator)/v.denominator) for v in Fy],[fmt(2*mp.atan(mp.mpf(t.numerator)/t.denominator)) for t in Ft])
    offer(n,src,None,None,None,'rad',fl=(xf,yf,af),lazy=lazy)
ROOT=mp.mpf(0)
def num(x):
    global ROOT
    if isinstance(x,list):return sum(num(c)*ROOT**i for i,c in enumerate(x))
    s=str(x)
    if '/' in s:
        a,b=s.split('/');return mp.mpf(a)/mp.mpf(b)
    return mp.mpf(s)
def do_witness(text,src):
    global ROOT
    w=yaml.load(text,Loader=yaml.CSafeLoader).get('witness',{})
    n=w.get('n',0)
    if not 1<=n<=1000:return
    if w.get('scalar',{}).get('kind')=='algebraic-number-field':
        lo,hi=map(num,w['scalar']['isolating_interval']);coeff=list(map(num,w['scalar']['minimal_polynomial']))
        ROOT=mp.findroot(lambda t:mp.polyval(coeff,t),(lo,hi),solver='secant')
    xs=[];ys=[];As=[]
    for s in w['squares']:
        if 'center' in s:
            x,y=map(num,s['center'])
            if 'basis' in s:a=mp.atan2(num(s['basis'][1]),num(s['basis'][0]))
            else:a=num(s.get('angle',0))*mp.pi/180
        else:
            cs=[[num(z) for z in c] for c in s['corners']]
            x=sum(c[0] for c in cs)/4;y=sum(c[1] for c in cs)/4
            a=mp.atan2(cs[1][1]-cs[0][1],cs[1][0]-cs[0][0])
        xs.append(fmt(x));ys.append(fmt(y));As.append(fmt(a))
    offer(n,src,xs,ys,As)
def do_evand_p(d,src):
    n=d['n'];sq=d['squares']
    if not 1<=n<=1000 or len(sq)!=n:return
    xs=[fmt(num(s[0])) for s in sq];ys=[fmt(num(s[1])) for s in sq];As=[fmt(num(s[2])*mp.pi/180) for s in sq]
    offer(n,src,xs,ys,As)
def opn(f):
    return gzip.open(f,'rt',errors='replace') if f.endswith('.gz') else open(f,errors='replace')
errs=collections.Counter()
for f in sorted(glob.glob(F+'data/ext/**/*',recursive=True)):
    if '/.git/' in f or not os.path.isfile(f):continue
    rel=os.path.relpath(f,F+'data')
    try:
        if re.search(r'\.cert(\.gz)?$',f):
            if os.path.getsize(f)>20e6:continue
            do_cert(opn(f).read(),rel)
        elif re.search(r'\.ya?ml(\.gz)?$',f) and '/witnesses/' in f and '.schema.' not in f:
            do_witness(opn(f).read(),rel)
        elif re.search(r'/data/p/[^/]*\.json$',f):
            do_evand_p(json.load(open(f)),rel)
        elif f.endswith('.npz') and 'Minuii' in f:
            z=np.load(f);n=len(z['th'])
            offer(n,rel,z['xy'][:,0].tolist(),z['xy'][:,1].tolist(),z['th'].tolist())
        elif f.endswith('.json') and os.path.getsize(f)<300e6:
            do_json(f,rel)
        elif f.endswith('.json.gz') and os.path.getsize(f)<50e6:
            walk(json.load(gzip.open(f,'rt')),rel)
        elif f.endswith('.csv') and os.path.getsize(f)<300e6:
            do_csv(f,rel)
    except Exception as e:errs[(rel.split('/')[0],type(e).__name__)]+=1
print(dict(errs))
agg=collections.defaultdict(lambda:dict(examined=0,unique=0,pass_=0,ns=set()))
for s,d in scan.stats.items():
    t='/'.join(s.split('/')[:2]);a=agg[t]
    a['examined']+=d['examined'];a['unique']+=d['unique'];a['pass_']+=d['pass_'];a['ns']|=d['ns']
for t,a in sorted(agg.items()):print(t,a['examined'],a['unique'],a['pass_'],len(a['ns']))
pickle.dump(dict(cand=dict(scan.cand),agg={t:{**a,'ns':sorted(a['ns'])} for t,a in agg.items()}),open(F+'data/cand_ext.pkl','wb'))
print('cand n',len(scan.cand),sum(len(v) for v in scan.cand.values()))
