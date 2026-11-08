"""Bounded CPU scan of Pell17 cores, partial upright borders and deletions.

All w with core count <=1000 are attempted. Full L borders and four prefix
orders cover every intermediate count; four coordinate extreme deletion orders
also test every smaller count. This is not exhaustive over arbitrary subsets.
"""
from pathlib import Path
import sys,json,math,hashlib,subprocess,io,time
import numpy as np
import pandas as pd
import pell_corrugation as p
from squares.core import read_submission,score
D=Path(__file__).resolve().parents[1]/"data"/"border-scan"
BASE=Path("/home/kit/out/public-20261104/merged-material-candidate.csv")
CPP=Path(__file__).resolve().parents[3]/"poc/cuda-loop/cuda/certifier"
def save(name,obj):
 (D/name).write_text(json.dumps(obj,indent=2))
def cpp(path,label):
 out=D/(label+"-cpp.json")
 subprocess.run([str(CPP),str(path),"--threads","4","--details",str(out)],check=True)
 j=json.loads(out.read_text())
 assert j["rejected"]==0,j
 return j
def main():
 D.mkdir(exist_ok=True,parents=True)
 start=time.time()
 assert hashlib.sha256(BASE.read_bytes()).hexdigest()=="007577524c36f240cbd33abc3aec0d198374e8d2be71e795717bf119461a7651"
 baseline=read_submission(BASE)
 print("baseline certification",flush=True)
 bs=score(baseline);save("baseline-official.json",bs)
 assert bs["accepted"]==1000 and abs(bs["score"]-22.25406709809781)<1e-10
 ref={r["n"]:float(r["L"]) for r in bs["results"]}
 pool={};records=[];counts={"screened":0}
 def offer(q,meta,polish=False):
  n=len(q)
  if not 1<=n<=1000:return
  counts["screened"]+=1
  L=p.extent(q)
  if L>=ref[n]+(0.003 if polish else -1e-9):return
  if n not in pool or L<pool[n][0]:
   pool[n]=(L,q.copy(),meta,polish)
 for w in range(1,18):
  b=math.floor(w*p.INV_S2)
  n0=(b+9)*(b+10)//2+(b+8)*(b+9)//2+b*(b+1)+w*(w+12)
  if n0>1000:break
  cache=D/f"core-w{w}.npz"
  if cache.exists():
   z=np.load(cache);q=z["q"];L=p.extent(q);meta=json.loads(str(z["meta"]))
  else:
   print("core",w,n0,flush=True)
   try:
    res,L0,meta=p.build_pell17_core(w,iterations=35)
    q=res.q;L=res.L
    valid,gap=p.sat_check(q)
    meta.update(sat_ok=valid,min_gap=gap)
    if not valid:raise ValueError("core SAT failed")
    np.savez(cache,q=q,meta=json.dumps(meta))
   except (RuntimeError,ValueError,IndexError) as e:
    records.append(dict(w=w,n0=n0,error=str(e)));save("cores.json",records);continue
  records.append(meta);save("cores.json",records)
  offer(q,dict(w=w,t=0,kind="core"),True)
  k=math.floor(L)
  for t in range(0,k+1):
   N=n0+2*k*t+t*t
   if N>1080:break
   wrap=p.add_l_border(q,t,k+t) if t else q.copy()
   for axis in (0,1):
    for rev in (False,True):
     order=np.argsort(wrap[:,axis],kind="stable")
     if rev:order=order[::-1]
     # Prefix deletions include all smaller n; only promising extents retained.
     for n in range(1,min(1000,N)+1):
      sub=wrap[order[:n]]
      offer(sub,dict(w=w,t=t,kind="extreme-prefix",axis=axis,reverse=rev), n>=N-3)
   if t:
    border=wrap[n0:]
    for axis in (0,1):
     for rev in (False,True):
      order=np.argsort(border[:,axis],kind="stable")
      if rev:order=order[::-1]
      for e in range(1,min(len(border),1000-n0)+1):
       offer(np.vstack([q,border[order[:e]]]),dict(w=w,t=t,kind="partial-border",axis=axis,reverse=rev),e>=len(border)-3)
  print("finished core",w,"pool",len(pool),flush=True)
 save("screening.json",dict(**counts,shortlist=sorted(pool)))
 wins={};wininfo={}
 # Also check the already committed Argon candidates, independently.
 committed=read_submission(Path(__file__).resolve().parents[1]/"data/output.csv")
 for n,rs in committed.items():
  q=np.array([[float(r[c]) for c in ("x","y","angle")] for r in rs])
  offer(q,dict(kind="committed-argon"),False)
 for n,(L,q,meta,polish) in sorted(pool.items()):
  print("candidate",n,L,ref[n],flush=True)
  options=[q]
  if polish:
   r=p.polish_band(q,L,iterations=20,position_trust=.05,angle_trust=.005)
   if r is not None:options.insert(0,r.q)
  for candidate in options:
   if p.extent(candidate)>=ref[n]-1e-9:continue
   text=p.csv_text({n:p.rows(candidate)})
   checks=p.certify_transport(text)
   check=checks[str(n)]
   if not all(c["ok"] and float(c["L"])<ref[n]-1e-9 for c in check.values()):continue
   path=D/f"win-{n}.csv";path.write_text(text)
   (D/f"win-{n}-pandas.csv").write_text(pd.read_csv(io.StringIO(text)).to_csv(index=False))
   cpp(path,f"win-{n}");cpp(D/f"win-{n}-pandas.csv",f"win-{n}-pandas")
   wins[n]=p.metric._parse_submission_csv(text)[n]
   wininfo[n]=dict(construction=meta,baseline_L=ref[n],certificates=check)
   print("WIN",n,flush=True);break
 save("wins.json",wininfo)
 merged=baseline|wins
 mergedpath=D/"merged.csv";mergedpath.write_text(p.csv_text(merged))
 pan=D/"merged-pandas.csv";pan.write_text(pd.read_csv(mergedpath).to_csv(index=False))
 results={}
 for mode,path in (("literal",mergedpath),("pandas",pan)):
  print("full certification",mode,flush=True)
  result=score(read_submission(path));save("merged-"+mode+"-official.json",result)
  assert result["accepted"]==1000
  cpp(path,"merged-"+mode)
  results[mode]=result["score"]
 report=dict(baseline_score=bs["score"],merged_scores=results,delta={m:s-bs["score"] for m,s in results.items()},wins=sorted(wins),seconds=time.time()-start,pandas_version=pd.__version__,counts=counts,coverage="All w with n0<=1000 attempted; upright L borders t<=k, four extreme-prefix deletion orders and four partial-border prefix orders, count<=1000; only shortlist polished. No arbitrary-subset exhaustive claim.")
 save("report.json",report);print(json.dumps(report),flush=True)
if __name__=="__main__":main()
