import concurrent.futures,subprocess,json,os,time,pathlib,signal
r=pathlib.Path(__file__).resolve().parent.parent
p='/mnt/e/squares-in-squares/.venv/bin/python'
env=os.environ.copy()
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:env[k]='1'
def run(j):
 folder=j['folder'];out=r/'poc'/folder/'data/regenerate-nov13';out.mkdir(parents=True,exist_ok=True)
 t=time.time()
 print('START',folder,flush=True)
 with (out/'run.log').open('w') as h:
  proc=subprocess.Popen([p,'-u']+j['args'],cwd=r/j.get('cwd','.'),env=env,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
  try: code=proc.wait(timeout=j['timeout']);status='completed' if code==0 else 'failed'
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   code=proc.returncode;status='runtime-cap'
 result=dict(**j,pid=proc.pid,status=status,returncode=code,seconds=round(time.time()-t,2))
 (out/'execution.json').write_text(json.dumps(result,indent=2))
 print('END',folder,status,result['seconds'],flush=True)
 return result
jobs=json.loads((r/'regeneration-proof/jobs2.json').read_text())
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
 results=list(ex.map(run,jobs))
(r/'regeneration-proof/executions2.json').write_text(json.dumps(results,indent=2))
print('ALL_RUNS_FINISHED',flush=True)
