"""Idempotent detached runtime job, independent timeout and durable evidence."""
import threading,argparse,fcntl,json,os,signal,subprocess,sys,time,tarfile,hashlib
from pathlib import Path

def atomic(path, data):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data,sort_keys=True));os.replace(tmp,path)
def launch(root,mode,seconds):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    lock=(root/"launch.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX)
    marker=root/"started.json"
    if marker.exists():return json.loads(marker.read_text())
    with (root/"supervisor.log").open("ab",buffering=0) as out:
        proc=subprocess.Popen([sys.executable,__file__,"supervise",str(root),mode,str(seconds)],
            stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,
            close_fds=True)
    row=dict(supervisor_pid=proc.pid,created=time.time(),mode=mode,seconds=seconds)
    atomic(marker,row)
    return row
def pack_evidence(root,mode,name):
    root=Path(root);bundle=root/name;tmp=root/(name+".tmp")
    with tarfile.open(tmp,"w:gz") as f:
        for item in ("started.json","pid.json","heartbeat.json","done.json","job.log"):
            if (root/item).exists():f.add(root/item,arcname=mode+"/"+item)
        if mode=="pilot" and (root.parent/"ftmc-results").exists():
            f.add(root.parent/"ftmc-results",arcname="ftmc-results")
    os.replace(tmp,bundle)

def supervise(root,mode,seconds):
    root=Path(root);signal.signal(signal.SIGHUP,signal.SIG_IGN)
    with (root/"job.log").open("ab",buffering=0) as out:
        cmd=[sys.executable,"-u",__file__,"smoke",str(root)] if mode=="smoke" else [sys.executable,"-u",str(root.parent/"ftmc_pilot.py")]
        env=dict(os.environ,FTMC_SECONDS=str(seconds),OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="2")
        job=subprocess.Popen(cmd,cwd=root.parent,env=env,stdin=subprocess.DEVNULL,
            stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        atomic(root/"pid.json",dict(pid=job.pid,started=time.time(),mode=mode))
        deadline=time.monotonic()+seconds+120
        next_snapshot=time.monotonic()+120
        snapshot=None
        killed=False
        while job.poll() is None:
            atomic(root/"heartbeat.json",dict(pid=job.pid,time=time.time(),alive=True))
            if time.monotonic()>deadline:
                killed=True;os.killpg(job.pid,signal.SIGTERM)
                try:job.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL)
                break
            if mode=="pilot" and time.monotonic()>next_snapshot and (snapshot is None or not snapshot.is_alive()):
                snapshot=threading.Thread(target=pack_evidence,args=(root,mode,"checkpoint.tar.gz"),daemon=True)
                snapshot.start();next_snapshot=time.monotonic()+120
            time.sleep(.5)
        code=job.wait()
        if snapshot is not None:snapshot.join(timeout=15)
    atomic(root/"done.json",dict(exitcode=code,watchdog_fired=killed,finished=time.time()))
    pack_evidence(root,mode,"evidence.tar.gz")

def smoke(root):
    import torch
    root=Path(root)
    assert torch.cuda.is_available() and "A100" in torch.cuda.get_device_name()
    from gensane_polish import quench
    print(json.dumps(dict(event="smoke_start",pid=os.getpid(),gpu=torch.cuda.get_device_name())),flush=True)
    torch.set_default_dtype(torch.float64)
    q=torch.tensor([[[x+.5,y+.5,0.] for x in range(3) for y in range(3)]],device="cuda")
    q,info=quench(q,[3.01],steps=8,seconds=4,clearance=2e-9)
    for i in range(3):
        z=torch.linalg.vector_norm(q);torch.cuda.synchronize()
        print(json.dumps(dict(event="smoke_progress",step=i+1,norm=float(z))),flush=True)
    print(json.dumps(dict(event="smoke_complete",pid=os.getpid(),finite=bool(torch.isfinite(q).all()))),flush=True)
if __name__=="__main__":
    action=sys.argv[1]
    if action=="launch":print(json.dumps(launch(sys.argv[2],sys.argv[3],float(sys.argv[4]))),flush=True)
    elif action=="supervise":supervise(sys.argv[2],sys.argv[3],float(sys.argv[4]))
    elif action=="smoke":smoke(sys.argv[2])
