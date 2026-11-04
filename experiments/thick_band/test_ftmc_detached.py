import json,os,subprocess,sys,time
from pathlib import Path
import ftmc_detached as d
def test_launch_detached_idempotent(tmp_path,monkeypatch):
    calls=[]
    class Proc:pid=123
    def spawn(cmd,**kw):
        calls.append(kw);return Proc()
    monkeypatch.setattr(d.subprocess,"Popen",spawn)
    first=d.launch(tmp_path,"smoke",3)
    assert d.launch(tmp_path,"smoke",3)==first
    assert len(calls)==1 and calls[0]["start_new_session"]
    assert calls[0]["stdin"]==subprocess.DEVNULL
def test_atomic(tmp_path):
    d.atomic(tmp_path/"x.json",{"pid":1})
    assert json.loads((tmp_path/"x.json").read_text())=={"pid":1}
    assert not (tmp_path/"x.tmp").exists()
def test_supervisor_timeout_no_gpu(tmp_path):
    # Real detached subprocess fixture validates independent TTL and durable tar.
    root=tmp_path/"root";root.mkdir()
    (tmp_path/"ftmc_pilot.py").write_text("import time;time.sleep(60)")
    d.supervise(root,"pilot",-29.9)
    row=json.loads((root/"done.json").read_text())
    assert row["watchdog_fired"] and row["exitcode"]!=0
    assert (root/"evidence.tar.gz").exists()
