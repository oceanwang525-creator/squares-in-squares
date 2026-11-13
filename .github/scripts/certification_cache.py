"""Trusted-main artifact provenance and context-bound official group evidence."""
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path
import importlib.metadata
import requests

NAME = "official-group-evidence-v1"
BOOTSTRAP_SHA = "32a65ad0a4f8556c5d79a09acea7716f068cb98354892d213b200bd93a50c1a2"  # populated only from an independently completed local full audit
ROOT = Path(__file__).resolve().parents[2]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def context(root=ROOT):
    files = ["poc/baseline/poc/src/squares/_official_metric.py",
             "poc/baseline/poc/src/squares/core.py", "poc/auto_union/poc/union.py",
             ".github/scripts/check_final_csv.py", ".github/scripts/certification_runtime.py",
             ".github/scripts/certification_cache.py"]
    # Exclude this one literal bootstrap digest to avoid a self-referential fingerprint.
    hashes = {}
    for name in files:
        data = (root/name).read_bytes()
        if name.endswith("certification_cache.py"):
            data = data.replace(BOOTSTRAP_SHA.encode(), b"") if BOOTSTRAP_SHA else data
        hashes[name] = digest(data)
    import mpmath
    from mpmath.libmp import BACKEND
    return dict(schema=1, files=hashes, python=list(sys.version_info[:3]),
        implementation=platform.python_implementation(), machine=platform.machine(),
        packages={p: importlib.metadata.version(p) for p in ("pandas","numpy","mpmath")},
        backend=BACKEND, mpmath_default_dps=mpmath.mp.dps,
        transport="pandas.read_csv-whole-file-default-to_csv-indexFalse;literal",
        official_aggregation="sorted-n-python-sum-original-worker-float")


def envelope(payload):
    raw=json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return dict(payload=payload, sha256=digest(raw))


def decode(raw, expected_context):
    data=json.loads(raw)
    if envelope(data["payload"])["sha256"] != data["sha256"]:
        raise ValueError("Tampered certificate manifest")
    p=data["payload"]
    if p["context"] != expected_context:
        print("CERT cache context changed; certify unknown groups",flush=True)
        return {}
    entries=p["entries"]
    for key, result in entries.items():
        if len(key)!=64 or any(x not in "0123456789abcdef" for x in key):
            raise ValueError("Invalid content key")
        if len(result)!=4 or not 1<=result[0]<=1000 or result[3]!="ok" or result[2] is None:
            raise ValueError("Invalid certificate entry")
        from decimal import Decimal
        if not Decimal(result[2]).is_finite() or Decimal(result[2])<=0:
            raise ValueError("Invalid side evidence")
        import math
        if not math.isfinite(result[1]) or result[1]<0:
            raise ValueError("Invalid per-group score")
    return {k:tuple(v) for k,v in entries.items()}


def api(path, binary=False):
    token=os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    headers={"Accept":"application/vnd.github+json"}
    if token:headers["Authorization"]="Bearer "+token
    r=requests.get("https://api.github.com/"+path,headers=headers,timeout=60)
    r.raise_for_status()
    return r.content if binary else r.json()


def trusted_run(run, repo, repo_id):
    return (run["head_branch"]=="main" and run["event"] in ("push","workflow_dispatch")
        and run["conclusion"]=="success" and run["head_repository"]["id"]==repo_id
        and run["repository"]["id"]==repo_id
        and run["path"]==".github/workflows/publish-certified-merge.yml")


def restore(ctx):
    bootstrap=ROOT/".github/certification-bootstrap.json"
    if bootstrap.exists():
        if not BOOTSTRAP_SHA or digest(bootstrap.read_bytes())!=BOOTSTRAP_SHA:
            raise ValueError("Bootstrap evidence does not match pinned audit")
        entries=decode(bootstrap.read_bytes(),ctx)
    else:entries={}
    repo=os.environ.get("GITHUB_REPOSITORY")
    if not repo:return entries
    info=api("repos/"+repo)
    artifacts=api("repos/"+repo+"/actions/artifacts?name="+NAME+"&per_page=100")
    for artifact in artifacts["artifacts"]:
        if artifact["expired"]:continue
        run=api("repos/"+repo+"/actions/runs/"+str(artifact["workflow_run"]["id"]))
        if not trusted_run(run,repo,info["id"]):continue
        # Only an ancestor trusted main run can seed the current invocation.
        if subprocess.run(["git","merge-base","--is-ancestor",run["head_sha"],"HEAD"],
                          cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
            continue
        raw=api("repos/"+repo+"/actions/artifacts/"+str(artifact["id"])+"/zip",True)
        if artifact.get("digest")!="sha256:"+digest(raw):
            raise ValueError("Artifact ZIP digest mismatch")
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            if z.namelist()!=["group-evidence.json"]:raise ValueError("Unexpected cache archive")
            if z.getinfo("group-evidence.json").file_size>20_000_000:raise ValueError("Oversized evidence")
            cached=decode(z.read("group-evidence.json"),ctx)
        if cached:
            entries.update(cached)
            print("CERT trusted artifact",artifact["id"],"run",run["id"],"entries",len(cached),flush=True)
            break
    return entries


def save(runtime, canonical):
    path=Path(os.environ.get("CERT_EVIDENCE_OUT","/tmp/squares-cert-evidence"))/"group-evidence.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(context=runtime.context, entries={k:v for k,v in runtime.cache.items() if v[3]=="ok"},
        canonical_sha256=digest(canonical.read_bytes()), source_sha=os.environ.get("GITHUB_SHA"),
        metrics=runtime.metrics)
    path.write_text(json.dumps(envelope(payload),sort_keys=True,separators=(",",":"),allow_nan=False))
    print("CERT evidence saved",str(path),len(payload["entries"]),flush=True)
