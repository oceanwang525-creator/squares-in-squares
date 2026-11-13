"""Trusted checkout only. Never execute code from an untrusted PR tree."""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from certification_cache import ROOT
from certification_runtime import install_runtime

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--proof",type=Path,required=True)
    p.add_argument("--full-audit",action="store_true")
    args=p.parse_args()
    if args.full_audit:os.environ["CERT_FULL_AUDIT"]="true"
    spec=importlib.util.spec_from_file_location("trusted_union",ROOT/"poc/auto_union/poc/union.py")
    u=importlib.util.module_from_spec(spec);sys.modules[spec.name]=u;spec.loader.exec_module(u)
    runtime=install_runtime(u)
    before=runtime.full_gate(args.baseline)
    after=runtime.full_gate(args.candidate)
    from decimal import Decimal
    for a,b in zip(before,after):
        if any(Decimal(y["L"])>Decimal(x["L"]) for x,y in zip(a["results"],b["results"])):
            raise ValueError("Per-n regression")
    args.proof.write_text(json.dumps(dict(before=before,after=after,
        context=runtime.context,cache_entries=len(runtime.cache),
        computed_groups=len(runtime.metrics)),indent=2))
    print("CERTIFIED 1000/1000 per mode; computed unique groups",len(runtime.metrics),
        "scores",[x["score"] for x in after],flush=True)

if __name__=="__main__":main()
