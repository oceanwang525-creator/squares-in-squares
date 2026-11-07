"""Recertify files with the reference interpreter, independent of SciPy."""
import argparse
import json
import time
from decimal import Decimal, localcontext
from pathlib import Path
import pandas as pd
import mpmath
from squares.core import certify
from squares._official_metric import _parse_submission_csv

def check(path):
    result = {}
    pandas_text = pd.read_csv(path).to_csv(index=False)
    path.with_suffix(".pandas.csv").write_text(pandas_text)
    for mode, text in (("literal",path.read_text()),("pandas",pandas_text)):
        for n, rows in _parse_submission_csv(text).items():
            try:
                info = dict(ok=True,L=certify(n,rows))
            except ValueError as error:
                info = dict(ok=False,error=str(error))
            result.setdefault(str(n),{})[mode]=info
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--outdir",type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    started=time.monotonic()
    assert pd.__version__=="3.1.0", pd.__version__
    report=json.loads((args.outdir/"results.json").read_text())
    baseline=check(args.outdir/"baseline-targets.csv")
    output=check(args.outdir/"output.csv")
    for n,info in report["summary"].items():
        info["baseline"]=baseline[n]
        info["output"]=output[n]
        comparisons={}
        for mode in ("literal","pandas"):
            b,o=baseline[n][mode],output[n][mode]
            if b["ok"] and o["ok"]:
                with localcontext() as ctx:
                    ctx.prec=80
                    delta=Decimal(o["L"])-Decimal(b["L"])
                comparisons[mode]=dict(delta_L=str(delta),strict_win=delta<0)
        info["comparison"]=comparisons
        info["verified_win"]=len(comparisons)==2 and all(r["strict_win"] for r in comparisons.values())
    report["search_pandas_version"]=report["pandas_version"]
    report["pandas_version"]=pd.__version__
    report["mpmath_version"]=mpmath.__version__
    report["reference_recertification_seconds"]=time.monotonic()-started
    report["reference_certification"]=True
    (args.outdir/"results.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(pandas=pd.__version__,mpmath=mpmath.__version__,
                         elapsed=report["reference_recertification_seconds"],
                         summary=report["summary"])))
if __name__=="__main__":main()
