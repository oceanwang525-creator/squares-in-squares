"""Write the two POC finals in the team's format (id,x,y,deg; s-prefixed; 75-digit degrees) with the
team's own converter.

  public_watch_nov13/final.csv : public rows from the watcher that still beat the incumbent
  pocket_squeeze/final.csv     : rows built by the pocket composer that beat incumbent and watcher rows
"""
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]          # repository root
WORK = Path(os.environ.get("WORK_DIR", "."))         # watcher working directory (state/, gate/, compose_cand*.json)
sys.path.insert(0, str(ROOT / "poc/auto_union/poc"))
import union as u   # noqa: E402

# groups that become invalid after the degrees conversion (found by pr_gate.py finals) are left out
EXCLUDE = {int(v) for v in os.environ.get("EXCLUDE", "").split(",") if v}
EPS = 2e-12      # uniform dilation of composed rows so touching squares keep a positive gap


def composed_rows(P):
    return [dict(square_id=str(i), x=repr(float(x) * (1 + EPS)), y=repr(float(y) * (1 + EPS)), angle=repr(float(t)))
            for i, (x, y, t) in enumerate(P)]


if __name__ == "__main__":
    sides = {int(k): tuple(map(float, v)) for k, v in json.loads((WORK / "gate/sides.json").read_text())["sides"].items()}
    best = json.loads((WORK / "state/best.json").read_text())
    watch = {}
    for k, v in best.items():
        n = int(k)
        if n not in EXCLUDE and float(v["L"]) < min(sides[n]):          # prefilter; the gate decides
            watch[n] = [dict(square_id=r["square_id"], x=r["x"].strip(), y=r["y"].strip(), angle=r["angle"].strip()) for r in v["rows"]]
    out1 = ROOT / "poc/public_watch_nov13/final.csv"
    out1.parent.mkdir(parents=True, exist_ok=True)
    out1.write_text(u.final_text(watch))
    print("public_watch_nov13/final.csv:", len(watch), "groups", out1.stat().st_size, "bytes")
    comp = {}
    for f in ("compose_cand2.json", "compose_cand.json"):
        p = WORK / f
        if not p.exists():
            continue
        for k, v in json.loads(p.read_text()).items():
            n = int(k)
            if n in comp:
                continue
            L = v["K"] + v["f"]
            ref = min(min(sides[n]), float(best[k]["L"]) if k in best else 9e9)
            if L * (1 + EPS) < ref - 1e-12:
                comp[n] = composed_rows(v["P"])
    out2 = ROOT / "poc/pocket_squeeze/final.csv"
    out2.parent.mkdir(parents=True, exist_ok=True)
    out2.write_text(u.final_text(comp))
    print("pocket_squeeze/final.csv:", len(comp), "groups", out2.stat().st_size, "bytes", sorted(comp))
