"""Certify poc/swarm_sametilt/final.csv vs. origin/main merged frontier (literal + pandas)."""
import subprocess, sys
from decimal import Decimal
from pathlib import Path
from poc.auto_union.poc import union

ref = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
a, b = union.load(Path("poc/swarm_sametilt/final.csv"))
files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", ref], text=True).split()
cands = [f for f in files if f.endswith(("final.csv", "merged.csv")) and "swarm_sametilt" not in f]
best = {}
for f in cands:
    txt = subprocess.check_output(["git", "show", f"{ref}:{f}"], text=True)
    try:
        g = union.parse_text(txt)
    except Exception:
        continue
    for n, rows in g.items():
        if n in a and len(rows) == n:
            r = union.check((n, rows, rows))
            if r and (n not in best or Decimal(r[0]) < best[n][0]):
                best[n] = (Decimal(r[0]), f)
tot = Decimal(0)
for n in sorted(a):
    r = union.check((n, a[n], b[n]))
    lit, pd_ = Decimal(r[0]), Decimal(r[1])
    inc, src = best.get(n, (None, None))
    dL = (lit - inc) if inc is not None else None
    ds = (lit - inc) / Decimal(n).sqrt() if inc is not None else Decimal(0)
    tot += ds
    print(f"n={n} lit==pd:{lit==pd_} L={str(lit)[:24]} inc={str(inc)[:24]} ({src}) dL={float(dL) if dL is not None else None:+.6e} ds={float(ds):+.6e}")
print("TOTAL ds:", float(tot))
