#!/usr/bin/env python3
"""Per-n comparison of incumbent L vs leaderboard best; writes docs/per_n_comparison.csv."""
import math
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[3]
D = R/"poc/leader_curve_analysis"
def our_L():
    df = pd.read_csv(R/"poc/pell_corrugation/data/border-scan/merged.csv")
    out = {}
    for n, g in df.groupby("n"):
        c = np.abs(np.cos(g.angle.values)); s = np.abs(np.sin(g.angle.values)); h = .5*(c+s)
        out[n] = max((g.x+h).max()-(g.x-h).min(), (g.y+h).max()-(g.y-h).min())
    return pd.Series(out, name="L_ours")
def build():
    lb = pd.read_csv(D/"data/perpuzzle_best.csv")
    lb["L"] = lb.L.astype(float); lb["team"] = lb.team.astype(str).str.replace(r"[^\x00-\x7f]", "", regex=True).str.strip()
    lb = lb.set_index("n")
    o = our_L()
    t = pd.DataFrame({"L_ours": o}).join(lb[["L","team"]].rename(columns={"L":"L_lb"}))
    t["n"] = t.index
    t["K"] = np.sqrt(t.n).astype(int); t["K"] = [math.isqrt(n) for n in t.n]
    t["D"] = t.n - t.K**2
    t["gap"] = t.L_ours - t.L_lb
    t["ours_mK"] = t.L_ours - t.K; t["lb_mK"] = t.L_lb - t.K
    t["beats"] = t.gap > 1e-9
    return t
if __name__ == "__main__":
    t = build(); t.to_csv(D/"docs/per_n_comparison.csv", index=False)
    print(len(t), t.beats.sum()); print(t.groupby("team").agg(cnt=("beats","size"), beats=("beats","sum"), gap=("gap",lambda x: x[x>1e-9].sum())))
    # score check
    for name, f in [("L",lambda t:t.L_ours),("L^2/n",lambda t:t.L_ours**2/t.n),("L/sqrt n",lambda t:t.L_ours/np.sqrt(t.n))]:
        print(name, f(t).sum())
