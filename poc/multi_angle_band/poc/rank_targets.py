"""Re-rank high-n gaps against current merged.csv, in memory (does not write research/ files)."""
import importlib.util, json, sys, pathlib
root = pathlib.Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("r", root/"poc/research/poc/rank_high_n_gaps.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
items, meta = m.audit_and_rank_high_n()
out = [{k: it[k] for k in ("n","k","j","L","excess_gap","incumbent_construction") if k in it} for it in items[:25]]
print(list(items[0].keys()))
for it in items[:25]:
    print(it["rank"] if "rank" in it else "", it["n"], it["k"], it["j"], round(it["L"],6), round(it["excess_gap"],6), it["incumbent_construction"])
json.dump(items[:25], open(root/"poc/multi_angle_band/docs/top25_ranking.json","w"), indent=1, default=str)
