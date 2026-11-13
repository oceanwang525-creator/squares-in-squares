"""PR gate using the team's own auto_union functions (never writes merged.csv).

  pr_gate.py baseline                      -> certify all 1000 incumbent groups (literal + pandas); save sides
  pr_gate.py finals final1.csv [final2..]  -> certify every group of the given finals, list strict wins
  pr_gate.py union out_dir final1.csv ...  -> build the union candidate from those finals and run the full gate
"""
import json, os, sys, time
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]          # repository root
sys.path.insert(0, str(ROOT / "poc/auto_union/poc"))
import union as u   # noqa: E402

GATE = Path(os.environ.get("GATE_DIR", "gate"))      # scratch directory for the certification cache
WORKERS = 8


def load_cache():
    p = GATE / "cache.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_cache(cache):
    GATE.mkdir(exist_ok=True)
    u.atomic(GATE / "cache.json", json.dumps(cache).encode())


def baseline(cache):
    t = time.time()
    incumbent, sides = u.baseline_checks(u.BASE, WORKERS, cache)
    save_cache(cache)
    out = dict(sha256=u.digest(u.BASE.read_bytes()),
               sides={str(n): [str(a), str(b)] for n, (a, b) in sides.items()})
    u.atomic(GATE / "sides.json", json.dumps(out).encode())
    score = lambda mode: float(sum(sides[n][mode] / Decimal(n).sqrt() - 1 for n in sides))
    print(f"baseline certified: 1000 groups in {time.time()-t:.0f}s; score literal {score(0):.15f} pandas {score(1):.15f}; sha256 {out['sha256'][:16]}", flush=True)
    return incumbent, sides


def finals(paths, cache, sides):
    res = {}
    for p in paths:
        a, b = u.load(Path(p))
        tasks = [(n, rows, b[n]) for n, rows in sorted(a.items())]
        u.certify_many(tasks, WORKERS, cache)
        save_cache(cache)
        wins = {}; bad = []; notwin = []
        for t in tasks:
            r = cache[u.key(t)]
            if r is None:
                bad.append(t[0]); continue
            if u.qualifies(r, sides[t[0]]):
                wins[t[0]] = (r, a[t[0]])
            else:
                notwin.append(t[0])
        res[p] = dict(wins=wins, invalid=bad, notwin=notwin)
        print(f"{p}: {len(tasks)} groups; strict wins {len(wins)}; valid but not winning {len(notwin)}; INVALID {bad}", flush=True)
    return res


if __name__ == "__main__":
    mode = sys.argv[1]
    cache = load_cache()
    incumbent, sides = baseline(cache)
    if mode == "baseline":
        sys.exit(0)
    if mode == "finals":
        finals(sys.argv[2:], cache, sides)
        sys.exit(0)
    if mode == "union":
        out_dir = Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)
        res = finals(sys.argv[3:], cache, sides)
        best = dict(incumbent); new_sides = dict(sides); src = {}
        for p, r in res.items():
            if r["invalid"]:
                raise SystemExit(f"invalid groups in {p}: {r['invalid']}")
            for n, (pair, rows) in r["wins"].items():
                pair = tuple(map(Decimal, pair))
                if pair[0] < new_sides[n][0] and pair[1] < new_sides[n][1]:
                    best[n], new_sides[n], src[n] = rows, pair, p
        improved = sorted(src)
        cand = out_dir / "union.csv"
        cand.write_bytes(u.csv_text(best).encode())
        t = time.time()
        reports = u.full_gate(cand)
        regress = []
        for m, rep in enumerate(reports):
            for row in rep["results"]:
                if Decimal(row["L"]) > sides[row["n"]][m]:
                    regress.append((m, row["n"]))
        strict = [n for n in improved if all(Decimal(next(r["L"] for r in rep["results"] if r["n"] == n)) < sides[n][m] for m, rep in enumerate(reports))]
        base_score = [float(sum(sides[n][m] / Decimal(n).sqrt() - 1 for n in sides)) for m in (0, 1)]
        summary = dict(incumbent_sha256=u.digest(u.BASE.read_bytes()), incumbent_score_literal=base_score[0], incumbent_score_pandas=base_score[1],
                       union_score_literal=reports[0]["score"], union_score_pandas=reports[1]["score"],
                       accepted=[r["accepted"] for r in reports], missing=[r["missing"] for r in reports], rejected=[r["rejected"] for r in reports],
                       improved=improved, strict_in_full_gate=strict, regressions=regress, sources=src,
                       per_n={str(n): dict(incumbent_literal=str(sides[n][0]), incumbent_pandas=str(sides[n][1]),
                                           new_literal=next(r["L"] for r in reports[0]["results"] if r["n"] == n),
                                           new_pandas=next(r["L"] for r in reports[1]["results"] if r["n"] == n)) for n in improved})
        u.atomic(out_dir / "gate.json", json.dumps(summary, indent=1).encode())
        print(f"FULL GATE in {time.time()-t:.0f}s: literal {reports[0]['score']!r} pandas {reports[1]['score']!r}; accepted {summary['accepted']} missing {summary['missing']} rejected {summary['rejected']}")
        print(f"incumbent {base_score[0]!r}; improved n {len(improved)}; strict in full gate {len(strict)}; regressions {regress}")
