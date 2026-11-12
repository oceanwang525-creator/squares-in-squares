#!/usr/bin/env python3
"""Build and certify the n=200..599 contribution against the tracked incumbent.

usage: finalize.py INCUMBENT_FULL.csv WORKING_FULL.csv OUT_FULL.csv [JOBS]

1. Every group of WORKING_FULL with 200 <= n <= 599 whose strings differ from
   the incumbent is gated per puzzle: kept only if the official L is strictly
   lower in BOTH transports (literal strings; pandas float round trip).
   Groups outside 200..599 are never taken from WORKING_FULL.
2. The union is written to OUT_FULL (not committed: ~27 MB) and scored whole
   with the official per-puzzle evaluator in both transports.  The incumbent is
   scored the same way in the same run.
3. Hard failures: any transport not 1000/1000, any n worse than the incumbent
   in either transport, any changed n outside 200..599.

Writes next to this POC:
  data/changed-rows.csv      winning groups only (n, square_id, x, y, angle)
  data/summary.json          scores, per-n old/new L, hashes
  docs/literal.json, docs/pandas.json          per-n official reports of the union
  docs/incumbent-literal.json, docs/incumbent-pandas.json   same for the incumbent
"""
import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path

POC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rows as R  # noqa: E402
import score_full as SF  # noqa: E402

LO, HI = 200, 599


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sides(rep):
    return {r["n"]: Decimal(r["L"]) for r in rep["results"] if r["L"] is not None}


def slim(rep):
    return dict(score=rep["score"], accepted=rep["accepted"], missing=rep["missing"], rejected=rep["rejected"],
                results=[dict(n=r["n"], L=r["L"], reason=r["reason"]) for r in rep["results"]])


def main():
    inc_path, work_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    jobs = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    inc = R.read_groups(inc_path)
    work = R.read_groups(work_path, set(range(LO, HI + 1)))
    changed = {}
    for n in range(LO, HI + 1):
        if work[n] == inc[n]:
            continue
        old, new = R.both(n, inc[n]), R.both(n, work[n])
        if new is not None and new[0] < old[0] and new[1] < old[1]:
            changed[n] = work[n]
    union = dict(inc)
    union.update(changed)
    R.write_full(out_path, union)
    lit, pan = SF.score_both(Path(out_path), jobs)
    ilit, ipan = SF.score_both(Path(inc_path), jobs)
    for tag, rep in (("union literal", lit), ("union pandas", pan), ("incumbent literal", ilit), ("incumbent pandas", ipan)):
        print(f"{tag}: score={rep['score']!r} accepted={rep['accepted']} missing={rep['missing']} rejected={rep['rejected']}")
        if (rep["accepted"], rep["missing"], rep["rejected"]) != (1000, 0, 0):
            raise SystemExit(f"{tag}: not 1000/1000")
    L, P, IL, IP = sides(lit), sides(pan), sides(ilit), sides(ipan)
    wins = []
    for n in range(1, 1001):
        if L[n] > IL[n] or P[n] > IP[n]:
            raise SystemExit(f"regression at n={n}")
        if L[n] < IL[n] or P[n] < IP[n]:
            if not (LO <= n <= HI and L[n] < IL[n] and P[n] < IP[n] and n in changed):
                raise SystemExit(f"unexpected change at n={n}")
            wins.append(n)
    if sorted(changed) != wins:
        raise SystemExit("changed set and strict-win set differ")
    (POC / "data").mkdir(exist_ok=True)
    (POC / "docs").mkdir(exist_ok=True)
    R.write_rows(POC / "data" / "changed-rows.csv", changed)
    summary = dict(
        incumbent=dict(literal_score=ilit["score"], pandas_score=ipan["score"], sha256=sha(inc_path)),
        union=dict(literal_score=lit["score"], pandas_score=pan["score"], sha256=sha(out_path)),
        accepted=dict(literal=lit["accepted"], pandas=pan["accepted"]),
        strict_wins=len(wins), regressions=0, range=[LO, HI],
        score_gain_literal=ilit["score"] - lit["score"],
        wins=[dict(n=n, old_literal=str(IL[n]), new_literal=str(L[n]), old_pandas=str(IP[n]), new_pandas=str(P[n]),
                   delta_L=float(L[n] - IL[n])) for n in wins],
        changed_rows_sha256=sha(POC / "data" / "changed-rows.csv"),
    )
    (POC / "data" / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for name, rep in (("literal", lit), ("pandas", pan), ("incumbent-literal", ilit), ("incumbent-pandas", ipan)):
        (POC / "docs" / f"{name}.json").write_text(json.dumps(slim(rep), indent=0) + "\n")
    print(f"strict wins: {len(wins)}; regressions: 0; gain {summary['score_gain_literal']:.3e}")


if __name__ == "__main__":
    main()
