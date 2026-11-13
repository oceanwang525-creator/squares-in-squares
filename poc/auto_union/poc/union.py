#!/usr/bin/env python3
"""Certified per-POC exports and fail-closed incumbent union."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, io, json, os, sys, tempfile
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from pathlib import Path
import pandas as pd
import mpmath as mp

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "poc/baseline/poc/src"))
from squares.core import certify, read_submission, score
from squares import _official_metric as metric

BASE = ROOT / "poc/pell_corrugation/data/border-scan/merged.csv"
FIELDS = ["id", "n", "square_id", "x", "y", "angle"]
_spec = importlib.util.spec_from_file_location("final_csv_structure", ROOT / ".github/scripts/check_final_csv.py")
_structure = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_structure)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def csv_text(groups):
    out = io.StringIO(newline="")
    w = csv.DictWriter(out, fieldnames=FIELDS, lineterminator="\n")
    w.writeheader()
    i = 0
    for n, rows in sorted(groups.items()):
        for row in sorted(rows, key=lambda r: int(r["square_id"])):
            w.writerow(dict(id=i, n=n, square_id=row["square_id"],
                            x=row["x"], y=row["y"], angle=row["angle"]))
            i += 1
    return out.getvalue()

def strip_prefix(value):
    return value[1:] if value.startswith("s") else value

def final_text(groups):
    """Canonical public final.csv format; prefix decimals to survive pandas."""
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["id", "x", "y", "deg"])
    with mp.workdps(95):
        for n, rows in sorted(groups.items()):
            for row in sorted(rows, key=lambda r: int(r["square_id"])):
                deg = mp.nstr(mp.mpf(row["angle"]) * 180 / mp.pi, 75)
                writer.writerow([f"{n:03}_{row['square_id']}",
                                 "s" + row["x"], "s" + row["y"], "s" + deg])
    return out.getvalue()

def parse_text(text):
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != ["id", "x", "y", "deg"]:
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("Expected id,x,y,deg or legacy incumbent schema")
        return metric._parse_submission_csv(text)
    _structure.validate(text)
    groups = {}
    with mp.workdps(95):
        for row in reader:
            if None in row or any(row[k] is None for k in ("id", "x", "y", "deg")):
                raise ValueError("Malformed CSV")
            parts = row["id"].split("_")
            if len(parts) != 2 or not all(p.isdigit() for p in parts):
                raise ValueError("Malformed square id")
            n, sid = map(int, parts)
            if not 1 <= n <= 1000:
                raise ValueError("n outside 1..1000")
            values = {k: strip_prefix(row[k]) for k in ("x", "y", "deg")}
            if not all(mp.isfinite(mp.mpf(v)) for v in values.values()):
                raise ValueError("Nonfinite coordinate")
            angle = mp.nstr(mp.mpf(values["deg"]) * mp.pi / 180, 75)
            groups.setdefault(n, []).append(dict(square_id=sid, x=values["x"],
                                                 y=values["y"], angle=angle))
    # A header-only final explicitly records no exported candidate groups.
    for n, rows in groups.items():
        error = metric._validate_puzzle(n, rows)
        if error:
            raise ValueError(error)
    return groups

def load(path):
    text = path.read_text()
    # Transport the original public CSV before stripping prefixes or converting units.
    transported = pd.read_csv(io.StringIO(text)).to_csv(index=False)
    return parse_text(text), parse_text(transported)

def check(task):
    n, rows, transported = task
    try:
        if len(rows) != n or len(transported) != n:
            raise ValueError("Incomplete group")
        a, b = Decimal(certify(n, rows)), Decimal(certify(n, transported))
        return str(a), str(b)
    except Exception:
        return None

def qualifies(result, limits):
    """Selection, not admission: replace only for a strict win in BOTH modes."""
    return result is not None and all(Decimal(x) < limit for x, limit in zip(result, limits))

def key(task):
    return digest(json.dumps(task, sort_keys=True).encode())

def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)

def certify_many(tasks, workers, cache):
    pending = {key(t): t for t in tasks if key(t) not in cache}
    print("uncached groups", len(pending), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (k, result) in enumerate(zip(pending, pool.map(check, pending.values(), chunksize=1))):
            cache[k] = result
            if i % 100 == 0: print("certified", i, "/", len(pending), flush=True)
    return cache

def baseline_checks(base, workers, cache):
    a, b = load(base)
    if set(a) != set(range(1,1001)) or set(b) != set(a):
        raise ValueError("Incumbent must contain exactly all 1000 groups")
    tasks = [(n, a[n], b[n]) for n in sorted(a)]
    certify_many(tasks, workers, cache)
    sides = {}
    for t in tasks:
        result = cache[key(t)]
        if result is None: raise ValueError("Invalid incumbent")
        sides[t[0]] = tuple(map(Decimal, result))
    return a, sides

def full_gate(path):
    reports = []
    for puzzles in load(path):
        r = score(puzzles)
        if r["accepted"] != 1000 or r["missing"] or r["rejected"]:
            raise ValueError("Full official certification failed")
        reports.append(r)
    return reports

def export(base, workers, audit):
    initial = base.read_bytes()
    cache_path = audit / "cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    incumbent, sides = baseline_checks(base, workers, cache)
    candidates, tasks, skipped = {}, [], {}
    folders = sorted(p for p in (ROOT / "poc").iterdir() if p.is_dir())
    for folder in folders:
        entries = []
        for path in sorted(folder.rglob("*.csv")):
            if path.name == "final.csv": continue
            try:
                a, b = load(path)
            except Exception as e:
                skipped[str(path.relative_to(ROOT))] = str(e); continue
            for n, rows in a.items():
                if n not in sides or n not in b: continue
                t = (n, rows, b[n])
                entries.append((t, str(path.relative_to(ROOT))))
                tasks.append(t)
        candidates[folder] = entries
    print("candidate files/groups", len(tasks), flush=True)
    certify_many(tasks, workers, cache)
    atomic(cache_path, json.dumps(cache).encode())
    results = {}
    for folder, entries in candidates.items():
        best, values, sources = {}, {}, {}
        for t, source in entries:
            result = cache[key(t)]
            if result is None: continue
            n, rows, _ = t
            a, b = map(Decimal, result)
            if n not in values or a < values[n]:
                best[n], values[n], sources[n] = rows, a, source
        final = folder / "final.csv"
        if best:
            data = final_text(best).encode()
            # Reparse the actual exported file before publishing any group.
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "final.csv"; p.write_bytes(data)
                literal, transported = load(p)
                actual_tasks = [(n, literal[n], transported[n]) for n in best]
                certify_many(actual_tasks, workers, cache)
                excluded = []
                for t in actual_tasks:
                    n = t[0]
                    if cache[key(t)] is None:
                        excluded.append(n)
                for n in excluded:
                    del best[n]
                    del sources[n]
                atomic(cache_path, json.dumps(cache).encode())
                # Subsetting independent groups does not change their exported bytes.
                data = final_text(best).encode()
            if base.read_bytes() != initial: raise ValueError("Incumbent changed during export")
            atomic(final, data)
            results[folder.name] = dict(count=len(best), bytes=len(data),
                ns=sorted(best), sources=sources, transport_excluded=excluded, sha256=digest(data))
        else:
            # --export regenerates valid candidate finals, including nonwinning groups.
            data = final_text({}).encode()
            if base.read_bytes() != initial: raise ValueError("Incumbent changed during export")
            atomic(final, data)
            results[folder.name] = dict(count=0, bytes=len(data), ns=[], sources={}, sha256=digest(data))
        print(folder.name, results[folder.name]["count"], results[folder.name]["bytes"], flush=True)
    if base.read_bytes() != initial: raise ValueError("Incumbent changed")
    atomic(audit / "exports.json", json.dumps(dict(baseline_sha256=digest(initial),
        pandas_version=pd.__version__, folders=results, skipped=skipped), indent=2).encode())

def union(base, workers, audit):
    initial = base.read_bytes()
    cache = {}
    best, sides = baseline_checks(base, workers, cache)
    original_sides = dict(sides)
    finals = []
    tasks = []
    for path in sorted((ROOT / "poc").glob("*/final.csv")):
        a, b = load(path)
        finals.append((path, a, b))
        tasks.extend((n, rows, b.get(n, [])) for n, rows in a.items())
    certify_many(tasks, workers, cache)
    for path, a, b in finals:
        for n, rows in a.items():
            if n not in original_sides or n not in b: raise ValueError("Invalid final group")
            result = cache[key((n, rows, b[n]))]
            if result is None: raise ValueError("Uncertified final: " + str(path))
            pair = tuple(map(Decimal, result))
            if pair[0] < sides[n][0] and pair[1] < sides[n][1]:
                best[n], sides[n] = rows, pair
    improved = [n for n in sides if sides[n][0] < original_sides[n][0]]
    with tempfile.TemporaryDirectory() as td:
        candidate = Path(td) / "union.csv"
        candidate.write_bytes(csv_text(best).encode() if improved else initial)
        reports = full_gate(candidate)
        for mode, report in enumerate(reports):
            for row in report["results"]:
                if Decimal(row["L"]) > original_sides[row["n"]][mode]:
                    raise ValueError("Per-n regression after full transport")
        if base.read_bytes() != initial: raise ValueError("Incumbent changed during certification")
        if improved: atomic(base, candidate.read_bytes())
    atomic(audit / "union.json", json.dumps(dict(improved=improved,
        before_sha256=digest(initial), after_sha256=digest(base.read_bytes()),
        reports=reports), indent=2).encode())
    print("UNION COMPLETE", len(improved), reports[0]["score"], digest(base.read_bytes()), flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--baseline", type=Path, default=BASE)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--audit", type=Path, default=ROOT/"poc/auto_union/docs")
    args=parser.parse_args()
    args.audit.mkdir(parents=True, exist_ok=True)
    if args.export: export(args.baseline, args.workers, args.audit)
    else: union(args.baseline, args.workers, args.audit)

if __name__ == "__main__":
    main()
