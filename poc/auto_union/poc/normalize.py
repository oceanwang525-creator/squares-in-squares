#!/usr/bin/env python3
"""Audit saved main-tree packings and normalize every final against one incumbent.

Only trusted main data is read. No POC search/import code is executed.
"""
import argparse
import csv
import io
import json
import platform
import re
import subprocess
from pathlib import Path

import numpy as np
import union as u


def saved_groups(path):
    """Decode documented saved-coordinate formats without executing POC code."""
    if path.suffix == ".csv":
        text = path.read_text()
        header = text.splitlines()[0] if text else ""
        if header == "square_index,x,y,theta":
            # slope_band/3 canonical_908, angles explicitly radians in its reader.
            rows = list(csv.DictReader(io.StringIO(text)))
            return {908: [dict(square_id=r["square_index"], x=r["x"],
                              y=r["y"], angle=r["theta"]) for r in rows]}
        if path.as_posix().endswith("slope_band/3/data/output.csv"):
            # Historical headerless six-column output; do not drop its first square.
            return u.parse_text(",".join(u.FIELDS) + "\n" + text)
        return None  # Native CSVs go through load() on their original bytes.
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        if (isinstance(data, list) and data
                and all(isinstance(r, dict) and
                        {"square_id", "x", "y", "angle"} <= r.keys() for r in data)):
            return {len(data): [{k: str(r[k]) for k in ("square_id", "x", "y", "angle")}
                                for r in data]}
        raise ValueError("report/metadata, not a saved coordinate-row list")
    if path.suffix in (".npy", ".npz"):
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as blob:
                if set(blob.files) != {"q", "meta"}:
                    raise ValueError("unsupported saved array schema")
                q = blob["q"]
                meta = json.loads(str(blob["meta"]))
                if not isinstance(meta, dict):
                    raise ValueError("expected metadata object")
        else:
            q = np.load(path, allow_pickle=False)
        if q.ndim != 2 or q.shape[1] not in (3, 4) or not np.isfinite(q).all():
            raise ValueError("not a finite Nx3 packing (optional fourth container L)")
        name = path.as_posix()
        match = re.search(r"(?:best_|slp_|q)(?:n)?(\d+)", path.stem)
        n = int(match.group(1)) if match else len(q)
        if "/k2p10_fill2/" in name:
            n = 635  # a_relax/b_fill add two squares to the 633-square core.
        if "/multi_angle_band/" in name:
            q = q[:n]  # Existing certify.py explicitly certifies this prefix.
        if len(q) != n or not 1 <= n <= 1000:
            raise ValueError(f"incomplete saved packing: {len(q)}/{n}")
        return {n: [dict(square_id=str(i), x=repr(float(r[0])),
                        y=repr(float(r[1])), angle=repr(float(r[2])))
                    for i, r in enumerate(q)]}
    raise ValueError("not a supported complete packing format")


def candidate(path):
    groups = saved_groups(path)
    if groups is None:
        return u.load(path)
    # Transport the decimal-preserving legacy CSV, not Python float objects.
    text = u.csv_text(groups)
    return u.parse_text(text), u.parse_text(u.pd.read_csv(io.StringIO(text)).to_csv(index=False))


def keep_existing(old_result, new_result, limits):
    """Keep a valid winning final unless its replacement strictly beats both modes."""
    return (u.qualifies(old_result, limits)
            and not u.qualifies(new_result, tuple(map(u.Decimal, old_result))))


def preserve_final_groups(existing, proposed, sides, workers, cache):
    """Keep original public row groups byte-for-byte on ties or mode tradeoffs."""
    if existing is None:
        return proposed, []
    texts = [existing.decode(), proposed.decode()]
    tasks, public = [], []
    for text in texts:
        u._structure.validate(text)
        a = u.parse_text(text)
        b = u.parse_text(u.pd.read_csv(io.StringIO(text)).to_csv(index=False))
        current = {n: (n, rows, b[n]) for n, rows in a.items()}
        tasks.extend(current.values())
        public.append((current, {n: [] for n in a}))
        # Canonical schema forbids quoted/newline coordinates, so preserve lines.
        for line in text.splitlines(keepends=True)[1:]:
            n = int(next(csv.reader([line]))[0].split("_")[0])
            public[-1][1][n].append(line.rstrip("\r\n") + "\n")
    u.certify_many(tasks, workers, cache)
    old_tasks, old_lines = public[0]
    new_tasks, new_lines = public[1]
    kept = []
    for n, task in old_tasks.items():
        new = cache[u.key(new_tasks[n])] if n in new_tasks else None
        if keep_existing(cache[u.key(task)], new, sides[n]):
            new_lines[n] = old_lines[n]
            kept.append(n)
    return ("id,x,y,deg\n" + "".join(
        line for n in sorted(new_lines) for line in new_lines[n])).encode(), kept


def normalize(audit, workers, previous):
    audit.mkdir(parents=True, exist_ok=True)
    initial = u.BASE.read_bytes()
    expected = subprocess.check_output(["git", "show", "HEAD:poc/pell_corrugation/data/border-scan/merged.csv"], cwd=u.ROOT)
    expected = u.digest(expected)
    assert u.digest(initial) == expected, "refresh/re-evaluate the baseline before continuing"
    assert u.pd.__version__ == "3.1.0"
    cache = {}
    cache_proof = {}
    if previous:
        prior = json.loads((previous / "normalization.json").read_text())
        assert prior["versions"] == dict(pandas=u.pd.__version__,
            mpmath=u.mp.__version__, numpy=np.__version__, python=platform.python_version())
        # Geometry is baseline-independent. Reuse only exact row-content tasks,
        # with identical checker code and versions. Compare against fresh sides.
        for name in ("core.py", "_official_metric.py"):
            rel = Path("poc/baseline/poc/src/squares") / name
            digest = u.digest((u.ROOT / rel).read_bytes())
            assert prior["cache_reuse"][str(rel)] == digest
            cache_proof[str(rel)] = digest
        cache = json.loads((previous / "cache.json").read_text())
        cache_proof.update(entries=len(cache), pandas=u.pd.__version__,
            mpmath=u.mp.__version__, numpy=np.__version__,
            prior_baseline_sha256=prior["baseline_before_sha256"],
            baseline_sha256=u.digest(initial),
            key="sha256 JSON exact n,literal rows,actual transported rows")
    for name in ("core.py", "_official_metric.py"):
        rel = Path("poc/baseline/poc/src/squares") / name
        cache_proof[str(rel)] = u.digest((u.ROOT / rel).read_bytes())
    incumbent, sides = u.baseline_checks(u.BASE, workers, cache)
    inventory, entries, tasks, before = [], {}, [], {}
    folders = sorted(p for p in (u.ROOT / "poc").iterdir() if p.is_dir())
    for folder in folders:
        final = folder / "final.csv"
        before[folder.name] = final.read_bytes() if final.exists() else None
        entries[folder] = []
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix not in (".csv", ".json", ".jsonl", ".npy", ".npz"):
                continue
            rel = str(path.relative_to(u.ROOT))
            record = dict(path=rel, sha256=u.digest(path.read_bytes()), bytes=path.stat().st_size)
            try:
                a, b = candidate(path)
                file_tasks = [(n, rows, b.get(n, [])) for n, rows in sorted(a.items())]
                record.update(status="candidate", groups=len(file_tasks), ns=sorted(a))
                entries[folder].extend((t, rel) for t in file_tasks)
                tasks.extend(file_tasks)
            except Exception as error:
                record.update(status="skipped", reason=str(error))
            inventory.append(record)
    u.atomic(audit / "inventory.json", json.dumps(inventory, indent=2).encode())
    print("DISCOVERY", len(folders), "folders", len(inventory), "files", len(tasks), "groups", flush=True)
    u.certify_many(tasks, workers, cache)
    u.atomic(audit / "cache.json", json.dumps(cache).encode())
    results = {}
    for folder, found in entries.items():
        best, values, sources = {}, {}, {}
        for task, source in found:
            n = task[0]
            result = cache[u.key(task)]
            if n in sides and u.qualifies(result, sides[n]):
                pair = tuple(map(u.Decimal, result))
                if n not in values or pair < values[n]:
                    best[n], values[n], sources[n] = task[1], pair, source
        # Conversion to public degrees can change the packing. Certify exactly
        # what will be written, in both literal and actual pandas modes.
        data = u.final_text(best).encode()
        final = folder / "final.csv"
        scratch = audit / "pending-final.csv"
        u.atomic(scratch, data)
        a, b = u.load(scratch)
        exported = [(n, rows, b[n]) for n, rows in sorted(a.items())]
        u.certify_many(exported, workers, cache)
        excluded = [t[0] for t in exported if not u.qualifies(cache[u.key(t)], sides[t[0]])]
        for n in excluded:
            del best[n]
            del sources[n]
        data = u.final_text(best).encode()
        data, preserved = preserve_final_groups(before[folder.name], data, sides, workers, cache)
        actual = u.parse_text(data.decode())
        for n in preserved:
            sources[n] = str(final.relative_to(u.ROOT))
        assert u.BASE.read_bytes() == initial
        u.atomic(final, data)
        results[folder.name] = dict(count=len(actual), ns=sorted(actual), sources=sources,
            preserved_existing_groups=preserved,
            conversion_excluded=excluded, sha256=u.digest(data),
            existed=before[folder.name] is not None,
            changed=before[folder.name] != data)
        print("FINAL", folder.name, len(best), flush=True)
    (audit / "pending-final.csv").unlink(missing_ok=True)
    u.atomic(audit / "cache.json", json.dumps(cache).encode())
    # Independent final sweep of the exact bytes, complete groups and strictness.
    final_tasks = []
    for folder in folders:
        path = folder / "final.csv"
        u._structure.validate(path.read_text())
        a, b = u.load(path)
        final_tasks.extend((n, rows, b[n]) for n, rows in a.items())
    u.certify_many(final_tasks, workers, cache)
    assert all(u.qualifies(cache[u.key(t)], sides[t[0]]) for t in final_tasks)
    assert u.BASE.read_bytes() == initial
    counts = dict(folders=len(folders),
        backfilled=sum(v is None for v in before.values()),
        trimmed=sum(v["existed"] and v["changed"] for v in results.values()),
        nonempty=sum(v["count"] > 0 for v in results.values()),
        header_only=sum(v["count"] == 0 for v in results.values()),
        winning_groups=sum(v["count"] for v in results.values()),
        candidate_paths=sum(v["status"] == "candidate" for v in inventory),
        audited_paths=len(inventory), candidate_groups=len(tasks),
        unique_candidate_groups=len({u.key(t) for t in tasks}))
    proof = dict(baseline_before_sha256=u.digest(initial), baseline_after_sha256=u.digest(u.BASE.read_bytes()),
        source_main=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=u.ROOT, text=True).strip(),
        versions=dict(pandas=u.pd.__version__, mpmath=u.mp.__version__, numpy=np.__version__, python=platform.python_version()),
        cache_reuse=cache_proof, counts=counts, folders=results,
        baseline_certified_groups=1000, baseline_modes=["official literal", "actual pandas 3.1.0 read_csv/to_csv"],
        final_strict_dual_gate=True, no_incumbent_write=True)
    u.atomic(audit / "normalization.json", json.dumps(proof, indent=2).encode())
    print("NORMALIZATION COMPLETE", json.dumps(counts), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    normalize(args.audit, args.workers, args.previous)
