#!/usr/bin/env python3
"""Read public notebook CSV outputs; never execute notebook code or submit."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from decimal import Decimal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".github/scripts"))
from certification_runtime import install_runtime, group_key

SPEC = importlib.util.spec_from_file_location("external_union", ROOT / "poc/auto_union/poc/union.py")
u = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(u)
PUBLIC = ["id", "x", "y", "deg"]
LEGACY = ["id", "n", "square_id", "x", "y", "angle"]
REF = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$")
PATTERN = r"(?i)(?:^|/)(?:submission|final|merged)[^/]*\.csv$"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    client = KaggleApi()
    client.authenticate()
    return client


def fetch(ref, destination):
    # Only data files, no pull/run/execute API. SDK also checks output traversal.
    client = api()
    client.kernels_output(ref, str(destination), file_pattern=PATTERN,
                          force=True, quiet=True)
    return [str(p) for p in sorted(destination.rglob("*.csv"))]


def discover(competition, limit):
    rows = api().kernels_list(competition=competition, sort_by="scoreAscending",
                             kernel_type="notebook", page_size=limit) or []
    return [row.to_dict() for row in rows if row is not None]


def child(conn, fn, args):
    # Forked input is data only. Exceptions do not bypass the parent's hard cap.
    try:
        os.setsid()
        conn.send(("ok", fn(*args)))
    except BaseException as error:
        conn.send(("error", type(error).__name__))
    finally:
        conn.close()


def launch(fn, args):
    receive, transmit = mp.Pipe(False)
    process = mp.Process(target=child, args=(transmit, fn, args))
    process.start()
    transmit.close()
    return process, receive, time.monotonic()


def stop(process, pipe):
    if process.is_alive():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            process.kill()
    process.join()
    pipe.close()


def bounded(fn, args, seconds):
    process, pipe, start = launch(fn, args)
    try:
        if not pipe.poll(seconds):
            raise TimeoutError("worker deadline")
        status, result = pipe.recv()
        if status != "ok":
            raise ValueError(result)
        return result
    finally:
        stop(process, pipe)


def read_groups(path, max_bytes):
    """Isolate groups before validation so one incomplete n cannot poison peers."""
    if path.stat().st_size > max_bytes:
        raise ValueError("CSV exceeds --max-csv-mb")
    text = path.read_text(encoding="utf-8-sig")
    literal = list(csv.DictReader(io.StringIO(text)))
    names = list(literal[0]) if literal else []
    if names not in (PUBLIC, LEGACY):
        raise ValueError("unsupported header")
    # Transport original bytes, not a post-parsing re-export.
    transported = list(csv.DictReader(io.StringIO(
        u.pd.read_csv(io.StringIO(text)).to_csv(index=False))))
    buckets = []
    dropped = []
    for rows in (literal, transported):
        groups = collections.defaultdict(list)
        invalid = set()
        for row in rows:
            try:
                n = int(row["id"].split("_")[0]) if names == PUBLIC else int(row["n"])
                if not 1 <= n <= 1000:
                    raise ValueError("n range")
                if None in row or any(value is None for value in row.values()):
                    invalid.add(n)
                    continue
                groups[n].append(row)
            except (ValueError, KeyError, TypeError, AttributeError):
                dropped.append("unidentifiable-row")
        parsed = {}
        for n, raw in sorted(groups.items()):
            if n in invalid:
                dropped.append(n)
                continue
            out = io.StringIO(newline="")
            writer = csv.DictWriter(out, fieldnames=names, lineterminator="\n")
            writer.writeheader()
            writer.writerows(raw)
            try:
                p = u.parse_text(out.getvalue())
                if set(p) != {n} or len(p[n]) != n:
                    raise ValueError("incomplete group")
                parsed[n] = p[n]
            except Exception:
                dropped.append(n)
        buckets.append(parsed)
    complete = set(buckets[0]) & set(buckets[1])
    return [(n, buckets[0][n], buckets[1][n]) for n in sorted(complete)], dropped


def official(n, literal, transported):
    pair = [u.metric._evaluate_puzzle_worker((n, rows))
            for rows in (literal, transported)]
    if not all(result[0] == n and result[3] == "ok" and result[2] is not None
               for result in pair):
        return None
    return pair


def certify(tasks, runtime, seconds, workers, log):
    """Per-pair killable processes, bounded in both time and concurrency."""
    pending = collections.deque(tasks)
    active = {}
    completed = 0
    try:
        while pending or active:
            while pending and len(active) < workers:
                key, n, a, b = pending.popleft()
                keys = [group_key((n, rows)) for rows in (a, b)]
                if all(k in runtime.cache for k in keys):
                    pair = [runtime.cache[k] for k in keys]
                    yield key, pair if all(r[3] == "ok" and r[2] is not None for r in pair) else None
                    continue
                proc, pipe, start = launch(official, (n, a, b))
                active[proc.pid] = (proc, pipe, start, key, n, a, b)
            for pid, (proc, pipe, start, key, n, a, b) in list(active.items()):
                if pipe.poll():
                    try:
                        status, result = pipe.recv()
                    except EOFError:
                        status, result = "error", "worker-exit"
                elif time.monotonic() - start > seconds:
                    status, result = "error", "deadline"
                elif not proc.is_alive():
                    status, result = "error", "worker-exit"
                else:
                    continue
                stop(proc, pipe)
                del active[pid]
                completed += 1
                if status == "ok" and result:
                    for rows, value in zip((a, b), result):
                        runtime.cache[group_key((n, rows))] = value
                log.write(json.dumps(dict(key=key, n=n, status=status,
                                          result=result)) + "\n")
                log.flush()
                if completed % 100 == 0:
                    print("certified", completed, "queued", len(pending) + len(active), flush=True)
                yield key, result if status == "ok" else None
            if active:
                time.sleep(.02)
    finally:
        for proc, pipe, *_ in active.values():
            stop(proc, pipe)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", default="squares-in-squares")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--group-seconds", type=float, default=90)
    parser.add_argument("--io-seconds", type=float, default=180)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-csv-mb", type=int, default=128)
    parser.add_argument("--baseline", type=Path, default=u.BASE)
    parser.add_argument("--output", type=Path, default=ROOT / "poc/external/final.csv")
    parser.add_argument("--work", type=Path, required=True,
                        help="fresh directory outside the checkout, keeps downloads and proofs")
    args = parser.parse_args()
    if not (1 <= args.limit <= 10 and 1 <= args.workers <= 16
            and args.group_seconds > 0 and args.io_seconds > 0 and args.max_csv_mb > 0):
        parser.error("positive limits required; --limit <=10 and --workers <=16")
    work = args.work.resolve()
    if work.exists():
        parser.error("--work must not already exist")
    if work.is_relative_to(ROOT):
        parser.error("--work must be outside checkout")
    work.mkdir(parents=True)
    if args.output.resolve() == args.baseline.resolve():
        parser.error("output cannot overwrite baseline")
    initial = args.baseline.read_bytes()
    baseline = work / "baseline.csv"
    baseline.write_bytes(initial)
    runtime = install_runtime(u)
    runtime.workers, runtime.seconds = args.workers, args.group_seconds
    before = runtime.full_gate(baseline)
    limits = [{x["n"]: Decimal(x["L"]) for x in report["results"]} for report in before]
    listing = bounded(discover, (args.competition, args.limit), args.io_seconds)
    sources, tasks, origins, files = [], {}, {}, {}
    for rank, metadata in enumerate(listing, 1):
        ref = metadata.get("ref", "")
        source = dict(rank=rank, url="https://www.kaggle.com/code/" + ref,
                      metadata=metadata, files=[])
        sources.append(source)
        if not REF.fullmatch(ref):
            source["error"] = "invalid-ref"
            continue
        destination = work / "downloads" / str(rank)
        destination.mkdir(parents=True)
        try:
            paths = bounded(fetch, (ref, destination), args.io_seconds)
        except (TimeoutError, ValueError, EOFError) as error:
            source["error"] = type(error).__name__
            continue
        for name in paths:
            path = Path(name).resolve()
            if not path.is_relative_to(destination.resolve()) or path.is_symlink():
                source["error"] = "unsafe-path"
                continue
            entry = dict(file=str(path.relative_to(destination)), sha256=sha(path))
            source["files"].append(entry)
            try:
                groups, dropped = bounded(read_groups, (path, args.max_csv_mb * 1024**2), args.io_seconds)
            except (TimeoutError, ValueError, EOFError) as error:
                entry["error"] = type(error).__name__
                continue
            entry.update(complete_groups=len(groups), dropped_structure=dropped)
            file_key = str(rank) + "/" + entry["file"]
            files[file_key] = (entry, {n for n, _, _ in groups}, {})
            for n, a, b in groups:
                key = u.key((n, a, b))
                tasks[key] = (key, n, a, b)
                origins.setdefault(key, []).append(file_key)
        print("downloaded rank", rank, ref, "CSV files", len(source["files"]), flush=True)
    # Exact parsed-byte de-duplication, no trust in notebook claims.
    chosen, winners = {}, {}
    with (work / "groups.jsonl").open("w") as log:
        for key, pair in certify(tasks.values(), runtime, args.group_seconds, args.workers, log):
            _, n, a, b = tasks[key]
            for file_key in origins[key]:
                files[file_key][2][n] = pair
            if pair and all(Decimal(pair[i][2]) < limits[i][n] for i in range(2)):
                chosen[n], winners[n] = origins[key][0], a
                for i in range(2):
                    limits[i][n] = Decimal(pair[i][2])
    for entry, ns, results in files.values():
        accepted = {n: pair for n, pair in results.items() if pair}
        entry["certified_groups"] = len(accepted)
        entry["uncertified_groups"] = sorted(ns - set(accepted))
        entry["certified_full_scores"] = ([sum(accepted[n][i][1] for n in range(1,1001))
                                            for i in range(2)]
                                           if set(accepted) == set(range(1,1001)) else None)
    # Certify actual public export after radians/degrees conversion.
    export = work / "export.csv"
    export.write_text(u.final_text(winners))
    exported = u.load(export)
    exported_tasks = [(str(n), n, exported[0][n], exported[1][n]) for n in winners]
    actual, dropped = {}, []
    with (work / "export-groups.jsonl").open("w") as log:
        for key, pair in certify(exported_tasks, runtime, args.group_seconds, args.workers, log):
            n = int(key)
            if pair and all(Decimal(pair[i][2]) < Decimal(before[i]["results"][n-1]["L"])
                            for i in range(2)):
                actual[n] = winners[n]
            else:
                dropped.append(n)
                del chosen[n]
    final_data = u.final_text(actual).encode()
    export.write_bytes(final_data)
    union = u.load(baseline)[0]
    union.update(u.load(export)[0])
    candidate = work / "candidate.csv"
    candidate.write_bytes(u.csv_text(union).encode() if actual else initial)
    after = runtime.full_gate(candidate)
    for a, b in zip(before, after):
        if any(Decimal(y["L"]) > Decimal(x["L"]) for x, y in zip(a["results"], b["results"])):
            raise ValueError("per-n regression")
    if args.baseline.read_bytes() != initial:
        raise ValueError("baseline changed during run")
    proof = dict(baseline_sha256=sha(baseline), candidate_sha256=sha(candidate),
                 final_sha256=hashlib.sha256(final_data).hexdigest(),
                 before_scores=[r["score"] for r in before],
                 after_scores=[r["score"] for r in after],
                 accepted_each_mode=1000, regressions=0, chosen=chosen,
                 dropped_export=dropped, sources=sources)
    (work / "audit.json").write_text(json.dumps(proof, indent=2))
    # Only the final output is written inside checkout, atomically after every gate.
    u.atomic(args.output, final_data)
    print("PASS", json.dumps(dict(wins=len(chosen), scores=proof["after_scores"],
                                 sha256=proof["final_sha256"])), flush=True)


if __name__ == "__main__":
    main()
