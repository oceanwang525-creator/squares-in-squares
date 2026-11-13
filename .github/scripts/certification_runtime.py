"""Invocation-local official-result reuse and bounded, observable Linux workers.

Only exact parsed group bytes share results. Nothing is loaded from disk caches.
The official geometry, precision escalation and per-puzzle score are unchanged.
"""
import hashlib
import json
import os
import resource
import csv
import io
from decimal import Decimal
from certification_cache import context, restore, save
import signal
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED


class CertificationDeadline(BaseException):
    """Must escape the official worker's broad Exception handler."""


def group_key(task):
    return hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()


def official_job(task, seconds):
    from squares import _official_metric as metric
    start = time.monotonic()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    precision = {}
    original = metric.working_dps
    def observed(dps):
        precision[dps] = precision.get(dps, 0) + 1
        return original(dps)
    def expired(signum, frame):
        raise CertificationDeadline()
    previous = signal.signal(signal.SIGALRM, expired)
    metric.working_dps = observed
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        result = metric._evaluate_puzzle_worker(task)
    except CertificationDeadline:
        raise TimeoutError(f"Official certification n={task[0]} exceeded {seconds}s")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        metric.working_dps = original
    after = resource.getrusage(resource.RUSAGE_SELF)
    return result, dict(n=task[0], seconds=round(time.monotonic()-start, 4),
        cpu_seconds=round(after.ru_utime+after.ru_stime-usage.ru_utime-usage.ru_stime, 4),
        peak_rss_kib=after.ru_maxrss, precision_contexts=precision)


class Runtime:
    def __init__(self, union, workers=None, seconds=180):
        self.union = union
        self.workers = workers or min(os.cpu_count() or 2, 8)
        self.seconds = seconds
        self.cache = {}
        self.metrics = []
        self.original_full_gate = union.full_gate
        self.context = context()
        self.cache = {} if os.environ.get("CERT_FULL_AUDIT") == "true" else restore(self.context)

    def evaluate(self, tasks, label):
        unique = {group_key(t): t for t in tasks}
        pending = {k: t for k, t in unique.items() if k not in self.cache}
        start = time.monotonic()
        print("CERT stage", label, "groups", len(tasks), "unique", len(unique),
              "cached", len(unique)-len(pending), "pending", len(pending),
              "workers", self.workers, "per_group_seconds", self.seconds, flush=True)
        if not pending:
            return
        # Longest groups first; completion order does not affect score summation.
        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(official_job, t, self.seconds): k
                       for k, t in sorted(pending.items(), key=lambda item: -item[1][0])}
            done_count = 0
            last_log = start
            while futures:
                done, _ = wait(futures, timeout=20, return_when=FIRST_COMPLETED)
                for future in done:
                    key = futures.pop(future)
                    result, info = future.result()  # timeout is fatal, never an excluded group
                    self.cache[key] = result
                    self.metrics.append(info)
                    done_count += 1
                now = time.monotonic()
                if now-last_log >= 20 or not futures:
                    cpu = sum(x["cpu_seconds"] for x in self.metrics)
                    print("CERT progress", label, done_count, "/", len(pending),
                          "elapsed_s", round(now-start, 2), "active_or_queued", len(futures),
                          "worker_cpu_s", round(cpu, 2),
                          "peak_worker_rss_kib", max((x["peak_rss_kib"] for x in self.metrics), default=0),
                          flush=True)
                    last_log = now
        slow = sorted(self.metrics, key=lambda x: -x["seconds"])[:5]
        print("CERT slowest", label, json.dumps(slow), flush=True)

    def full_gate(self, path):
        start = time.monotonic()
        text = path.read_text()
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames != ["id", "x", "y", "deg"]:
            if reader.fieldnames != ["id", "n", "square_id", "x", "y", "angle"]:
                raise ValueError("Unexpected canonical schema")
            for index, row in enumerate(reader):
                if None in row or any(v is None for v in row.values()) or row["id"] != str(index):
                    raise ValueError("Invalid canonical row IDs or CSV envelope")
        modes = self.union.load(path)  # actual pandas whole-file roundtrip, before parsing
        print("CERT transport", path.name, "seconds", round(time.monotonic()-start, 3),
              "pandas", self.union.pd.__version__, flush=True)
        reports = []
        for mode, groups in zip(("literal", "pandas"), modes):
            if set(groups) != set(range(1, 1001)):
                raise ValueError("Full certification requires exactly 1000 complete groups")
            for n, rows in groups.items():
                error = self.union.metric._validate_puzzle(n, rows)
                if error:
                    raise ValueError(error)
            tasks = [(n, groups[n]) for n in sorted(groups)]
            self.evaluate(tasks, path.name+"-"+mode)
            results = [self.cache[group_key(t)] for t in tasks]
            if any(n != t[0] or reason != "ok" or side is None
                   for t, (n, value, side, reason) in zip(tasks, results)):
                raise ValueError("Full official certification failed")
            reports.append(dict(score=sum(r[1] for r in results), max_n=1000,
                accepted=1000, missing=0, rejected=0,
                results=[dict(n=n, score=value, L=side, reason=reason)
                         for n, value, side, reason in results]))
        return reports

    def certify_many(self, tasks, workers, cache):
        missing = {self.union.key(t): t for t in tasks if self.union.key(t) not in cache}
        singles = [(n, rows) for n, a, b in missing.values() for rows in (a, b)]
        self.evaluate(singles, "union-inputs")
        for k, (n, a, b) in missing.items():
            results = [self.cache[group_key((n, rows))] for rows in (a, b)]
            cache[k] = (tuple(r[2] for r in results)
                        if len(a) == len(b) == n and all(r[3] == "ok" and r[2] is not None for r in results)
                        else None)
        return cache


def install_runtime(union):
    runtime = Runtime(union)
    union.full_gate = runtime.full_gate
    union.certify_many = runtime.certify_many
    return runtime
