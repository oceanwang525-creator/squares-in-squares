#!/usr/bin/env python3
"""Trusted-main union, exact dual-transport gate, and guarded publication.

Never run this write-enabled entry point on a pull request checkout.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from decimal import Decimal

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = "poc/pell_corrugation/data/border-scan/merged.csv"
NAME = "koboshchan"
EMAIL = "91501777+koboshchan@users.noreply.github.com"
COMMIT_MESSAGE = "Auto-union certified POC finals"
EXPECTED = set(range(1, 1001))

# Script directory is not guaranteed on sys.path when imported by tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from certification_runtime import install_runtime
from certification_cache import save


def git(*args, cwd=ROOT, check=True):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def value(*args, cwd=ROOT):
    return git(*args, cwd=cwd).stdout.strip()


def identity_args():
    return ["-c", "user.name=" + NAME, "-c", "user.email=" + EMAIL]


def own_commit_push(event, event_name, read=value):
    """Only an exact, single canonical-only automation commit may be skipped."""
    if event_name != "push" or event.get("size", len(event.get("commits", []))) != 1:
        return False
    commits = event.get("commits", [])
    if len(commits) != 1:
        return False
    before, after = event.get("before", ""), event.get("after", "")
    if not re.fullmatch(r"[0-9a-f]{40}", before) or not re.fullmatch(r"[0-9a-f]{40}", after):
        return False
    if set(before) == {"0"} or commits[0].get("id") != after:
        return False
    if read("rev-list", "--count", before + ".." + after) != "1":
        return False
    if read("show", "-s", "--format=%P", after) != before:
        return False
    if read("show", "-s", "--format=%B", after) != COMMIT_MESSAGE:
        return False
    if read("show", "-s", "--format=%an%n%ae%n%cn%n%ce", after) != "\n".join([NAME, EMAIL, NAME, EMAIL]):
        return False
    return read("diff", "--name-only", before, after, "--") == CANONICAL


def validate_report(report):
    if (report["accepted"], report["missing"], report["rejected"]) != (1000, 0, 0):
        raise ValueError("Official certification must accept all 1000 groups")
    rows = report["results"]
    if len(rows) != 1000 or {r["n"] for r in rows} != EXPECTED:
        raise ValueError("Official certification has missing/duplicate groups")
    for row in rows:
        if row["reason"] != "ok" or row["L"] is None:
            raise ValueError("Rejected group")
        side = Decimal(row["L"])
        if not side.is_finite() or side <= 0:
            raise ValueError("Invalid certified side")
    score = Decimal(str(report["score"]))
    if not score.is_finite():
        raise ValueError("Invalid score")
    return {row["n"]: Decimal(row["L"]) for row in rows}


def validate_transition(before, after, changed):
    if len(before) != 2 or len(after) != 2:
        raise ValueError("Both literal and actual pandas reports are required")
    mode_wins = []
    for old, new in zip(before, after):
        a, b = validate_report(old), validate_report(new)
        if any(b[n] > a[n] for n in EXPECTED):
            raise ValueError("Per-n regression against initial canonical")
        wins = [n for n in sorted(EXPECTED) if b[n] < a[n]]
        if changed:
            if not wins or not Decimal(str(new["score"])) < Decimal(str(old["score"])):
                raise ValueError("Changed bytes require a strictly better official score")
        elif wins or b != a or new["score"] != old["score"]:
            raise ValueError("No-op bytes must have identical certification")
        mode_wins.append(wins)
    if mode_wins[0] != mode_wins[1]:
        raise ValueError("Every selected group must strictly improve in both modes")
    return mode_wins[0]


def fetch_main():
    git("fetch", "--no-tags", "origin", "refs/heads/main")
    return value("rev-parse", "FETCH_HEAD")


def publish_snapshot(snapshot, score, source_sha):
    """Normal fast-forward mirror push; never overwrite a concurrent publisher."""
    if fetch_main() != source_sha:
        print("Main advanced; skipping stale publication", flush=True)
        return False
    with tempfile.TemporaryDirectory(prefix="certified-mirror-") as temp:
        target = Path(temp) / "tree"
        filename = f"merged-{score:.11f}.csv"
        # Fetch the current branch, not a stale checkout remote-tracking ref.
        exists = git("ls-remote", "--exit-code", "--heads", "origin", "refs/heads/merge", check=False)
        if exists.returncode not in (0, 2):
            raise RuntimeError(exists.stderr)
        if exists.returncode == 0:
            git("fetch", "--no-tags", "origin", "refs/heads/merge")
            parent = value("rev-parse", "FETCH_HEAD")
            files = value("ls-tree", "-r", "--name-only", parent).splitlines()
            blob = hashlib.sha256(snapshot).hexdigest()
            if files == [filename]:
                current = subprocess.check_output(["git", "show", f"{parent}:{filename}"], cwd=ROOT)
                if hashlib.sha256(current).hexdigest() == blob:
                    print("Mirror bytes and score filename unchanged; no commit", flush=True)
                    return True
            git("worktree", "add", "--detach", str(target), parent)
            git("rm", "-rf", "--", ".", cwd=target)
        else:
            git("worktree", "add", "--detach", str(target), source_sha)
            git("switch", "--orphan", "certified-mirror", cwd=target)
        try:
            (target / filename).write_bytes(snapshot)
            git("add", "--", filename, cwd=target)
            if value("diff", "--cached", "--name-only", cwd=target) == "":
                raise ValueError("Unexpected empty mirror change")
            if value("ls-files", cwd=target).splitlines() != [filename]:
                raise ValueError("Mirror must contain exactly the certified CSV")
            git(*identity_args(), "commit", "-m", f"Publish {score:.11f} from {source_sha}", cwd=target)
            if fetch_main() != source_sha:
                print("Main advanced before publication; skipping stale mirror", flush=True)
                return False
            git("push", "--atomic", "origin", source_sha + ":refs/heads/main", "HEAD:refs/heads/merge", cwd=target)
            print("Published exact canonical from", source_sha, filename, flush=True)
            return True
        finally:
            git("worktree", "remove", "--force", str(target))


def main():
    if os.environ.get("GITHUB_REF") != "refs/heads/main":
        raise ValueError("Only trusted main may execute this publisher")
    if os.environ.get("GITHUB_EVENT_NAME") not in ("push", "workflow_dispatch"):
        raise ValueError("Unsupported event")
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    if value("rev-parse", "HEAD") == event.get("after") and own_commit_push(event, os.environ["GITHUB_EVENT_NAME"]):
        print("Exact automation-only canonical commit; no recursive union", flush=True)
        return
    initial_sha = value("rev-parse", "HEAD")
    if fetch_main() != initial_sha:
        print("Main advanced before certification; next run owns latest source", flush=True)
        return
    if value("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Publisher requires a clean trusted checkout")
    canonical = ROOT / CANONICAL
    initial = canonical.read_bytes()
    spec = importlib.util.spec_from_file_location("certified_auto_union", ROOT / "poc/auto_union/poc/union.py")
    union = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = union
    spec.loader.exec_module(union)
    if union.pd.__version__ != "3.1.0":
        raise ValueError("Actual pandas 3.1.0 is required")
    runtime = install_runtime(union)
    with tempfile.TemporaryDirectory(prefix="certified-union-") as temp:
        audit = Path(temp)
        print("Certifying initial canonical in literal and actual pandas modes", flush=True)
        before = union.full_gate(canonical)
        for report in before:
            validate_report(report)
        union.union(canonical, min(os.cpu_count() or 2, 8), audit)
        proof = json.loads((audit / "union.json").read_text())
        snapshot = canonical.read_bytes()
        changed = snapshot != initial
        if proof["before_sha256"] != hashlib.sha256(initial).hexdigest() or proof["after_sha256"] != hashlib.sha256(snapshot).hexdigest():
            raise ValueError("Union proof does not match exact canonical bytes")
        wins = validate_transition(before, proof["reports"], changed)
        if sorted(proof["improved"]) != wins:
            raise ValueError("Union improvement list differs from full certification")
        save(runtime, canonical)
        print("Dual certification", json.dumps({
            "source_sha": initial_sha, "changed": changed, "wins": wins,
            "scores": [r["score"] for r in proof["reports"]],
            "sha256": proof["after_sha256"],
        }), flush=True)
        if fetch_main() != initial_sha:
            print("Main advanced during certification; no canonical or mirror push", flush=True)
            return
        source_sha = initial_sha
        if changed:
            dirty = value("diff", "--name-only").splitlines()
            if dirty != [CANONICAL]:
                raise ValueError("Union modified files other than canonical")
            git("add", "--", CANONICAL)
            if value("diff", "--cached", "--name-only").splitlines() != [CANONICAL]:
                raise ValueError("Only canonical may be committed")
            git(*identity_args(), "commit", "-m", COMMIT_MESSAGE)
            # Recheck after commit as well; the normal push is the final race guard.
            if fetch_main() != initial_sha:
                print("Main advanced before canonical push; discarding stale local commit", flush=True)
                return
            git("push", "origin", "HEAD:refs/heads/main")
            source_sha = value("rev-parse", "HEAD")
            print("Canonical committed", source_sha, flush=True)
        else:
            print("No strict improvement; canonical bytes and main history unchanged", flush=True)
        publish_snapshot(snapshot, proof["reports"][1]["score"], source_sha)


if __name__ == "__main__":
    main()
