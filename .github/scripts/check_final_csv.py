#!/usr/bin/env python3
"""Trusted-base, data-only PR structure gate. Not a geometry or strict-win gate."""
import csv
import io
import math
import os
import re
import subprocess
import sys

def validate(text):
    if text.splitlines()[:1] != ["id,x,y,deg"]:
        raise ValueError("expected exact header id,x,y,deg")
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    next(reader)
    groups = {}
    for line, row in enumerate(reader, 2):
        if len(row) != 4:
            raise ValueError(f"line {line}: expected four columns")
        match = re.fullmatch(r"([0-9]{3}|1000)_(0|[1-9][0-9]*)", row[0])
        if not match:
            raise ValueError(f"line {line}: invalid id")
        n, index = map(int, match.groups())
        if not 1 <= n <= 1000 or not 0 <= index < n:
            raise ValueError(f"line {line}: id out of range")
        indices = groups.setdefault(n, set())
        if index in indices:
            raise ValueError(f"line {line}: duplicate id/group")
        indices.add(index)
        for value in row[1:]:
            if not math.isfinite(float(value[1:] if value.startswith("s") else value)):
                raise ValueError(f"line {line}: coordinates must be finite numbers")
    for n, indices in groups.items():
        if indices != set(range(n)):
            raise ValueError(f"n={n}: incomplete group ({len(indices)}/{n})")

def git(*args):
    return subprocess.check_output(["git", *args])

def check_pr(base, head):
    if not all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in (base, head)):
        raise ValueError("invalid commit SHA")
    changed = git("diff", "--name-only", "-z", "--no-renames",
                  f"{base}...{head}", "--").decode().split("\0")
    folders = sorted({p.split("/")[1] for p in changed
                      if p.startswith("poc/") and len(p.split("/")) >= 3})
    checked = []
    for folder in folders:
        directory = f"poc/{folder}"
        # Fully removed approaches have no head folder to publish a final in.
        if not git("ls-tree", head, "--", directory):
            continue
        path = directory + "/final.csv"
        kind = git("ls-tree", head, "--", path).decode()
        if not kind.startswith(("100644 blob ", "100755 blob ")):
            raise ValueError(f"{path!r}: every touched POC must have a regular final.csv in PR head")
        try:
            validate(git("show", f"{head}:{path}").decode("utf-8"))
        except (ValueError, csv.Error) as error:
            raise ValueError(f"{path!r}: {error}") from error
        checked.append(path)
    return checked

def main():
    paths = check_pr(os.environ["BASE_SHA"], os.environ["HEAD_SHA"])
    print(f"Validated {len(paths)} touched POC final.csv file(s), structure only.")
    print("Separate merge gate required: official literal + actual pandas certification "
          "and strictly smaller L in BOTH modes versus fresh main; no ties or snapshots.")

if __name__ == "__main__":
    try:
        main()
    except (ValueError, csv.Error, UnicodeError, subprocess.CalledProcessError) as error:
        print(f"final.csv structure check failed: {error}", file=sys.stderr)
        sys.exit(1)
