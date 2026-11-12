#!/usr/bin/env python3
"""Clone or refresh the external packing banks used by the resweep.

Data only: repositories are cloned, never executed. The resweep's Nov 10
snapshot is refreshed to the current HEAD so new witness/certificate commits
are visible to scan_new.py.
"""
from __future__ import annotations

import argparse
import os
import subprocess

REPOS = [
    "https://github.com/Minuii/squares-in-squares",
    "https://github.com/jlevy/squares",
    "https://github.com/evand/square-packing",
    "https://github.com/CacinieP/sis-cloud-artifacts",
]


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for url in REPOS:
        name = url.rsplit("/", 2)[-2] + "_" + url.rsplit("/", 1)[-1]
        target = os.path.join(args.out, name)
        if os.path.isdir(os.path.join(target, ".git")):
            run("git", "-C", target, "fetch", "--depth", "1", "origin")
            run("git", "-C", target, "reset", "--hard", "FETCH_HEAD")
            print("updated", target)
        else:
            run("git", "clone", "--depth", "1", url, target)
            print("cloned", target)


if __name__ == "__main__":
    main()
