#!/usr/bin/env python3
"""Clone new witness repos of the lab that files exact upper bounds.

The lab publishes one repository per improvement, named `s<n>-upper-bound`
(with optional run/issue suffixes). This script lists the account's repos via
the GitHub API, clones or refreshes the `s*-upper-bound*` ones, then prints the
ones that are new relative to a saved inventory. Scanning and certification are
done by `scan_and_merge.py` in `poc/external_bank_expansion`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.request

USER = "kenaznia-doga"
API = f"https://api.github.com/users/{USER}/repos?per_page=100&sort=pushed"


def get(url: str):
    request = urllib.request.Request(
        url, headers={"User-Agent": "curl/8.4.0",
                      "Accept": "application/vnd.github+json"})
    return json.load(urllib.request.urlopen(request, timeout=60))


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--inventory", required=True,
                        help="JSON file listing repos already scanned")
    args = parser.parse_args()

    repos = [r for r in get(API)
             if "upper-bound" in r["name"] or r["name"].startswith("s")
             and "-upper" in r["name"]]
    known = set()
    if os.path.exists(args.inventory):
        known = set(json.load(open(args.inventory, encoding="utf-8")))
    new = [r for r in repos if r["full_name"] not in known]
    print(json.dumps(dict(repos=len(repos), known=len(known), new=len(new)), indent=1))
    for repo in new:
        print(" ", repo["full_name"], repo["pushed_at"], repo["size"])
    os.makedirs(args.out, exist_ok=True)
    for repo in repos:
        target = os.path.join(args.out, repo["full_name"].replace("/", "_"))
        if os.path.isdir(os.path.join(target, ".git")):
            run("git", "-C", target, "fetch", "--depth", "1", "origin")
            run("git", "-C", target, "reset", "--hard", "-q", "FETCH_HEAD")
        else:
            run("git", "clone", "--depth", "1", "--quiet",
                repo["clone_url"], target)
    with open(args.inventory, "w", encoding="utf-8") as handle:
        json.dump(sorted(r["full_name"] for r in repos), handle, indent=1)
    print("cloned", len(repos), "into", args.out)


if __name__ == "__main__":
    main()
