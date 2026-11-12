#!/usr/bin/env python3
"""One harvest cycle: newest kernel outputs, live feed, and team pushes.

Steps
1. List the competition kernels (several sort orders) and fetch CSV outputs of
   every kernel whose ``lastRunTime`` is newer than the saved state.
2. Poll the live tracker feed and print any side below the incumbent.
3. Fetch ``origin`` and report whether the team landed a new ``merged.csv`` or
   a new certified report to diff per ``n``.

Nothing is merged here: this only stages data and reports candidates. The
merge is done by ``poc/kernel_refresh_morning/poc/scan_and_merge.py`` (kernels)
or ``poc/team_union_nov12`` style per-``n`` diff (team), and every candidate is
gated by the unchanged official metric.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FEED = "https://dogaozden.com/math/live/square-packing/data/feed.json"
USER_AGENT = "curl/8.4.0"


def request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})


def list_kernels() -> list[dict]:
    seen: dict[str, dict] = {}
    for page in (1, 2, 3):
        for sort in ("DATE_RUN", "DATE_CREATED", "HOTNESS", "SCORE", "VOTES"):
            body = json.dumps({"kernelFilterCriteria": {
                "search": "",
                "listRequest": {"pageSize": 100, "page": page,
                                "sortBy": sort, "competitionId": 166661}}}).encode()
            req = urllib.request.Request(
                "https://www.kaggle.com/api/i/kernels.KernelsService/ListKernels",
                body, {"content-type": "application/json",
                       "User-Agent": USER_AGENT})
            try:
                payload = json.load(urllib.request.urlopen(req, timeout=60))
            except Exception:  # noqa: BLE001
                continue
            for kernel in payload.get("kernels", []):
                seen.setdefault(kernel["scriptUrl"], kernel)
    return list(seen.values())


def fetch_outputs(owner: str, slug: str, target: str) -> int:
    token = None
    files: list[dict] = []
    while True:
        url = ("https://www.kaggle.com/api/v1/kernels/output"
               f"?userName={urllib.parse.quote(owner)}"
               f"&kernelSlug={urllib.parse.quote(slug)}")
        if token:
            url += "&pageToken=" + urllib.parse.quote(token)
        payload = json.load(urllib.request.urlopen(request(url), timeout=60))
        files.extend(payload.get("files", []))
        token = payload.get("nextPageToken")
        if not token:
            break
    got = 0
    os.makedirs(target, exist_ok=True)
    for entry in files:
        name = entry.get("fileName", "")
        if not name.endswith(".csv") or name.startswith("."):
            continue
        dest = os.path.join(target, name.replace("/", "__"))
        try:
            with urllib.request.urlopen(request(entry["url"]), timeout=300) as src, \
                    open(dest, "wb") as sink:
                sink.write(src.read())
            got += 1
        except Exception as error:  # noqa: BLE001
            print("  download failed", name, error)
    return got


def walk(obj, ours, out, depth=0):
    if depth > 8:
        return
    if isinstance(obj, dict):
        if "n" in obj and ("side" in obj or "L" in obj):
            try:
                n = int(obj["n"])
                side = Decimal(str(obj.get("side", obj.get("L"))))
            except Exception:  # noqa: BLE001
                n = None
            if n is not None and n in ours and side < ours[n] - Decimal("1e-12"):
                out.append((n, side, (ours[n] - side) / Decimal(n).sqrt()))
        for value in obj.values():
            walk(value, ours, out, depth + 1)
    elif isinstance(obj, list):
        for value in obj[:5000]:
            walk(value, ours, out, depth + 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--since", default="",
                        help="override the last-run timestamp")
    args = parser.parse_args()

    state = {"last_run": ""}
    if os.path.exists(args.state):
        state = json.load(open(args.state, encoding="utf-8"))
    cut = args.since or state.get("last_run", "")

    kernels = list_kernels()
    fresh = [k for k in kernels
             if k.get("lastRunTime", "") > cut and k.get("hasDataOutputFiles")]
    fresh.sort(key=lambda k: k["lastRunTime"])
    print(json.dumps(dict(kernels=len(kernels), fresh=len(fresh), since=cut), indent=1))
    for kernel in fresh:
        path = kernel["scriptUrl"].replace("/code/", "")
        owner, slug = path.split("/", 1)
        target = os.path.join(args.out, f"{owner}__{slug}")
        got = fetch_outputs(owner, slug, target)
        print("fetched", path, got, "csv")

    if kernels:
        state["last_run"] = max(k["lastRunTime"] for k in kernels)
    with open(args.state, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=1)

    report = json.load(open(args.report, encoding="utf-8"))
    ours = {int(r["n"]): Decimal(r["L"]) for r in report["results"]
            if r["L"] is not None}
    feed = json.loads(urllib.request.urlopen(request(FEED), timeout=90).read())
    hits: list = []
    walk(feed, ours, hits)
    unique = {(n, side): (n, side, gain) for n, side, gain in hits}
    hits = sorted(unique.values(), key=lambda h: -h[2])
    print(json.dumps(dict(feed_at=feed.get("at"), feed_candidates=len(hits),
                          feed_gain=str(sum(h[2] for h in hits))), indent=1))
    for n, side, gain in hits[:20]:
        print(f"  feed n={n} gain={gain:.3e} side={side}")

    subprocess.run(["git", "-C", str(REPO), "fetch", "origin"], check=False)
    log = subprocess.run(["git", "-C", str(REPO), "log", "--oneline", "-3",
                          "origin/main"], check=False, capture_output=True,
                         text=True)
    print("origin/main:")
    print(log.stdout.rstrip())


if __name__ == "__main__":
    main()
