#!/usr/bin/env python3
"""Poll the live square-packing tracker for sides below the tracked incumbent.

The tracker JSON has per-`n` results with a certified or candidate side, plus a
`drawings` map with 5-decimal poses. Any side strictly below the incumbent is
printed; the caller then fetches the author's exact witness repo and certifies
it. Data only: this never runs third-party code.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from decimal import Decimal

USER_AGENT = "Mozilla/5.0 (authorized square packing research)"
FEED = "https://dogaozden.com/math/live/square-packing/data/feed.json"


def get(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    return urllib.request.urlopen(request, timeout=90).read()


def walk(obj, ours, out, depth=0):
    if depth > 8:
        return
    if isinstance(obj, dict):
        if "n" in obj and ("side" in obj or "L" in obj):
            try:
                n = int(obj["n"])
                side = Decimal(str(obj.get("side", obj.get("L"))))
            except Exception:  # noqa: BLE001 - malformed rows are skipped
                n = None
            if n is not None and n in ours and side < ours[n]:
                out.append(dict(
                    n=n, side=str(side), gain=str((ours[n] - side) / Decimal(n).sqrt()),
                    found_by=obj.get("found_by"), found_at=obj.get("found_at"),
                    verdict=obj.get("verdict"), record=obj.get("record"),
                ))
        for value in obj.values():
            walk(value, ours, out, depth + 1)
    elif isinstance(obj, list):
        for value in obj[:5000]:
            walk(value, ours, out, depth + 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True,
                        help="official literal report of the incumbent")
    parser.add_argument("--out", default="")
    parser.add_argument("--feed", default=FEED)
    args = parser.parse_args()

    report = json.load(open(args.report, encoding="utf-8"))
    ours = {int(r["n"]): Decimal(r["L"]) for r in report["results"]
            if r["L"] is not None}
    feed = json.loads(get(args.feed))
    hits: list[dict] = []
    walk(feed, ours, hits)
    # De-duplicate identical (n, side) rows from results/drawings/events.
    unique = {(h["n"], h["side"]): h for h in hits}
    hits = sorted(unique.values(), key=lambda h: -Decimal(h["gain"]))
    print(json.dumps(dict(
        feed_at=feed.get("at"), status_at=feed.get("status_at"),
        results=len(feed.get("results", [])),
        drawings=len(feed.get("drawings", {})),
        candidates=len(hits),
        total_gain=str(sum(Decimal(h["gain"]) for h in hits)),
    ), indent=1))
    for hit in hits[:40]:
        print(f"n={hit['n']:4d} gain={Decimal(hit['gain']):.6e} "
              f"side={hit['side']} verdict={hit['verdict']} "
              f"by={hit['found_by']} at={hit['found_at']}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(hits, handle, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
