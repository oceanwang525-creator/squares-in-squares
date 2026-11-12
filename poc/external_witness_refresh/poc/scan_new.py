#!/usr/bin/env python3
"""Harvest new witness/certificate packings from refreshed external repos.

Adapted from poc/public_resweep_other/poc/scan_ext.py. Handles rational
corner witnesses, center-basis witnesses, algebraic-number-field scalars,
rational .cert files, Minuii .npz files, and generic CSV/JSON layouts.
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import lzma
import os
import re
from decimal import Decimal
from fractions import Fraction

import mpmath as mp
import numpy as np
import yaml

mp.mp.dps = 70


def fmt(value):
    return mp.nstr(value, 65)


def rat(text):
    text = str(text).strip()
    if "/" in text:
        a, b = text.split("/")
        return Fraction(int(a), int(b))
    return Fraction(text)


def num(value, root):
    if isinstance(value, list):
        return sum(num(c, root) * root ** i for i, c in enumerate(value))
    text = str(value)
    if "/" in text:
        a, b = text.split("/")
        return mp.mpf(a) / mp.mpf(b)
    return mp.mpf(text)


def robust_root(coeffs, lo, hi):
    def value(t):
        return mp.polyval(coeffs, t)

    flo, fhi = value(lo), value(hi)
    if flo == 0:
        return lo
    if fhi == 0:
        return hi
    if flo * fhi > 0:
        raise ValueError("root is not bracketed")
    for _ in range(400):
        mid = (lo + hi) / 2
        fmid = value(mid)
        if fmid == 0:
            return mid
        if flo * fmid < 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return (lo + hi) / 2


class Scanner:
    def __init__(self, ours: dict[int, Decimal], tol: float = 1e-9):
        self.ours = ours
        self.tol = tol
        self.candidates: dict[int, list[dict]] = collections.defaultdict(list)
        self.seen: set[tuple[int, str]] = set()

    def offer(self, n, source, xs, ys, angles, float_arrays=None):
        if n not in self.ours or len(xs) != n:
            return
        if float_arrays is None:
            xf = np.array([float(v) for v in xs])
            yf = np.array([float(v) for v in ys])
            af = np.array([float(v) for v in angles])
        else:
            xf, yf, af = float_arrays
        if not (np.isfinite(xf).all() and np.isfinite(yf).all()
                and np.isfinite(af).all()):
            return
        half = 0.5 * (np.abs(np.cos(af)) + np.abs(np.sin(af)))
        side = max((xf + half).max() - (xf - half).min(),
                   (yf + half).max() - (yf - half).min())
        if side >= float(self.ours[n]) - self.tol:
            return
        key = (n, source)
        if key in self.seen:
            return
        self.seen.add(key)
        self.candidates[n].append(dict(
            source=source, float_side=float(side),
            rows=[dict(square_id=str(i), x=str(xs[i]), y=str(ys[i]),
                       angle=str(angles[i])) for i in range(n)],
        ))

    def do_cert(self, text, source):
        lines = [line for line in text.splitlines()
                 if line.strip() and not line.startswith("#")]
        header = lines[0].split()
        if len(header) != 2:
            return
        n = int(header[0])
        if not 1 <= n <= 1000 or len(lines) - 1 != n:
            return
        rows = [line.split() for line in lines[1:]]
        if any(len(row) != 3 for row in rows):
            return
        fx = [rat(row[0]) for row in rows]
        fy = [rat(row[1]) for row in rows]
        ft = [rat(row[2]) for row in rows]
        xs = [fmt(mp.mpf(v.numerator) / v.denominator) for v in fx]
        ys = [fmt(mp.mpf(v.numerator) / v.denominator) for v in fy]
        angles = [fmt(2 * mp.atan(mp.mpf(t.numerator) / t.denominator)) for t in ft]
        xf = np.array([float(v) for v in fx])
        yf = np.array([float(v) for v in fy])
        af = np.array([2 * np.arctan(float(v)) for v in ft])
        self.offer(n, source, xs, ys, angles, (xf, yf, af))

    def do_witness(self, text, source):
        payload = yaml.load(text, Loader=yaml.CSafeLoader).get("witness", {})
        n = payload.get("n", 0)
        if not 1 <= n <= 1000:
            return
        root = mp.mpf(0)
        if payload.get("scalar", {}).get("kind") == "algebraic-number-field":
            coeff = [num(v, root) for v in payload["scalar"]["minimal_polynomial"]]
            lo, hi = [num(v, root) for v in payload["scalar"]["isolating_interval"]]
            root = robust_root(coeff, lo, hi)
        xs, ys, angles = [], [], []
        for square in payload["squares"]:
            if "center" in square:
                x, y = [num(v, root) for v in square["center"]]
                if "basis" in square:
                    angle = mp.atan2(num(square["basis"][1], root),
                                     num(square["basis"][0], root))
                else:
                    angle = num(square.get("angle", 0), root) * mp.pi / 180
            else:
                corners = [[num(v, root) for v in c] for c in square["corners"]]
                x = sum(c[0] for c in corners) / 4
                y = sum(c[1] for c in corners) / 4
                angle = mp.atan2(corners[1][1] - corners[0][1],
                                 corners[1][0] - corners[0][0])
            xs.append(fmt(x))
            ys.append(fmt(y))
            angles.append(fmt(angle))
        self.offer(n, source, xs, ys, angles)

    def walk(self, payload, source, hint=None, depth=0):
        if depth > 6:
            return
        if isinstance(payload, dict):
            n = None
            for key in ("n", "N", "puzzle", "n_squares"):
                if isinstance(payload.get(key), int):
                    n = payload[key]
            if n is None:
                n = hint
            for key, value in payload.items():
                if n is not None and is_tri(value, n):
                    self.offer(n, source, [r[0] for r in value],
                               [r[1] for r in value], [r[2] for r in value])
                elif (n is not None and is_dicts(value, n)
                      and all(("angle" in r or "theta" in r) for r in value)):
                    self.offer(n, source, [r["x"] for r in value],
                               [r["y"] for r in value],
                               [r.get("angle", r.get("theta")) for r in value])
                else:
                    child = int(key) if isinstance(key, str) and key.isdigit() else None
                    self.walk(value, source,
                              child or (n if not isinstance(value, (list, dict)) else None),
                              depth + 1)
        elif isinstance(payload, list):
            for value in payload[:5000]:
                self.walk(value, source, hint, depth + 1)


def is_tri(value, n):
    return (isinstance(value, list) and len(value) == n and n > 0
            and all(isinstance(row, (list, tuple)) and len(row) >= 3
                    and all(isinstance(v, (int, float)) for v in row[:3])
                    for row in value))


def is_dicts(value, n):
    return (isinstance(value, list) and len(value) == n and n > 0
            and all(isinstance(row, dict) and {"x", "y"} <= set(row) for row in value))


def read_text(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", errors="replace").read()
    if path.endswith(".xz"):
        return lzma.open(path, "rt", errors="replace").read()
    return open(path, errors="replace").read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--incumbent-report", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--since-regex", default=r"2026-11-1[012]",
                        help="only scan paths matching this regex; use . for all")
    parser.add_argument("--tol", type=float, default=1e-9,
                        help="float prefilter margin below the incumbent L")
    parser.add_argument("--known-best", action="store_true", default=True,
                        help="also scan packing/witnesses/known-best")
    args = parser.parse_args()

    report = json.load(open(args.incumbent_report, encoding="utf-8"))
    ours = {int(row["n"]): Decimal(row["L"]) for row in report["results"]
            if row["L"] is not None}
    scanner = Scanner(ours, tol=args.tol)

    files = [path for path in glob.glob(os.path.join(args.repo_root, "**", "*"),
                                        recursive=True)
             if os.path.isfile(path) and re.search(args.since_regex, path)]
    if args.known_best:
        files += glob.glob(os.path.join(args.repo_root, "**",
                                        "packing", "witnesses", "known-best", "*.yaml"),
                           recursive=True)
    files = sorted(set(files))
    print("files", len(files), flush=True)

    errors = collections.Counter()
    for path in files:
        source = os.path.relpath(path, args.repo_root)
        try:
            if re.search(r"\.cert(\.gz)?$", path):
                if os.path.getsize(path) < 20e6:
                    scanner.do_cert(read_text(path), source)
            elif re.search(r"\.ya?ml(\.gz)?$", path):
                text = read_text(path)
                if "witness" in text:
                    scanner.do_witness(text, source)
            elif path.endswith(".npz"):
                data = np.load(path)
                n = len(data["th"])
                scanner.offer(n, source, data["xy"][:, 0].tolist(),
                              data["xy"][:, 1].tolist(), data["th"].tolist())
            elif path.endswith(".json"):
                scanner.walk(json.load(open(path)), source)
            elif path.endswith(".json.gz"):
                scanner.walk(json.loads(read_text(path)), source)
        except Exception as error:  # noqa: BLE001
            errors[type(error).__name__] += 1
    print("errors", dict(errors), flush=True)

    serializable = {str(n): entries for n, entries in scanner.candidates.items()}
    with open(args.candidates, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=1)
    print("candidate n", len(scanner.candidates), "candidates",
          sum(len(v) for v in scanner.candidates.values()), flush=True)
    for n in sorted(scanner.candidates):
        best = min(scanner.candidates[n], key=lambda entry: entry["float_side"])
        print(f"n={n:4d} float_gap={float(ours[n]) - best['float_side']:.6f} "
              f"{best['source']}")


if __name__ == "__main__":
    main()
