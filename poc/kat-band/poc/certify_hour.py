"""CPU-only post-run gates. Geometry is unit-square CENTER x,y and RADIANS.

NPZ full layouts require x:(n,3); filename starts n<N>-; optional n scalar.
Components under ribbon-archive/ or gadget/ are certified at their LOCAL count,
never compared to a full-box baseline and never called a fullboxwin.
No floating slack or C++ simulated roundtrip can authorize a win.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from collections import Counter
from decimal import Decimal

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "poc/baseline/poc/src"))
from squares import _official_metric as metric

PUBLIC_SHA = "007577524c36f240cbd33abc3aec0d198374e8d2be71e795717bf119461a7651"
DEFAULT_CPP = "/home/kit/ftmc-hour-prep/certify_hour_strict"
TRUSTED_CPP_SHA = "68e8f1bad1f3025a8243da0453e95f3255b7303561680831645f30482ce1abb5"
KNOWN_UNTRUSTED_CPP_SHA = "7d2e01cd5033cac94a5352516a1f39a6c551d1aaa75040a977dff1df5bcf011d"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def require_transport():
    if pd.__version__ != "3.1.0":
        raise RuntimeError("Requires actual pandas 3.1.0, not a simulated parser")


def load_csv(p):
    return metric._parse_submission_csv(Path(p).read_text())


def write_csv(p, groups):
    with Path(p).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "n", "square_id", "x", "y", "angle"])
        i = 0
        for n, rows in sorted(groups.items()):
            for r in rows:
                w.writerow([i, n, r["square_id"], r["x"], r["y"], r["angle"]])
                i += 1


def strict_screen(rows):
    """All pairs, four axes, long double, zero allowed overlap, no broadphase."""
    q = np.array([[r[k] for k in ("x", "y", "angle")] for r in rows],
                 dtype=np.longdouble)
    if q.shape != (len(rows), 3) or not np.isfinite(q).all():
        return {"passed": False, "reason": "nonfinite/shape"}
    c, s = np.cos(q[:, 2]), np.sin(q[:, 2])
    u = np.column_stack((c, s))
    v = np.column_stack((-s, c))
    minimum = np.longdouble(np.inf)
    overlaps = 0
    for i in range(len(q)-1):
        d = q[i+1:, :2] - q[i, :2]
        best = np.full(len(d), -np.inf, dtype=np.longdouble)
        for owner in (False, True):
            for basis in (u, v):
                a = basis[i+1:] if owner else np.broadcast_to(basis[i], d.shape)
                hi = (np.abs(a @ u[i]) + np.abs(a @ v[i])) / 2
                hj = (np.abs(np.sum(a*u[i+1:], axis=1)) +
                      np.abs(np.sum(a*v[i+1:], axis=1))) / 2
                best = np.maximum(best, np.abs(np.sum(d*a, axis=1))-hi-hj)
        minimum = min(minimum, np.min(best))
        overlaps += int(np.sum(best < 0))
    return {"passed": overlaps == 0, "overlap_pairs": overlaps,
            "all_pairs": len(rows)*(len(rows)-1)//2,
            "minimum_gap": str(minimum), "arithmetic": "numpy.longdouble, zero slack"}


def checks(csv_path, out, cpp):
    require_transport()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    raw = load_csv(csv_path)
    transport = out / "actual-pandas310.csv"
    # This exact read_csv -> to_csv is the required transport, default parser.
    pd.read_csv(csv_path).to_csv(transport, index=False)
    modes = {}
    for mode, path in (("literal", Path(csv_path)), ("pandas310", transport)):
        groups = load_csv(path)
        detail = out / (mode + "-cpp.json")
        cmd = [str(cpp), str(path), "--max-n", str(max(groups)),
               "--threads", "2", "--details", str(detail)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        save(out / (mode + "-process.json"),
             {"command": cmd, "exit": result.returncode,
              "stdout": result.stdout, "stderr": result.stderr})
        cp = {}
        if result.returncode == 0 and detail.exists():
            cp = {r["n"]: r for r in json.loads(detail.read_text())["results"]}
            for r in cp.values():
                # Pin the independently audited exhaustive screen, never the Gemini gate.
                r["independent_trusted"] = sha(cpp) == TRUSTED_CPP_SHA
                r["binary_sha256"] = sha(cpp)
                r["provenance_blocker"] = (None if r["independent_trusted"] else
                    "unvetted binary or Gemini false-pass lineage")
        modes[mode] = {}
        for n, rows in groups.items():
            official = metric._evaluate_puzzle_worker((n, rows))
            modes[mode][str(n)] = {
                "official": dict(zip(("n", "score", "L", "reason"), official)),
                "cpp": cp.get(n, {"reason": "process_failed", "L": None}),
                "screen": strict_screen(rows)}
        save(out / (mode + "-checks.json"), modes[mode])
    save(out / "input.json", {"source": str(csv_path), "sha256": sha(csv_path),
                             "transport_sha256": sha(transport)})
    return modes


def gates(modes, n, screen=True):
    return all(
        m[str(n)]["official"]["reason"] == "ok" and
        m[str(n)]["cpp"]["reason"] == "ok" and
        (not screen or m[str(n)]["screen"]["passed"])
        for m in modes.values())


def full_win(candidate, baseline, n):
    return (all(m[str(n)]["cpp"].get("independent_trusted", False) for m in candidate.values()) and
            gates(candidate, n) and gates(baseline, n, screen=False) and
            all(Decimal(candidate[k][str(n)][engine]["L"]) <
                Decimal(baseline[k][str(n)][engine]["L"])
                for k in ("literal", "pandas310") for engine in ("official", "cpp")))


def prepare(baseline, out, cpp):
    if sha(baseline) != PUBLIC_SHA:
        raise ValueError("Baseline is not the latest certified public23 merge")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    groups = load_csv(baseline)
    ranking = []
    for n in range(700, 1001):
        if math.isqrt(n)**2 == n:
            continue
        squares, L = metric._build_squares(n, groups[n])
        ranking.append({"n": n, "k": math.isqrt(n), "j": n-math.isqrt(n)**2,
                        "L": metric._format_detail(L),
                        "normalized_excess": float(L)/math.sqrt(n)-1})
    ranking.sort(key=lambda r: (-r["normalized_excess"], r["n"]))
    extra = [r["n"] for r in ranking[:3]]
    primary = sorted(set([908, 909, 910, 911, 971, 972, 973] + extra))
    targets = sorted(set(primary + [n+1 for n in primary if n < 1000]))
    write_csv(out/"baseline-targets.csv", {n: groups[n] for n in targets})
    checked = checks(out/"baseline-targets.csv", out/"baseline-checks", cpp)
    if not all(gates(checked, n, screen=False) for n in targets):
        raise ValueError("Baseline target certification failed")
    # Compatible with FTMC's dict[n] -> centered geometry, not corner coordinates.
    save(out/"baseline.json", {str(n): [[float(r[k]) for k in ("x","y","angle")]
                                       for r in groups[n]] for n in targets})
    save(out/"baseline-metadata.json", {
        "schema": 1, "source": str(baseline), "source_sha256": sha(baseline),
        "metric_sha256": sha(ROOT/"poc/baseline/poc/src/squares/_official_metric.py"),
        "cpp_sha256": sha(cpp), "pandas": pd.__version__,
        "cpp_warning": "Exhaustive long-double screen, not interval proof; official literal and pandas mandatory",
        "primary": primary, "extra": extra, "targets": targets,
        "ranking": ranking, "checks": checked,
        "references": {str(n): {k: checked[k][str(n)]["official"]["L"] for k in checked}
                       for n in targets}})
    return checked


def unpack(source, out):
    source = Path(source)
    if source.is_dir():
        return source
    dest = Path(out)/"evidence"
    dest.mkdir()
    with tarfile.open(source) as tf:
        members = tf.getmembers()
        if sum(m.size for m in members) > 2_000_000_000:
            raise ValueError("Evidence exceeds 2GB")
        for m in members:
            p = Path(m.name)
            if p.is_absolute() or ".." in p.parts or not (m.isfile() or m.isdir()):
                raise ValueError("Unsafe archive member")
        tf.extractall(dest, members=members, filter="data")
    return dest


def certify(source, baseline, out, cpp):
    out = Path(out)
    base = prepare(baseline, out, cpp)
    root = unpack(source, out)
    records = []
    counts = {}
    for i, path in enumerate(sorted(root.rglob("*.npz"))):
        component = any(s in path.parts for s in ("ribbon-archive", "gadget", "gadgets"))
        match = re.match(r"n(\d+)(?:-|_|\.)", path.name)
        record = {"source": str(path), "sha256": sha(path), "component": component,
                  "fullboxwin": False}
        try:
            with np.load(path, allow_pickle=False) as data:
                q = np.asarray(data["core"] if component and "core" in data else data["x"])
                n = len(q) if component else (int(data["n"]) if "n" in data else
                                              int(match.group(1)) if match else 0)
            record["n"] = n
            if not 1 <= n <= 1000 or q.shape != (n, 3) or not np.isfinite(q).all():
                raise ValueError("shape/count/nonfinite or missing filename n")
            rows = [{"square_id": j, **{k: repr(float(q[j, z]))
                    for z, k in enumerate(("x", "y", "angle"))}} for j in range(n)]
            d = out/f"candidate-{i:05d}"
            d.mkdir()
            write_csv(d/"candidate.csv", {n: rows})
            checked = checks(d/"candidate.csv", d/"checks", cpp)
            record["checks"] = checked
            record["geometry_gates_pass"] = gates(checked, n)
            record["verified"] = gates(checked, n) and all(
                m[str(n)]["cpp"].get("independent_trusted", False) for m in checked.values())
            if not component and str(n) not in base["literal"]:
                raise ValueError("n is outside certified baseline targets")
            record["fullboxwin"] = not component and full_win(checked, base, n)
            record["reasons"] = ([] if all(m[str(n)]["cpp"].get("independent_trusted", False)
                                           for m in checked.values()) else
                                  ["independent_cpp_provenance_unresolved"]) + [
                f"{mode}:{engine}:{values[str(n)][engine].get('reason', 'overlap')}"
                for mode, values in checked.items() for engine in ("official", "cpp")
                if values[str(n)][engine]["reason"] != "ok"]
            for mode, values in checked.items():
                if not values[str(n)]["screen"]["passed"]:
                    record["reasons"].append(mode+":strict_allpair_overlap")
            if record["verified"] and not component and not record["fullboxwin"]:
                record["reasons"].append("not_strictly_better_all_modes")
        except Exception as e:
            record["error"] = str(e)
            record["verified"] = False
        records.append(record)
        nkey = str(record.get("n", "unknown"))
        count = counts.setdefault(nkey, Counter())
        count["retained"] += 1
        count["verified"] += bool(record.get("verified"))
        count["fullboxwin"] += record["fullboxwin"]
        count["rejected"] += not bool(record.get("verified"))
        for reason in record.get("reasons", []) + ([record["error"]] if "error" in record else []):
            count[reason] += 1
        save(out/"certification.json", {"records": records, "per_n": counts})
    save(out/"certification.json", {"records": records, "per_n": counts})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--outputdir", required=True, type=Path)
    p.add_argument("--evidence", type=Path, help="Evidence tar or results directory")
    p.add_argument("--cpp", default=DEFAULT_CPP, type=Path)
    a = p.parse_args()
    require_transport()
    if a.evidence:
        certify(a.evidence, a.baseline, a.outputdir, a.cpp)
    else:
        prepare(a.baseline, a.outputdir, a.cpp)


if __name__ == "__main__":
    main()
