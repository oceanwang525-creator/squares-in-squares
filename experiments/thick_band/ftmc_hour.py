"""Round-robin FTMC search. No incumbent writes, certificates or submissions.

Run beside the bundled modules with baseline.json, or pass --baseline.
Saved full layouts are floating SAT-screened candidates, NEVER certified wins.
GPU initialization and baseline preparation precede the search clock.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time
import traceback

import numpy as np

D = Path(__file__).resolve().parent


def schedule(lane, n):
    """Replayable broad mutation schedule; every target has its own lane."""
    modes = ("angle", "shear", "slide", "taper")
    amplitudes = (1e-4, 3e-4, .001, .003, .01, .03)
    slacks = (.002, .005, .015, .03, .05)
    return dict(seed=n*1000003+lane, mode=modes[(lane+lane//6+n) % 4],
                amplitude=amplitudes[(lane+n) % len(amplitudes)],
                slack=slacks[(lane+2*n) % len(slacks)])


def strict_layout(q, n):
    from band_polish import sat_check
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (n, 3):
        return False, "exact_count_or_shape"
    if not np.isfinite(q).all():
        return False, "nonfinite"
    if not sat_check(q)[0]:
        return False, "strict_allpair_overlap"
    return True, "strict_float_screen_only"


class Evidence:
    def __init__(self, out):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.counts = Counter()
        self.serial = 0

    def log(self, event, **data):
        row = dict(event=event, time=time.time(), **data)
        text = json.dumps(row, allow_nan=False, default=str)
        print(text, flush=True)
        with (self.out/"events.jsonl").open("a") as f:
            f.write(text+"\n")

    def reject(self, reason, **meta):
        self.counts[reason] += 1
        self.log("reject", reason=reason, **meta)

    def save(self, q, n, reference, **meta):
        from band_polish import extent
        valid, reason = strict_layout(q, n)
        if not valid:
            self.reject(reason, n=n, **meta)
            return False
        self.serial += 1
        L = float(extent(q))
        record = dict(n=n, L=L, reference_L=reference,
                      delta=reference-L if reference is not None else None,
                      prospective_win=reference is not None and L < reference-1e-9,
                      certified_win=False, needs_official_certificate=True,
                      screen="strict_allpair_float_not_certificate", **meta)
        path = self.out/f"n{n}-candidate{self.serial:07d}.npz"
        np.savez_compressed(path, x=np.ascontiguousarray(q), L=L,
                            reference_L=reference, meta=json.dumps(record))
        self.counts["saved"] += 1
        self.log("candidate", file=path.name, **record)
        return True

    def checkpoint(self, **state):
        path = self.out/"checkpoint.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(counts=dict(self.counts),
                                       saved=self.serial, **state), allow_nan=False))
        os.replace(tmp, path)


def run(args):
    import torch
    from band_polish import extent
    from band_relax import canonicalize, half_extent
    from gensane_polish import quench, unit_feasible
    from gensane_recovery import recover
    from ftmc_moves import split, mutate
    from core_refill import refill
    from local_ruin import local_ruin
    from ribbon_archive import RibbonArchive

    out = Evidence(args.output)
    blob = Path(args.baseline).read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    raw = json.loads(blob)
    targets = sorted(map(int, raw)) if not args.targets else list(dict.fromkeys(
        int(x) for x in args.targets.split(",")))
    if not targets or any(str(n) not in raw for n in targets):
        raise ValueError("targets must have immutable baseline references")
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("this search requires an A100")
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    # Force context creation and kernel initialization before the timed search.
    warm = torch.ones((1,), device="cuda")
    (warm+warm).sum().item()
    torch.cuda.synchronize()
    states = {}
    refs = {int(k): float(extent(np.asarray(v))) for k, v in raw.items()}
    for n in sorted(map(int, raw)):
        original = np.asarray(raw[str(n)], dtype=np.float64)
        if original.shape != (n, 3) or not np.isfinite(original).all():
            raise ValueError(f"invalid baseline shape {n}")
        t, _ = canonicalize(torch.tensor(original)[None])
        safe, side, ok = unit_feasible(t, clearance=2e-11, max_expand=1.000001)
        if not bool(ok[0]):
            raise ValueError(f"baseline float repair {n}")
        q = np.ascontiguousarray(safe[0].numpy())
        passed, reason = strict_layout(q, n)
        if not passed:
            raise ValueError(f"baseline screen {n}: {reason}")
        states[n] = dict(q=q, side=float(side[0]), lane=0, mutation_lane=0, gate=None)
    archive = RibbonArchive(Path(args.output)/"ribbon-archive")
    start = time.monotonic()
    end = start+args.seconds
    out.log("hardware", gpu=torch.cuda.get_device_name(), budget=args.seconds,
            baseline_sha256=digest, targets=targets, initialization_excluded=True)
    generation = 0
    last_checkpoint = start

    def remaining(deadline, cap):
        return max(0., min(cap, deadline-time.monotonic()-.25))

    def offer(core, n, topology, stage):
        if len(core):
            archive.offer_components(core, structural_identity=f"n{n}-{topology}",
                outer_bounds=[0, 0, refs[n], refs[n]],
                contact_metadata=dict(stage=stage, label="local_gadget_proxy"))

    def polish(q, n, deadline, meta, core_count=None, topology=None):
        # Save refill/recreation output even when no time is left for polishing.
        out.save(q, n, refs.get(n), stage="full_seed", **meta)
        budget = remaining(deadline, 10)
        if budget < .2:
            return
        z = torch.tensor(np.ascontiguousarray(q)[None], device="cuda")
        z, _ = quench(z, [refs[n]-1e-5], steps=1200,
                      seconds=budget, clearance=2e-9)
        z, _, valid = unit_feasible(z, clearance=2e-9, max_expand=1.03)
        if not bool(valid[0]):
            out.reject("full_unit_feasible", n=n, **meta)
            return
        candidate = z[0].cpu().numpy()
        out.save(candidate, n, refs[n], stage="full_quench", **meta)
        budget = remaining(deadline, 12)
        if budget >= .2:
            r = recover(candidate, seconds=budget, iterations=6,
                        position_trust=.02, angle_trust=.001, target=refs[n]-1e-8)
            if r["x"] is not None:
                out.save(r["x"], n, refs[n], stage="recovery", **meta)
                candidate = r["x"]
        if core_count is not None and topology is not None:
            # Refill places the core first; quench and recovery preserve row IDs.
            # Never re-split by angle: that would change membership after polishing.
            offer(candidate[:core_count], n, topology, "post-LP-fixed-core-rows")
        else:
            # Ruin changes row membership. Keep its proxies separately labeled.
            core, _ = split(candidate)
            offer(core, n, f"insertion-membership-{meta['generation']}",
                  "post-LP-incompatible-membership")

    try:
        while time.monotonic() < end-.5:
            for n in targets:
                if time.monotonic() >= end-.5:
                    break
                generation += 1
                s = states[n]
                lane = s["lane"]
                s["lane"] += 1
                deadline = min(end, time.monotonic()+args.target_seconds)
                meta = dict(generation=generation, lane=lane, source_n=n,
                            operator="mutation")
                out.log("generation", n=n, **meta)
                try:
                    q, side = s["q"], s["side"]
                    core, u = split(q)
                    if s["gate"] is None or (not s["gate"] and lane % 8 == 0):
                        offer(core, n, "baseline-core-order-v1", "baseline")
                        budget = remaining(deadline, 8)
                        if budget < .2:
                            continue
                        ident = refill(core, side, side, len(u), hints=u[:, :2],
                                       candidate_cap=6000, time_limit=budget)
                        passed = ident.x is not None and strict_layout(ident.x, n)[0]
                        s["gate"] = bool(passed)
                        out.log("identity", n=n, passed=bool(passed),
                                diagnostics=ident.diagnostics)
                        if ident.x is not None:
                            out.save(ident.x, n, refs[n], stage="identity", **meta)
                    # Occasional true insertion, with independent n+1 reference.
                    if lane % 5 == 4 and n+1 in states:
                        target = n+1
                        budget = remaining(deadline, 5)
                        if budget >= .2:
                            r = local_ruin(q, side, target, seed=n*1000003+lane,
                                           remove_count=10, time_limit=budget)
                            imeta = dict(generation=generation, lane=lane,
                                         source_n=n, operator="insertion",
                                         target_n=target, remove_requested=10,
                                         add_requested=11)
                            imeta.update(removed_actual=len(r["meta"]["removed_indices"]),
                                         add_actual_requested=r["meta"].get("refill_target"))
                            out.log("local_ruin", **imeta, diagnostics=r["meta"])
                            if r["x"] is not None:
                                polish(r["x"], target, deadline, imeta)
                            else:
                                out.reject("insertion_"+r["meta"]["reason"], **imeta)
                    elif s["gate"]:
                        mutation_lane = s["mutation_lane"]
                        s["mutation_lane"] += 1
                        params = schedule(mutation_lane, n)
                        work = side+params.pop("slack")
                        c, hints, mutation = mutate(q, side, **params)
                        c[:, :2] += (work-side)/2
                        h = hints[:, :2]+(work-side)/2
                        if not len(c):
                            out.reject("empty_core", n=n, **meta)
                            continue
                        budget = remaining(deadline, 6)
                        if budget < .2:
                            continue
                        x = torch.tensor(np.ascontiguousarray(c)[None], device="cuda")
                        x, _ = quench(x, [work], steps=900, seconds=budget, clearance=2e-10)
                        anchor = x.detach().mean(dim=1)[:, :2].clone()
                        x, cL, valid = unit_feasible(x, clearance=2e-10, max_expand=1.01)
                        x[:, :, :2] += anchor[:, None, :]-x.mean(dim=1)[:, None, :2]
                        hh = half_extent(x)[:, :, None]
                        inside = bool(((x[:, :, :2]-hh) >= 0).all() and
                                      ((x[:, :, :2]+hh) <= work).all())
                        if not bool(valid[0]) or not inside or float(cL[0]) > work:
                            out.reject("core_feasibility", n=n, **meta)
                            continue
                        c = x[0].cpu().numpy()
                        topology = f"taper-seed-{mutation['seed']}" if mutation["mode"] == "taper" else "baseline-core-order-v1"
                        offer(c, n, topology, "core")
                        budget = remaining(deadline, 8)
                        if budget < .2:
                            continue
                        filled = refill(c, work, work, n-len(c), hints=h,
                                        candidate_cap=6000, time_limit=budget)
                        out.log("refill", n=n, **meta, mutation=mutation,
                                diagnostics=filled.diagnostics)
                        if filled.x is None:
                            out.reject("refill_no_solution", n=n, **meta)
                        else:
                            polish(filled.x, n, deadline, meta, core_count=len(c),
                                   topology=topology)
                    else:
                        out.reject("identity_gate_failed", n=n, **meta)
                except Exception:
                    out.reject("generation_exception", n=n, **meta,
                               traceback=traceback.format_exc())
                finally:
                    now = time.monotonic()
                    if now-last_checkpoint >= args.checkpoint_seconds:
                        out.checkpoint(elapsed=now-start, generation=generation,
                                       lanes={str(k): v["lane"] for k, v in states.items()},
                                       baseline_sha256=digest)
                        out.log("checkpoint", elapsed=now-start, generation=generation)
                        last_checkpoint = now
                    # Failed gates/operators must never form an uncontrolled busy loop.
                    time.sleep(min(.1, max(0., end-time.monotonic())))
        out.log("finished", seconds=time.monotonic()-start, generations=generation,
                counts=dict(out.counts), certified_wins=0)
    finally:
        out.checkpoint(elapsed=time.monotonic()-start, generation=generation,
                       baseline_sha256=digest)
        if hashlib.sha256(Path(args.baseline).read_bytes()).hexdigest() != digest:
            raise RuntimeError("external baseline modification detected")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", default=str(D/"baseline.json"))
    p.add_argument("--output", default=str(D/"ftmc-results"))
    p.add_argument("--seconds", type=float, default=float(os.getenv("FTMC_SECONDS", "3600")))
    p.add_argument("--targets", default="", help="comma-separated subset; default all baseline keys")
    p.add_argument("--target-seconds", type=float, default=38.)
    p.add_argument("--checkpoint-seconds", type=float, default=30.)
    return p


def main():
    args = parser().parse_args()
    if not all(np.isfinite(v) and v > 0 for v in
               (args.seconds, args.target_seconds, args.checkpoint_seconds)):
        raise ValueError("time budgets must be finite and positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        run(args)
    except BaseException:
        (output/"ERROR.txt").write_text(traceback.format_exc())
        raise
    finally:
        with tarfile.open(output.parent/"ftmc-results.tar.gz", "w:gz") as f:
            f.add(output, arcname="ftmc-results")


if __name__ == "__main__":
    main()
