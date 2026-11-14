#!/usr/bin/env python3
"""Test staircase splicing (vstrip, hcol, peel) + relax_and_compress + brick_mechwalk."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gstep_lib as gl  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox  # noqa: E402


def main() -> None:
    t0 = time.time()
    bricks = gl.load_source_bricks()
    for nm in ("B5_11x20", "B6_10x21_mw972_walk"):
        b = bricks[nm]
        print(f"Source {nm}: {b['w']}x{b['h']} g=+{b['g']} N={len(b['x'])} f={b['f']:.6f}", flush=True)

    synth_summary = []

    for src_name, out_prefix in (
        ("B6_10x21_mw972_walk", "B7"),
        ("B5_11x20", "B6"),
    ):
        src = bricks[src_name]
        v_cands = gl.splice_vertical_strip(src)
        h_cands = gl.splice_horizontal_column(src)
        print(f"[{src_name}] v_cands={len(v_cands)} {[t for t,_ in v_cands]} | h_cands={len(h_cands)} {[t for t,_ in h_cands]}", flush=True)

        # Compress best vertical splice (and horizontal splice)
        for group_tag, cands, f_sched in (
            ("vstrip", v_cands[:3], (0.76, 0.72, 0.69, 0.67, 0.655)),
            ("hcol", h_cands[:3], (0.76, 0.72, 0.69, 0.67, 0.655)),
        ):
            best_b = None
            for tag, cand in cands:
                pen0, _ = cpp_eval_sat_pen_bbox(cand["x"], cand["y"], cand["th"])
                comp = gl.relax_and_compress_brick(cand, f_targets=f_sched, n_trials=8, seed=101)
                if comp is not None:
                    print(f"  [{group_tag}] {tag}: pen0={pen0:.4f} -> comp {comp['w']}x{comp['h']} g=+{comp['g']} fx={comp['fx']:.6f} fy={comp['fy']:.6f} f={comp['f']:.6f}", flush=True)
                    if best_b is None or comp["f"] < best_b["f"]:
                        best_b = comp
                else:
                    print(f"  [{group_tag}] {tag}: pen0={pen0:.4f} -> FAILED to reach feasibility", flush=True)
            if best_b is not None:
                walked = gl.brick_mechwalk(best_b, f"{out_prefix}_{best_b['w']}x{best_b['h']}_{group_tag}", max_rounds=2, save=True)
                print(f"  ==> WALKED {walked['tag']}: {walked['w']}x{walked['h']} g=+{walked['g']} f={walked['f']:.6f}", flush=True)
                synth_summary.append({
                    "name": walked["tag"], "w": walked["w"], "h": walked["h"], "g": walked["g"],
                    "fx": walked["fx"], "fy": walked["fy"], "f": walked["f"],
                })
                # Also test peeling 1 row/col from walked brick!
                for side in ("top", "bottom", "left", "right"):
                    pb = gl.peel_outer_unit_layer(walked, side)
                    if pb is None:
                        continue
                    pc = gl.relax_and_compress_brick(pb, f_targets=(0.74, 0.71, 0.685, 0.67), n_trials=8, seed=202)
                    if pc is not None:
                        pw = gl.brick_mechwalk(pc, f"{out_prefix}_{pc['w']}x{pc['h']}_{group_tag}_peel_{side}", max_rounds=2, save=True)
                        print(f"    ==> PEELED ({side}) {pw['tag']}: {pw['w']}x{pw['h']} g=+{pw['g']} f={pw['f']:.6f}", flush=True)
                        synth_summary.append({
                            "name": pw["tag"], "w": pw["w"], "h": pw["h"], "g": pw["g"],
                            "fx": pw["fx"], "fy": pw["fy"], "f": pw["f"],
                        })

    gl.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (gl.DOCS_DIR / "synth_summary.json").write_text(json.dumps(synth_summary, indent=2), encoding="utf-8")
    print(f"DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
