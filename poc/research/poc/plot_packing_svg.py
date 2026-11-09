"""Standalone SVG vector graphics renderer for square packing solutions.

Renders packing solutions inside a bounding container [0, L] x [0, L],
styling upright squares in light blue (#a0c4ff) with dark borders and
tilted squares in bright coral (#ff6b6b) with center orientation indicators.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def compute_square_corners(cx: float, cy: float, angle: float) -> list[tuple[float, float]]:
    """Compute the 4 corners of a unit square centered at (cx, cy) rotated by angle.
    
    Returns corners in counter-clockwise order:
    (top-right, top-left, bottom-left, bottom-right) in local frame.
    """
    c = math.cos(angle)
    s = math.sin(angle)
    h = 0.5
    dx1, dy1 = h * c - h * s, h * s + h * c
    dx2, dy2 = -h * c - h * s, -h * s + h * c
    return [
        (cx + dx1, cy + dy1),
        (cx + dx2, cy + dy2),
        (cx - dx1, cy - dy1),
        (cx - dx2, cy - dy2),
    ]


def is_square_tilted(angle: float, threshold: float = 1e-3) -> bool:
    """Return True if angle is not an integer multiple of pi/2 (within threshold)."""
    # Normalize angle to [0, pi/2)
    half_pi = math.pi / 2.0
    mod = angle % half_pi
    dev = min(mod, half_pi - mod)
    return dev > threshold


def normalize_packing_layout(
    layout: Sequence[Dict[str, Any]],
    L: float,
    auto_align: bool = True,
) -> Tuple[List[Dict[str, float]], float, float]:
    """Extract and optionally translate coordinates so packing sits inside [0, L] x [0, L].
    
    Returns:
        (normalized_squares, shift_x, shift_y)
    """
    parsed: List[Dict[str, float]] = []
    all_corners: List[Tuple[float, float]] = []

    for item in layout:
        # Support dict with various keys: x/cx, y/cy, angle/theta/a
        x = float(item.get("x", item.get("cx", 0.0)))
        y = float(item.get("y", item.get("cy", 0.0)))
        angle = float(item.get("angle", item.get("theta", item.get("a", 0.0))))
        parsed.append({"x": x, "y": y, "angle": angle})
        all_corners.extend(compute_square_corners(x, y, angle))

    if not parsed or not auto_align:
        return parsed, 0.0, 0.0

    min_x = min(pt[0] for pt in all_corners)
    max_x = max(pt[0] for pt in all_corners)
    min_y = min(pt[1] for pt in all_corners)
    max_y = max(pt[1] for pt in all_corners)

    span_x = max_x - min_x
    span_y = max_y - min_y

    # Check if translation is needed to fit into [0, L] x [0, L]
    needs_shift = (
        min_x < -1e-5
        or min_y < -1e-5
        or max_x > L + 1e-5
        or max_y > L + 1e-5
    )

    if needs_shift:
        shift_x = -min_x + max(0.0, (L - span_x) / 2.0)
        shift_y = -min_y + max(0.0, (L - span_y) / 2.0)
    else:
        shift_x = 0.0
        shift_y = 0.0

    shifted: List[Dict[str, float]] = []
    for item in parsed:
        shifted.append({
            "x": item["x"] + shift_x,
            "y": item["y"] + shift_y,
            "angle": item["angle"],
        })

    return shifted, shift_x, shift_y


def render_packing_svg(
    layout: Sequence[Dict[str, Any]],
    L: float,
    *,
    pad: Optional[float] = None,
    view_size_px: int = 800,
    upright_color: str = "#a0c4ff",
    tilted_color: str = "#ff6b6b",
    upright_stroke: str = "#1d3557",
    tilted_stroke: str = "#780000",
    container_stroke: str = "#111827",
    container_bg: str = "#ffffff",
    tilt_threshold: float = 1e-3,
    show_orientation: bool = True,
    auto_align: bool = True,
    y_axis_up: bool = True,
    title: Optional[str] = None,
) -> str:
    """Render packing layout as a standalone, valid SVG vector graphic.
    
    Args:
        layout: List of square descriptors with keys 'x', 'y', 'angle'.
        L: Bounding container side length [0, L] x [0, L].
        pad: Padding outside [0, L] container walls. Defaults to max(0.6, L * 0.03).
        view_size_px: Output pixel size for SVG width/height attributes.
        upright_color: Fill color for upright squares (#a0c4ff).
        tilted_color: Fill color for tilted squares (#ff6b6b).
        upright_stroke: Border stroke for upright squares.
        tilted_stroke: Border stroke for tilted squares.
        container_stroke: Stroke color for outer container walls.
        container_bg: Background color inside outer container.
        tilt_threshold: Radian threshold below which a square is upright.
        show_orientation: Whether to draw orientation pointer on tilted squares.
        auto_align: Whether to translate squares to [0, L] x [0, L] if shifted.
        y_axis_up: If True, Cartesian y=0 is at bottom, matching math coordinates.
        title: Optional metadata / header title string.
    
    Returns:
        XML string of the complete standalone SVG.
    """
    squares, _, _ = normalize_packing_layout(layout, L, auto_align=auto_align)
    n = len(squares)

    if pad is None:
        pad = max(0.6, L * 0.035)

    # Dynamic scaling for strokes so they remain proportional to unit squares
    wall_stroke = max(0.04, L * 0.0035)
    sq_stroke = max(0.02, L * 0.0018)
    arrow_stroke = max(0.035, L * 0.0025)
    center_dot_r = 0.065
    arrow_len = 0.35
    arrow_head_len = 0.11
    arrow_head_half_w = 0.055

    # SVG viewBox coordinates
    vb_min_x = -pad
    vb_min_y = -pad
    vb_w = L + 2.0 * pad
    vb_h = L + 2.0 * pad

    tilted_count = 0
    upright_count = 0

    elements: List[str] = []

    # Canvas background outside container
    elements.append(
        f'  <rect x="{vb_min_x:.4f}" y="{vb_min_y:.4f}" width="{vb_w:.4f}" height="{vb_h:.4f}" '
        f'fill="#f8fafc" />'
    )

    # Container interior background
    elements.append(
        f'  <rect x="0.0000" y="0.0000" width="{L:.4f}" height="{L:.4f}" '
        f'fill="{container_bg}" />'
    )

    # Container grid accents (subtle 1-unit markings)
    elements.append('  <!-- Container Grid & Backing -->')

    # Draw squares
    sq_elements: List[str] = []
    orientation_elements: List[str] = []

    for sq in squares:
        cx = sq["x"]
        cy = sq["y"]
        angle = sq["angle"]

        tilted = is_square_tilted(angle, threshold=tilt_threshold)
        if tilted:
            tilted_count += 1
            fill = tilted_color
            stroke = tilted_stroke
        else:
            upright_count += 1
            fill = upright_color
            stroke = upright_stroke

        # Compute corners in Cartesian frame
        corners_cart = compute_square_corners(cx, cy, angle)

        # Convert corners to SVG coordinates
        corners_svg: List[Tuple[float, float]] = []
        for px, py in corners_cart:
            svg_x = px
            svg_y = (L - py) if y_axis_up else py
            corners_svg.append((svg_x, svg_y))

        pts_str = " ".join(f"{x:.4f},{y:.4f}" for x, y in corners_svg)
        sq_elements.append(
            f'    <polygon points="{pts_str}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sq_stroke:.4f}" stroke-linejoin="round" />'
        )

        # Orientation indicator on tilted squares
        if tilted and show_orientation:
            svg_cx = cx
            svg_cy = (L - cy) if y_axis_up else cy

            # Direction angle in SVG coordinates
            svg_angle = -angle if y_axis_up else angle
            cos_a = math.cos(svg_angle)
            sin_a = math.sin(svg_angle)

            # Shaft end (tip)
            tip_x = svg_cx + arrow_len * cos_a
            tip_y = svg_cy + arrow_len * sin_a

            # Arrowhead vertices
            base_x = tip_x - arrow_head_len * cos_a
            base_y = tip_y - arrow_head_len * sin_a
            s1_x = base_x - arrow_head_half_w * sin_a
            s1_y = base_y + arrow_head_half_w * cos_a
            s2_x = base_x + arrow_head_half_w * sin_a
            s2_y = base_y - arrow_head_half_w * cos_a

            # Center origin dot
            orientation_elements.append(
                f'    <circle cx="{svg_cx:.4f}" cy="{svg_cy:.4f}" r="{center_dot_r:.4f}" '
                f'fill="#1e293b" />'
            )
            # Shaft line
            orientation_elements.append(
                f'    <line x1="{svg_cx:.4f}" y1="{svg_cy:.4f}" x2="{tip_x:.4f}" y2="{tip_y:.4f}" '
                f'stroke="#1e293b" stroke-width="{arrow_stroke:.4f}" stroke-linecap="round" />'
            )
            # Arrowhead
            orientation_elements.append(
                f'    <polygon points="{tip_x:.4f},{tip_y:.4f} {s1_x:.4f},{s1_y:.4f} {s2_x:.4f},{s2_y:.4f}" '
                f'fill="#1e293b" />'
            )

    elements.append('  <!-- Unit Squares -->')
    elements.append('  <g id="squares">')
    elements.extend(sq_elements)
    elements.append('  </g>')

    if orientation_elements:
        elements.append('  <!-- Orientation Indicators -->')
        elements.append('  <g id="orientations">')
        elements.extend(orientation_elements)
        elements.append('  </g>')

    # Outer square container [0, L] x [0, L] walls drawn on top for crispness
    elements.append('  <!-- Outer Container Walls [0, L] x [0, L] -->')
    elements.append(
        f'  <rect x="0.0000" y="0.0000" width="{L:.4f}" height="{L:.4f}" '
        f'fill="none" stroke="{container_stroke}" stroke-width="{wall_stroke:.4f}" stroke-linecap="square" />'
    )

    # Optional descriptive title header
    if title:
        elements.append(
            f'  <!-- Title: {title} -->'
        )

    header_info = f"n={n}, L={L:.6f} ({upright_count} upright, {tilted_count} tilted)"

    svg_content = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb_min_x:.4f} {vb_min_y:.4f} {vb_w:.4f} {vb_h:.4f}" '
        f'width="{view_size_px}" height="{view_size_px}">',
        f'  <title>{header_info}</title>',
        f'  <desc>Packing solution of {n} unit squares in container of side {L:.6f}. Upright: {upright_count}, Tilted: {tilted_count}.</desc>',
        '\n'.join(elements),
        '</svg>',
    ]

    return '\n'.join(svg_content)


def save_packing_svg(
    layout: Sequence[Dict[str, Any]],
    L: float,
    output_path: Union[str, Path],
    **kwargs: Any,
) -> Path:
    """Render packing and write standalone SVG vector graphic to disk."""
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    svg_str = render_packing_svg(layout, L, **kwargs)
    out_p.write_text(svg_str, encoding="utf-8")
    return out_p


# ---------------------------------------------------------------------------
# Data Loaders & Incumbent / Candidate Fetchers
# ---------------------------------------------------------------------------

def load_csv_packing(csv_path: Union[str, Path], target_n: int) -> List[Dict[str, Any]]:
    """Load square rows for target n from a competition CSV."""
    p = Path(csv_path)
    if not p.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    results: List[Dict[str, Any]] = []
    with p.open(mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["n"]) == target_n:
                results.append({
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "angle": float(row["angle"]),
                    "square_id": row.get("square_id", ""),
                })
    return results


def load_incumbent_L(json_path: Union[str, Path], target_n: int) -> float:
    """Load certified side length L for target n from official/cpp JSON report."""
    p = Path(json_path)
    if not p.is_file():
        return 0.0
    with p.open(mode="r", encoding="utf-8") as f:
        data = json.load(f)
    if "results" in data:
        for r in data["results"]:
            if r.get("n") == target_n:
                return float(r.get("L", 0.0))
    return 0.0


def generate_all_research_plots(
    plots_dir: Union[str, Path] = Path("poc/research/docs/plots"),
    repo_root: Optional[Path] = None,
) -> Dict[str, Path]:
    """Generate all required incumbent and idea candidate plots.
    
    1. Known tilted incumbents: n=147, n=290, n=739
    2. Best real tilted candidate generated by Idea 1 (n=155)
    3. Best real tilted candidate generated by Idea 3 (n=132 and n=156)
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]

    out_dir = Path(plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: Dict[str, Path] = {}

    merged_csv = repo_root / "poc/pell_corrugation/data/border-scan/merged.csv"
    official_json = repo_root / "poc/pell_corrugation/data/border-scan/merged-pandas-official.json"
    idea1_json = repo_root / "poc/research/data/large_n_scratch_band_results.json"
    idea3_json = repo_root / "poc/research/data/recursive_refill_results.json"

    # Known incumbents: n=147, n=290, n=739
    incumbent_targets = [
        (147, 12.656854249492426),
        (290, 17.5355339059328518),
        (739, 27.690921182142139),
    ]

    for n_val, default_L in incumbent_targets:
        L_val = load_incumbent_L(official_json, n_val) or default_L
        layout = load_csv_packing(merged_csv, n_val)
        if not layout:
            raise RuntimeError(f"Could not load incumbent packing for n={n_val} from {merged_csv}")

        # Save with both naming styles for clear discoverability
        path1 = save_packing_svg(
            layout,
            L_val,
            out_dir / f"n{n_val}_incumbent.svg",
            title=f"Incumbent n={n_val} (L={L_val:.6f})",
        )
        path2 = save_packing_svg(
            layout,
            L_val,
            out_dir / f"incumbent_n{n_val}.svg",
            title=f"Incumbent n={n_val} (L={L_val:.6f})",
        )
        generated[f"incumbent_n{n_val}"] = path1

    # Idea 1: Best real tilted candidate
    # In large_n_scratch_band_results.json, n=155 has 11 tilted squares (family: tilted_band, delta_L = 0.047739)
    if idea1_json.is_file():
        with idea1_json.open(mode="r", encoding="utf-8") as f:
            d1 = json.load(f)
        # Find best tilted candidate
        tilted_candidates = []
        for k_str, val in d1.get("targets", {}).items():
            sqs = val.get("layout", [])
            tilted_sqs = [s for s in sqs if is_square_tilted(float(s.get("angle", 0.0)))]
            if tilted_sqs:
                tilted_candidates.append((val.get("delta_L", 999.0), int(k_str), val))
        tilted_candidates.sort(key=lambda x: x[0])

        if tilted_candidates:
            _, best_n1, best_data1 = tilted_candidates[0]
            layout1 = best_data1["layout"]
            L1 = float(best_data1.get("best_L", 0.0))
            p1 = save_packing_svg(
                layout1,
                L1,
                out_dir / f"idea1_best_n{best_n1}.svg",
                title=f"Idea 1 Best Tilted: n={best_n1} (L={L1:.6f})",
            )
            # Also save canonical alias
            save_packing_svg(
                layout1,
                L1,
                out_dir / "idea1_best_tilted.svg",
                title=f"Idea 1 Best Tilted: n={best_n1} (L={L1:.6f})",
            )
            generated["idea1_best"] = p1

    # Idea 3: Best real tilted candidate
    # In recursive_refill_results.json, n=132 has min delta_L = 0.014421, n=156 has delta_L = 0.020935
    if idea3_json.is_file():
        with idea3_json.open(mode="r", encoding="utf-8") as f:
            d3 = json.load(f)
        tilted_candidates3 = []
        for item in d3.get("results", []):
            sqs = item.get("candidate_packing", [])
            tilted_sqs = [s for s in sqs if is_square_tilted(float(s.get("angle", 0.0)))]
            if tilted_sqs:
                tilted_candidates3.append((item.get("delta_L", 999.0), item.get("n"), item))
        tilted_candidates3.sort(key=lambda x: x[0])

        if tilted_candidates3:
            # Top candidate (n=132)
            _, best_n3, best_data3 = tilted_candidates3[0]
            layout3 = best_data3["candidate_packing"]
            L3 = float(best_data3.get("L", 0.0))
            p3 = save_packing_svg(
                layout3,
                L3,
                out_dir / f"idea3_best_n{best_n3}.svg",
                title=f"Idea 3 Best Tilted: n={best_n3} (L={L3:.6f})",
            )
            save_packing_svg(
                layout3,
                L3,
                out_dir / "idea3_best_tilted.svg",
                title=f"Idea 3 Best Tilted: n={best_n3} (L={L3:.6f})",
            )
            generated["idea3_best"] = p3

            # Also plot n=156 if present
            n156_cand = [c for c in tilted_candidates3 if c[1] == 156]
            if n156_cand:
                _, n156_val, d156 = n156_cand[0]
                save_packing_svg(
                    d156["candidate_packing"],
                    float(d156["L"]),
                    out_dir / f"idea3_best_n{n156_val}.svg",
                    title=f"Idea 3 Tilted: n={n156_val} (L={float(d156['L']):.6f})",
                )

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render square packing solutions as standalone SVG graphics."
    )
    parser.add_argument(
        "--generate-all",
        action="store_true",
        default=True,
        help="Generate all standard incumbent and idea plots in docs/plots/",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="poc/research/docs/plots",
        help="Target directory for generated SVG plots",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Optional input CSV or JSON layout file",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Target n when reading CSV file",
    )
    parser.add_argument(
        "--L",
        type=float,
        default=None,
        help="Bounding side length L",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output SVG file path",
    )

    args = parser.parse_args()

    if args.file and args.out:
        in_p = Path(args.file)
        if in_p.suffix.lower() == ".csv":
            if args.n is None:
                raise ValueError("Must specify --n with CSV input")
            layout = load_csv_packing(in_p, args.n)
        else:
            with in_p.open("r", encoding="utf-8") as f:
                d = json.load(f)
            layout = d.get("layout", d.get("candidate_packing", d))

        if args.L is None:
            raise ValueError("Must specify container side length --L")

        out = save_packing_svg(layout, args.L, args.out)
        print(f"Rendered SVG saved to: {out}")
        return

    # Default action: generate all research plots
    repo_root = Path(__file__).resolve().parents[3]
    plots_path = repo_root / args.plots_dir
    res = generate_all_research_plots(plots_path, repo_root=repo_root)
    print(f"Generated {len(res)} SVG plots in {plots_path}:")
    for name, p in res.items():
        print(f"  - {name}: {p.name}")


if __name__ == "__main__":
    main()
