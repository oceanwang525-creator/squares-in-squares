from fractions import Fraction
import importlib.util
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scan = load_module("scan_pockets_scan4", HERE / "poc" / "scan_pockets.py")


def lattice(offset, step, limit=Fraction(3, 2)):
    out = set()
    k = 0
    while offset + k * step < limit:
        out.add(offset + k * step)
        k += 1
    return out


def test_complementary_fine_phase_has_no_old_site():
    old = Fraction(1, 64)
    new = lattice(Fraction(0), Fraction(1, 32))
    assert old not in new
    assert Fraction(0) in new


def test_coarse_sites_are_a_subset_of_the_finer_lattice():
    old = lattice(Fraction(1, 64), Fraction(1, 16))
    new = lattice(Fraction(1, 64), Fraction(1, 32))
    assert old <= new


def test_window_pairs_use_exact_integer_width_floor():
    pairs = scan.window_pairs(ng=40, step=0.03125, maxw=1.0)
    widths = 0.03125 * (pairs[:, 1] - pairs[:, 0])
    assert np.all(widths >= 1.0 - 1e-12)
    assert np.all(widths <= 1.0 + 1e-12)


def test_final_structure_uses_trusted_checker():
    checker = load_module(
        "check_final_csv_scan4", REPO / ".github" / "scripts" / "check_final_csv.py"
    )
    final = HERE / "final.csv"
    if not final.exists():
        return
    checker.validate(final.read_text(encoding="utf-8"))


def test_pr72_donor_overlay_group_partition():
    donors = load_module(
        "make_augmented_donors_scan4",
        HERE / "poc" / "make_augmented_donors.py",
    )
    selected = set()
    for expected, chosen in donors.PR_GROUPS.values():
        assert chosen <= expected
        selected |= chosen
    assert selected == {442, 485, 530, 577, 588, 634, 695, 861, 926}
