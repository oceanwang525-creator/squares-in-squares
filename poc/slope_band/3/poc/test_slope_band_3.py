"""Unit tests for Slope-Band Dual-Pocket Symmetric Composite POC (POC 3)."""
import sys
import unittest
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4] if len(Path(__file__).resolve().parents) > 4 else Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "poc/pell_corrugation/poc"))
sys.path.insert(0, str(REPO / "poc/baseline/poc/src"))

import numpy as np
import slope_band_3 as sb


class SlopeBandDualPocketTests(unittest.TestCase):
    def test_transposition_symmetry_mapping(self):
        # A test square at (5.0, 20.0, 0.42)
        test_nw = np.array([[5.0, 20.0, 0.42]])
        mapped_se = sb.apply_transposition_symmetry(test_nw)
        # Expected: x2 = 20.0 - 3.0 = 17.0, y2 = 5.0 + 2.0 = 7.0, theta2 = -0.42
        self.assertAlmostEqual(mapped_se[0, 0], 17.0)
        self.assertAlmostEqual(mapped_se[0, 1], 7.0)
        self.assertAlmostEqual(mapped_se[0, 2], -0.42)

    def test_canonical_908_symmetry(self):
        q_908, L_908, meta_908 = sb.build_dual_pocket_908()
        self.assertEqual(len(q_908), 908)
        self.assertEqual(meta_908["tilted_squares"], 200)
        self.assertEqual(meta_908["upright_squares"], 708)
        self.assertEqual(meta_908["squares_per_pocket"], 100)

        sym_ok, sym_res = sb.verify_transposition_symmetry(q_908)
        self.assertTrue(sym_ok)
        self.assertLess(sym_res, 1e-7)

    def test_dual_pocket_908_extent_and_sat(self):
        q_908, L_908, meta_908 = sb.build_dual_pocket_908()
        # Verify L <= 30.637030
        self.assertLessEqual(L_908, 30.637030)
        # Verify beats baseline
        baseline = float(sb.PORTFOLIO_BASELINE[908])
        self.assertLess(L_908, baseline)
        self.assertLess(L_908 - baseline, -0.065)

        sat_ok, sat_gap = sb.sat_check(q_908)
        self.assertTrue(sat_ok)
        self.assertGreater(sat_gap, -1e-9)

    def test_dual_pocket_composite_968_geometry(self):
        q_968, L_968, meta_968 = sb.build_dual_pocket_composite_968()
        self.assertEqual(len(q_968), 968)
        self.assertEqual(meta_968["core_squares"], 908)
        self.assertEqual(meta_968["border_squares"], 60)
        self.assertEqual(meta_968["tilted_squares"], 200)
        self.assertEqual(meta_968["upright_squares"], 768)

        # Container side length check
        self.assertLessEqual(L_968, 31.637031)
        baseline_968 = float(sb.PORTFOLIO_BASELINE[968])
        self.assertLess(L_968, baseline_968)
        # Check massive deficit drop: ~ -0.070
        self.assertLess(L_968 - baseline_968, -0.065)

        # Check SAT clearance
        sat_ok, sat_gap = sb.sat_check(q_968)
        self.assertTrue(sat_ok)
        self.assertGreater(sat_gap, -1e-9)

    def test_falsification_gate(self):
        res = sb.run_falsification_test()
        self.assertEqual(res["gate_status"], "PASS")
        self.assertTrue(res["n_908"]["sat_ok"])
        self.assertTrue(res["n_908"]["symmetry_ok"])
        self.assertTrue(res["n_908"]["beats_baseline"])
        self.assertTrue(res["n_908"]["matches_incumbent"])
        self.assertTrue(res["n_968"]["sat_ok"])
        self.assertTrue(res["n_968"]["beats_baseline"])
        self.assertLess(float(res["n_968"]["delta_L_baseline"]), -0.065)
