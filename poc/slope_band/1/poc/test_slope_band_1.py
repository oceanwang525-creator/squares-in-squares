"""Unit tests for slope_band POC 1: exact 1:2 rational slope band generator."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slope_band_1 as sb


class SlopeBandTests(unittest.TestCase):
    def test_mathematical_constants(self):
        self.assertAlmostEqual(sb.THETA_1_2, math.atan(0.5), places=12)
        self.assertAlmostEqual(sb.COS_THETA, 2.0 / math.sqrt(5.0), places=12)
        self.assertAlmostEqual(sb.SIN_THETA, 1.0 / math.sqrt(5.0), places=12)
        self.assertAlmostEqual(sb.H_THETA, 3.0 / math.sqrt(5.0), places=12)
        self.assertAlmostEqual(sb.H_45, math.sqrt(2.0), places=12)
        self.assertAlmostEqual(sb.DELTA_L_ADVANTAGE, math.sqrt(2.0) - 3.0 / math.sqrt(5.0), places=12)
        self.assertGreater(sb.DELTA_L_ADVANTAGE, 0.0725)
        self.assertLess(sb.DELTA_L_ADVANTAGE, 0.0726)

    def test_2x1_tooth_geometry_invariants(self):
        step = np.array([2.0, 1.0])
        # Integer step (2, 1) is strictly orthogonal to the band normal n
        normal_proj = float(step @ sb.NORMAL)
        self.assertAlmostEqual(normal_proj, 0.0, places=12)

        # Integer step (2, 1) advances along the band tangent by exactly sqrt(5)
        tangent_proj = float(step @ sb.TANGENT)
        self.assertAlmostEqual(tangent_proj, math.sqrt(5.0), places=12)

        # Invariance of 2y - x along the flank staircase
        y0, x0 = 3.0, 5.0
        y1, x1 = y0 + 1.0, x0 + 2.0
        self.assertEqual(2.0 * y0 - x0, 2.0 * y1 - x1)

    def test_convergents_sqrt5(self):
        # Convergent 9/4: 4 teeth * sqrt(5) ≈ 8.94427 vs 9 squares
        p1, q1, m1 = sb.CONVERGENTS_SQRT5[1]
        self.assertEqual((p1, q1), (9, 4))
        self.assertAlmostEqual(abs(m1), abs(9.0 - 4.0 * math.sqrt(5.0)), places=10)

        # Convergent 38/17: 17 teeth * sqrt(5) ≈ 38.013156 vs 38 squares
        p2, q2, m2 = sb.CONVERGENTS_SQRT5[2]
        self.assertEqual((p2, q2), (38, 17))
        self.assertAlmostEqual(abs(m2), abs(38.0 - 17.0 * math.sqrt(5.0)), places=10)
        self.assertLess(abs(m2), 0.014)

    def test_unpolished_prototype_generation(self):
        q, meta = sb.generate_slope_band_unpolished(L0=15.0, w=4, l=9, margin=0.01)
        self.assertGreaterEqual(len(q), 150)
        self.assertEqual(meta["n_tilted"], 36)
        self.assertGreater(meta["n_upright"], 100)

        # SAT check verifies no inter-square overlaps
        is_sat, min_gap = sb.sat_check(q, clearance=-1e-9)
        self.assertTrue(is_sat, f"SAT failed with min_gap={min_gap}")

        # Check container extent within nominal bounds
        L = sb.extent(q)
        self.assertLessEqual(L, 15.0 + 0.10)

    def test_falsification_gate(self):
        # Runs falsification test against rigid upright teeth
        res = sb.falsification_test(L0=15.0, w=4, l=9, position_trust=0.10)
        # Falsified: rigid 2x1 teeth cause hyperstatic LP infeasibility (Status 8)
        self.assertFalse(res["passed"])
        self.assertIn("infeasible", res["reason"].lower())
