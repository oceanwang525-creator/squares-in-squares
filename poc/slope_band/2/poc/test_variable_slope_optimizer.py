"""Unit tests and mathematical invariants for variable-slope band optimizer."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import variable_slope_optimizer as vso


class VariableSlopeOptimizerTests(unittest.TestCase):
    def test_extent_and_normalize_box(self):
        # Two squares at (1, 1, 0) and (2, 2, 0)
        q = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]])
        # Extent along each axis: from (1 - 0.5) to (2 + 0.5) is 2.0
        ext = vso.extent(q)
        self.assertAlmostEqual(ext, 2.0, places=9)

        q_norm = vso.normalize_box(q, margin=0.1)
        h = 0.5
        min_x = np.min(q_norm[:, 0] - h)
        self.assertAlmostEqual(min_x, 0.1, places=9)

    def test_sat_collision_detection(self):
        # Overlapping squares at distance 0.8 (< 1.0)
        q_overlap = np.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
        is_sat, min_gap = vso.sat_check(q_overlap)
        self.assertFalse(is_sat)
        self.assertLess(min_gap, 0.0)

        # Touching/separated squares at distance 1.05
        q_separate = np.array([[0.0, 0.0, 0.0], [1.05, 0.0, 0.0]])
        is_sat, min_gap = vso.sat_check(q_separate)
        self.assertTrue(is_sat)
        self.assertGreaterEqual(min_gap, 0.049)

    def test_variable_slope_theta_constants(self):
        # Verify 24.10 degrees corresponds to arctan(17/38)
        rad = vso.THETA_24_10_RAD
        deg = rad * 180.0 / math.pi
        self.assertAlmostEqual(deg, 24.1004, places=3)
        self.assertAlmostEqual(math.atan(17.0 / 38.0), rad, places=4)

    def test_piece_gap_gradient_computation(self):
        q = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, vso.THETA_24_10_RAD]])
        axis = (0, 0, 1)  # owner=0, k=0, direction=1
        for signs in vso.SIGNS:
            g, grad = vso.piece_gap_gradient(q, 0, 1, axis, signs)
            self.assertEqual(len(grad), 6)
            self.assertTrue(np.isfinite(g))
            self.assertTrue(np.all(np.isfinite(grad)))

    def test_variable_slope_pocket_construction(self):
        # Build mini pocket
        cols = (5, 7, 8, 7, 5)
        n_upright = 50
        q, free_mask = vso.build_variable_slope_pocket(
            n_upright=n_upright,
            column_counts=cols,
            theta_0=vso.THETA_24_10_RAD,
            nw_center=(5.0, 10.0),
        )
        total_tilted = sum(cols) * 2
        self.assertEqual(len(q), n_upright + total_tilted)
        self.assertEqual(np.sum(free_mask), total_tilted)
        self.assertEqual(np.sum(~free_mask), n_upright)

        # Check angle of tilted squares
        tilted_angles = q[free_mask, 2]
        self.assertTrue(np.all(np.abs(np.abs(tilted_angles) - vso.THETA_24_10_RAD) < 1e-6))

    def test_solve_epigraph_lp_small(self):
        # A small 4-square cluster with 2 free angles near theta=24.10 deg
        th = vso.THETA_24_10_RAD
        q = np.array([
            [1.0, 1.0, 0.0],
            [2.1, 1.0, 0.0],
            [1.0, 2.5, th],
            [2.1, 2.5, th],
        ])
        free_mask = np.array([False, False, True, True])
        res, msg = vso.solve_epigraph_lp(
            q=q,
            upper_L=5.0,
            free_mask=free_mask,
            position_trust=0.1,
            angle_trust=0.01,
            clearance=1e-8,
        )
        self.assertIsNotNone(res, f"LP failed: {msg}")
        self.assertEqual(res.shape, q.shape)
        # Upright squares must preserve angle=0.0
        self.assertEqual(res[0, 2], 0.0)
        self.assertEqual(res[1, 2], 0.0)

    def test_rows_and_csv_formatting(self):
        q = np.array([[1.5, 2.5, vso.THETA_24_10_RAD]])
        formatted = vso.rows(q)
        self.assertEqual(len(formatted), 1)
        self.assertIn("x", formatted[0])
        self.assertIn("y", formatted[0])
        self.assertIn("angle", formatted[0])

        csv_out = vso.csv_text({1: formatted})
        self.assertTrue(csv_out.startswith("id,n,square_id,x,y,angle"))
        self.assertIn("0,1,0,", csv_out)


if __name__ == "__main__":
    unittest.main()
