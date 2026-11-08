"""Unit tests for Pell-17 flank perimeter tooth deletion POC (POC 1)."""
import sys
import unittest
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4] if len(Path(__file__).resolve().parents) > 4 else Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "poc/pell_corrugation/poc"))
sys.path.insert(0, str(REPO / "poc/baseline/poc/src"))

import vacancy_deletion as vd
import pell_corrugation as pc


class VacancyDeletionTests(unittest.TestCase):
    def test_normalize_tooth_spec(self):
        a1 = 18
        # Outer SW tooth index 17 -> (17, 0)
        # Outer NE tooth index 0 -> (0, 17)
        # Middle tooth index 9 -> (9, 8)
        spec = [17, 0, (9, 8)]
        normalized = vd.normalize_tooth_spec(spec, a1)
        expected = {(17, 0), (0, 17), (9, 8)}
        self.assertEqual(normalized, expected)

        with self.assertRaises(ValueError):
            vd.normalize_tooth_spec([18], a1)

    def test_uncorrugated_tooth_deletion_geometry(self):
        w, l, a1, a2, b = 13, 25, 18, 17, 9
        # Full seed should be 739 squares
        q_full, L0_full = vd.build_staircase_band_tooth_deleted(w, l, a1, a2, b, b)
        self.assertEqual(len(q_full), 739)

        # Remove 1 tooth (17, 0)
        q_del1, L0_del1 = vd.build_staircase_band_tooth_deleted(
            w, l, a1, a2, b, b, deleted_teeth_a1=[(17, 0)]
        )
        self.assertEqual(len(q_del1), 738)
        self.assertEqual(L0_full, L0_del1)

        # Removed tooth coordinates should not appear
        tooth_coord = [L0_full - 0.5 - 17, 0.5, 0.0]
        dists = [abs(p[0] - tooth_coord[0]) + abs(p[1] - tooth_coord[1]) for p in q_del1]
        self.assertGreater(min(dists), 0.5)

    def test_corrugated_tooth_deletion_geometry(self):
        w, l, a1, a2, b = 13, 25, 18, 17, 9
        # Remove 2 teeth (SW and NE perimeter teeth)
        q_del2, L0 = vd.build_corrugated_band_tooth_deleted(
            w, l, a1, a2, b, b, deleted_teeth_a1=[(17, 0), (0, 17)]
        )
        self.assertEqual(len(q_del2), 737)

    def test_small_pell17_core_tooth_deletion(self):
        # Fast w=6 Pell-17 core: b=4, a1=13, a2=12, l=18 (base n0=297)
        res_del, L0, meta = vd.build_pell17_tooth_deleted_core(
            w=6, deleted_teeth_a1=[(12, 0)], corrugated=True, iterations=15, clearance=5e-9
        )
        self.assertEqual(len(res_del.q), 296)
        sat_ok, sat_pen = pc.sat_check(res_del.q)
        self.assertTrue(sat_ok)
        self.assertLessEqual(res_del.L, L0)
        self.assertEqual(meta["teeth_deleted_count"], 1)

    def test_fast_falsification_gate_execution(self):
        # Run 2 iterations to quickly test the end-to-end pipeline and falsification evaluation
        summary = vd.run_falsification_test(uncorrugated=True, iterations=2)
        self.assertEqual(summary["target_n"], 909)
        self.assertEqual(summary["core_squares"], 738)
        self.assertEqual(summary["border_squares"], 171)
        self.assertEqual(summary["total_squares"], 909)
        self.assertTrue(summary["sat_ok"])
        # Compactness should be preserved: bounding box does not dilate
        self.assertFalse(summary["box_dilated"])
        self.assertTrue(summary["compactness_preserved"])
        # Falsification gate against public incumbent: record whether it beats public
        self.assertIn("falsified_against_public", summary)


if __name__ == "__main__":
    unittest.main()
