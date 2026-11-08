"""Unit and gate tests for Vacancy Deletion POC 2: Band-column end deletion."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "poc/pell_corrugation/poc"))

import band_vacancy as bv
import pell_corrugation as pc


class BandVacancyTests(unittest.TestCase):
    def test_find_pinch_columns(self):
        pinch_cols, du_se, du_nw = bv.find_pinch_columns()
        self.assertGreater(len(pinch_cols), 0)
        self.assertIn(6, pinch_cols)
        self.assertIn(18, pinch_cols)

    def test_vacancy_core_square_counts(self):
        # 1. Intact Pell-17 core has 739 squares
        q0, L0, meta0 = bv.build_band_vacancy_core(deletions=[])
        self.assertEqual(len(q0), 739)
        self.assertEqual(meta0["deleted_count"], 0)

        # 2. Deleting 1 square from column 0 end yields 738 squares
        del1 = [bv.BandDeletionSpec(column=0, square=0, label="c0_s0")]
        q1, L1, meta1 = bv.build_band_vacancy_core(deletions=del1)
        self.assertEqual(len(q1), 738)
        self.assertEqual(meta1["deleted_count"], 1)

        # 3. Deleting 2 squares from column 0 (both ends) yields 737 squares
        del2 = [
            bv.BandDeletionSpec(column=0, square=0, label="c0_s0"),
            bv.BandDeletionSpec(column=0, square=12, label="c0_s12"),
        ]
        q2, L2, meta2 = bv.build_band_vacancy_core(deletions=del2)
        self.assertEqual(len(q2), 737)
        self.assertEqual(meta2["deleted_count"], 2)

        # 4. Deleting 1 square from pinch column yields 738 squares
        pinch_cols, _, _ = bv.find_pinch_columns()
        del_pinch = [bv.BandDeletionSpec(column=pinch_cols[0], square=0, label="pinch_s0")]
        qp, Lp, metap = bv.build_band_vacancy_core(deletions=del_pinch)
        self.assertEqual(len(qp), 738)

    def test_border_wrapping_target_counts(self):
        del1 = [bv.BandDeletionSpec(column=0, square=0, label="c0_s0")]
        q1, _, _ = bv.build_band_vacancy_core(deletions=del1)

        # t=3 border wraps 738 core into 909
        q_909 = bv.wrap_vacancy_core(q1, t=3, k_total=30)
        self.assertEqual(len(q_909), 909)

        # t=4 border wraps 738 core with 1 extra square into 971
        q_971 = bv.wrap_vacancy_core(q1, t=4, k_total=31, extra_squares=1)
        self.assertEqual(len(q_971), 971)

        # Deleting 2 squares and wrapping with t=3 yields 908
        del2 = [
            bv.BandDeletionSpec(column=0, square=0, label="c0_s0"),
            bv.BandDeletionSpec(column=0, square=12, label="c0_s12"),
        ]
        q2, _, _ = bv.build_band_vacancy_core(deletions=del2)
        q_908 = bv.wrap_vacancy_core(q2, t=3, k_total=30)
        self.assertEqual(len(q_908), 908)

    def test_falsification_gate_and_blocker(self):
        """Falsification test: band-column end deletion cannot shrink global L below incumbents."""
        del1 = [bv.BandDeletionSpec(column=0, square=0, label="c0_s0")]
        q1, L0, _ = bv.build_band_vacancy_core(deletions=del1, collapse_shift=0.25)

        # Polish core (fast 5 iterations)
        res_core = pc.polish_band(q1, L0, iterations=5, clearance=5e-9)
        self.assertIsNotNone(res_core)
        valid, gap = pc.sat_check(res_core.q)
        self.assertTrue(valid)
        L_core = res_core.L

        # Wrap with t=3 for n=909
        q_909 = bv.wrap_vacancy_core(res_core.q if res_core is not None else q1, t=3, k_total=30)
        L_909 = pc.extent(q_909)

        # Incumbent for n=909 is 30.65685425
        inc_909 = float(bv.INCUMBENTS[909])
        # Delta must be positive (fails gate): container side length is held by the other
        # 24 intact columns that bridge rigidly between flank staircases a1 and a2.
        self.assertGreater(L_909, inc_909)
        delta_L = L_909 - inc_909
        self.assertGreater(delta_L, 0.030, f"Expected L_909 ({L_909}) to exceed incumbent ({inc_909}) by >0.030")


if __name__ == "__main__":
    unittest.main()
