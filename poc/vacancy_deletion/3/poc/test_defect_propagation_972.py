"""Unit tests for vacancy deletion POC 3: Defect propagation on public n=972 incumbent."""
import math
import sys
import unittest
from decimal import Decimal
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import defect_propagation_972 as dp


class DefectPropagation972Tests(unittest.TestCase):
    def test_incumbent_structure_and_algebraic_decomposition(self):
        q = dp.load_incumbent_972()
        self.assertEqual(len(q), 972)

        struct = dp.analyze_incumbent_structure(q)
        self.assertEqual(struct["upright_count"], 739)
        self.assertEqual(struct["tilted_count"], 233)
        self.assertEqual(len(struct["columns"]), 13)
        self.assertEqual(
            struct["column_lengths"],
            [13, 13, 21, 21, 21, 21, 21, 21, 21, 21, 13, 13, 13],
        )

        # Uncorrugated extent must match exact L45 + 4
        # L45 = (18 + 17)/2 + 1 + 13/sqrt(2) = 27.692388155425... -> + 4 = 31.692388155425...
        L_measured = dp.extent(q)
        self.assertAlmostEqual(L_measured, 31.692388155428, places=5)

    def test_defect_candidate_generation_971_and_970(self):
        q = dp.load_incumbent_972()
        struct = dp.analyze_incumbent_structure(q)

        # 1-square deletion (n=971)
        q_971, del_ids_1 = dp.create_defect_candidate(
            q, struct, delete_count=1, col_idx=0, end="min", propagate_shift=False
        )
        self.assertEqual(len(q_971), 971)
        self.assertEqual(len(del_ids_1), 1)

        # 2-square deletion (n=970)
        q_970, del_ids_2 = dp.create_defect_candidate(
            q, struct, delete_count=2, col_idx=0, end="min", propagate_shift=False
        )
        self.assertEqual(len(q_970), 970)
        self.assertEqual(len(del_ids_2), 2)

    def test_defect_shift_and_sat_preservation(self):
        q = dp.load_incumbent_972()
        struct = dp.analyze_incumbent_structure(q)

        # Create candidate with defect shift applied
        q_cand, _ = dp.create_defect_candidate(
            q, struct, delete_count=1, col_idx=0, end="min", propagate_shift=True
        )
        self.assertEqual(len(q_cand), 971)

        # Verify candidate normalization and bounding box
        q_norm = dp.normalize_box(q_cand, margin=5e-9)
        self.assertGreater(dp.extent(q_norm), 31.0)
        self.assertLess(dp.extent(q_norm), 32.0)

    def test_csv_formatting_roundtrip(self):
        q = dp.load_incumbent_972()[:10]
        puzzle_rows = dp.rows(q)
        self.assertEqual(len(puzzle_rows), 10)
        self.assertIn("square_id", puzzle_rows[0])
        self.assertIn("x", puzzle_rows[0])
        self.assertIn("y", puzzle_rows[0])
        self.assertIn("angle", puzzle_rows[0])

        csv_out = dp.csv_text({10: puzzle_rows})
        self.assertTrue(csv_out.startswith("id,n,square_id,x,y,angle"))


if __name__ == "__main__":
    unittest.main()
