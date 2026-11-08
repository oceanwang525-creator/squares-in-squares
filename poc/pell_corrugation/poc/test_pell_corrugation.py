"""Unit tests and official literal + pandas 3.1.0 certification gates for pell_corrugation."""
import json
import sys
import unittest
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pell_corrugation as pc


class PellCorrugationTests(unittest.TestCase):
    def test_pell17_out_of_phase_corrugation_core(self):
        # Fast Pell-17 core at w=6, b=4, a=(13,12), l=18 (k0=17, n0=297)
        res, L0, meta = pc.build_pell17_core(6, iterations=25, clearance=5e-9)
        self.assertEqual(len(res.q), 297)
        self.assertTrue(pc.sat_check(res.q)[0])
        self.assertLess(res.L, L0 - 0.0013)
        self.assertEqual(meta["D"], 17)

    def test_saved_artifacts_and_official_certification(self):
        self.assertEqual(pc.pd.__version__, "3.1.0")
        out_csv = pc.HERE / "output.csv"
        pan_csv = pc.HERE / "output-pandas.csv"
        res_json = pc.HERE / "results.json"
        self.assertTrue(out_csv.exists())
        self.assertTrue(pan_csv.exists())
        self.assertTrue(res_json.exists())

        text = out_csv.read_text()
        puzzles = pc.metric._parse_submission_csv(text)
        expected_counts = {n: n for n in pc.ALL_BRIEF_TARGETS}
        self.assertEqual({n: len(v) for n, v in puzzles.items()}, expected_counts)

        checks = pc.certify_transport(text)
        for n in pc.ALL_BRIEF_TARGETS:
            entry = checks[str(n)]
            self.assertTrue(entry["literal"]["ok"], f"Literal cert failed for n={n}: {entry}")
            self.assertTrue(entry["actual_pandas"]["ok"], f"Pandas cert failed for n={n}: {entry}")

        # Verify certified win on n=910 against public/targeted incumbent
        inc_910 = Decimal(pc.PUBLIC_TARGET_INCUMBENTS[910])
        L_lit_910 = Decimal(checks["910"]["literal"]["L"])
        L_pan_910 = Decimal(checks["910"]["actual_pandas"]["L"])
        self.assertLess(L_lit_910 - inc_910, Decimal("-0.000039"))
        self.assertLess(L_pan_910 - inc_910, Decimal("-0.000039"))

        data = json.loads(res_json.read_text())
        self.assertEqual(data["public_incumbent_wins"], 1)
        self.assertEqual(data["portfolio_baseline_wins"], 10)


if __name__ == "__main__":
    unittest.main()
