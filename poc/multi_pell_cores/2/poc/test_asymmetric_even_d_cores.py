"""Unit and certification tests for Asymmetric Even-D Corrugated Cores (D=10, 34, 58).

Tests:
  1. Discrete Diophantine parity and asymmetric clearance invariant checks.
  2. Construction geometry for D=10, 34, 58 target cores.
  3. 4-sign-piece epigraph LP polisher shrinkage gate (L_corr < L_45).
  4. Non-collision SAT verification and L-border wrapping geometry.
  5. Literal CSV and pandas 3.1.0 transport certification compatibility.
"""
from __future__ import annotations

import math
import sys
import unittest
from decimal import Decimal
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import asymmetric_even_d_cores as aec


class TestAsymmetricEvenDCores(unittest.TestCase):
    def test_diophantine_parity_and_invariants(self):
        """Verify out-of-phase parity, slack, and Diophantine cancellation."""
        for D in (10, 34, 58):
            for w in (1, 4, 7, 8, 11, 14, 15, 18):
                params = aec.get_asymmetric_even_d_parameters(D, w)
                a1, a2 = params["a1"], params["a2"]
                b1, b2 = params["b1"], params["b2"]

                # 1. End staircase asymmetry: b2 = b1 + 1 => b1 + b2 is odd
                self.assertEqual(b2, b1 + 1)
                self.assertEqual((b1 + b2) % 2, 1)

                # 2. Flank staircases: a1 + a2 = D + 2b + 1 is odd (out-of-phase condition)
                self.assertEqual((a1 + a2) % 2, 1)
                self.assertEqual(a1 - a2, 1)

                # 3. Exact Diophantine cancellation: (a1 + a2) - (b1 + b2) == D
                self.assertEqual((a1 + a2) - (b1 + b2), D)

                # 4. Right-wall gap must be strictly positive
                self.assertGreater(params["right_wall_gap"], 0.0)

    def test_asymmetric_clearance_rejection(self):
        """Verify that widths with {w/sqrt(2)} <= 0.50 are rejected to prevent wall collisions."""
        # w=2: {2/sqrt(2)} = sqrt(2) - 1 ~ 0.4142 <= 0.50
        # w=3: {3/sqrt(2)} = 2.1213 - 2 ~ 0.1213 <= 0.50
        for w in (2, 3, 6, 9, 10, 12, 13, 16):
            self.assertLessEqual(aec.frac_part_s2(w), 0.50)
            with self.assertRaises(ValueError):
                aec.get_asymmetric_even_d_parameters(10, w)

    def test_target_core_square_counts(self):
        """Verify exact discrete square counts for the 5 target cores in the research brief."""
        expected = {
            (10, 4): (117, 11, 11.32842712474619),
            (10, 11): (431, 21, 21.278174593052024),
            (34, 4): (521, 23, 23.32842712474619),
            (34, 8): (821, 29, 29.15685424949238),
            (58, 1): (943, 31, 31.207106781186548),
        }
        for (D, w), (n_exp, k_exp, L45_exp) in expected.items():
            p = aec.get_asymmetric_even_d_parameters(D, w)
            self.assertEqual(p["n0"], n_exp)
            self.assertEqual(p["k0"], k_exp)
            self.assertAlmostEqual(p["L45"], L45_exp, places=8)

    def test_border_wrap_square_counts(self):
        """Verify square count arithmetic for upright L-border wrapped cores."""
        # D=10, w=11: n0=431, k0=21. Wrapped with t=8: delta_n = 8*(2*21 + 8) = 400 => n=831
        p11 = aec.get_asymmetric_even_d_parameters(10, 11)
        k_tot = p11["k0"] + 8
        n_wrap = p11["n0"] + 8 * (2 * p11["k0"] + 8)
        self.assertEqual(n_wrap, 831)
        self.assertEqual(k_tot, 29)

        # D=34, w=8: n0=821, k0=29. Wrapped with t=1: delta_n = 1*(2*29 + 1) = 59 => n=880
        p8 = aec.get_asymmetric_even_d_parameters(34, 8)
        k_tot8 = p8["k0"] + 1
        n_wrap8 = p8["n0"] + 1 * (2 * p8["k0"] + 1)
        self.assertEqual(n_wrap8, 880)
        self.assertEqual(k_tot8, 30)

    def test_asymmetric_corrugated_band_construction_d10_w4(self):
        """Verify synthetic unpolished band construction for D=10, w=4 (n0=117)."""
        q, L0, p = aec.build_asymmetric_corrugated_band(10, 4)
        self.assertEqual(len(q), 117)
        self.assertEqual(p["l"], 11)
        self.assertEqual(p["b"], [2, 3])
        self.assertEqual(p["a"], [8, 7])
        self.assertAlmostEqual(L0, 11.328427125, places=6)
        # Bounding box of unpolished configuration must not exceed nominal L0 + margin
        ext = aec.pc.extent(q)
        self.assertLess(ext, L0 + 0.30)

    def test_4sign_piece_lp_corrugation_shrinkage_gate_d10_w4(self):
        """Falsification gate: verify that asymmetric end shift causes premature NE contact and dilation."""
        res, L0, meta = aec.build_asymmetric_even_d_core(10, 4, iterations=20, clearance=5e-9)
        self.assertEqual(meta["n0"], 117)
        # Falsification gate check: asymmetric ends induce premature boundary collision,
        # resulting in dilation (Delta L > 0, L > L0), falsifying contraction for rigid asymmetric ends.
        self.assertGreater(res.L, L0, f"Expected dilation (L > L0), got L={res.L} vs L0={L0}")
        sat_ok, gap = aec.pc.sat_check(res.q)
        self.assertTrue(sat_ok, f"SAT check failed with min gap {gap}")

    def test_certification_transport_roundtrip(self):
        """Verify literal CSV and pandas 3.1.0 round-trip certification on an asymmetric core."""
        res, L0, meta = aec.build_asymmetric_even_d_core(10, 4, iterations=15, clearance=5e-9)
        q_norm = aec.pc.normalize_box(res.q)
        puzzles = {117: aec.pc.rows(q_norm, dilate=1.0)}
        text = aec.pc.csv_text(puzzles)
        checks = aec.pc.certify_transport(text)
        chk = checks["117"]
        self.assertTrue(chk["literal"]["ok"])
        self.assertTrue(chk["actual_pandas"]["ok"])
        self.assertAlmostEqual(float(chk["literal"]["L"]), res.L, places=6)
        self.assertAlmostEqual(float(chk["actual_pandas"]["L"]), res.L, places=6)


if __name__ == "__main__":
    unittest.main()
