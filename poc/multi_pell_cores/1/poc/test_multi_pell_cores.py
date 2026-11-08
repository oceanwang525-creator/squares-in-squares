"""Unit tests for multi_pell_cores POC 1: Symmetric odd-D corrugated cores (D=3, 27, 51)."""
from __future__ import annotations

import math
import sys
import unittest
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multi_pell_cores as mpc


class MultiPellCoresTests(unittest.TestCase):
    def test_out_of_phase_parity_theorem(self):
        """Verify that all symmetric odd-D cores have odd (a1 + a2) out-of-phase staircases."""
        for D in (3, 27, 51):
            for w in mpc.SWEET_SPOT_WIDTHS[D]:
                p = mpc.get_symmetric_odd_d_params(D, w)
                self.assertEqual((p.a1 + p.a2) % 2, 1, f"Failed parity for D={D}, w={w}")
                self.assertEqual(p.a1 - p.a2, 1, f"Failed symmetry for D={D}, w={w}")
                self.assertEqual(p.b1, p.b2, f"Failed symmetric ends for D={D}, w={w}")
                self.assertEqual((p.a1 + p.a2) - (p.b1 + p.b2), D, f"D mismatch for D={D}, w={w}")

    def test_d3_sweet_spot_parameters(self):
        """Verify Diophantine D=3 parameters across all feasible widths."""
        expected = {
            2: dict(b=1, a1=3, a2=2, l=4, k0=4, n0=19, L45=4.914213562),
            3: dict(b=2, a1=4, a2=3, l=5, k0=6, n0=37, L45=6.621320344),
            6: dict(b=4, a1=6, a2=5, l=8, k0=10, n0=104, L45=10.742640687),
            9: dict(b=6, a1=8, a2=7, l=11, k0=14, n0=205, L45=14.863961031),
            10: dict(b=7, a1=9, a2=8, l=12, k0=16, n0=257, L45=16.571067812),
            13: dict(b=9, a1=11, a2=10, l=15, k0=20, n0=406, L45=20.692388155),
            16: dict(b=11, a1=13, a2=12, l=18, k0=24, n0=589, L45=24.813708499),
            20: dict(b=14, a1=16, a2=15, l=22, k0=30, n0=906, L45=30.642135624),
        }
        for w, exp in expected.items():
            p = mpc.get_symmetric_odd_d_params(3, w)
            self.assertEqual(p.b1, exp["b"])
            self.assertEqual(p.a1, exp["a1"])
            self.assertEqual(p.a2, exp["a2"])
            self.assertEqual(p.l, exp["l"])
            self.assertEqual(p.k0, exp["k0"])
            self.assertEqual(p.n0, exp["n0"])
            self.assertAlmostEqual(p.L45, exp["L45"], places=7)
            self.assertAlmostEqual(p.slack, 3.0 / math.sqrt(2.0) - 2.0, places=9)

    def test_d27_sweet_spot_parameters(self):
        """Verify Hybrid D=27 parameters across all feasible widths."""
        expected = {
            2: dict(b=1, a1=15, a2=14, l=21, k0=16, n0=269, L45=16.914213562),
            3: dict(b=2, a1=16, a2=15, l=22, k0=18, n0=328, L45=18.621320344),
            6: dict(b=4, a1=18, a2=17, l=25, k0=22, n0=494, L45=22.742640687),
            9: dict(b=6, a1=20, a2=19, l=28, k0=26, n0=694, L45=26.863961031),
            10: dict(b=7, a1=21, a2=20, l=29, k0=28, n0=787, L45=28.571067812),
        }
        for w, exp in expected.items():
            p = mpc.get_symmetric_odd_d_params(27, w)
            self.assertEqual(p.b1, exp["b"])
            self.assertEqual(p.a1, exp["a1"])
            self.assertEqual(p.a2, exp["a2"])
            self.assertEqual(p.l, exp["l"])
            self.assertEqual(p.k0, exp["k0"])
            self.assertEqual(p.n0, exp["n0"])
            self.assertAlmostEqual(p.L45, exp["L45"], places=7)
            self.assertAlmostEqual(p.slack, 27.0 / math.sqrt(2.0) - 19.0, places=9)

    def test_d51_sweet_spot_parameters(self):
        """Verify Triple-Pell D=51 parameters and n0=907 sub-record geometry."""
        expected = {
            2: dict(b=1, a1=27, a2=26, l=38, k0=28, n0=807, L45=28.914213562),
            3: dict(b=2, a1=28, a2=27, l=39, k0=30, n0=907, L45=30.621320344),
        }
        for w, exp in expected.items():
            p = mpc.get_symmetric_odd_d_params(51, w)
            self.assertEqual(p.b1, exp["b"])
            self.assertEqual(p.a1, exp["a1"])
            self.assertEqual(p.a2, exp["a2"])
            self.assertEqual(p.l, exp["l"])
            self.assertEqual(p.k0, exp["k0"])
            self.assertEqual(p.n0, exp["n0"])
            self.assertAlmostEqual(p.L45, exp["L45"], places=7)
            self.assertAlmostEqual(p.slack, 51.0 / math.sqrt(2.0) - 36.0, places=9)

    def test_band_construction_d3_w2(self):
        """Verify geometric band construction for D=3, w=2 (n0=19)."""
        p = mpc.get_symmetric_odd_d_params(3, 2)
        q, L0 = mpc.build_corrugated_band(p.w, p.l, p.a1, p.a2, p.b1, p.b2, delta=0.0035)
        self.assertEqual(len(q), 19)
        self.assertAlmostEqual(L0, 4.914213562, places=7)
        self.assertLessEqual(mpc.extent(q), L0 + 0.10)

    def test_band_construction_d51_w3_seed907(self):
        """Verify n0=907 core seed generation for D=51, w=3 with L45 < public n908 record."""
        p = mpc.get_symmetric_odd_d_params(51, 3)
        q, L0 = mpc.build_corrugated_band(p.w, p.l, p.a1, p.a2, p.b1, p.b2, delta=0.0028)
        self.assertEqual(len(q), 907)
        self.assertAlmostEqual(L0, 30.621320344, places=7)
        # 30.621320344 is algebraically smaller than public n=908 incumbent 30.63702956
        self.assertLess(L0, 30.63702956)

    def test_l_border_wrapping_d27(self):
        """Verify wrapping D=27, w=6 core (n0=494) with t=8 upright border produces n=910."""
        p = mpc.get_symmetric_odd_d_params(27, 6)
        self.assertEqual(p.n0, 494)
        self.assertEqual(p.k0, 22)
        # N = n0 + 2*k0*t + t^2 = 494 + 2*22*8 + 64 = 494 + 352 + 64 = 910
        t = 8
        expected_total = p.n0 + 2 * p.k0 * t + t * t
        self.assertEqual(expected_total, 910)

        q, L0 = mpc.build_corrugated_band(p.w, p.l, p.a1, p.a2, p.b1, p.b2, delta=0.0030)
        wrapped = mpc.add_l_border(q, t=t, k_total=p.k0 + t)
        self.assertEqual(len(wrapped), 910)


if __name__ == "__main__":
    unittest.main()
