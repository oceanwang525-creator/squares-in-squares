"""Unit tests for poc/swarm_mechwalk/poc/mechwalk_lib.py."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

POC_DIR = Path(__file__).resolve().parents[1] / "poc"
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import mechwalk_lib as ml  # noqa: E402


class MechwalkLibTests(unittest.TestCase):
    def test_exact_jacobian_finite_difference_pairs_and_walls(self):
        rng = np.random.default_rng(7)
        theta0, theta1, theta2 = 0.37, -0.21, 0.55
        ax, ay = np.cos(theta0), np.sin(theta0)
        cth = np.cos(theta0 - theta1)
        sth = np.sin(theta0 - theta1)
        rad = 0.5 * (1.0 + abs(cth) + abs(sth))

        h = ml.half_extent(np.array([theta0, theta1, theta2]))
        L = 5.0
        x = np.array([h[0], h[0] + rad * ax, L - h[2]])
        y = np.array([h[0], h[0] + rad * ay, L - h[2]])
        theta = np.array([theta0, theta1, theta2])

        jac, info = ml.contact_jacobian(x, y, theta, contact_tol=1e-5)
        self.assertEqual(info["pair_contacts"], 1)
        self.assertEqual(info["wall_contacts"], 4)

        def eval_gaps(cx, cy, cth_arr):
            ch = ml.half_extent(cth_arr)
            dx, dy = cx[1] - cx[0], cy[1] - cy[0]
            c, s = np.cos(cth_arr), np.sin(cth_arr)
            axes = ((c[0], s[0]), (-s[0], c[0]), (c[1], s[1]), (-s[1], c[1]))
            r = 0.5 * (1.0 + abs(np.cos(cth_arr[1] - cth_arr[0])) + abs(np.sin(cth_arr[1] - cth_arr[0])))
            pg = max(abs(dx * u + dy * v) - r for u, v in axes)
            return np.array([
                pg,
                cx[0] - ch[0],
                cy[0] - ch[0],
                L - cx[2] - ch[2],
                L - cy[2] - ch[2],
            ])

        direction = rng.normal(size=9)
        eps = 1e-6
        gp = eval_gaps(x + eps * direction[:3], y + eps * direction[3:6], theta + eps * direction[6:])
        gm = eval_gaps(x - eps * direction[:3], y - eps * direction[3:6], theta - eps * direction[6:])
        num = (gp - gm) / (2.0 * eps)
        ana = jac @ direction
        np.testing.assert_allclose(ana, num, atol=1e-7)

    def test_multi_pair_jacobian_all_rows_match_finite_difference(self):
        # 4 rotated squares forming multiple pair contacts (rows > 0)
        rng = np.random.default_rng(19)
        th = np.array([0.22, -0.35, 0.48, -0.17])
        x = np.array([1.0, 2.05, 1.05, 2.10])
        y = np.array([1.0, 1.05, 2.05, 2.10])
        jac, info = ml.contact_jacobian(x, y, th, contact_tol=0.35)
        self.assertGreaterEqual(info["pair_contacts"], 4)

        def active_pair_gaps(cx, cy, cth_arr):
            c, s = np.cos(cth_arr), np.sin(cth_arr)
            out = []
            for i in range(4):
                for j in range(i + 1, 4):
                    dx, dy = cx[j] - cx[i], cy[j] - cy[i]
                    if abs(dx) > 1.55 or abs(dy) > 1.55:
                        continue
                    r = 0.5 * (1.0 + abs(np.cos(cth_arr[j] - cth_arr[i])) + abs(np.sin(cth_arr[j] - cth_arr[i])))
                    axes = ((c[i], s[i]), (-s[i], c[i]), (c[j], s[j]), (-s[j], c[j]))
                    g = max(abs(dx * u + dy * v) - r for u, v in axes)
                    if g <= 0.35:
                        out.append(g)
            return np.array(out)

        g0 = active_pair_gaps(x, y, th)
        self.assertEqual(len(g0), info["pair_contacts"])
        direction = rng.normal(size=12)
        eps = 1e-6
        gp = active_pair_gaps(x + eps * direction[:4], y + eps * direction[4:8], th + eps * direction[8:])
        gm = active_pair_gaps(x - eps * direction[:4], y - eps * direction[4:8], th - eps * direction[8:])
        num = (gp - gm) / (2.0 * eps)
        ana = (jac @ direction)[: info["pair_contacts"]]
        np.testing.assert_allclose(ana, num, atol=1e-7)

    def test_header_only_final_csv(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "final.csv"
            ml.write_final_csv({}, p)
            self.assertEqual(p.read_text(encoding="utf-8"), "id,x,y,deg\n")

    def test_certify_canonical_and_write_final_n2(self):
        x = np.array([0.5, 1.5 + 1e-9])
        y = np.array([0.5, 0.5])
        th = np.array([0.0, 0.0])
        ver = ml.certify_canonical(2, x, y, th)
        self.assertTrue(ver["ok"])
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "final.csv"
            ml.write_final_csv({2: {"rows": ver["rows"]}}, p)
            text = p.read_text(encoding="utf-8")
            self.assertIn("002_0", text)
            self.assertIn("002_1", text)


if __name__ == "__main__":
    unittest.main()
