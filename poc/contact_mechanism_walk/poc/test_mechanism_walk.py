from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "mechanism_walk", Path(__file__).with_name("mechanism_walk.py")
)
mw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mw)


class MechanismWalkTests(unittest.TestCase):
    def test_grid_contact_jacobian(self):
        x = np.array([0.5, 1.5, 0.5, 1.5])
        y = np.array([0.5, 0.5, 1.5, 1.5])
        theta = np.zeros(4)
        jac, info = mw.contact_jacobian(x, y, theta)
        self.assertGreaterEqual(info["pair_contacts"], 4)
        self.assertGreaterEqual(info["wall_contacts"], 8)
        self.assertEqual(jac.shape[1], 12)

    def test_mode_move_is_first_order_contact_preserving(self):
        # The first pair touches but sits well away from every wall. Moving
        # both members horizontally is therefore an exact contact mechanism.
        x = np.array([2.0, 3.0, 9.5])
        y = np.array([2.0, 2.0, 9.5])
        theta = np.zeros(3)
        jac, _ = mw.contact_jacobian(x, y, theta)
        mode = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertLess(np.max(np.abs(jac @ mode)), 1e-12)

    def test_pair_jacobian_matches_rotation_finite_difference(self):
        theta0, theta1 = 0.37, -0.21
        ax, ay = np.cos(theta0), np.sin(theta0)
        cth = np.cos(theta0) * np.cos(theta1) + np.sin(theta0) * np.sin(theta1)
        sth = np.cos(theta0) * np.sin(theta1) - np.sin(theta0) * np.cos(theta1)
        radius = 0.5 * (1.0 + abs(cth) + abs(sth))
        x = np.array([5.0, 5.0 + radius * ax])
        y = np.array([5.0, 5.0 + radius * ay])
        theta = np.array([theta0, theta1])
        jac, _ = mw.contact_jacobian(x, y, theta)

        def selected_gap(cx, cy, angle):
            dx, dy = cx[1] - cx[0], cy[1] - cy[0]
            c, s = np.cos(angle), np.sin(angle)
            axes = (
                (c[0], s[0]),
                (-s[0], c[0]),
                (c[1], s[1]),
                (-s[1], c[1]),
            )
            return max(
                abs(dx * axis_x + dy * axis_y)
                - float(mw._axis_half_extent(angle, axis_x, axis_y).sum())
                for axis_x, axis_y in axes
            )

        direction = np.array([0.4, -0.7, 0.2, 0.5, -0.9, 0.3])
        epsilon = 1e-6
        plus = selected_gap(
            x + epsilon * direction[:2],
            y + epsilon * direction[2:4],
            theta + epsilon * direction[4:],
        )
        minus = selected_gap(
            x - epsilon * direction[:2],
            y - epsilon * direction[2:4],
            theta - epsilon * direction[4:],
        )
        numerical = (plus - minus) / (2.0 * epsilon)
        analytic = float((jac[0] @ direction)[0])
        self.assertAlmostEqual(analytic, numerical, places=8)

    def test_soft_modes_have_full_state_length(self):
        x = np.array([0.5, 1.5, 0.5, 1.5])
        y = np.array([0.5, 0.5, 1.5, 1.5])
        theta = np.zeros(4)
        jac, _ = mw.contact_jacobian(x, y, theta)
        _, modes = mw.soft_modes(jac, k=2)
        self.assertEqual(modes.shape, (2, 12))
        self.assertAlmostEqual(float(np.linalg.norm(modes[0])), 1.0, places=10)

    def test_remove_rigid_motion_annihilates_rotation(self):
        x = np.array([0.5, 2.5, 5.0])
        y = np.array([1.5, 3.0, 0.5])
        cx, cy = x - np.mean(x), y - np.mean(y)
        rigid = np.concatenate((-cy, cx, np.ones(3)))
        residual = mw.remove_rigid_motion(rigid, x, y)
        self.assertLess(np.linalg.norm(residual), 1e-12)

    def test_header_only_final(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final.csv"
            mw.write_final({}, path)
            self.assertEqual(path.read_text(), "id,x,y,deg\n")


if __name__ == "__main__":
    unittest.main()
