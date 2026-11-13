import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import normalize as n


class SavedPackingTests(unittest.TestCase):
    def test_metadata_is_not_coordinates(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "report.json"
            p.write_text('{"n": 1, "L": 1}')
            with self.assertRaises(ValueError):
                n.candidate(p)

    def test_coordinate_json(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "best_1.json"
            p.write_text('[{"square_id":"0","x":"0.5","y":"0.5","angle":"0"}]')
            a, b = n.candidate(p)
            self.assertEqual(len(a[1]), 1)
            self.assertIsNotNone(n.u.check((1, a[1], b[1])))

    def test_headerless_csv_keeps_first_row(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "slope_band/3/data/output.csv"
            p.parent.mkdir(parents=True)
            p.write_text("0,1,0,0.5,0.5,0\n")
            a, b = n.candidate(p)
            self.assertEqual(len(a[1]), 1)

    def test_npy_incomplete_does_not_become_smaller_problem(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "best_n635.npy"
            np.save(p, np.array([[0.5, 0.5, 0.0]]))
            with self.assertRaises(ValueError):
                n.candidate(p)

    def test_npy_pickle_forbidden(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "best_1.npy"
            np.save(p, np.array([{"x": 1}], dtype=object))
            with self.assertRaises(ValueError):
                n.candidate(p)

    def test_npy_container_column_not_angle(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "slp1.npy"
            np.save(p, np.array([[0.5, 0.5, 0.0, 1.0]]))
            a, b = n.candidate(p)
            self.assertEqual(a[1][0]["angle"], "0.0")
            self.assertIsNotNone(n.u.check((1, a[1], b[1])))

    def test_npz_q_meta(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "core-w1.npz"
            np.savez(p, q=np.array([[0.5, 0.5, 0.0]]), meta='{"w":1}')
            a, b = n.candidate(p)
            self.assertIsNotNone(n.u.check((1, a[1], b[1])))

    def test_npz_unknown_schema(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "partial.npz"
            np.savez(p, core=np.array([[0.5, 0.5, 0.0]]))
            with self.assertRaises(ValueError):
                n.candidate(p)

    def test_npz_pickle_forbidden(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "core-w1.npz"
            np.savez(p, q=np.array([{}], dtype=object), meta="{}")
            with self.assertRaises(ValueError):
                n.candidate(p)

    def test_keep_existing_dual_winner_on_tie(self):
        self.assertTrue(n.keep_existing(("2", "2"), ("2", "2"), (3, 3)))

    def test_keep_existing_on_mode_tradeoff(self):
        self.assertTrue(n.keep_existing(("2", "2"), ("1.9", "2.1"), (3, 3)))

    def test_keep_existing_if_proposed_missing(self):
        self.assertTrue(n.keep_existing(("2", "2"), None, (3, 3)))

    def test_replace_only_strict_dual_improvement(self):
        self.assertFalse(n.keep_existing(("2", "2"), ("1.9", "1.9"), (3, 3)))

    def test_do_not_keep_old_tie_or_invalid(self):
        self.assertFalse(n.keep_existing(("3", "3"), None, (3, 3)))
        self.assertFalse(n.keep_existing(None, ("2", "2"), (3, 3)))


if __name__ == "__main__":
    unittest.main()
