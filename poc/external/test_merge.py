import importlib.util
from pathlib import Path
import tempfile
import time
import unittest

spec = importlib.util.spec_from_file_location("external", Path(__file__).with_name("merge.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ExternalTests(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "submission.csv"
            p.write_text(text)
            return m.read_groups(p, 1024 * 1024)

    def test_salvage_complete_peer(self):
        rows, dropped = self.parse("id,x,y,deg\n001_0,s0,s0,s0\n002_0,s0,s0,s0\n")
        self.assertEqual([n for n, _, _ in rows], [1])
        self.assertIn(2, dropped)

    def test_duplicate_is_not_complete(self):
        rows, dropped = self.parse("id,x,y,deg\n002_0,s0,s0,s0\n002_0,s2,s0,s0\n")
        self.assertEqual(rows, [])

    def test_transport_original(self):
        rows, _ = self.parse("id,x,y,deg\n001_0,s0.123456789012345678901,s0,s0\n")
        self.assertEqual(rows[0][1], rows[0][2])
        self.assertEqual(rows[0][1][0]["x"], "0.123456789012345678901")

    def test_nonfinite_dropped(self):
        rows, _ = self.parse("id,x,y,deg\n001_0,snan,s0,s0\n")
        self.assertEqual(rows, [])

    def test_overlap_rejected(self):
        a = [{"square_id": i, "x": "0", "y": "0", "angle": "0"} for i in range(2)]
        self.assertIsNone(m.official(2, a, a))

    def test_hard_deadline(self):
        start = time.monotonic()
        with self.assertRaises(TimeoutError):
            m.bounded(time.sleep, (10,), .1)
        self.assertLess(time.monotonic()-start, 3)

    def test_size_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "submission.csv"
            p.write_text("id,x,y,deg\n")
            with self.assertRaises(ValueError):
                m.read_groups(p, 1)


if __name__ == "__main__":
    unittest.main()
