import importlib.util
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from decimal import Decimal
import io

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

    def test_merge_first_only_best_and_fallback(self):
        rows = lambda x: [{"square_id": 0, "x": str(x), "y": "0", "angle": "0"}]
        tasks = [("worse", 1, rows(2), rows(2)), ("best", 1, rows(1), rows(1))]
        calls = []
        def fake_cert(batch, *args):
            batch = list(batch)
            calls.append([t[0] for t in batch])
            for key, n, a, b in batch:
                yield key, None if key == "best" else [(1, 0., "2", "ok")]*2
        with patch.object(m, "score_only", side_effect=lambda rows: Decimal(rows[0]["x"])), patch.object(m, "certify", side_effect=fake_cert), patch.object(m, "bounded", return_value=[(1, 0., "1", "ok")]*2):
            results = list(m.merge_first(tasks, None, [{1: Decimal(3)}]*2, 1, 1, io.StringIO()))
        self.assertEqual(calls, [["best"], ["worse"]])
        self.assertEqual(results[0][0], "worse")

    def test_merge_first_does_not_certify_losers(self):
        rows = lambda x: [{"square_id": 0, "x": str(x), "y": "0", "angle": "0"}]
        tasks = [("worse", 1, rows(2), rows(2)), ("best", 1, rows(1), rows(1)), ("behind_main", 2, rows(9), rows(9))]
        calls = []
        def fake_cert(batch, *args):
            batch = list(batch)
            calls.extend(t[0] for t in batch)
            for key, n, a, b in batch:
                yield key, [(n, 0., "1", "ok")]*2
        with patch.object(m, "score_only", side_effect=lambda rows: Decimal(rows[0]["x"])), patch.object(m, "certify", side_effect=fake_cert), patch.object(m, "bounded", return_value=[(1, 0., "1", "ok")]*2):
            list(m.merge_first(tasks, None, [{1: Decimal(3), 2: Decimal(4)}]*2, 1, 1, io.StringIO()))
        self.assertEqual(calls, ["best"])

    def test_size_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "submission.csv"
            p.write_text("id,x,y,deg\n")
            with self.assertRaises(ValueError):
                m.read_groups(p, 1)


if __name__ == "__main__":
    unittest.main()
