import io
import tarfile
import tempfile
import subprocess
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import certify_hour as c


def rows(x):
    return [{"square_id": i, "x": str(a), "y": "0", "angle": "0"}
            for i, a in enumerate(x)]


def mock_modes(L="3", passed=True):
    return {k: {"2": {"official": {"reason": "ok", "L": L},
                        "cpp": {"reason": "ok", "L": L, "independent_trusted": True},
                        "screen": {"passed": passed}}}
            for k in ("literal", "pandas310")}


class CertificationTests(unittest.TestCase):
    def test_touch_and_overlap(self):
        self.assertTrue(c.strict_screen(rows([0, 1]))["passed"])
        self.assertFalse(c.strict_screen(rows([0, .999999999999]))["passed"])

    def test_far_pair_allpairs_count(self):
        self.assertEqual(c.strict_screen(rows([0, 2, 4]))["all_pairs"], 3)

    def test_no_equality_win(self):
        self.assertFalse(c.full_win(mock_modes(), mock_modes(), 2))
        self.assertTrue(c.full_win(mock_modes("2"), mock_modes("3"), 2))

    def test_strict_reject_even_official_cpp_pass(self):
        self.assertFalse(c.full_win(mock_modes("2", False), mock_modes(), 2))

    def test_one_transport_rejected(self):
        m = mock_modes("2")
        m["pandas310"]["2"]["official"]["reason"] = "overlap"
        self.assertFalse(c.full_win(m, mock_modes(), 2))

    def test_one_transport_worse(self):
        m = mock_modes("2")
        m["pandas310"]["2"]["official"]["L"] = "4"
        self.assertFalse(c.full_win(m, mock_modes(), 2))

    def cpp_rows(self, rs):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"q.csv"
            c.write_csv(p, {2: rs})
            output = Path(d)/"out.json"
            r = subprocess.run([c.DEFAULT_CPP, str(p), "--details", str(output)],
                               capture_output=True)
            return r.returncode, json.loads(output.read_text()) if output.exists() else None

    def test_cpp_sub_1e10_overlap(self):
        rc, obj = self.cpp_rows(rows([0, ".999999999999"]))
        self.assertEqual(rc, 0)
        self.assertEqual(obj["results"][0]["reason"], "strict allpair overlap")

    def test_cpp_touch(self):
        rc, obj = self.cpp_rows(rows([0, 1]))
        self.assertEqual(obj["results"][0]["reason"], "ok")

    def test_cpp_rotated_separated_overlap(self):
        rs = rows([0, 4])
        for r in rs:
            r["angle"] = ".7853981633974483"
        self.assertEqual(self.cpp_rows(rs)[1]["results"][0]["reason"], "ok")
        rs[1]["x"] = ".4"
        self.assertEqual(self.cpp_rows(rs)[1]["results"][0]["reason"], "strict allpair overlap")

    def test_cpp_counts_nan_ids(self):
        self.assertEqual(self.cpp_rows(rows([0]))[1]["results"][0]["reason"], "exact count mismatch")
        rs = rows([0, 1]); rs[1]["x"] = "nan"
        self.assertEqual(self.cpp_rows(rs)[0], 2)
        rs = rows([0, 1]); rs[1]["square_id"] = 0
        self.assertEqual(self.cpp_rows(rs)[1]["results"][0]["reason"], "invalid square ids")

    def test_cpp_untrusted_cannot_win(self):
        m = mock_modes("2")
        m["literal"]["2"]["cpp"]["independent_trusted"] = False
        self.assertFalse(c.full_win(m, mock_modes(), 2))

    def test_bad_version(self):
        with patch.object(c.pd, "__version__", "3.0.6"):
            with self.assertRaises(RuntimeError):
                c.require_transport()

    def test_csv_roundtrip_literal(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"x.csv"
            c.write_csv(p, {2: rows([0, 1])})
            self.assertEqual(c.load_csv(p)[2][1]["x"], "1")

    def test_unsafe_tar(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"bad.tar"
            with tarfile.open(p, "w") as t:
                info = tarfile.TarInfo("../escape")
                info.size = 1
                t.addfile(info, io.BytesIO(b"x"))
            with self.assertRaises(ValueError):
                c.unpack(p, d)

    def test_baseline_guard(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"bad.csv"
            p.write_text("bad")
            with self.assertRaises(ValueError):
                c.prepare(p, Path(d)/"out", "/nonexistent")


if __name__ == "__main__":
    unittest.main()
