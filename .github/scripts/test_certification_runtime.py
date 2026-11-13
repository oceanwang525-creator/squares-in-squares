import importlib.util
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import certification_runtime as c
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "poc/baseline/poc/src"))
from squares import _official_metric as metric
from squares.core import grid


class RuntimeTests(unittest.TestCase):
    def make_runtime(self, groups=None):
        if groups is None:
            groups = {n: [{"square_id": "0", "x": str(n)}] for n in range(1, 1001)}
        union = SimpleNamespace(load=lambda path: (groups, groups), pd=SimpleNamespace(__version__="3.1.0"),
            key=c.group_key, full_gate=lambda path: None, metric=SimpleNamespace(_validate_puzzle=lambda n,rows: None))
        with patch.object(c, "context", return_value={}), patch.object(c, "restore", return_value={}):
            runtime = c.Runtime(union, workers=1, seconds=1)
        return runtime, groups

    def fake_job(self, task, seconds):
        n, rows = task
        return (n, float(n)/1000, str(n), "ok"), dict(n=n, seconds=0.1,
            cpu_seconds=0.1, peak_rss_kib=100, precision_contexts={50: 1})

    def test_exact_reuse_and_transport_changes_are_not_cached(self):
        runtime, groups = self.make_runtime()
        tasks = [(1, groups[1])]
        with patch.object(c, "official_job", self.fake_job), patch.object(c, "ProcessPoolExecutor") as pool:
            from concurrent.futures import Future
            def submit(fn, *args):
                f=Future(); f.set_result(fn(*args)); return f
            pool.return_value.__enter__.return_value.submit.side_effect=submit
            runtime.evaluate(tasks, "first")
            runtime.evaluate(tasks, "same")
            self.assertEqual(pool.return_value.__enter__.return_value.submit.call_count, 1)
            runtime.evaluate([(1, [{"square_id": "0", "x": "1.000000000001"}])], "changed")
            self.assertEqual(pool.return_value.__enter__.return_value.submit.call_count, 2)

    def test_full_gate_order_and_scores_match_official_aggregation(self):
        runtime, groups = self.make_runtime()
        for n, rows in groups.items():
            runtime.cache[c.group_key((n, rows))]=self.fake_job((n, rows), 1)[0]
        with patch.object(Path, "read_text", return_value="id,n,square_id,x,y,angle\n"):
            reports=runtime.full_gate(Path("unused.csv"))
        self.assertEqual(len(reports),2)
        self.assertEqual(reports[0]["score"],sum(float(n)/1000 for n in range(1,1001)))
        self.assertEqual(reports[0]["accepted"],1000)
        self.assertEqual(reports[0]["results"][-1]["n"],1000)

    def test_missing_group_and_invalid_result_fail_closed(self):
        runtime, groups=self.make_runtime()
        groups.pop(1)
        with patch.object(Path, "read_text", return_value="id,n,square_id,x,y,angle\n"), self.assertRaises(ValueError):runtime.full_gate(Path("missing.csv"))
        runtime, groups=self.make_runtime()
        for n, rows in groups.items():
            runtime.cache[c.group_key((n,rows))]=(n,0.0,None,"invalid")
        with patch.object(Path, "read_text", return_value="id,n,square_id,x,y,angle\n"), self.assertRaises(ValueError):runtime.full_gate(Path("invalid.csv"))

    def test_candidate_pair_reuses_literal_and_pandas_but_invalid_is_not_accepted(self):
        runtime, groups=self.make_runtime()
        a=grid(1)
        runtime.cache[c.group_key((1,a))]=(1,0.0,"1","ok")
        cache={}
        runtime.certify_many([(1,a,a)],1,cache)
        self.assertEqual(cache[c.group_key((1,a,a))],("1","1"))
        runtime.cache[c.group_key((1,a))]=(1,0.0,None,"overlap")
        cache={}
        runtime.certify_many([(1,a,a)],1,cache)
        self.assertIsNone(cache[c.group_key((1,a,a))])

    def test_deadline_is_fatal_and_precision_context_is_restored(self):
        def slow(task):
            try:time.sleep(1)
            except Exception:return (1,0.0,"1","ok")
        original=metric.working_dps
        with patch.object(metric,"_evaluate_puzzle_worker",slow):
            with self.assertRaises(TimeoutError):c.official_job((1,grid(1)),0.01)
        self.assertIs(metric.working_dps,original)

    def test_real_official_result_identical_instrumented(self):
        expected=metric._evaluate_puzzle_worker((9,grid(9)))
        result, info=c.official_job((9,grid(9)),10)
        self.assertEqual(result,expected)
        self.assertEqual(info["n"],9)
        self.assertGreaterEqual(info["cpu_seconds"],0)

    def test_real_process_pool_smoke(self):
        runtime, groups=self.make_runtime()
        a=grid(4)
        runtime.evaluate([(4,a)],"real-pool")
        self.assertEqual(runtime.cache[c.group_key((4,a))],metric._evaluate_puzzle_worker((4,a)))


if __name__=="__main__":unittest.main()
