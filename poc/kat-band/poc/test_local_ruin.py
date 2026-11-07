import unittest
from unittest.mock import patch
from itertools import count
import numpy as np
from local_ruin import local_ruin, _clear
from band_polish import sat_check


class LocalRuinTests(unittest.TestCase):
    def test_exact_n_from_n_minus_one(self):
        # Large empty lower-left patch, with immutable upright obstacles.
        source = np.array([[3.5,3.5,0.], [2.5,3.5,0.]])
        before = source.tobytes()
        r = local_ruin(source, 4., 3, attempt_cap=60, candidate_cap=30,
                       time_limit=3.)
        self.assertEqual(r['meta']['reason'], 'ok')
        x = r['x']
        self.assertEqual(x.shape, (3,3))
        self.assertEqual(x.dtype, np.float64)
        self.assertTrue(x.flags.c_contiguous)
        self.assertTrue(sat_check(x)[0])
        self.assertEqual(x[:2].tobytes(), before)
        self.assertEqual(source.tobytes(), before)
        self.assertEqual(r['meta']['removed_indices'], [])

    def test_tilted_ruin_outside_untouched(self):
        source = np.array([[.8,.8,.2], [3.5,3.5,0.]])
        before = source.copy()
        r = local_ruin(source, 4., 2, remove_count=1, attempt_cap=60,
                       time_limit=3.)
        self.assertIsNotNone(r['x'], r['meta'])
        ids = r['meta']['outside_indices']
        self.assertEqual(r['x'][:len(ids)].tobytes(), source[ids].tobytes())
        np.testing.assert_array_equal(source, before)

    def test_removed_hint_recreated(self):
        source = np.array([[1.,1.,.2]])
        r = local_ruin(source, 2., 1, attempt_cap=40)
        self.assertEqual(r['meta']['removed_indices'], [0])
        self.assertIsNotNone(r['x'], r['meta'])
        self.assertEqual(len(r['x']), 1)
        np.testing.assert_array_equal(source, [[1.,1.,.2]])

    def test_low_admission_generation_budget_keeps_hint_for_milp(self):
        source = np.array([[.8,.8,.2], [3.5,3.5,0.]])
        hint = source[0].copy()
        # Deterministic clock and very low admission: only the removed hint
        # survives obstacle screening. Sampling must stop early, not consume
        # the overall deadline and throw the feasible hint away.
        ticks = count()
        with patch('local_ruin.time.monotonic', side_effect=lambda: next(ticks)*.002), \
             patch('local_ruin._clear', side_effect=lambda row, others: np.array_equal(row, hint)):
            r = local_ruin(source, 4., 2, time_limit=1., attempt_cap=100000,
                           candidate_cap=600)
        self.assertTrue(r['meta']['generation_budget_hit'])
        self.assertFalse(r['meta']['timed_out'])
        self.assertEqual(r['meta']['candidates'], 1)
        self.assertIsNotNone(r['meta']['milp_status'])
        self.assertEqual(r['meta']['reason'], 'ok')
        self.assertEqual(r['x'].shape, (2,3))
        self.assertTrue(sat_check(r['x'])[0])

    def test_impossible_never_returns_wrong_count(self):
        r = local_ruin([[.5,.5,0]], 1., 2, attempt_cap=20)
        self.assertIsNone(r['x'])

    def test_strict_overlap_rejected(self):
        source = np.array([[.5,.5,0], [1.5-1e-12,.5,0]])
        r = local_ruin(source, 2., 2)
        self.assertEqual(r['meta']['reason'], 'invalid_source')
        self.assertFalse(_clear(source[0], source[1:]))
        self.assertTrue(_clear([.5,.5,0], np.array([[1.5,.5,0]])))

    def test_budget_and_cap(self):
        r = local_ruin([], 4., 1, time_limit=1e-9)
        self.assertIsNone(r['x'])
        self.assertTrue(r['meta']['timed_out'])
        r = local_ruin([], 4., 1, candidate_cap=1, attempt_cap=10)
        self.assertTrue(r['meta']['candidate_cap_hit'])
        self.assertEqual(len(r['x']), 1)

    def test_bad_arguments(self):
        for kw in ({'target':2.5}, {'target':1, 'candidate_cap':0},
                   {'target':1, 'remove_count':33}):
            with self.assertRaises(ValueError): local_ruin([], 4., **kw)

    def test_deterministic(self):
        kw = dict(L=4., target=1, seed=7, attempt_cap=30)
        a = local_ruin([], **kw)
        b = local_ruin([], **kw)
        np.testing.assert_array_equal(a['x'], b['x'])


if __name__ == '__main__': unittest.main()
