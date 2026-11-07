import unittest
import numpy as np
from core_refill import refill, _valid


class RefillTests(unittest.TestCase):
    def test_identity_grid(self):
        core=np.empty((0,3),dtype=np.float64)
        old=np.array([[x+.5,y+.5] for x in range(3) for y in range(2)])
        before=old.copy()
        r=refill(core,3.,2.,6,old_uprights=old)
        self.assertEqual(r.diagnostics['reason'],'ok')
        self.assertEqual(r.uprights.shape,(6,3))
        self.assertEqual(r.uprights.dtype,np.float64)
        self.assertTrue(_valid(r.uprights,3,2))
        np.testing.assert_array_equal(old,before)

    def test_tilted_surround_identity(self):
        core=np.array([[2.5,2.5,np.pi/4]])
        old=np.array([[x+.5,y+.5] for x in range(5) for y in range(5)
                      if x in (0,4) or y in (0,4)],dtype=float)
        before=core.copy()
        r=refill(core,5,5,16,hints=old,time_limit=10)
        self.assertEqual(r.diagnostics['reason'],'ok')
        self.assertTrue(_valid(np.vstack((core,r.uprights)),5,5))
        np.testing.assert_array_equal(r.x,np.vstack((core,r.uprights)))
        np.testing.assert_array_equal(core,before)

    def test_impossible_count_refuses(self):
        r=refill([],1,1,2)
        self.assertIsNone(r.uprights)
        self.assertTrue(r.diagnostics['graph_optimal'])
        self.assertEqual(r.diagnostics['selected'],1)

    def test_cap_is_explicit(self):
        r=refill([],4,4,1,candidate_cap=2)
        self.assertTrue(r.diagnostics['candidate_cap_hit'])
        self.assertFalse(r.diagnostics['graph_optimal'])
        self.assertEqual(len(r.uprights),1)

    def test_no_overlap_mask(self):
        core=np.array([[.5,.5,0],[1.5-1e-12,.5,0]])
        r=refill(core,2,2,0)
        self.assertIsNone(r.uprights)
        self.assertEqual(r.diagnostics['reason'],'invalid_core')

    def test_budget(self):
        r=refill([],30,30,900,time_limit=1e-9)
        self.assertIsNone(r.uprights)
        self.assertTrue(r.diagnostics['timed_out'])

    def test_noninteger_walls(self):
        # Wall lattice candidates remain strictly compatible at noninteger W.
        r=refill([],2.4,2.4,5,time_limit=3)
        self.assertIsNone(r.uprights)
        self.assertTrue(r.diagnostics['graph_optimal'])
        self.assertEqual(r.diagnostics['selected'],4)

    def test_bad_arguments(self):
        for kw in ({'depth':5},{'time_limit':0},{'candidate_cap':0}):
            with self.assertRaises(ValueError): refill([],2,2,1,**kw)


if __name__=='__main__': unittest.main()
