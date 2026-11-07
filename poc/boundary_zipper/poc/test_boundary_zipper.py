import unittest
import numpy as np
import boundary_zipper as z

class Tests(unittest.TestCase):
    def test_sat(self):
        a=np.array([.5,.5,0.])
        self.assertEqual(z.overlap(a,np.array([[1.5,.5,0.],[1.4,.5,0.],[3.,3.,.7]])).tolist(),
                         [False,True,False])
    def test_all_extrema_movable(self):
        q=z.array(z.grid(100))
        lo,hi=z.bounds(q)
        h=.5*(np.abs(np.cos(q[:,2]))+np.abs(np.sin(q[:,2])))
        extrema=np.any(np.isclose(q[:,:2]-h[:,None],lo) |
                       np.isclose(q[:,:2]+h[:,None],hi),axis=1)
        for variant in (0,1):
            mask=z.connected_cohort(q,z.seam_mask(q,lo,hi,variant))
            self.assertTrue(mask[extrema].all())
    def test_full_donor_identity_control(self):
        data=z.load(z.DEFAULTS[0],{907,909,971})
        for n,rows in data.items():
            q=z.array(rows)
            lo,hi=z.bounds(q)
            for variant in (0,1):
                old=q[z.connected_cohort(q,z.seam_mask(q,lo,hi,variant))]
                self.assertTrue(all(not z.overlap(a,old[i+1:]).any() for i,a in enumerate(old)),n)
    def test_constructed_fallback_count(self):
        for n in (908,910,972):
            rows=z.grid(n)
            self.assertEqual(len(rows),n)
            self.assertEqual({int(r["square_id"]) for r in rows},set(range(n)))

if __name__=="__main__":unittest.main()
