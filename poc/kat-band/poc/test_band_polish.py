import unittest
import numpy as np
from band_polish import (sat_check,extent,chosen_axis,piece_gap_gradient,
                         fixed_angle_polish,free_angle_polish,acceptable)

def grid(pitch=1.12,angle=0.):
    q=np.array([[x*pitch+.5,y*pitch+.5,angle] for y in range(2) for x in range(2)])
    c,s=np.cos(angle),np.sin(angle)
    q[:,:2]=q[:,:2]@np.array([[c,s],[-s,c]])
    return q

class Tests(unittest.TestCase):
    def test_exact_touch_grid(self):
        q=grid(1.)
        self.assertTrue(sat_check(q)[0])
        out=fixed_angle_polish(q)
        self.assertLessEqual(out.L,extent(q))
        self.assertTrue(sat_check(out.q)[0])

    def test_fixed_strict_improvement(self):
        q=grid()
        result=fixed_angle_polish(q)
        self.assertTrue(sat_check(result.q)[0])
        self.assertLess(result.L,extent(q)-.05)
        np.testing.assert_array_equal(q[:,2],result.q[:,2])
        self.assertTrue(all(a["L"]>=b["L"] for a,b in zip(result.history,result.history[1:])))

    def test_rotated_fixture_radians(self):
        q=grid(angle=.213)
        self.assertTrue(sat_check(q)[0])
        result=fixed_angle_polish(q)
        self.assertLessEqual(result.L,extent(q))
        self.assertTrue(sat_check(result.q)[0])

    def test_free_single_rotated_square(self):
        q=np.array([[20.,-3.,.27]])
        result=free_angle_polish(q,iterations=15,angle_trust=.025)
        self.assertTrue(sat_check(result.q)[0])
        self.assertLess(result.L,extent(q)-.05)
        self.assertGreater(result.accepted,0)

    def test_free_kink_aligned_and_rotated(self):
        for angle in (0.,.21):
            q=grid(angle=angle)
            result=free_angle_polish(q,iterations=6)
            self.assertTrue(sat_check(result.q)[0])
            self.assertLessEqual(result.L,extent(q))

    def test_derivatives_all_axis_owners_signs(self):
        q=np.array([[1.1,.3,.173],[2.2,1.7,.641]])
        for owner in (0,1):
            for axisdim in (0,1):
                for direction in (-1,1):
                    axis=(owner,axisdim,direction)
                    for signs in ((-1,-1),(-1,1),(1,-1),(1,1)):
                        _,grad=piece_gap_gradient(q,0,1,axis,signs)
                        numerical=[]
                        for k in range(6):
                            a=q.copy();b=q.copy()
                            a.flat[k]+=1e-6;b.flat[k]-=1e-6
                            numerical.append((piece_gap_gradient(a,0,1,axis,signs)[0]-
                                              piece_gap_gradient(b,0,1,axis,signs)[0])/2e-6)
                        np.testing.assert_allclose(grad,numerical,atol=2e-9,rtol=2e-9)

    def test_all_four_sign_pieces_equal_support(self):
        q=np.array([[0.,0.,.3],[1.7,.9,.3]])
        axis=chosen_axis(q,0,1)
        # At aligned relative angles, both signs of sin are required at the kink.
        gaps=[piece_gap_gradient(q,0,1,axis,s)[0]
              for s in ((-1,-1),(-1,1),(1,-1),(1,1))]
        owner,k,direction=axis
        phi=q[owner,2]+k*np.pi/2
        a=direction*np.array([np.cos(phi),np.sin(phi)])
        r=q[1-owner,2]-phi
        exact=(q[1,:2]-q[0,:2])@a-.5-.5*(abs(np.cos(r))+abs(np.sin(r)))
        self.assertAlmostEqual(min(gaps),exact,places=14)

    def test_invalid_rejected_even_if_L_better(self):
        q=np.array([[0.,0.,0.],[.9,0.,0.]])
        self.assertFalse(sat_check(q)[0])
        self.assertFalse(acceptable(q,100.))
        with self.assertRaises(ValueError): fixed_angle_polish(q)
        with self.assertRaises(ValueError): free_angle_polish(q)
        self.assertFalse(acceptable(np.array([[np.nan,0.,0.]]),100.))
        self.assertFalse(acceptable(grid(),1.))

    def test_external_certificate_gates_each_candidate(self):
        calls=[]
        def cert(q):
            calls.append(q.copy())
            return len(calls)==1
        q=grid()
        result=free_angle_polish(q,validator=cert)
        np.testing.assert_array_equal(result.q,q)
        self.assertGreater(len(calls),1)

    def test_near_parallel_non_axis_separation(self):
        q=np.array([[0.,0.,np.pi/4],[.73,.73,np.pi/4]])
        self.assertTrue(sat_check(q)[0])
        q[1,:2]=.6
        self.assertFalse(sat_check(q)[0])

if __name__=="__main__":
    unittest.main()
