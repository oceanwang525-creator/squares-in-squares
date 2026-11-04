"""Small CPU gate only. No broad enumeration or optimization."""
import json
import sys
import unittest
from pathlib import Path
import numpy as np
import mpmath as mp
from band_geometry import (BandSpec, generate, validate, high_precision_rows,
                           write_csv, enumerate_specs, overlap_matrix)

ROOT = Path(__file__).resolve().parent


def stenlund():
    return BandSpec(3+4*np.sqrt(2),3,8,offset=(-.25,-.25))


def wainwright():
    r=1/np.sqrt(2)
    d=r-2/3
    return BandSpec(3+8*r/3,2,4,offset=(-r/3,-r/3),
                    layer_shifts=((-d,-d),(0,-d),(0,d),(-d,d)))


class GeometryTests(unittest.TestCase):
    def test_published_topology(self):
        for spec, count, depths in [(stenlund(),66,(5,6,2,2)),
                                     (wainwright(),19,(2,3,1,1))]:
            p, spec=generate(spec)
            self.assertEqual(len(p),count)
            self.assertEqual(spec.corner_depths,depths)
            self.assertEqual(validate(p,spec.side),
                             dict(n=count,inside=True,overlap_pairs=0))

    def test_svg_oracle(self):
        # Source-authored coordinates are a test oracle, never generation input.
        sys.path.insert(0,"/home/kit/squares/tools")
        from parse_svg_packing import parse_svg
        for spec,n in [(stenlund(),66),(wainwright(),19)]:
            p,_=generate(spec)
            rr,*_=parse_svg(f"/home/kit/squares/svgs/square-{n}.svg",50)
            q=np.array([[float(r["x"]),float(r["y"])] for r in rr])
            distances=np.linalg.norm(p[:,None,:2]-q[None,:,:2],axis=-1)
            nearest=distances.argmin(axis=1)
            self.assertEqual(len(set(nearest)),n)
            self.assertLess(distances.min(axis=1).max(),3e-14)
            # Square angles agree modulo pi/2.
            angles=np.array([float(rr[i]["theta"]) for i in nearest])
            delta=(p[:,2]-angles+np.pi/4)%(np.pi/2)-np.pi/4
            self.assertLess(np.abs(delta).max(),1e-12)

    def test_float_screen(self):
        p=np.array([[0.,0.,0.],[1.,0.,0.],[.9,0.,0.]])
        self.assertFalse(overlap_matrix(p[:1],p[1:2]).any())
        self.assertTrue(overlap_matrix(p[:1],p[2:]).any())

    def test_reject_escape(self):
        with self.assertRaisesRegex(ValueError,"escapes"):
            generate(BandSpec(3.,3,10))

    def test_variable_angle_cap_and_phases(self):
        # Tiny API smoke, not a main generation sweep.
        spec=BandSpec(10.,2,8,theta=.65,offset=(-.2,-.1),
                      phase=(.1,.1),end_trims=((0,1),(1,0)),
                      grid_phases=((.03,.02),)*4)
        p,spec=generate(spec)
        self.assertTrue(validate(p,spec.side)["inside"])
        self.assertEqual(validate(p,spec.side)["overlap_pairs"],0)
        self.assertEqual(len(high_precision_rows(spec)),len(p))

    def test_lazy_enumerator(self):
        specs=enumerate_specs(10.,widths=(3,),offsets=((0.,0.),),
                              length_adjustments=(0,),cap_modes=("full",))
        first=next(specs)
        self.assertEqual(first.width,3)
        p,spec=generate(first)
        self.assertEqual(validate(p,spec.side)["overlap_pairs"],0)
        self.assertEqual(list(specs),[])


def artifacts():
    mp.mp.dps=90
    _,s66=generate(stenlund())
    _,s19=generate(wainwright())
    r=1/mp.sqrt(2)
    d=r-mp.mpf(2)/3
    rows66=high_precision_rows(s66,exact_side=3+4*mp.sqrt(2),
                              exact_theta=mp.pi/4)
    rows19=high_precision_rows(s19,exact_side=3+8*r/3,
                              exact_theta=mp.pi/4,
                              exact_offset=(-r/3,-r/3),
                              exact_layer_shifts=((-d,-d),(0,-d),(0,d),(-d,d)))
    write_csv(ROOT/"stenlund66.csv",{66:rows66})
    write_csv(ROOT/"wainwright19.csv",{19:rows19})
    write_csv(ROOT/"reproduced-bands.csv",{19:rows19,66:rows66})


if __name__=="__main__":
    artifacts()
    unittest.main()
