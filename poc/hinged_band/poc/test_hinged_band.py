"""Local CPU invariants and actual-reference-pandas certificates."""
import unittest
import numpy as np
import hinged_band as h

class ConstructionTests(unittest.TestCase):
    def test_independent_hinge_and_slide(self):
        p, info = h.hinged(31,7,.015,.12,.25,1)
        q, _ = h.hinged(31,7,.015,-.12,.25,1)
        self.assertEqual(len(set(p[:,2])),2)
        self.assertTrue(min(info["half_counts"])>0)
        self.assertFalse(len(h.conflicts(p)))
        self.assertFalse(np.array_equal(p,q))

    def test_sourced_atom_gates(self):
        for atom,n in ((h.atom17,17),(h.atom37,37)):
            a=atom()
            self.assertEqual(len(a),n)
            c=h.certify_transport(h.csv_text({n:h.rows(a)}))
            self.assertTrue(all(v["ok"] for v in c[str(n)].values()))

    def test_constructive_fallback(self):
        puzzles={}
        for n in (908,910,972):
            p,info=h.fallback(n)
            self.assertEqual(len(p),n)
            self.assertEqual(info["atoms"],["Hamalainen17","Cantrell37"])
            self.assertFalse(len(h.conflicts(p)))
            puzzles[n]=h.rows(p)
        c=h.certify_transport(h.csv_text(puzzles))
        self.assertTrue(all(v["ok"] for e in c.values() for v in e.values()))

    def test_final_artifact(self):
        self.assertEqual(h.pd.__version__,"3.1.0")
        text=(h.HERE/"output.csv").read_text()
        ps=h.metric._parse_submission_csv(text)
        self.assertEqual({n:len(v) for n,v in ps.items()},{908:908,910:910,972:972})
        c=h.certify_transport(text)
        self.assertTrue(all(v["ok"] for e in c.values() for v in e.values()))

if __name__=="__main__":
    unittest.main()
