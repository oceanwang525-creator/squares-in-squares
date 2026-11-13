import importlib.util, tempfile, unittest
from unittest.mock import patch
from decimal import Decimal
from pathlib import Path
spec=importlib.util.spec_from_file_location("union",Path(__file__).with_name("union.py"))
u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
class UnionTests(unittest.TestCase):
 def test_transport_regression_is_excluded(self):
  limits=(Decimal("1"),Decimal("1"))
  self.assertFalse(u.qualifies(None,limits))
  self.assertFalse(u.qualifies(("1","1.000000000000000000000000000001"),limits))
  self.assertFalse(u.qualifies(("1","0.99"),limits))
  self.assertFalse(u.qualifies(("0.99","1"),limits))
  self.assertFalse(u.qualifies(("1","1"),limits))
  self.assertTrue(u.qualifies(("0.99","0.99"),limits))
 def test_complete(self):
  rows=[dict(square_id="0",x="0.5",y="0.5",angle="0")]
  self.assertIsNotNone(u.check((1,rows,rows)))
 def test_incomplete(self):
  self.assertIsNone(u.check((2,[],[])))
 def test_overlap(self):
  rows=[dict(square_id=str(i),x="0.5",y="0.5",angle="0") for i in range(2)]
  self.assertIsNone(u.check((2,rows,rows)))
 def test_roundtrip(self):
  rows=[dict(square_id="0",x="0.5",y="0.5",angle="0")]
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/"test.csv"; p.write_text(u.csv_text({1:rows}))
   a,b=u.load(p);self.assertIsNotNone(u.check((1,a[1],b[1])))
 def test_metadata_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/"test.csv";p.write_text("n,L\n1,1\n")
   with self.assertRaises(ValueError):u.load(p)

 def test_no_improvement_keeps_bytes(self):
  with tempfile.TemporaryDirectory() as td:
   base=Path(td)/"merged.csv"; original=b"unchanged bytes\r\n";base.write_bytes(original)
   report=dict(score=1,results=[dict(n=1,L="1")])
   with patch.object(u,"ROOT",Path(td)), patch.object(u,"baseline_checks",return_value=({1:[]},{1:(Decimal(1),Decimal(1))})), patch.object(u,"full_gate",return_value=[report,report]):
    u.union(base,1,Path(td))
   self.assertEqual(base.read_bytes(),original)
 def test_full_failure_never_writes(self):
  with tempfile.TemporaryDirectory() as td:
   base=Path(td)/"merged.csv";base.write_bytes(b"original")
   with patch.object(u,"ROOT",Path(td)), patch.object(u,"baseline_checks",return_value=({1:[]},{1:(Decimal(1),Decimal(1))})), patch.object(u,"full_gate",side_effect=ValueError("bad")):
    with self.assertRaises(ValueError):u.union(base,1,Path(td))
   self.assertEqual(base.read_bytes(),b"original")

 def test_fourcol_roundtrip(self):
  rows=[dict(square_id="0",x="0.5",y="0.5",angle="0")]
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/"final.csv";p.write_text(u.final_text({1:rows}))
   self.assertEqual(p.read_text().splitlines()[0],"id,x,y,deg")
   a,b=u.load(p);self.assertIsNotNone(u.check((1,a[1],b[1])))
 def test_header_only_loads(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/"final.csv";p.write_text("id,x,y,deg\n")
   self.assertEqual(u.load(p), ({}, {}))

 def test_union_skips_header_only(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td); final=root/"poc/empty/final.csv";final.parent.mkdir(parents=True);final.write_text("id,x,y,deg\n")
   base=root/"merged.csv"; original=b"unchanged bytes\r\n";base.write_bytes(original)
   report=dict(score=1,results=[dict(n=1,L="1")])
   with patch.object(u,"ROOT",root), patch.object(u,"baseline_checks",return_value=({1:[]},{1:(Decimal(1),Decimal(1))})), patch.object(u,"full_gate",return_value=[report,report]), patch.object(u,"certify_many") as cert:
    u.union(base,1,root)
    self.assertEqual(cert.call_args.args[0], [])
   self.assertEqual(base.read_bytes(), original)

 def test_export_zero_wins_replaces_snapshot_with_header(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td); folder=root/"poc/empty";folder.mkdir(parents=True)
   final=folder/"final.csv";final.write_text("snapshot")
   audit=root/"audit";audit.mkdir()
   base=root/"merged.csv";base.write_bytes(b"incumbent")
   with patch.object(u,"ROOT",root), patch.object(u,"baseline_checks",return_value=({1:[]},{1:(Decimal(1),Decimal(1))})), patch.object(u,"certify_many"):
    u.export(base,1,audit)
   self.assertEqual(final.read_bytes(),b"id,x,y,deg\n")
   self.assertEqual(base.read_bytes(),b"incumbent")

 def test_export_drops_transport_tie(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td); folder=root/"poc/one";folder.mkdir(parents=True)
   (folder/"source.csv").write_text("fixture")
   audit=root/"audit";audit.mkdir()
   base=root/"merged.csv";base.write_bytes(b"incumbent")
   rows=[dict(square_id="0",x="0.5",y="0.5",angle="0")]
   def certify(tasks, workers, cache):
    for task in tasks: cache[u.key(task)]=("0.9","1")
   with patch.object(u,"ROOT",root), patch.object(u,"baseline_checks",return_value=({1:rows},{1:(Decimal(1),Decimal(1))})), patch.object(u,"load",return_value=({1:rows},{1:rows})), patch.object(u,"certify_many",side_effect=certify):
    u.export(base,1,audit)
   self.assertEqual((folder/"final.csv").read_bytes(),b"id,x,y,deg\n")

 def test_fourcol_rejects_bad_groups(self):
  for text in ["id,x,y,deg\n002_0,s0,s0,s0\n", "id,x,y,deg\n002_0,s0,s0,s0\n002_0,s2,s0,s0\n", "id,x,y,deg\n001_0,nan,0,0\n", "id,x,y,deg\n1001_0,0,0,0\n"]:
   with tempfile.TemporaryDirectory() as td:
    p=Path(td)/"final.csv";p.write_text(text)
    with self.assertRaises(ValueError):u.load(p)
if __name__=="__main__": unittest.main()
