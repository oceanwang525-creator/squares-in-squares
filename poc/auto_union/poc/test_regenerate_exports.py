import unittest
import regenerate_exports as e
class ExplicitRowsTests(unittest.TestCase):
 def row(self):return dict(x='0.5',y='0.5',angle='0')
 def test_explicit_n(self):self.assertEqual(len(e.row_groups({'n':1,'rows':[self.row()]})),1)
 def test_mapping_n(self):self.assertEqual(len(e.row_groups({'1':{'rows':[self.row()]}})),1)
 def test_incomplete(self):self.assertEqual(e.row_groups({'n':2,'rows':[self.row()]}),[])
 def test_missing_n(self):self.assertEqual(e.row_groups({'rows':[self.row()]}),[])
 def test_metadata_not_rows(self):self.assertEqual(e.row_groups({'n':1,'rows':[{'score':1}]}),[])
if __name__=='__main__':unittest.main()
