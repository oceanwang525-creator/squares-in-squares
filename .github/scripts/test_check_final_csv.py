import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("checker", Path(__file__).with_name("check_final_csv.py"))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

class StructureTests(unittest.TestCase):
    def test_header_only_valid(self):
        c.validate("id,x,y,deg\n")
    def test_complete_valid(self):
        c.validate("id,x,y,deg\n001_0,s0,s0,s0\n")
    def test_complete_worse_spread_group_allowed(self):
        c.validate("id,x,y,deg\n002_0,s0,s0,s0\n002_1,s100,s0,s0\n")

    def test_reject_bad_data(self):
        for text in ["", "n,square_id,x,y,angle\n", '"id",x,y,deg\n',
                     "id,x,y,deg\n002_0,0,0,0\n",
                     "id,x,y,deg\n002_0,0,0,0\n002_0,1,0,0\n",
                     "id,x,y,deg\n001_0,nan,0,0\n"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                c.validate(text)

    def test_all_touched_folders_not_just_changed_final(self):
        with tempfile.TemporaryDirectory() as td:
            old = os.getcwd()
            os.chdir(td)
            try:
                def git(*args):
                    return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode().strip()
                def write(path, text):
                    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)
                def commit():
                    git("add", "."); git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
                    return git("rev-parse", "HEAD")
                git("init", "-q")
                write("README.md", "base")
                base = commit()
                write("poc/a/final.csv", "id,x,y,deg\n")
                write("poc/b/code.py", "# not executed")
                head = commit()
                with self.assertRaisesRegex(ValueError, "poc/b/final.csv"):
                    c.check_pr(base, head)
                write("poc/b/final.csv", "id,x,y,deg\n")
                head = commit()
                self.assertEqual(len(c.check_pr(base, head)), 2)
                # A final need not itself change, but must exist in head.
                base = head
                write("poc/b/code.py", "# changed")
                head = commit()
                self.assertEqual(c.check_pr(base, head), ["poc/b/final.csv"])
                Path("poc/b/final.csv").unlink()
                head = commit()
                with self.assertRaises(ValueError):
                    c.check_pr(base, head)
                Path("poc/b/code.py").unlink()
                head = commit()
                self.assertEqual(c.check_pr(base, head), [])
                # Symlinks must never redirect trusted code reads.
                write("poc/b/code.py", "# returned")
                Path("poc/b/final.csv").symlink_to("../../a/final.csv")
                head = commit()
                with self.assertRaises(ValueError):
                    c.check_pr(base, head)
            finally:
                os.chdir(old)

if __name__ == "__main__":
    unittest.main()
