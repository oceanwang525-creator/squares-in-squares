import json,shutil,tarfile
import pytest
from pathlib import Path
from ftmc_bundle import REQUIRED,validate,build,verify

@pytest.fixture
def stage(tmp_path):
    source=Path(__file__).parent
    for n in REQUIRED-{"baseline.json"}:shutil.copy(source/n,tmp_path/n)
    (tmp_path/"baseline.json").write_text(json.dumps({str(n):[[0,0,0]]*n for n in [911,973,908]}))
    return tmp_path

def test_complete_roundtrip(stage):
    p=stage/"bundle.tar.gz";m=build(stage,p);assert verify(p)==m
def test_archive_dependency_missing(stage):
    with pytest.raises(ValueError,match="ribbon_archive"):validate(stage,REQUIRED-{"ribbon_archive.py"})
def test_transitive_missing(stage):
    (stage/"band_polish.py").write_text("import hidden_dependency")
    (stage/"hidden_dependency.py").write_text("x=1")
    with pytest.raises(ValueError,match="hidden_dependency"):validate(stage,REQUIRED)
def test_tampered_bundle(stage):
    p=stage/"bundle.tar.gz";build(stage,p)
    (stage/"baseline.json").write_text("{}")
    with tarfile.open(stage/"bad.tar.gz","w:gz") as out:
        with tarfile.open(p) as old:
            for m in old.getmembers():
                if m.name=="baseline.json":out.add(stage/"baseline.json",arcname=m.name)
                else:out.addfile(m,old.extractfile(m))
    with pytest.raises((ValueError,KeyError)):verify(stage/"bad.tar.gz")
