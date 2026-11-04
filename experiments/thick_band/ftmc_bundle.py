"""Offline content manifest and import-closure validation for flat pilot bundles."""
import ast,hashlib,importlib.util,json,tarfile,tempfile,sys
from pathlib import Path
ROOT="ftmc_pilot.py"
REQUIRED={"ftmc_pilot.py","band_geometry.py","band_relax.py","band_polish.py",
"gensane_polish.py","gensane_recovery.py","ftmc_moves.py","core_refill.py",
"ribbon_archive.py","baseline.json"}
def validate(directory, names):
    directory=Path(directory); names=set(names)
    missing=REQUIRED-names
    if missing:raise ValueError("missing required files "+repr(sorted(missing)))
    for name in names:
        if Path(name).name!=name or not (directory/name).is_file():
            raise ValueError("unsafe or missing member "+name)
        if name.endswith(".py"):
            tree=ast.parse((directory/name).read_text(),filename=name)
            compile(tree,name,"exec")
            for node in ast.walk(tree):
                mods=[]
                if isinstance(node,ast.Import):mods=[x.name.split(".")[0] for x in node.names]
                elif isinstance(node,ast.ImportFrom) and node.module:mods=[node.module.split(".")[0]]
                for module in mods:
                    local=directory/(module+".py")
                    if local.exists() and local.name not in names:
                        raise ValueError("missing local dependency "+local.name+" imported by "+name)
                    if not local.exists() and module not in {'numpy','scipy','torch'} and importlib.util.find_spec(module) is None:
                        raise ValueError("unavailable external dependency "+module)
    baseline=json.loads((directory/"baseline.json").read_text())
    for n in (911,973,908):
        q=baseline[str(n)]
        if len(q)!=n or any(len(row)!=3 for row in q):raise ValueError("baseline shape")
    return {n:hashlib.sha256((directory/n).read_bytes()).hexdigest() for n in sorted(names)}
def build(directory, destination):
    hashes=validate(directory,REQUIRED)
    manifest={"version":1,"entrypoint":ROOT,"external_dependencies":["numpy","scipy","torch"],"sha256":hashes}
    with tarfile.open(destination,"w:gz") as out:
        for name in sorted(REQUIRED):out.add(Path(directory)/name,arcname=name)
        import io
        data=json.dumps(manifest,sort_keys=True,indent=2).encode()
        member=tarfile.TarInfo("manifest.json");member.size=len(data)
        out.addfile(member,io.BytesIO(data))
    verify(destination)
    return manifest
def verify(bundle):
    with tempfile.TemporaryDirectory() as d:
        with tarfile.open(bundle) as f:
            names=[x.name for x in f.getmembers()]
            if len(names)!=len(set(names)) or set(names)!=REQUIRED|{"manifest.json"}:
                raise ValueError("bundle member mismatch")
            for member in f.getmembers():
                if not member.isfile() or Path(member.name).name!=member.name:
                    raise ValueError("unsafe bundle member")
            f.extractall(d,filter="data")
        manifest=json.loads((Path(d)/"manifest.json").read_text())
        if manifest["sha256"]!=validate(d,REQUIRED):raise ValueError("checksum mismatch")
        return manifest
if __name__=="__main__":
    print(json.dumps(build(sys.argv[1],sys.argv[2]),sort_keys=True,indent=2))
