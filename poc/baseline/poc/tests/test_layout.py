"""Repository-layout regressions, including CPU-only entrypoint checks."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[4]


def test_package_data_paths():
    package = ROOT / "poc/baseline/poc"
    config = tomllib.loads((package / "pyproject.toml").read_text())
    for paths in config["tool"]["setuptools"]["data-files"].values():
        for path in paths:
            assert (package / path).is_file()
    for line in (package / "MANIFEST.in").read_text().splitlines():
        assert (package / line.removeprefix("include ")).is_file()


def test_approach_readme_links():
    for readme in [ROOT / "README.md", *ROOT.glob("poc/*/README.md"),
                   *ROOT.glob("poc/*/docs/**/*.md")]:
        for url in re.findall(r"\]\(([^ )]+)\)", readme.read_text()):
            if ":" in url or url.startswith("#"):
                continue
            assert (readme.parent / url.split("#")[0]).exists(), (readme, url)


@pytest.mark.parametrize("directory", ["cuda-loop", "kat-algos", "gemini"])
def test_cuda_build_graph_without_gpu(directory):
    result = subprocess.run(
        ["make", "-n", "-C", str(ROOT / "poc" / directory / "cuda"),
         "all", "ARCH=sm_80"], capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script", [
    "cuda-loop/poc/demo.py", "cuda-loop/poc/endless.py",
    "cuda-loop/poc/full_run.py", "cuda-loop/poc/families.py",
    "kat-algos/poc/algos_bench.py", "kat-algos/poc/tests/algos_smoke.py",
    "gemini/poc/gemini_full_run.py", "kat-band/poc/seed_cli.py",
    "hinged_band/poc/hinged_band.py",
    "boundary_zipper/poc/boundary_zipper.py",
    "diamond_core/poc/diamond_core.py", "fault_cut/poc/fault_cut.py",
    "shelf_knapsack/poc/shelf_knapsack.py",
])
def test_cpu_entrypoint_help(script, tmp_path):
    # No kernels or optimization run. Also verifies imports outside repository cwd.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "poc/baseline/poc/src")
    result = subprocess.run(
        [sys.executable, str(ROOT / "poc" / script), "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
