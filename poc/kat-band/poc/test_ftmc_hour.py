"""CPU-only helper tests. Never allocate GPU or touch real baselines."""
import json
import numpy as np
import pytest
from ftmc_hour import Evidence, configure_cuda, parser, schedule, strict_layout


def test_schedule_replay_and_coverage():
    rows = [schedule(i, 911) for i in range(120)]
    assert rows == [schedule(i, 911) for i in range(120)]
    assert {r["mode"] for r in rows} == {"angle", "shear", "slide", "taper"}
    assert min(r["amplitude"] for r in rows) == 1e-4
    assert max(r["amplitude"] for r in rows) == .03
    assert {r["slack"] for r in rows} == {.002, .005, .015, .03, .05}
    assert len({r["seed"] for r in rows}) == 120
    assert schedule(0, 908)["seed"] != schedule(0, 909)["seed"]


def test_exact_count_finite_and_strict_allpair():
    q = np.array([[.5, .5, 0], [1.5, .5, 0]])
    assert strict_layout(q, 2)[0]
    assert strict_layout(q, 3)[1] == "exact_count_or_shape"
    overlapping = q.copy()
    overlapping[1, 0] -= 1e-10
    assert strict_layout(overlapping, 2)[1] == "strict_allpair_overlap"
    q[1, 0] = np.nan
    assert strict_layout(q, 2)[1] == "nonfinite"


def test_save_worse_and_prospective_never_certified(tmp_path):
    e = Evidence(tmp_path)
    q = np.array([[.5, .5, 0], [1.5, .5, 0]])
    before = q.copy()
    assert e.save(q, 2, 1.9, stage="refill", operator="mutation")
    assert e.save(q, 2, 2.1, stage="refill", operator="insertion", source_n=1)
    blobs = sorted(tmp_path.glob("*.npz"))
    assert len(blobs) == 2
    for path in blobs:
        with np.load(path, allow_pickle=False) as data:
            record = json.loads(str(data["meta"]))
            assert record["certified_win"] is False
            assert record["needs_official_certificate"] is True
            np.testing.assert_array_equal(data["x"], before)
    events = [json.loads(x) for x in (tmp_path/"events.jsonl").read_text().splitlines()]
    assert events[0]["prospective_win"] is False
    assert events[1]["prospective_win"] is True
    assert events[1]["source_n"] == 1 and events[1]["n"] == 2
    np.testing.assert_array_equal(q, before)


def test_rejects_do_not_persist_and_are_counted(tmp_path):
    e = Evidence(tmp_path)
    assert not e.save(np.zeros((2, 3)), 2, 1.)
    assert not e.save(np.zeros((1, 3)), 2, 1.)
    assert e.counts["strict_allpair_overlap"] == 1
    assert e.counts["exact_count_or_shape"] == 1
    assert not list(tmp_path.glob("*.npz"))
    e.checkpoint(elapsed=1., generation=2)
    state = json.loads((tmp_path/"checkpoint.json").read_text())
    assert state["saved"] == 0 and state["generation"] == 2
    assert not (tmp_path/"checkpoint.tmp").exists()


def test_default_hour_and_interface(monkeypatch):
    monkeypatch.delenv("FTMC_SECONDS", raising=False)
    a = parser().parse_args([])
    assert a.seconds == 3600
    assert a.output.endswith("ftmc-results")
    assert a.baseline.endswith("baseline.json")
    assert a.targets == ""
    a = parser().parse_args(["--seconds", "2", "--targets", "908,909",
                             "--target-seconds", "1", "--checkpoint-seconds", ".5"])
    assert a.seconds == 2 and a.target_seconds == 1


def test_first_six_mutation_visits_cover_all_amplitudes_each_target():
    expected = {1e-4, 3e-4, .001, .003, .01, .03}
    for n in [908, 909, 910, 911, 971, 972, 973, 1001]:
        assert {schedule(i, n)["amplitude"] for i in range(6)} == expected
        # Across subsequent blocks, the same amplitude explores every mode.
        for amplitude in expected:
            assert {schedule(i, n)["mode"] for i in range(24)
                    if schedule(i, n)["amplitude"] == amplitude} == {
                        "angle", "shear", "slide", "taper"}
        assert len({schedule(i, n)["mode"] for i in range(6)}) == 4


class FakeCuda:
    def __init__(self, name="NVIDIA A100-SXM4-80GB", available=True):
        self.name = name
        self.available = available
        self.fractions = []

    def is_available(self):
        return self.available

    def get_device_name(self):
        assert self.available
        return self.name

    def set_per_process_memory_fraction(self, fraction):
        self.fractions.append(fraction)


def test_gpu_defaults_preserve_a100_and_no_allocator_cap():
    args = parser().parse_args([])
    assert args.gpu_name == "A100" and args.cuda_memory_fraction is None
    cuda = FakeCuda()
    assert configure_cuda(cuda) == cuda.name
    assert cuda.fractions == []


@pytest.mark.parametrize("name", ["RTX 4070 SUPER", "NVIDIA GeForce RTX 4070 SUPER",
                                  "NVIDIA RTX 4070 SUPER", "GeForce RTX 4070 SUPER"])
def test_explicit_desktop_gpu_and_optional_memory_cap(name):
    args = parser().parse_args(["--gpu-name", "RTX 4070 SUPER",
                                "--cuda-memory-fraction", ".06"])
    cuda = FakeCuda(name)
    assert configure_cuda(cuda, args.gpu_name, args.cuda_memory_fraction) == name
    assert cuda.fractions == [.06]


@pytest.mark.parametrize("name,required", [
    ("NVIDIA GeForce RTX 4070 SUPER", "A100"),
    ("NVIDIA A100-SXM4-80GB", "RTX 4070 SUPER"),
    ("NVIDIA GeForce RTX 4070", "RTX 4070 SUPER"),
    ("NVIDIA GeForce RTX 4070 SUPER Ti", "RTX 4070 SUPER"),
])
def test_gpu_mismatch_never_sets_memory_cap(name, required):
    cuda = FakeCuda(name)
    with pytest.raises(RuntimeError):
        configure_cuda(cuda, required, .06)
    assert cuda.fractions == []


def test_missing_cuda_never_falls_back_to_cpu():
    cuda = FakeCuda(available=False)
    with pytest.raises(RuntimeError, match="CPU fallback is disabled"):
        configure_cuda(cuda, "RTX 4070 SUPER", .06)
    assert cuda.fractions == []


@pytest.mark.parametrize("fraction", [0, -.1, 1.01, float("nan"), float("inf")])
def test_bad_cuda_memory_fraction(fraction):
    cuda = FakeCuda()
    with pytest.raises(ValueError, match="memory fraction"):
        configure_cuda(cuda, memory_fraction=fraction)
    assert cuda.fractions == []


def test_gpu_override_allowlist():
    with pytest.raises(ValueError, match="unsupported GPU"):
        configure_cuda(FakeCuda(), "RTX 4090")
    with pytest.raises(SystemExit):
        parser().parse_args(["--gpu-name", "RTX 4090"])
