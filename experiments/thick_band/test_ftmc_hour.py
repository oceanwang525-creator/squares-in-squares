"""CPU-only helper tests. Never allocate GPU or touch real baselines."""
import json
import numpy as np
import pytest
from ftmc_hour import Evidence, parser, schedule, strict_layout


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
