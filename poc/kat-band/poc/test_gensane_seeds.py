import json
import numpy as np
import pytest
from gensane_seeds import build_portfolio


def grid(k=4):
    return np.array([(x+.5, y+.5, 0.) for x in range(k) for y in range(k)])


def test_exact_count_24_baseline_and_reproducibility():
    q = grid()
    before = q.copy()
    p, meta = build_portfolio(q, 4., 24, np.random.default_rng(41))
    p2, meta2 = build_portfolio(q, 4., 24, np.random.default_rng(41))
    assert p.shape == (24, 16, 3)
    assert p.dtype == np.float64 and np.isfinite(p).all()
    np.testing.assert_array_equal(p[0], before)
    np.testing.assert_array_equal(q, before)
    np.testing.assert_array_equal(p, p2)
    assert meta == meta2
    json.dumps(meta)
    assert all(not m["certified"] and m["n"] == 16 for m in meta)
    assert any(m["kind"] == "analytic_band" for m in meta)
    for lane in p:
        assert len(np.unique(np.round(lane[:, :2], 10), axis=0)) == 16


def test_strip_restores_exact_n_and_retains_unselected():
    q = grid()
    q[0, 2] = .23
    p, meta = build_portfolio(q, 4., 12, 7)
    for lane, m in zip(p, meta):
        if m["kind"] != "atlas_strip_reinsert":
            continue
        idx = m["removed_indices"]
        keep = np.ones(len(q), dtype=bool)
        keep[idx] = False
        assert 0 in idx
        assert m["removed_count"] == m["inserted_count"] == len(idx)
        np.testing.assert_array_equal(lane[keep], q[keep])


def test_tiny_and_single_square():
    p, meta = build_portfolio(grid(), 4., 2, 9)
    assert np.max(np.abs(p[1]-p[0])) < .002
    p, meta = build_portfolio(np.array([[.5, .5, 0]]), 1., 24, 3)
    assert p.shape == (24, 1, 3) and np.isfinite(p).all()


@pytest.mark.parametrize("count", [0, 25, True, 1.5])
def test_invalid_lane_count(count):
    with pytest.raises(ValueError):
        build_portfolio(grid(), 4, count, 0)


def test_bad_baselines():
    for q in (np.zeros((0,3)), np.zeros((2,3)), np.array([[np.nan,0,0]])):
        with pytest.raises(ValueError):
            build_portfolio(q, 4, 1, 0)


def test_large_tilted_strip_capacity_is_bounded():
    L = 31.
    # 100 tilted cells, plus an upright grid. Unique centers, not a packing.
    tilted = np.array([(3.+i*.24, 3.+i*.24, np.pi/4)
                       for i in range(100)])
    upright = np.array([(x+.5, y+.5, 0.)
                        for x in range(30) for y in range(30)])
    q = np.concatenate((tilted, upright))
    p, meta = build_portfolio(q, L, 24, 19)
    assert p.shape == (24, len(q), 3)
    assert (p[:, :, :2] >= -2).all()
    assert (p[:, :, :2] <= L+2).all()
    restored = False
    for lane, m in zip(p, meta):
        if m['kind'] != 'atlas_strip_reinsert':
            continue
        assert m['reconstructed_count'] <= m['rectangle_capacity']
        assert m['reconstructed_count'] + m['restored_original_count'] == m['removed_count']
        restored |= m['restored_original_count'] > 0
        assert len(np.unique(np.round(lane[:, :2], 10), axis=0)) == len(q)
    assert restored
