import json
import numpy as np
import pytest
from ribbon_archive import RibbonArchive, LABELS


def offer(a, q, **kw):
    return a.offer(q, structural_identity=kw.pop('structural_identity', 'ribbon-A'),
                   outer_bounds=kw.pop('outer_bounds', [0, 0, 10, 10]), **kw)


def pair(distance):
    return np.array([[2., 2., 0.], [2.+distance, 2., 0.]])


def test_dominance_immutable_roundtrip(tmp_path):
    a = RibbonArchive(tmp_path)
    q = pair(1.3)
    original = q.copy()
    first = offer(a, q, contact_metadata={'source': 'proposal'})
    assert first['labels'] == LABELS and first['strict_sat']
    np.testing.assert_array_equal(q, original)
    loaded = a.load(first)
    assert loaded.dtype == np.float64 and loaded.flags.c_contiguous
    assert not loaded.flags.writeable
    np.testing.assert_array_equal(loaded, original)
    assert offer(a, q) is None
    second = offer(a, pair(1.1))
    assert second['width'] < first['width']
    assert len(a.entries()) == 1
    assert offer(a, pair(1.2)) is None
    # Readers with a previous index remain valid after Pareto pruning.
    np.testing.assert_array_equal(a.load(first), original)
    assert not list(tmp_path.glob('.archive-*'))


def test_tradeoff_and_incompatible_context(tmp_path):
    a = RibbonArchive(tmp_path)
    assert offer(a, np.array([[2, 2, 0.]]))
    # Rotating one square increases both bbox extents: dominated, same context.
    assert offer(a, np.array([[2, 2, .3]])) is None
    assert offer(a, np.array([[2, 2, .3]]), structural_identity='other')
    assert offer(a, np.array([[2, 2, .3]]), outer_bounds=[0, 0, 11, 10])
    assert offer(a, np.array([[2, 2, .3]]), grouping_signature={'roles': [7]})
    assert len(a.entries()) == 4
    b = RibbonArchive(tmp_path / 'tradeoff')
    assert offer(b, pair(1.1))
    vertical = pair(1.1)[:, [1, 0, 2]]
    assert offer(b, vertical)
    assert len(b.entries()) == 2


def test_invalid_and_overlap(tmp_path):
    a = RibbonArchive(tmp_path)
    assert offer(a, pair(.99)) is None
    assert not a.entries()
    for q in [np.zeros((0, 3)), np.zeros((2, 2)), [[1, 2, np.nan]]]:
        with pytest.raises(ValueError):
            offer(a, q)
    with pytest.raises(ValueError):
        offer(a, pair(1.1), contact_metadata={'bad': float('nan')})
    with pytest.raises(ValueError):
        offer(a, pair(1.1), structural_identity='')
    with pytest.raises(ValueError):
        a.load({'file': '../escape.npz'})


def test_components_keep_masked_gain(tmp_path):
    a = RibbonArchive(tmp_path)
    # Second ribbon controls full width; improvement in first still survives.
    q = np.vstack((pair(1.3), pair(1.3) + [0, 5, 0]))
    old = q.copy()
    assert all(a.offer_components(q, structural_identity='parent', outer_bounds=[0,0,10,10]))
    q[1, 0] -= .1
    results = a.offer_components(q, structural_identity='parent', outer_bounds=[0,0,10,10])
    assert results[0] is not None and results[1] is None
    assert len(a.entries()) == 2
    np.testing.assert_array_equal(old[2:], q[2:])


def test_failed_index_write_keeps_previous_index(tmp_path, monkeypatch):
    import ribbon_archive
    a = RibbonArchive(tmp_path)
    first = offer(a, pair(1.3))
    before = a.index.read_bytes()
    atomic = ribbon_archive._atomic
    def fail_index(path, write):
        if path.name == 'index.json':
            raise OSError('simulated crash before index replacement')
        return atomic(path, write)
    monkeypatch.setattr(ribbon_archive, '_atomic', fail_index)
    with pytest.raises(OSError):
        offer(a, pair(1.1))
    assert a.index.read_bytes() == before
    np.testing.assert_array_equal(a.load(first), pair(1.3))
    assert json.loads(before)['version'] == 1


def test_touching_allowed_but_no_overlap_tolerance(tmp_path):
    a = RibbonArchive(tmp_path)
    assert offer(a, pair(1.0))
    assert offer(a, pair(1.0 - 1e-12)) is None


def test_group_roles_not_interchangeable(tmp_path):
    a = RibbonArchive(tmp_path)
    assert offer(a, pair(1.3), grouping_signature={'indices': [0,1]})
    assert offer(a, pair(1.1), grouping_signature={'indices': [2,3]})
    assert len(a.entries()) == 2
