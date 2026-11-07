"""Local ribbon Pareto snapshots, never full packing incumbents.

Widths/heights are local_gadget_proxy metrics, not f1/f2 or a full-box gain.
Every snapshot requires_refill_and_official_certificate. SAT here is only a
strict floating-point screen, not an interval certificate. Call offer_components
before refill and on each LP result (after stripping uprights). Stable square
ordering and caller-supplied structural_identity are required across proposals;
change the identity if square roles, gadget placement constraints or topology
semantics change. No rotation, permutation or f1/f2 equivalence is inferred.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from contextlib import contextmanager

@contextmanager
def _writer_lock(path):
    with path.open('a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

import numpy as np

try:
    from .band_polish import sat_check
    from .ftmc_moves import components
except ImportError:
    from band_polish import sat_check
    from ftmc_moves import components

LABELS = ['local_gadget_proxy', 'requires_refill_and_official_certificate']
_SAFE = re.compile(r'[0-9a-f]{64}\.npz\Z')


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _atomic(path, write):
    fd, name = tempfile.mkstemp(prefix='.archive-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as out:
            write(out)
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
        if os.name != 'nt':
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class RibbonArchive:
    """Crash-safe NPZ blobs followed by an atomic JSON index; no pickle.

    A crash before index replacement can leave harmless unindexed blobs. Blobs
    are never overwritten/deleted by pruning, so old readers remain valid.
    Writers are serialized with a process lock on this POSIX experiment host.
    Outer bounds are exact compatibility context, not containment certification.
    """

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index = self.directory / 'index.json'

    def entries(self):
        if not self.index.exists():
            return []
        data = json.loads(self.index.read_text())
        if data.get('version') != 1:
            raise ValueError('unsupported archive version')
        for entry in data['entries']:
            if not _SAFE.fullmatch(entry['file']):
                raise ValueError('unsafe archive filename')
        return data['entries']

    def load(self, entry):
        name = entry['file']
        if not _SAFE.fullmatch(name):
            raise ValueError('unsafe archive filename')
        with np.load(self.directory / name, allow_pickle=False) as blob:
            q = np.array(blob['core'], dtype=np.float64, order='C', copy=True)
        if hashlib.sha256(q.tobytes()).hexdigest() != entry['geometry_sha256']:
            raise ValueError('archive geometry checksum mismatch')
        q.flags.writeable = False
        return q

    def offer(self, core, *, structural_identity, outer_bounds,
              grouping_signature=None, contact_metadata=None):
        """Keep a strict local width/height Pareto improvement in its context.

        Returns an entry or None for overlap/equality/dominance. The caller's
        geometry and full incumbent are never modified. Contact metadata is
        observational, not identity; structural contact roles belong explicitly
        in structural_identity. Bounds and grouping signature must match exactly.
        """
        q = np.array(core, dtype=np.float64, order='C', copy=True)
        if q.ndim != 2 or q.shape[1] != 3 or not len(q) or not np.isfinite(q).all():
            raise ValueError('core must be nonempty finite (n,3) radians')
        if not isinstance(structural_identity, str) or not structural_identity:
            raise ValueError('explicit structural identity required')
        bounds = np.asarray(outer_bounds, dtype=np.float64)
        if bounds.shape != (4,) or not np.isfinite(bounds).all():
            raise ValueError('outer_bounds must be (xmin,ymin,xmax,ymax)')
        if np.any(bounds[2:] <= bounds[:2]):
            raise ValueError('outer bounds must have positive dimensions')
        groups = [g.tolist() for g in components(q)]
        context = dict(square_count=len(q), structural_identity=structural_identity,
                       outer_bounds=bounds.tolist(), component_partition=groups,
                       grouping_signature=grouping_signature)
        key = hashlib.sha256(_json(context).encode()).hexdigest()
        valid, gap = sat_check(q)
        if not valid:
            return None
        # Outward rounded local AABB, no tolerance or implied enclosing box gain.
        wide = q.astype(np.longdouble)
        half = (np.abs(np.cos(wide[:, 2])) + np.abs(np.sin(wide[:, 2]))) / 2
        lo = np.nextafter(np.min(wide[:, :2] - half[:, None], axis=0).astype(float), -np.inf)
        hi = np.nextafter(np.max(wide[:, :2] + half[:, None], axis=0).astype(float), np.inf)
        width, height = np.nextafter(hi - lo, np.inf)
        geometry_hash = hashlib.sha256(q.tobytes()).hexdigest()
        metadata = dict(contact_metadata=contact_metadata, strict_sat=True,
                        minimum_sat_gap=None if not np.isfinite(gap) else gap)
        _json(metadata)  # reject non-JSON/NaN metadata before writing anything
        blob_id = hashlib.sha256((key + geometry_hash + _json(metadata)).encode()).hexdigest()
        entry = dict(file=blob_id + '.npz', geometry_sha256=geometry_hash,
                     compatibility_key=key, context=context, width=float(width),
                     height=float(height), bbox=[*lo.tolist(), *hi.tolist()],
                     labels=list(LABELS), **metadata)
        with _writer_lock(self.directory / '.lock'):
            entries = self.entries()
            peers = [e for e in entries if e['compatibility_key'] == key]
            if any(e['width'] <= width and e['height'] <= height for e in peers):
                return None
            kept = [e for e in entries if not (
                e['compatibility_key'] == key and width <= e['width'] and height <= e['height'])]
            _atomic(self.directory / entry['file'], lambda out: np.savez(out, core=q))
            _atomic(self.index, lambda out: out.write(
                _json(dict(version=1, entries=kept + [entry])).encode()))
        return entry

    def offer_components(self, core, *, structural_identity, outer_bounds,
                         contact_metadata=None):
        """Archive every ftmc component, including gains masked by other ribbons.

        Partition and original indices prevent mixing unrelated component roles.
        The parent must keep square indexing stable, or use a new identity.
        """
        q = np.array(core, dtype=np.float64, order='C', copy=True)
        if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all():
            raise ValueError('core must be finite (n,3) radians')
        groups = components(q)
        partition = [g.tolist() for g in groups]
        return [self.offer(q[g], structural_identity=structural_identity,
                           outer_bounds=outer_bounds,
                           grouping_signature=dict(parent_count=len(q),
                               partition=partition, member_indices=g.tolist()),
                           contact_metadata=contact_metadata) for g in groups]
