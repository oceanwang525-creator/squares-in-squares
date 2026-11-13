"""Ctypes wrapper around highd_ils.cpp (/tmp/libhighd_ils.so), capped at <= 4 CPU threads."""
from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CPP_SRC = HERE / "highd_ils.cpp"
CPP_LIB = Path("/tmp/libhomology_ils.so")

_LIB: ctypes.CDLL | None = None


def get_cpp_lib(force_rebuild: bool = False) -> ctypes.CDLL:
    """Compile (if needed) and load /tmp/libhighd_ils.so."""
    global _LIB
    if _LIB is not None and not force_rebuild:
        return _LIB
    need_build = (
        force_rebuild
        or (not CPP_LIB.exists())
        or (CPP_SRC.stat().st_mtime > CPP_LIB.stat().st_mtime)
    )
    if need_build:
        cmd = [
            "g++",
            "-O3",
            "-march=native",
            "-shared",
            "-fPIC",
            "-pthread",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(CPP_SRC),
            "-o",
            str(CPP_LIB),
        ]
        subprocess.run(cmd, check=True)

    lib = ctypes.CDLL(str(CPP_LIB))
    c_dp = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    c_ip = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    c_u8p = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")

    lib.eval_sym_sat_energy_grad.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        c_dp,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        c_dp,
    ]
    lib.eval_sym_sat_energy_grad.restype = ctypes.c_double

    lib.expand_sym_coords.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, c_dp, c_dp]
    lib.expand_sym_coords.restype = None

    lib.eval_sat_pen_bbox.argtypes = [
        ctypes.c_int,
        c_dp,
        c_dp,
        c_dp,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.eval_sat_pen_bbox.restype = ctypes.c_double

    lib.batch_ils_solve.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        c_dp,
        c_dp,
        ctypes.c_int,
        c_dp,
        c_ip,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        c_dp,
        c_dp,
        c_dp,
        c_dp,
    ]
    lib.batch_ils_solve.restype = None

    lib.batch_masked_melt_solve.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        c_dp,
        c_dp,
        c_u8p,
        ctypes.c_int,
        c_dp,
        c_ip,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        c_dp,
        c_dp,
        c_dp,
        c_dp,
    ]
    lib.batch_masked_melt_solve.restype = None

    _LIB = lib
    return lib


def sym_dim(n: int, sym_mode: int) -> tuple[int, int, int]:
    """Return (m_orbits, has_center, dim) for sym_mode in {1, 2, 4}."""
    has_c = 0 if sym_mode == 1 else (n % sym_mode)
    if sym_mode in (2, 4) and has_c > 1:
        raise ValueError(f"sym_mode={sym_mode} requires n % sym_mode in {{0, 1}}, got n={n}")
    m_orb = n if sym_mode == 1 else (n - has_c) // sym_mode
    return m_orb, has_c, 3 * m_orb + has_c


def cpp_eval_sat_pen_bbox(
    x: np.ndarray, y: np.ndarray, th: np.ndarray, side: float = 1.0
) -> tuple[float, float]:
    """Evaluate exact SAT penetration and bounding box L via C++ spatial hash."""
    lib = get_cpp_lib()
    xa = np.ascontiguousarray(x, dtype=np.float64)
    ya = np.ascontiguousarray(y, dtype=np.float64)
    ta = np.ascontiguousarray(th, dtype=np.float64)
    out_L = ctypes.c_double(0.0)
    pen = lib.eval_sat_pen_bbox(len(xa), xa, ya, ta, float(side), ctypes.byref(out_L))
    return float(pen), float(out_L.value)


def cpp_expand_sym(
    n: int, sym_mode: int, L: float, q_red: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand reduced orbit coords q_red around (L/2, L/2) into full (x, y, th) in [0, L]^2."""
    lib = get_cpp_lib()
    q_arr = np.ascontiguousarray(q_red, dtype=np.float64)
    z_full = np.empty(3 * n, dtype=np.float64)
    lib.expand_sym_coords(n, sym_mode, float(L), q_arr, z_full)
    return z_full[:n].copy(), z_full[n : 2 * n].copy(), z_full[2 * n : 3 * n].copy()


def full_to_free_sym(x: np.ndarray, y: np.ndarray, th: np.ndarray, L: float) -> np.ndarray:
    """Center full [0, L]^2 coordinates around L/2 for sym_mode=1."""
    hc = 0.5 * L
    return np.concatenate([x - hc, y - hc, th]).astype(np.float64)


def run_cpp_batch_ils(
    n: int,
    sym_mode: int,
    q_list: list[np.ndarray],
    L_sched_list: list[list[float]] | list[np.ndarray],
    ils_iters_per_L: list[int],
    a_start: float = 0.90,
    n_melt_stages: int = 3,
    n_threads: int = 4,
    base_seed: int = 19001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run batch of C++ ILS + L-continuation trajectories across <= 4 CPU threads."""
    lib = get_cpp_lib()
    n_threads = min(4, max(1, int(n_threads)))
    b = len(q_list)
    _, _, dim = sym_dim(n, sym_mode)
    n_L = len(ils_iters_per_L)

    q_batch = np.zeros((b, dim), dtype=np.float64)
    L_batch = np.zeros((b, n_L), dtype=np.float64)
    for i in range(b):
        assert len(q_list[i]) == dim, f"seed {i} len {len(q_list[i])} != dim {dim}"
        assert len(L_sched_list[i]) == n_L
        q_batch[i] = q_list[i]
        L_batch[i] = L_sched_list[i]

    q_feas = np.zeros_like(q_batch)
    ils_arr = np.ascontiguousarray(ils_iters_per_L, dtype=np.int32)
    out_Lf = np.empty(b, dtype=np.float64)
    out_E = np.empty(b, dtype=np.float64)
    out_pen = np.empty(b, dtype=np.float64)
    out_Lb = np.empty(b, dtype=np.float64)

    q_flat = np.ascontiguousarray(q_batch.ravel())
    qf_flat = np.ascontiguousarray(q_feas.ravel())
    L_flat = np.ascontiguousarray(L_batch.ravel())

    lib.batch_ils_solve(
        b,
        n,
        sym_mode,
        q_flat,
        qf_flat,
        n_L,
        L_flat,
        ils_arr,
        float(a_start),
        int(n_melt_stages),
        int(n_threads),
        int(base_seed),
        out_Lf,
        out_E,
        out_pen,
        out_Lb,
    )
    return (
        q_flat.reshape(b, dim),
        qf_flat.reshape(b, dim),
        out_Lf,
        out_E,
        out_pen,
        out_Lb,
    )


def run_cpp_masked_melt(
    n: int,
    sym_mode: int,
    q_list: list[np.ndarray],
    mask_list: list[np.ndarray],
    L_sched_list: list[list[float]] | list[np.ndarray],
    ils_iters_per_L: list[int],
    a_start: float = 0.85,
    n_melt_stages: int = 4,
    masked_iters: int = 180,
    full_iters: int = 240,
    n_threads: int = 4,
    base_seed: int = 29001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run batch of frozen-exterior masked-pocket melts + full unfreeze settle (<= 4 threads)."""
    lib = get_cpp_lib()
    n_threads = min(4, max(1, int(n_threads)))
    b = len(q_list)
    _, _, dim = sym_dim(n, sym_mode)
    n_L = len(ils_iters_per_L)

    q_batch = np.zeros((b, dim), dtype=np.float64)
    m_batch = np.zeros((b, dim), dtype=np.uint8)
    L_batch = np.zeros((b, n_L), dtype=np.float64)
    for i in range(b):
        assert len(q_list[i]) == dim and len(mask_list[i]) == dim and len(L_sched_list[i]) == n_L
        q_batch[i] = q_list[i]
        m_batch[i] = mask_list[i]
        L_batch[i] = L_sched_list[i]

    q_feas = np.zeros_like(q_batch)
    ils_arr = np.ascontiguousarray(ils_iters_per_L, dtype=np.int32)
    out_Lf = np.empty(b, dtype=np.float64)
    out_E = np.empty(b, dtype=np.float64)
    out_pen = np.empty(b, dtype=np.float64)
    out_Lb = np.empty(b, dtype=np.float64)

    q_flat = np.ascontiguousarray(q_batch.ravel())
    qf_flat = np.ascontiguousarray(q_feas.ravel())
    m_flat = np.ascontiguousarray(m_batch.ravel())
    L_flat = np.ascontiguousarray(L_batch.ravel())

    lib.batch_masked_melt_solve(
        b, n, sym_mode, q_flat, qf_flat, m_flat, n_L, L_flat, ils_arr,
        float(a_start), int(n_melt_stages), int(masked_iters), int(full_iters),
        int(n_threads), int(base_seed), out_Lf, out_E, out_pen, out_Lb,
    )
    return (
        q_flat.reshape(b, dim),
        qf_flat.reshape(b, dim),
        out_Lf,
        out_E,
        out_pen,
        out_Lb,
    )
