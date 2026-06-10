import json
from pathlib import Path
from typing import Optional

import numpy as np
import numba
from numba import njit, prange

_P0 = np.int64(1)
_P1 = np.int64(2_654_435_761)
_P2 = np.int64(805_459_861)


@njit(cache=True, fastmath=True, inline='always')
def _hash3(ix: int, iy: int, iz: int, T: int) -> int:
    h = (ix * _P0) ^ (iy * _P1) ^ (iz * _P2)
    return h & (T - 1)


@njit(cache=True, fastmath=True)
def _forward_one(x: float, y: float, z: float,
                 features: np.ndarray,        # (L, T, F) float32
                 res: np.ndarray,             # (L,) int64
                 W1: np.ndarray, b1: np.ndarray,   # decoder hidden
                 W2: np.ndarray, b2: np.ndarray,   # decoder output
                 has_hidden: bool) -> float:
    L = features.shape[0]
    T = features.shape[1]
    F = features.shape[2]
    in_dim = L * F

    # Clamp + map to [0, 1].
    if x < -1.0: x = -1.0
    elif x > 1.0: x = 1.0
    if y < -1.0: y = -1.0
    elif y > 1.0: y = 1.0
    if z < -1.0: z = -1.0
    elif z > 1.0: z = 1.0
    ux = (x + 1.0) * 0.5
    uy = (y + 1.0) * 0.5
    uz = (z + 1.0) * 0.5

    h_buf = np.empty(in_dim, dtype=np.float32)

    for Li in range(L):
        N = res[Li]
        sx = ux * (N - 1.0)
        sy = uy * (N - 1.0)
        sz = uz * (N - 1.0)
        ix = np.int64(sx); iy = np.int64(sy); iz = np.int64(sz)
        fx = sx - ix; fy = sy - iy; fz = sz - iz

        h000 = _hash3(ix,     iy,     iz,     T)
        h100 = _hash3(ix + 1, iy,     iz,     T)
        h010 = _hash3(ix,     iy + 1, iz,     T)
        h110 = _hash3(ix + 1, iy + 1, iz,     T)
        h001 = _hash3(ix,     iy,     iz + 1, T)
        h101 = _hash3(ix + 1, iy,     iz + 1, T)
        h011 = _hash3(ix,     iy + 1, iz + 1, T)
        h111 = _hash3(ix + 1, iy + 1, iz + 1, T)

        omfx = 1.0 - fx; omfy = 1.0 - fy; omfz = 1.0 - fz
        for Fi in range(F):
            f000 = features[Li, h000, Fi]
            f100 = features[Li, h100, Fi]
            f010 = features[Li, h010, Fi]
            f110 = features[Li, h110, Fi]
            f001 = features[Li, h001, Fi]
            f101 = features[Li, h101, Fi]
            f011 = features[Li, h011, Fi]
            f111 = features[Li, h111, Fi]

            c00 = f000 * omfx + f100 * fx
            c10 = f010 * omfx + f110 * fx
            c01 = f001 * omfx + f101 * fx
            c11 = f011 * omfx + f111 * fx
            c0  = c00 * omfy + c10 * fy
            c1  = c01 * omfy + c11 * fy
            h_buf[Li * F + Fi] = c0 * omfz + c1 * fz

    # Decoder.
    if has_hidden:
        hidden = W1.shape[0]
        out = b2[0]
        for j in range(hidden):
            acc = b1[j]
            for i in range(in_dim):
                acc += W1[j, i] * h_buf[i]
            if acc < 0.0:
                acc = 0.0
            out += W2[0, j] * acc
        return out
    else:
        out = b2[0]
        for i in range(in_dim):
            out += W2[0, i] * h_buf[i]
        return out


@njit(cache=True, fastmath=True, parallel=True)
def _forward_batch(pts: np.ndarray,            # (B, 3) float32
                   features: np.ndarray,
                   res: np.ndarray,
                   W1: np.ndarray, b1: np.ndarray,
                   W2: np.ndarray, b2: np.ndarray,
                   has_hidden: bool,
                   out: np.ndarray) -> None:
    B = pts.shape[0]
    for i in prange(B):
        out[i] = _forward_one(pts[i, 0], pts[i, 1], pts[i, 2],
                              features, res, W1, b1, W2, b2, has_hidden)


@njit(cache=True, fastmath=True)
def _forward_loop_bench(x: float, y: float, z: float,
                        n_iters: int,
                        features: np.ndarray, res: np.ndarray,
                        W1: np.ndarray, b1: np.ndarray,
                        W2: np.ndarray, b2: np.ndarray,
                        has_hidden: bool) -> float:
    acc = 0.0
    for _ in range(n_iters):
        acc += _forward_one(x, y, z, features, res, W1, b1, W2, b2, has_hidden)
    return acc


class SDFInference:

    def __init__(self, npz_path: str | Path):
        z = np.load(npz_path, allow_pickle=False)
        cfg = json.loads(str(z['cfg']))
        self.cfg = cfg
        self.has_hidden = cfg.get('hidden', 0) > 0

        # fp16 → fp32 for fast math.
        self.features = z['features'].astype(np.float32)
        self.res = z['resolutions'].astype(np.int64)

        if self.has_hidden:
            self.W1 = z['W1'].astype(np.float32)
            self.b1 = z['b1'].astype(np.float32)
            self.W2 = z['W2'].astype(np.float32)
            self.b2 = z['b2'].astype(np.float32)
        else:
            self.W1 = np.zeros((1, 1), dtype=np.float32)
            self.b1 = np.zeros((1,), dtype=np.float32)
            self.W2 = z['decoder_W'].astype(np.float32)
            self.b2 = z['decoder_b'].astype(np.float32)

    def query_one(self, x: float, y: float, z: float) -> float:
        return float(_forward_one(x, y, z, self.features, self.res,
                                  self.W1, self.b1, self.W2, self.b2,
                                  self.has_hidden))

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        pts = np.ascontiguousarray(pts, dtype=np.float32)
        out = np.empty(pts.shape[0], dtype=np.float32)
        _forward_batch(pts, self.features, self.res,
                       self.W1, self.b1, self.W2, self.b2,
                       self.has_hidden, out)
        return out

    def warmup(self) -> None:
        _ = self.query_one(0.0, 0.0, 0.0)
        _ = self(np.zeros((4, 3), dtype=np.float32))

    def bench_single_point(self, n_iters: int = 100_000) -> float:
        import time
        _forward_loop_bench(0.1, 0.2, 0.3, 100, self.features, self.res,
                            self.W1, self.b1, self.W2, self.b2, self.has_hidden)
        t0 = time.perf_counter()
        _forward_loop_bench(0.1, 0.2, 0.3, n_iters, self.features, self.res,
                            self.W1, self.b1, self.W2, self.b2, self.has_hidden)
        dt = time.perf_counter() - t0
        return dt / n_iters * 1e9
