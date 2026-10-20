"""Plane geometry on depth clouds: the surfaces a scene is made of.

A depth camera looking at a workspace sees a few large flat things — the
table, a cabinet door, a wall — and each one is thousands of points that
agree on a normal. That agreement is a far better orientation reference
than any small fiducial: a 57 mm tag seen from a metre is 32 pixels wide
and its pose flips between two mirror solutions from frame to frame
(measured 34 deg of scatter, 2026-09-16), while the door it is stuck to
holds its normal to a degree.

Pure numpy; nothing here touches ROS.
"""

import numpy as np


def cloud_from_depth(depth_m, k, stride=1):
    """Depth image (metres, 0 = invalid) -> (points (N,3) in the camera's
    optical frame, u (N,), v (N,)) for the valid pixels."""
    d = np.asarray(depth_m, dtype=float)[::stride, ::stride]
    kk = np.asarray(k, dtype=float)
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    u = u * stride
    v = v * stride
    ok = np.isfinite(d) & (d > 0)
    z = d[ok]
    x = (u[ok] - kk[0, 2]) / kk[0, 0] * z
    y = (v[ok] - kk[1, 2]) / kk[1, 1] * z
    return np.stack([x, y, z], axis=1), u[ok], v[ok]


def fit_plane(points, tol_m=0.004, iters=300, max_points=40000, rng=None):
    """RANSAC plane through `points` -> (unit normal, centroid, inlier fraction).

    The normal's sign is arbitrary; callers orient it (toward the camera,
    upward, ...). Points beyond `max_points` are subsampled first — a
    histogram's worth of evidence is plenty and a 400k-point cloud is not
    worth the wait. Returns None when there are too few points."""
    P = np.asarray(points, dtype=float)
    if len(P) < 3:
        return None
    rng = np.random.default_rng(0) if rng is None else rng
    if len(P) > max_points:
        P = P[rng.choice(len(P), max_points, replace=False)]
    best_n, best_inl = 0, None
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        inl = np.abs((P - a) @ n) < tol_m
        cnt = int(inl.sum())
        if cnt > best_n:
            best_n, best_inl = cnt, inl
    if best_inl is None or best_n < 3:
        return None
    Q = P[best_inl]
    c = Q.mean(axis=0)
    _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
    n = vt[2] / np.linalg.norm(vt[2])
    return n, c, best_n / float(len(P))


def ray_plane(ray, normal, point_on_plane):
    """The point where the ray from the origin along `ray` meets the plane."""
    ray = np.asarray(ray, dtype=float)
    n = np.asarray(normal, dtype=float)
    denom = n @ ray
    if abs(denom) < 1e-12:
        return None
    s = (n @ np.asarray(point_on_plane, dtype=float)) / denom
    return ray * s


def angle_deg(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
