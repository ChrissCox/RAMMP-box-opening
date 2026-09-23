"""The wrist camera's timestamp lag, and how to measure it.

A depth frame is stamped by the camera driver; the arm's joint states are
stamped by the arm driver. If those two clocks disagree by a constant dt,
a frame taken while the camera moves at v is placed v*dt from where it
was really taken — at half a metre a second, 30 ms is 15 mm. That bias is
why the precise detector only trusts STILL frames today, and why the arm
has to stop above the box before it can aim.

The lag is a constant, so it can be measured once: record frames during
a motion over a static box, with the camera pose looked up at the frame's
stamp plus a sweep of candidate offsets, and pick the offset at which the
moving frames agree with the still ones. `best_offset` is that pick, on
data the recorder (scripts/record_scan_frames.py --lag-sweep) writes.
"""

import numpy as np

# candidate offsets, seconds: +/- 80 ms in 10 ms steps
OFFSETS_S = np.round(np.arange(-0.08, 0.0801, 0.01), 3)
# a frame counts as MOVING when the camera travelled more than this since
# the previous one (the still filter's own threshold, depth_source)
MOVING_M = 0.004


def best_offset(offsets, fixes, moving):
    """Which offset makes the moving frames agree with the still ones.

    offsets: (K,) candidate seconds. fixes: (N, K, 3) the box top found in
    frame n with the camera pose taken at stamp + offsets[k], nan where the
    frame yielded no fix. moving: (N,) bool. Returns (offset_s, table) where
    table lists (offset, rms_mm, n_moving_fixes) per candidate, or (None,
    table) when there is not enough to decide."""
    offsets = np.asarray(offsets, float)
    fixes = np.asarray(fixes, float)
    moving = np.asarray(moving, bool)
    still = ~moving
    table = []
    for k, off in enumerate(offsets):
        ref_pts = fixes[still, k, :]
        ref_pts = ref_pts[np.isfinite(ref_pts).all(axis=1)]
        mov_pts = fixes[moving, k, :]
        mov_pts = mov_pts[np.isfinite(mov_pts).all(axis=1)]
        if len(ref_pts) < 3 or len(mov_pts) < 3:
            table.append((float(off), float("nan"), int(len(mov_pts))))
            continue
        ref = np.median(ref_pts, axis=0)
        rms = float(np.sqrt(np.mean(np.sum((mov_pts - ref) ** 2, axis=1))))
        table.append((float(off), 1000.0 * rms, int(len(mov_pts))))
    usable = [(rms, off) for off, rms, n in table if np.isfinite(rms)]
    if not usable:
        return None, table
    return min(usable)[1], table


def moving_flags(trans_by_frame, threshold_m=MOVING_M):
    """(N,) bool: which frames the camera was moving through, from the
    camera translation at each frame's own stamp."""
    t = np.asarray(trans_by_frame, float)
    if len(t) == 0:
        return np.zeros(0, bool)
    step = np.r_[0.0, np.linalg.norm(np.diff(t, axis=0), axis=1)]
    return step > threshold_m
