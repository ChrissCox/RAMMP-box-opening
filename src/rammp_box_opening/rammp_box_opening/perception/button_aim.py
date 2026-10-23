"""Aiming the press from staging: the button's circle, nothing else.

At staging the wrist camera hangs 12-14 cm above the lid and the button is
the largest circle in the frame. The plateau detector (depth_source) gated
that circle behind the whole box top — height band, footprint against the
model's dims, solid fill, no image border — and at 25 cm every one of those
is a way to refuse a box the camera is looking straight at (2026-09-17: a
4.1-inch OXO refused 103/104 frames as "footprint 0.11x0.10 vs 0.07x0.07"
with the button dead centre in the image). The scene camera has already
said which thing is the box; from here the only question is where its
button is, and a circle of the button's diameter, at lid height, sitting
on a flat lid, answers it without knowing the box's footprint at all.

Owner's design (2026-09-17): scene OWL finds the box, the arm flies over
it, then OpenCV's circle alone aims the press; the OWL-gated wrist scan is
the fallback when the scene camera finds nothing.

The reach-out press (flatbot: the box 0.80-0.89 m out, where the hand
cannot point straight down) sees the button from a tilted pose, 29-39 deg
off the lid's normal. There the button is an ellipse, the disc is not flat
in camera depth and an image annulus is not a lid annulus, so a camera
axis more than OBLIQUE_MIN_DEG off straight down takes find_button_oblique:
the frame is rectified onto a horizontal plane at the lid's prior height,
where the button is a circle again, and every gate is judged in base z.
The lid's height is measured in the frame, not trusted — the scene camera
has put a lid top 21 mm below table_z + dims.z.
"""

import math
from dataclasses import dataclass

import numpy as np

from rammp_box_opening.perception.depth_source import (
    DEPTH_MAX_M,
    DEPTH_MIN_M,
    TOP_RESIDUAL_MAX_M,
)

MIN_RANGE_M = 0.08  # closer than this the button fills the frame; no aim
RADIUS_WINDOW = (0.7, 1.4)  # of the model diameter's pixel radius, as button_circle_refine
DISC_STD_MAX_M = 0.006  # the button top is flat
ANNULUS = (1.15, 1.5)  # of r: the ring around the button is LID, not table
ANNULUS_TOL_M = 0.012
ANNULUS_MIN_FRAC = 0.8
AGREE_M = 0.004  # hits that agree within this (planar) count together
MIN_HITS = 3
KNOB_MAX_UP_M = 0.025  # a popped knob's top is at most this far above the lid (met at ~10 and at 17.7 mm)
# The press point is the CENTRE of the seam circle, and nothing overrides
# it (owner: "make sure the gripper goes to the center of the circle
# segmented out by openCV"). For four days a "dimple" detector did: the
# lowest patch of depth inside the disc, taken for the place a finger
# presses, after a +9.6 mm touch on 2026-09-17 was read as "pads on a ring
# around a dimple". That surface was the button itself, already popped
# (this knob stands ~10 mm proud when up: the aim frame of 2026-09-21 13:35
# has the whole disc +10 mm above its own lid). The patch was stereo noise
# at the sensor's 1 mm steps: it fired on one real aim in four, moved that
# press 13 mm off the centre, and that press missed.

# The oblique path (find_button_oblique).
OBLIQUE_MIN_DEG = 10.0  # optical axis this far off straight down: rectify first
OBLIQUE_MAX_INCIDENCE_DEG = 50.0  # view ray vs the lid's normal at the button, beyond which no aim
# The same angle at the prior (or the optical axis' hit) is only a sanity
# bound: a prior 10-12 cm outward reads 10-14 deg more than the button does.
OBLIQUE_PRIOR_MAX_DEG = 65.0
OBLIQUE_BUTTON_PX = 24.0  # the button's radius in the rectified image (fixes the scale)
OBLIQUE_HALF_M = 0.16  # rectified half-extent: a 12 cm prior error plus the annulus
OBLIQUE_RADIUS_WINDOW = (0.85, 1.2)  # of OBLIQUE_BUTTON_PX; a lid off its prior scales the circle
OBLIQUE_MAX_CIRCLES = 8
LID_PRIOR_TOL_M = 0.035  # the measured lid within this of its prior height
PRIOR_MAX_M = 0.15  # a circle farther than this from prior_xy is not the button
DISC_MIN_FILL = 0.5  # of the disc's rectified cells with valid depth
ANNULUS_MIN_FILL = 0.3  # of the annulus' cells; a popped knob hides its far side
KNOB_NOISE_M = 0.0025  # disc and ring medians closer than this are one height
KNOB_SPLIT_M = 0.006  # disc this far above its lid: two circles the centroid tells apart
# a first frame is still when the camera's pose this long before its stamp
# is its pose at it: over two of the wrist camera's frame intervals (the
# OAK's ~75-90 ms, run logs 2026-10-22), so a stricter test than two
# consecutive frames — and the reach-out aims after a 0.46 s at-rest read
FIRST_LOOKBACK_S = 0.2
OBLIQUE_WINDOW = 3.0  # of the circle's rectified radius: the cells a candidate's gates read


@dataclass(frozen=True)
class ButtonSighting:
    xyz: tuple  # the PRESS POINT, base frame: the circle's centre xy, the disc's median z
    uv: tuple  # the seam circle's pixel centre — the press point's pixel
    r_px: float
    # top-down: the camera z-depth at the centre pixel; oblique: |P - C|,
    # the press point's distance from the camera along its ray
    range_m: float
    diameter_m: float
    # how far the disc stands above the lid ring around it, in THIS frame
    # (nan without depth on the ring): ~0 closed, ~+10 mm with the knob up.
    # A difference inside one frame — the mount and the table cancel.
    above_lid_m: float = float("nan")
    lid_z_m: float = float("nan")  # the lid ring's measured base z (oblique path)
    incidence_deg: float = 0.0  # view ray vs the lid's +z normal at the press point
    oblique: bool = False  # found by find_button_oblique


def circle_gates(depth, k, rot_cam, trans_cam, u, v, r, lid_z):
    """Why this Hough circle is not the button, or None (it may be).

    depth is metres at colour pixels. lid_z: the expected lid top height
    in base. The centre must sit at lid height; the disc must be flat; the
    annulus around it must ALSO be at lid height — a rounded lid corner
    passes the first two and fails the third (the real frame of 2026-09-17
    had four of them at Hough radii 25-34 px)."""
    h, w = depth.shape
    ui, vi = int(round(u)), int(round(v))
    if ui - 3 < 0 or vi - 3 < 0 or ui + 4 > w or vi + 4 > h:
        return "at the image border"
    win = depth[vi - 3 : vi + 4, ui - 3 : ui + 4]
    valid = win[(win > DEPTH_MIN_M) & (win < DEPTH_MAX_M) & np.isfinite(win)]
    if valid.size < 5:
        return "no depth at the centre"
    zc = float(np.median(valid))
    kk = np.asarray(k, float)
    pc = np.array([(u - kk[0, 2]) / kk[0, 0] * zc, (v - kk[1, 2]) / kk[1, 1] * zc, zc])
    p = np.asarray(rot_cam, float) @ pc + np.asarray(trans_cam, float)
    # at lid height — or ABOVE it by as much as a popped knob stands: an
    # open box's button is still its button, and the mission must see it to
    # refuse to press it shut (tasks.press_demo.refuse_open_box)
    if p[2] - lid_z < -TOP_RESIDUAL_MAX_M or p[2] - lid_z > KNOB_MAX_UP_M:
        return "centre %.0f mm from lid height" % (1000 * (p[2] - lid_z))
    v0, u0, d2 = _window_d2(h, w, u, v, ANNULUS[1] * r)
    win = depth[v0 : v0 + d2.shape[0], u0 : u0 + d2.shape[1]]
    disc = win[d2 < (0.8 * r) ** 2]
    disc = disc[(disc > DEPTH_MIN_M) & (disc < DEPTH_MAX_M) & np.isfinite(disc)]
    if disc.size < 20 or float(np.std(disc)) > DISC_STD_MAX_M:
        return "disc not flat (%d px, std %.1f mm)" % (disc.size, 1000 * float(np.std(disc)) if disc.size else 0.0)
    ring = (d2 > (ANNULUS[0] * r) ** 2) & (d2 < (ANNULUS[1] * r) ** 2)
    rd = win[ring]
    ok = (rd > DEPTH_MIN_M) & (rd < DEPTH_MAX_M) & np.isfinite(rd)
    if ok.sum() < 30:
        return "no depth around the circle"
    # heights of the ring's points, in base z (the camera looks down; each
    # pixel's own depth is lifted)
    rv, ru = np.nonzero(ring)
    ring_pts = _lift(depth, kk, rot_cam, trans_cam, ru[ok] + u0, rv[ok] + v0)
    frac = float(np.mean(np.abs(ring_pts[:, 2] - lid_z) < ANNULUS_TOL_M))
    if frac < ANNULUS_MIN_FRAC:
        return "ring around it is not lid (%.0f%% at lid height) — a corner?" % (100 * frac)
    return None


def _window_d2(h, w, u, v, r_max):
    """(v0, u0, d2): the squared pixel distance from (u, v) over the image's
    window that holds every pixel nearer than r_max — outside it no
    strict d2 < r_max ** 2 test can hold, so masks over it are the
    full-frame masks, in the same row-major order. A centre off the frame
    gets an empty window (this numpy's mgrid refuses a stop below its
    start)."""
    m = int(math.ceil(r_max)) + 1
    v0 = min(h, max(0, int(math.floor(v)) - m))
    u0 = min(w, max(0, int(math.floor(u)) - m))
    v1 = max(v0, min(h, int(math.ceil(v)) + m + 1))
    u1 = max(u0, min(w, int(math.ceil(u)) + m + 1))
    yy, xx = np.mgrid[v0:v1, u0:u1]
    return v0, u0, (xx - u) ** 2 + (yy - v) ** 2


def _lift(depth, k, rot_cam, trans_cam, us, vs):
    """Base-frame points for the pixels (us, vs) from their own depth."""
    kk = np.asarray(k, float)
    zc = depth[vs, us]
    pc = np.c_[(us - kk[0, 2]) / kk[0, 0] * zc, (vs - kk[1, 2]) / kk[1, 1] * zc, zc]
    return pc @ np.asarray(rot_cam, float).T + np.asarray(trans_cam, float)


def disc_height(depth, k, rot_cam, trans_cam, u, v, r):
    """The base-frame height of the button's top: the median over the disc
    inside the seam circle (u, v, r), or nan without depth there. One pixel
    at the centre is noisy by 2-3 mm at 25 cm; the disc is what the closed
    pads meet."""
    h, w = depth.shape
    v0, u0, d2 = _window_d2(h, w, u, v, r)
    win = depth[v0 : v0 + d2.shape[0], u0 : u0 + d2.shape[1]]
    disc = d2 < r ** 2
    disc &= (win > DEPTH_MIN_M) & (win < DEPTH_MAX_M) & np.isfinite(win)
    vs, us = np.nonzero(disc)
    if len(us) < 50:
        return float("nan")
    return float(np.median(_lift(depth, k, rot_cam, trans_cam, us + u0, vs + v0)[:, 2]))


def ring_height(depth, k, rot_cam, trans_cam, u, v, r):
    """The base-frame height of the LID around the button: the median over
    the annulus the gates judge (ANNULUS), or nan."""
    h, w = depth.shape
    v0, u0, d2 = _window_d2(h, w, u, v, ANNULUS[1] * r)
    win = depth[v0 : v0 + d2.shape[0], u0 : u0 + d2.shape[1]]
    ring = (d2 > (ANNULUS[0] * r) ** 2) & (d2 < (ANNULUS[1] * r) ** 2)
    ring &= (win > DEPTH_MIN_M) & (win < DEPTH_MAX_M) & np.isfinite(win)
    vs, us = np.nonzero(ring)
    if len(us) < 30:
        return float("nan")
    return float(np.median(_lift(depth, k, rot_cam, trans_cam, us + u0, vs + v0)[:, 2]))


def view_tilt_deg(rot_cam):
    """The angle between the camera's optical axis and straight down."""
    return math.degrees(math.acos(max(-1.0, min(1.0, -float(np.asarray(rot_cam, float)[2, 2])))))


def find_button(color_bgr, depth, k, rot_cam, trans_cam, table_z, model, dist=None, lid_z=None, prior_xy=None):
    """The button in one still frame: (ButtonSighting, None) or (None, why).
    The sighting's xyz is the PRESS POINT: the seam circle's centre.

    `dist` (the colour stream's plumb_bob coefficients) corrects the
    centre for lens distortion before it becomes a ray — the stream is not
    rectified, and near the image edge a pinhole lift was 1-2 mm off
    (2026-09-23). The depth is still read at the raw pixel, where the
    aligned depth lives.

    `lid_z`: the lid top's expected base z (default table_z + dims.z) and
    `prior_xy`: where the button is expected in base xy. Only the oblique
    path uses either: the top-down staging aim is the one it was, its lid at
    table_z + dims.z whatever lid_z says. A camera axis more than
    OBLIQUE_MIN_DEG off straight down goes to find_button_oblique with the
    same arguments."""
    if rot_cam is not None and view_tilt_deg(rot_cam) > OBLIQUE_MIN_DEG:
        return find_button_oblique(
            color_bgr, depth, k, rot_cam, trans_cam, table_z, model, dist=dist, lid_z=lid_z, prior_xy=prior_xy
        )
    import cv2

    if color_bgr is None or depth is None or k is None:
        return None, "no frame"
    kk = np.asarray(k, float)
    lid_z = float(table_z) + float(model.dims[2])
    rng = float(trans_cam[2]) - lid_z  # the camera looks straight down at staging
    if rng < MIN_RANGE_M:
        return None, "camera %.0f mm above the lid — too close to aim" % (1000 * rng)
    r_px = kk[0, 0] * (float(model.button_diameter_m) / 2.0) / rng
    r_lo, r_hi = int(RADIUS_WINDOW[0] * r_px), int(math.ceil(RADIUS_WINDOW[1] * r_px))
    if r_lo < 6:
        return None, "button would be %.0f px — too far to aim" % r_px
    gray = cv2.medianBlur(cv2.cvtColor(np.ascontiguousarray(color_bgr), cv2.COLOR_BGR2GRAY), 3)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=2 * r_px, param1=90, param2=14,
        minRadius=r_lo, maxRadius=r_hi,
    )
    if circles is None:
        return None, "no circle of %.0f-%.0f px radius in the frame" % (r_lo, r_hi)
    whys = []
    for u, v, r in circles[0][:8]:  # Hough's own order: strongest first
        why = circle_gates(depth, kk, rot_cam, trans_cam, float(u), float(v), float(r), lid_z)
        if why is not None:
            whys.append(why)
            continue
        ui, vi = int(round(u)), int(round(v))
        win = depth[vi - 3 : vi + 4, ui - 3 : ui + 4]
        zc = float(np.median(win[(win > DEPTH_MIN_M) & (win < DEPTH_MAX_M) & np.isfinite(win)]))
        # the centre's xy from its own (undistorted) pixel at its own range;
        # the height is the disc's median — the honest "top" for the
        # calibration line
        uu, vv = float(u), float(v)
        if dist is not None and np.any(np.asarray(dist, float)):
            uu, vv = cv2.undistortPoints(np.array([[[uu, vv]]], np.float64), kk, np.asarray(dist, float), P=kk)[0, 0]
        pc = np.array([(uu - kk[0, 2]) / kk[0, 0] * zc, (vv - kk[1, 2]) / kk[1, 1] * zc, zc])
        p = np.asarray(rot_cam, float) @ pc + np.asarray(trans_cam, float)
        z_disc = disc_height(depth, kk, rot_cam, trans_cam, float(u), float(v), float(r))
        z_top = z_disc if np.isfinite(z_disc) else float(p[2])
        z_ring = ring_height(depth, kk, rot_cam, trans_cam, float(u), float(v), float(r))
        return ButtonSighting(
            xyz=(float(p[0]), float(p[1]), z_top), uv=(float(u), float(v)), r_px=float(r),
            range_m=zc, diameter_m=2.0 * float(r) * zc / kk[0, 0],
            above_lid_m=float(z_disc - z_ring),
        ), None
    return None, "%d circle(s), none the button: %s" % (len(circles[0]), "; ".join(whys[:3]))


def _project(pts, k, rot_cam, trans_cam, dist, shape):
    """Raw pixels (N, 2) of base-frame points (N, 3); nan behind the camera
    or, by the pinhole, beyond a quarter frame outside the image (there a
    distortion polynomial can fold a point back into the frame)."""
    pc = (np.atleast_2d(np.asarray(pts, float)) - np.asarray(trans_cam, float)) @ np.asarray(rot_cam, float)
    return _project_cam(pc, k, dist, shape)


def _project_cam(pc, k, dist, shape):
    """_project for points (N, 3) already in the camera frame."""
    import cv2

    kk = np.asarray(k, float)
    out = np.full((len(pc), 2), np.nan, pc.dtype)
    z = np.where(pc[:, 2] > 1e-6, pc[:, 2], np.nan)
    u = kk[0, 0] * pc[:, 0] / z + kk[0, 2]
    v = kk[1, 1] * pc[:, 1] / z + kk[1, 2]
    h, w = shape
    with np.errstate(invalid="ignore"):
        near = (u > -0.25 * w) & (u < 1.25 * w) & (v > -0.25 * h) & (v < 1.25 * h)
    if dist is None:
        out[near] = np.c_[u[near], v[near]]
    elif near.any():
        uv, _ = cv2.projectPoints(pc[near].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), kk, dist)
        out[near] = uv.reshape(-1, 2)
    return out


def _pixel_rays(us, vs, k, dist):
    """Normalised camera coordinates (x/z, y/z) of raw pixels."""
    import cv2

    kk = np.asarray(k, float)
    if dist is None:
        return (us - kk[0, 2]) / kk[0, 0], (vs - kk[1, 2]) / kk[1, 1]
    nrm = cv2.undistortPoints(np.c_[us, vs].astype(np.float64).reshape(-1, 1, 2), kk, dist).reshape(-1, 2)
    return nrm[:, 0], nrm[:, 1]


def _incidence_deg(p, cam):
    """The angle between the ray p -> camera and the lid's +z normal."""
    ray = np.asarray(cam, float) - np.asarray(p, float)
    return math.degrees(math.acos(max(-1.0, min(1.0, ray[2] / float(np.linalg.norm(ray))))))


def _oblique_circle(lifted, cu, cv, r, s, c0, g, h0, cam, rb):
    """((P, h_disc, h_ring, f), None) for the rectified circle (cu, cv, r),
    or (None, why). lifted: (n, n, 3) base points of the cells' depth; f
    scales the rectified circle to metric at the height it lies at. Only the
    cells within OBLIQUE_WINDOW * r of the circle are read: every sample a
    gate counts lies within 2.5 rb of the centre and a knob's height of it,
    and such a sample's ray crosses h0 no farther out than that."""
    half = int(math.ceil(OBLIQUE_WINDOW * r))
    i0, j0 = max(0, int(cv) - half), max(0, int(cu) - half)
    win = lifted[i0 : int(cv) + half + 1, j0 : int(cu) + half + 1]
    yy = np.arange(i0, i0 + win.shape[0], dtype=np.float32)[:, None]  # the window's cells
    xx = np.arange(j0, j0 + win.shape[1], dtype=np.float32)[None, :]
    zz = win[..., 2]
    fin = np.isfinite(zz)
    cell = (xx - cu) ** 2 + (yy - cv) ** 2 < (0.8 * r) ** 2
    zd = zz[cell & fin]
    if zd.size < max(20, DISC_MIN_FILL * cell.sum()):
        return None, "too little depth on the disc (%d of %d cells)" % (zd.size, cell.sum())
    h_circle = float(np.median(zd))
    if not -LID_PRIOR_TOL_M - TOP_RESIDUAL_MAX_M <= h_circle - h0 <= LID_PRIOR_TOL_M + KNOB_MAX_UP_M:
        return None, "disc %.0f mm from the lid's prior height" % (1000 * (h_circle - h0))
    p_rect = np.array([g[0] + (cu - c0) / s, g[1] + (cv - c0) / s, h0])

    def centre(h):
        # the circle on h0 is a circle at h seen through C: slide it along
        # the rays (exact for a circle parallel to the plane)
        return cam + (p_rect - cam) * (cam[2] - h) / (cam[2] - h0)

    def levels(p):
        # disc and lid ring by each sample's OWN base xy: metric, whatever the view
        q = np.where(fin, (win[..., 0] - p[0]) ** 2 + (win[..., 1] - p[1]) ** 2, np.inf)
        return q, zz[q < (0.8 * rb) ** 2], zz[(q > (ANNULUS[0] * rb) ** 2) & (q < (ANNULUS[1] * rb) ** 2)]

    # Which height the circle lies at: the disc's ("disc"), the lid ring's
    # ("ring"), or halfway between ("mid").
    at = "disc"
    p = centre(h_circle)
    q, _, zr = levels(p)
    if zr.size >= 30:
        h_lid = float(np.median(zr))
        if h_circle - h_lid < -KNOB_NOISE_M:
            # Recessed: the circle is the hole's rim, on the lid, and through
            # it shows the floor's far part (the near rim hides the rest).
            # Slid to the floor's height the centre went tan(incidence) x
            # the depth away from the camera: 8 mm at -10 mm and 39 deg.
            at = "ring"
        elif h_circle - h_lid > KNOB_SPLIT_M:
            # A proud knob shows two circles: its top's edge, at the disc's
            # height, and its base on the lid, at the ring's — and they meet
            # the rays tan(incidence) * height apart. Its top is wholly in
            # view: the centroid of the points nearer its height than the
            # lid's says which.
            with np.errstate(invalid="ignore"):
                top = (q < (2.5 * rb) ** 2) & (np.abs(zz - h_circle) < np.abs(zz - h_lid))
            if top.sum() >= 20:
                cxy = win[top][:, :2].mean(0)
                at = "ring" if np.hypot(*(centre(h_lid)[:2] - cxy)) < np.hypot(*(p[:2] - cxy)) else "disc"
        elif h_circle - h_lid > KNOB_NOISE_M:
            # A knob a few mm proud: too low for its top's points to stand
            # out of the depth noise, so which circle Hough took is unknown.
            # Halfway between them is off by at most half their split
            # (tan(incidence) x height / 2: 2.4 mm at 6 mm and 39 deg).
            at = "mid"
        h_circle = {"disc": h_circle, "ring": h_lid, "mid": 0.5 * (h_circle + h_lid)}[at]
    for _ in range(2):  # the heights around the moved centre, and the centre again
        p = centre(h_circle)
        _, zd, zr = levels(p)
        if zd.size < 20 or zr.size < 30:
            break
        h_d, h_r = float(np.median(zd)), float(np.median(zr))
        h_circle = {"disc": h_d, "ring": h_r, "mid": 0.5 * (h_d + h_r)}[at]
    p = centre(h_circle)
    _, zd, zr = levels(p)
    h_disc = float(np.median(zd)) if zd.size else h_circle
    h_ring = float(np.median(zr)) if zr.size else h_circle

    def cells(area, h):  # the rectified cells a metric area at height h covers
        return area * (s * (cam[2] - h0) / (cam[2] - h)) ** 2

    expect = cells(math.pi * (0.8 * rb) ** 2, h_disc)
    if zd.size < max(20, DISC_MIN_FILL * expect):
        return None, "too little depth on the disc (%d of ~%.0f cells)" % (zd.size, expect)
    if float(np.std(zd)) > DISC_STD_MAX_M:
        return None, "disc not flat in base z (std %.1f mm)" % (1000 * float(np.std(zd)))
    expect = cells(math.pi * (ANNULUS[1] ** 2 - ANNULUS[0] ** 2) * rb ** 2, h_ring)
    if zr.size < max(30, ANNULUS_MIN_FILL * expect):
        return None, "too little depth around the circle (%d of ~%.0f cells)" % (zr.size, expect)
    frac = float(np.mean(np.abs(zr - h_ring) < ANNULUS_TOL_M))
    if frac < ANNULUS_MIN_FRAC:
        return None, "ring around it is not one lid (%.0f%% at its height) — a corner?" % (100 * frac)
    # The same ring by where each cell's RAY crosses its plane. Past the
    # lid's far rim the box hides the table from this view, so no sample's
    # own xy lands there and the ring above reads all lid for a circle
    # straddling that rim; its rays went on past the lid, to land lower. A
    # ray met higher up is not counted: a popped knob hides the far side.
    fr = (cam[2] - h_ring) / (cam[2] - h0)
    rx = cam[0] + (g[0] + (xx - c0) / s - cam[0]) * fr - p[0]
    ry = cam[1] + (g[1] + (yy - c0) / s - cam[1]) * fr - p[1]
    rho2 = rx ** 2 + ry ** 2
    zb = zz[fin & (rho2 > (ANNULUS[0] * rb) ** 2) & (rho2 < (ANNULUS[1] * rb) ** 2)]
    on_lid = int(np.sum(np.abs(zb - h_ring) < ANNULUS_TOL_M))
    past = int(np.sum(zb <= h_ring - ANNULUS_TOL_M))
    if on_lid + past < 30:
        return None, "no lid seen around the circle (%d cells)" % (on_lid + past)
    if on_lid < ANNULUS_MIN_FRAC * (on_lid + past):
        return None, "ring around it is not one lid (%.0f%% of its rays meet it) — a corner?" % (
            100.0 * on_lid / (on_lid + past)
        )
    if abs(h_ring - h0) > LID_PRIOR_TOL_M:
        return None, "lid ring %.0f mm from its prior height" % (1000 * (h_ring - h0))
    up = h_disc - h_ring
    if up < -TOP_RESIDUAL_MAX_M or up > KNOB_MAX_UP_M:
        return None, "disc %.0f mm from the lid around it" % (1000 * up)
    return (np.r_[p[:2], h_disc], h_disc, h_ring, (cam[2] - h_circle) / (cam[2] - h0)), None


def _prior_miss_m(cu, cv, s, c0, g, h0, cam, prior_xy):
    """How near the rectified circle (cu, cv) can come to prior_xy at any
    height the gates allow it: the circle on h0 slid along the rays over
    that band of heights is a segment in xy."""
    p_rect = np.array([g[0] + (cu - c0) / s, g[1] + (cv - c0) / s, h0])
    ends = [
        (cam + (p_rect - cam) * (cam[2] - h) / (cam[2] - h0))[:2]
        for h in (h0 + LID_PRIOR_TOL_M + KNOB_MAX_UP_M, h0 - LID_PRIOR_TOL_M - TOP_RESIDUAL_MAX_M)
    ]
    seg = ends[1] - ends[0]
    w = np.asarray(prior_xy, float) - ends[0]
    t = min(1.0, max(0.0, float(w @ seg) / max(float(seg @ seg), 1e-12)))
    return float(np.hypot(*(w - t * seg)))


def find_button_oblique(color_bgr, depth, k, rot_cam, trans_cam, table_z, model, dist=None, lid_z=None, prior_xy=None):
    """The button seen from a tilted pose: (ButtonSighting, None) or (None, why).

    The frame is resampled onto the horizontal plane z = h0 (lid_z, default
    table_z + dims.z) on a metric grid centred on prior_xy (else where the
    optical axis meets the plane). There a horizontal circle is a circle.
    Each cell's depth is lifted along its own raw pixel's ray, so every gate
    reads base z and the lid ring's height is measured, not assumed: it
    must lie within LID_PRIOR_TOL_M of h0. The circle found on h0 is then
    moved along the camera's rays to the disc's measured height h:
    P = C + (P' - C) (Cz - h) / (Cz - h0)."""
    import cv2

    if color_bgr is None or depth is None or k is None:
        return None, "no frame"
    kk = np.asarray(k, float)
    rot = np.asarray(rot_cam, float)
    cam = np.asarray(trans_cam, float)
    dd = np.asarray(dist, float).ravel() if dist is not None and np.any(np.asarray(dist, float)) else None
    h0 = float(table_z) + float(model.dims[2]) if lid_z is None else float(lid_z)
    rb = float(model.button_diameter_m) / 2.0
    hh, ww = depth.shape[:2]
    if cam[2] - h0 < MIN_RANGE_M:  # the range is at least this
        return None, "camera %.0f mm above the lid — too close to aim" % (1000 * (cam[2] - h0))
    if prior_xy is not None:
        g = np.array([float(prior_xy[0]), float(prior_xy[1]), h0])
        uv = _project(g, kk, rot, cam, dd, (hh, ww))[0]
        if not (np.all(np.isfinite(uv)) and 0 <= uv[0] <= ww - 1 and 0 <= uv[1] <= hh - 1):
            return None, "the prior button (%.3f, %.3f) projects outside the image" % (g[0], g[1])
    else:
        axis = rot[:, 2]
        if axis[2] > -1e-3:
            return None, "the optical axis does not meet the lid plane"
        g = cam + axis * (h0 - cam[2]) / axis[2]
    inc = _incidence_deg(g, cam)  # at the prior, not the button: a sanity bound only
    if inc > OBLIQUE_PRIOR_MAX_DEG:
        return None, "the lid seen %.0f deg off its normal — too oblique to aim" % inc

    s = OBLIQUE_BUTTON_PX / rb  # rectified px per metre
    n = 2 * int(round(OBLIQUE_HALF_M * s)) + 1
    c0 = (n - 1) / 2.0
    ax = ((np.arange(n) - c0) / s).astype(np.float32)
    # the cells in the camera frame, in float32 and by rows and columns
    # (columns along base x, rows along base y): R^T (g - C) + x R^T ex + y R^T ey
    rf = rot.astype(np.float32)
    pc = (rot.T @ (g - cam)).astype(np.float32) + ax[None, :, None] * rf[0] + ax[:, None, None] * rf[1]
    uv = _project_cam(pc.reshape(-1, 3), kk, dd, (hh, ww))
    with np.errstate(invalid="ignore"):
        valid = (uv[:, 0] >= 0) & (uv[:, 0] <= ww - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= hh - 1)
    if valid.sum() < 100:
        return None, "the lid plane around the button is not in the frame"
    uvf = np.where(valid[:, None], uv, np.float32(-1.0)).astype(np.float32)
    mx = uvf[:, 0].reshape(n, n)
    my = uvf[:, 1].reshape(n, n)
    rect = cv2.remap(np.ascontiguousarray(color_bgr), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    vgrid = valid.reshape(n, n)
    gray[~vgrid] = np.uint8(np.median(gray[vgrid]))  # no edge where the frame ends
    gray = cv2.medianBlur(gray, 3)
    # depth by nearest pixel, lifted along that pixel's own ray
    us = np.rint(uvf[valid, 0]).astype(int)
    vs = np.rint(uvf[valid, 1]).astype(int)
    d = np.asarray(depth)[vs, us].astype(float)
    ok = np.isfinite(d) & (d > DEPTH_MIN_M) & (d < DEPTH_MAX_M)
    if ok.sum() < 100:
        return None, "no depth on the lid plane around the button"
    xn, yn = _pixel_rays(us[ok], vs[ok], kk, dd)
    lifted = np.full((n * n, 3), np.nan, np.float32)
    lifted[np.flatnonzero(valid)[ok]] = np.c_[xn * d[ok], yn * d[ok], d[ok]] @ rot.T + cam
    lifted = lifted.reshape(n, n, 3)

    r0 = OBLIQUE_BUTTON_PX
    r_lo, r_hi = int(OBLIQUE_RADIUS_WINDOW[0] * r0), int(math.ceil(OBLIQUE_RADIUS_WINDOW[1] * r0))
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=2 * r0, param1=90, param2=14,
        minRadius=r_lo, maxRadius=r_hi,
    )
    if circles is None:
        return None, "no circle of %d-%d px radius on the rectified lid plane" % (r_lo, r_hi)
    cands = [(float(u), float(v), float(r)) for u, v, r in circles[0]]  # strongest first
    if prior_xy is not None:
        # nearest the prior first (the scene camera chose that box); the grid
        # is centred on the prior, so a cell's offset from c0 is its offset
        # from it ON h0. The sort is stable: equally near, strongest first.
        # A lid above h0 meets the rays nearer the camera than h0 does, so
        # the cut is where the circle can lie at any allowed height; the
        # metric check after the correction is the one that holds.
        off = [math.hypot(u - c0, v - c0) / s for u, v, _ in cands]
        near = sorted(
            (i for i in range(len(cands)) if _prior_miss_m(*cands[i][:2], s, c0, g, h0, cam, prior_xy) <= PRIOR_MAX_M),
            key=lambda i: off[i],
        )
        if not near:
            return None, "%d circle(s), all over %.0f mm from the prior" % (len(cands), 1000 * PRIOR_MAX_M)
        cands = [cands[i] for i in near]
    cands = cands[:OBLIQUE_MAX_CIRCLES]
    whys = []
    for cu, cv, r in cands:
        hit, why = _oblique_circle(lifted, cu, cv, r, s, c0, g, h0, cam, rb)
        if hit is None:
            whys.append(why)
            continue
        p, h_disc, h_ring, f = hit
        if prior_xy is not None and math.hypot(p[0] - prior_xy[0], p[1] - prior_xy[1]) > PRIOR_MAX_M:
            whys.append("%.0f mm from the prior" % (1000 * math.hypot(p[0] - prior_xy[0], p[1] - prior_xy[1])))
            continue
        inc = _incidence_deg(p, cam)
        if inc > OBLIQUE_MAX_INCIDENCE_DEG:
            whys.append("seen %.0f deg off its normal — too oblique" % inc)
            continue
        rng = float(np.linalg.norm(p - cam))
        diameter = 2.0 * r / s * f
        uv_p = _project(p, kk, rot, cam, dd, (hh, ww))[0]
        return ButtonSighting(
            xyz=(float(p[0]), float(p[1]), h_disc), uv=(float(uv_p[0]), float(uv_p[1])),
            r_px=float(kk[0, 0] * diameter / 2.0 / rng), range_m=rng, diameter_m=float(diameter),
            above_lid_m=h_disc - h_ring, lid_z_m=h_ring, incidence_deg=inc, oblique=True,
        ), None
    return None, "%d circle(s) on the rectified lid plane, none the button: %s" % (len(cands), "; ".join(whys[:3]))


def agreeing_median(points, tol=AGREE_M, n=MIN_HITS):
    """The median of the last n points when they agree pairwise within tol
    (planar), else None."""
    if len(points) < n:
        return None
    last = np.asarray(points[-n:], float)
    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(*(last[i, :2] - last[j, :2])) > tol:
                return None
    return tuple(float(v) for v in np.median(last, axis=0))


class ButtonAimer:
    """Frames from a D405Grabber, still ones only, until MIN_HITS agree.
    lid_z and prior_xy go to find_button (see there)."""

    def __init__(self, grab, model, table_z, lid_z=None, prior_xy=None):
        self.grab = grab
        self.model = model
        self.table_z = float(table_z)
        self.lid_z = None if lid_z is None else float(lid_z)
        self.prior_xy = None if prior_xy is None else (float(prior_xy[0]), float(prior_xy[1]))
        self.last_why = None
        self.frames = 0
        self.hits = 0
        self.points = []
        self.above_lid = []
        self._last_stamp = None
        self._last_depth_stamp = None  # the depth that last voted (depth_paired)
        self._last_cam = None
        self.last_sighting = None
        self.pair_dt_ms = []  # each hit's depth-minus-colour stamp (ms), when known

    def tick(self):
        """One frame if there is a new one: True when a fix is ready."""
        from rammp_box_opening.perception.d405 import frame_snapshot
        from rammp_box_opening.perception.depth_source import (
            camera_is_still, camera_pose_at, depth_key, depth_paired, pair_dt_ms,
        )

        g = self.grab
        # one read of the frame: the dedup key, the pose's stamp and the
        # pixels judged all belong to it even if a newer one lands meanwhile
        f = frame_snapshot(g)
        if f.color is None or f.depth is None or f.k is None or f.color_stamp is None:
            self.last_why = "no wrist frame yet"
            return False
        stamp = (f.color_stamp.sec, f.color_stamp.nanosec)
        if stamp == self._last_stamp:
            return False
        paired, why = depth_paired(f, self._last_depth_stamp)
        if not paired:
            # not consumed: its depth may not have landed yet, and the next
            # tick tries this colour frame again; a stalled depth never aims
            self.last_why = why
            return False
        self._last_stamp = stamp
        self._last_depth_stamp = depth_key(f)
        self.frames += 1
        cam = camera_pose_at(g, stamp=f.color_stamp)
        if cam is None:
            self.last_why = "no camera pose for the frame"
            return False
        prev = self._last_cam
        if prev is None:
            # the first frame has no frame before it: the camera's pose
            # FIRST_LOOKBACK_S before its stamp stands in for one (TF keeps
            # the history) — the same still test, over at least one frame
            # interval, so a parked arm's first frame votes
            off = float(getattr(g, "stamp_offset_s", 0.0) or 0.0)
            prev = camera_pose_at(g, offset_s=off - FIRST_LOOKBACK_S, stamp=f.color_stamp)
        still = camera_is_still(prev, cam)
        self._last_cam = cam
        if not still:
            self.last_why = "camera moving"
            return False
        rot, trans = cam
        s, why = find_button(
            f.color, f.depth, f.k, rot, trans, self.table_z, self.model, dist=f.dist,
            lid_z=self.lid_z, prior_xy=self.prior_xy,
        )
        if s is None:
            self.last_why = why
            return False
        self.hits += 1
        self.last_sighting = s
        self.points.append(s.xyz)
        self.above_lid.append(s.above_lid_m)
        dt = pair_dt_ms(f)
        if dt is not None:
            self.pair_dt_ms.append(dt)
        return agreeing_median(self.points) is not None

    def button_above_lid_m(self):
        """How far the button stands above its own lid, over the hits the
        fix was made from (median), or None when none of them could say."""
        vals = [v for v in self.above_lid[-MIN_HITS:] if np.isfinite(v)]
        return float(np.median(vals)) if vals else None

    def fix(self):
        return agreeing_median(self.points)

    def status(self):
        return "%d/%d still frames found the button circle%s" % (
            self.hits, self.frames, "" if self.last_why is None else " (last reject: %s)" % self.last_why
        )
