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


@dataclass(frozen=True)
class ButtonSighting:
    xyz: tuple  # the PRESS POINT, base frame: the circle's centre xy, the disc's median z
    uv: tuple  # the seam circle's pixel centre — the press point's pixel
    r_px: float
    range_m: float
    diameter_m: float
    # how far the disc stands above the lid ring around it, in THIS frame
    # (nan without depth on the ring): ~0 closed, ~+10 mm with the knob up.
    # A difference inside one frame — the mount and the table cancel.
    above_lid_m: float = float("nan")


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
    yy, xx = np.mgrid[0:h, 0:w]
    d2 = (xx - u) ** 2 + (yy - v) ** 2
    disc = depth[d2 < (0.8 * r) ** 2]
    disc = disc[(disc > DEPTH_MIN_M) & (disc < DEPTH_MAX_M) & np.isfinite(disc)]
    if disc.size < 20 or float(np.std(disc)) > DISC_STD_MAX_M:
        return "disc not flat (%d px, std %.1f mm)" % (disc.size, 1000 * float(np.std(disc)) if disc.size else 0.0)
    ring = (d2 > (ANNULUS[0] * r) ** 2) & (d2 < (ANNULUS[1] * r) ** 2)
    rd = depth[ring]
    ok = (rd > DEPTH_MIN_M) & (rd < DEPTH_MAX_M) & np.isfinite(rd)
    if ok.sum() < 30:
        return "no depth around the circle"
    # heights of the ring's points, in base z (the camera looks down; each
    # pixel's own depth is lifted)
    rv, ru = np.nonzero(ring)
    ring_pts = _lift(depth, kk, rot_cam, trans_cam, ru[ok], rv[ok])
    frac = float(np.mean(np.abs(ring_pts[:, 2] - lid_z) < ANNULUS_TOL_M))
    if frac < ANNULUS_MIN_FRAC:
        return "ring around it is not lid (%.0f%% at lid height) — a corner?" % (100 * frac)
    return None


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
    yy, xx = np.mgrid[0:h, 0:w]
    disc = (xx - u) ** 2 + (yy - v) ** 2 < r ** 2
    disc &= (depth > DEPTH_MIN_M) & (depth < DEPTH_MAX_M) & np.isfinite(depth)
    vs, us = np.nonzero(disc)
    if len(us) < 50:
        return float("nan")
    return float(np.median(_lift(depth, k, rot_cam, trans_cam, us, vs)[:, 2]))


def ring_height(depth, k, rot_cam, trans_cam, u, v, r):
    """The base-frame height of the LID around the button: the median over
    the annulus the gates judge (ANNULUS), or nan."""
    h, w = depth.shape
    yy, xx = np.mgrid[0:h, 0:w]
    d2 = (xx - u) ** 2 + (yy - v) ** 2
    ring = (d2 > (ANNULUS[0] * r) ** 2) & (d2 < (ANNULUS[1] * r) ** 2)
    ring &= (depth > DEPTH_MIN_M) & (depth < DEPTH_MAX_M) & np.isfinite(depth)
    vs, us = np.nonzero(ring)
    if len(us) < 30:
        return float("nan")
    return float(np.median(_lift(depth, k, rot_cam, trans_cam, us, vs)[:, 2]))


def find_button(color_bgr, depth, k, rot_cam, trans_cam, table_z, model, dist=None):
    """The button in one still frame: (ButtonSighting, None) or (None, why).
    The sighting's xyz is the PRESS POINT: the seam circle's centre.

    `dist` (the colour stream's plumb_bob coefficients) corrects the
    centre for lens distortion before it becomes a ray — the stream is not
    rectified, and near the image edge a pinhole lift was 1-2 mm off
    (2026-09-23). The depth is still read at the raw pixel, where the
    aligned depth lives."""
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
    """Frames from a D405Grabber, still ones only, until MIN_HITS agree."""

    def __init__(self, grab, model, table_z):
        self.grab = grab
        self.model = model
        self.table_z = float(table_z)
        self.last_why = None
        self.frames = 0
        self.hits = 0
        self.points = []
        self.above_lid = []
        self._last_stamp = None
        self._last_cam = None
        self.last_sighting = None

    def tick(self):
        """One frame if there is a new one: True when a fix is ready."""
        from rammp_box_opening.perception.depth_source import camera_is_still, camera_pose_at

        g = self.grab
        if g.color is None or g.depth is None or g.k is None or g.color_stamp is None:
            self.last_why = "no wrist frame yet"
            return False
        stamp = (g.color_stamp.sec, g.color_stamp.nanosec)
        if stamp == self._last_stamp:
            return False
        self._last_stamp = stamp
        self.frames += 1
        cam = camera_pose_at(g)
        if cam is None:
            self.last_why = "no camera pose for the frame"
            return False
        still = camera_is_still(self._last_cam, cam)
        self._last_cam = cam
        if not still:
            self.last_why = "camera moving"
            return False
        rot, trans = cam
        s, why = find_button(g.color, g.depth, g.k, rot, trans, self.table_z, self.model, dist=getattr(g, "dist", None))
        if s is None:
            self.last_why = why
            return False
        self.hits += 1
        self.last_sighting = s
        self.points.append(s.xyz)
        self.above_lid.append(s.above_lid_m)
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
