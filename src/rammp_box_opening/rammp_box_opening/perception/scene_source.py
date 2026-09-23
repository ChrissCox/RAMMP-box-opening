"""Finding the box from the scene camera, before the arm moves.

The fixed scene camera sees the whole workspace from a metre away. That is
the wrong camera for AIMING a press — a lid is 30 pixels wide from there —
and exactly the right one for the question the mission used to answer by
flying the wrist camera around: where, roughly, is the box? A coarse pose
from here sends the arm straight to staging above the box; the wrist camera
then takes its precise fix from 12 cm, where it is good.

Semantics first, geometry second. The scene is not the same every time and
carries other containers, plates and cables, so a box-sized plateau alone
is not enough: the OWL node's scene instance says WHICH thing is the box
(a pixel box), and the depth points inside that box that sit at lid height
above the table give its position. No smoothness or footprint gates — the
scene depth at a metre is noisier than those gates were tuned for, and the
wrist fix that follows carries the precision.
"""

from dataclasses import dataclass

import numpy as np

from rammp_box_opening.perception.depth_source import container_pose_from_top
from rammp_box_opening.perception.planes import fit_plane

LID_BAND_M = 0.035  # the top must sit this close to table + dims.z
TOP_SLAB_M = 0.015  # the lid top: points within this of the highest surface
TOP_PCTL = 97  # "highest" as a percentile, so flying pixels do not define it
MIN_LID_POINTS = 40
YAW_MIN_POINTS = 150  # fewer than this and the min-area rect's yaw is noise
# the top slab must be a FACE, not a rim: a taller container's near face
# crosses the lid band as a thin strip, and a strip must not read as a box
MIN_FOOTPRINT_FRAC = 0.4  # of the model's shorter side, as a 10-90 % spread
# ... and not much LARGER than the model either: a black 19 cm tray next to
# the 10 cm OXO passed every other gate and the OWL's best box flipped
# between the two (2026-09-17). The longer spread must fit the longer side.
MAX_FOOTPRINT_FRAC = 1.4


@dataclass(frozen=True)
class SceneFix:
    pose: object  # ContainerPose (bottom-centre origin, yaw)
    top_xyz: tuple  # the lid top's centre in base_link
    n_points: int
    score: float  # the OWL score behind it
    table_z_scene: float  # the table height the scene depth itself shows, or nan


def robust_yaw(xy):
    """The square's orientation, mod 90 deg, as the angle whose axis-aligned
    3rd-97th percentile box is tightest. A min-area rectangle is set by the
    outermost points, so one stray sets it; a 3 % trim is not. The trim
    must stay small, though: a square's projection barely changes width
    between 10 % and 90 %, so a 10-90 box hardly knows the angle at all."""
    xy = np.asarray(xy, float)

    def area(theta):
        c, s = np.cos(-theta), np.sin(-theta)
        r = xy @ np.array([[c, -s], [s, c]]).T
        lo, hi = np.percentile(r, [3, 97], axis=0)
        return float(np.prod(hi - lo))

    coarse = np.radians(np.arange(0.0, 90.0, 1.0))
    best = coarse[int(np.argmin([area(t) for t in coarse]))]
    fine = best + np.radians(np.arange(-1.0, 1.0, 0.1))
    best = fine[int(np.argmin([area(t) for t in fine]))]
    return float(best % (np.pi / 2))


def box_from_scene_points(pts_base, uv, bbox, table_z, model):
    """Coarse container pose from scene depth points inside a pixel box.

    pts_base: (N, 3) depth points in base_link; uv: their colour pixels
    (N, 2); bbox: (x0, y0, x1, y1) in colour pixels. Returns
    (ContainerPose, top_xyz, n_points) or (None, why, 0)."""
    x0, y0, x1, y1 = bbox
    inside = (uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)
    if not inside.any():
        return None, "no depth inside the box", 0
    lid_z = float(table_z) + float(model.dims[2])
    z = pts_base[:, 2]
    band = inside & (np.abs(z - lid_z) < LID_BAND_M)
    if band.sum() < MIN_LID_POINTS:
        return None, "only %d depth points at lid height inside the box (need %d)" % (int(band.sum()), MIN_LID_POINTS), int(band.sum())
    # The TOP face is the highest surface in the pixel box. From the scene
    # camera's oblique view the box's near face is in the box too, and its
    # upper part sits inside the lid band: taking every band point dragged
    # the centre 5 cm toward the camera (live, 2026-09-16). Keep only the
    # slab just under the highest points.
    top_z = float(np.percentile(z[band], TOP_PCTL))
    lid = band & (z > top_z - TOP_SLAB_M)
    n = int(lid.sum())
    if n < MIN_LID_POINTS:
        return None, "only %d depth points on the top face (need %d)" % (n, MIN_LID_POINTS), n
    top = pts_base[lid]
    centre = np.median(top, axis=0)
    # Flying pixels at lid height land inside the box too, and one stray a
    # hand-width away is enough to stretch a min-area rectangle across the
    # gate. So the footprint is measured as the 10th-90th percentile spread
    # along the slab's principal axes, which a stray cannot move, and the
    # rectangle that gives the yaw is fitted to the core between the 5th
    # and 95th percentiles only.
    xy = top[:, :2] - centre[:2]
    _w, vecs = np.linalg.eigh(np.cov(xy.T))
    proj = xy @ vecs  # columns: minor, major axis
    lo10, hi90 = np.percentile(proj, [10, 90], axis=0)
    spread = hi90 - lo10
    need = MIN_FOOTPRINT_FRAC * min(float(model.dims[0]), float(model.dims[1]))
    if spread.min() < need:
        return None, (
            "the top slab is a strip %.0fx%.0f mm across, not a face (need %.0f mm) — "
            "something taller crossing lid height?" % (1000 * spread[0], 1000 * spread[1], 1000 * need)
        ), n
    too_big = MAX_FOOTPRINT_FRAC * max(float(model.dims[0]), float(model.dims[1]))
    if spread.max() > too_big:
        return None, (
            "the top is %.0fx%.0f mm, the container %.0fx%.0f — another container"
            % (1000 * spread[0], 1000 * spread[1], 1000 * model.dims[0], 1000 * model.dims[1])
        ), n
    yaw = robust_yaw(xy) if n >= YAW_MIN_POINTS else 0.0
    pose = container_pose_from_top((centre[0], centre[1], centre[2]), yaw, model, table_z)
    return pose, (float(centre[0]), float(centre[1]), float(centre[2])), n


@dataclass(frozen=True)
class SceneBoxFix:
    pose: object  # ContainerPose, or None when no box was this container
    top: tuple  # the lid top's centre (base) — None without a pose
    n: int  # lid points behind it
    box: object  # the OWL candidate that won ([x0, y0, x1, y1]), or None
    tried: tuple  # why each refused candidate was refused, in order


def scene_fix_from_boxes(boxes, pts_base, uv, table_z, model):
    """The OWL's candidates (score, [x0, y0, x1, y1]) best first, each
    lifted to the lid slab (box_from_scene_points); the first that is this
    container's size wins. What the live locate() does with every message,
    and what an offline replay of recorded frames computes the same way."""
    tried = []
    for score, box in boxes:
        pose, top, n = box_from_scene_points(pts_base, uv, box, table_z, model)
        if pose is not None:
            return SceneBoxFix(pose, top, n, box, tuple(tried))
        tried.append("%s (OWL %.2f)" % (top, score))
    return SceneBoxFix(None, None, 0, None, tuple(tried))


def table_from_scene_points(pts_base, uv, image_h, row_frac=0.6):
    """The table height the scene depth shows, from a plane fit over the
    lower part of the image; nan when there is no clean plane. Logged
    beside the surveyed height so a wrong survey, or a knocked camera,
    shows up in every run."""
    low = uv[:, 1] > row_frac * image_h
    if low.sum() < 500:
        return float("nan")
    # a few thousand points are plenty for a table plane, and this number
    # is only PRINTED: over the full lower image (~150k points) the RANSAC
    # cost 1.6-2.3 s of a 3.9 s locate (profiled live 2026-09-17)
    fit = fit_plane(pts_base[low], tol_m=0.008, iters=150, max_points=3000)
    if fit is None or fit[2] < 0.3:
        return float("nan")
    n, c, _frac = fit
    if abs(n[2]) < 0.9:
        return float("nan")  # the dominant plane down there is not horizontal
    return float(c[2])


class SceneLocator:
    """The ROS side: the scene grabber, the scene OWL rung and the
    calibration, answering `locate()` with a SceneFix or None."""

    def __init__(self, node, cfg, model, table_z, T_base_link, spins=True, tag_px=None):
        from rammp_box_opening.perception.owl_source import OwlRung
        from rammp_box_opening.perception.scene import SceneGrabber

        self.node = node
        # spins=False: the node is spun elsewhere (the mission's perception
        # thread); locate() then only waits, it must not spin it too
        self.spins = bool(spins)
        self.cfg = cfg
        self.model = model
        self.table_z = float(table_z)
        self.T_base_link = np.asarray(T_base_link, dtype=float)
        self.grab = SceneGrabber(node, keep=3, need_depth=True)
        self.owl = OwlRung(node, cfg, camera="scene")
        self.last_why = None
        self.last_timing = {}
        # where the calibration saw the cabinet's tag (scene_calib.scene_tag_px):
        # checked once per locate, before any fix is trusted
        self.tag_px = tag_px
        self.moved_note = None  # "tag not in view" and the like, for the log

    def ready(self):
        return not self.grab.missing() and self.grab.depth_to_color() is not None and self.grab.link_to_color() is not None

    def locate(self, timeout_s):
        """Enable the scene OWL, wait for its box, lift the depth inside it."""
        import time

        import rclpy

        from rammp_box_opening.perception.owl_source import bboxes_in_msg

        t0 = time.monotonic()
        self.owl.enable()
        self.last_timing = {}  # where a locate's seconds went (the SCENE line prints it)
        try:
            heard = False  # any message at all from the scene instance
            t_ready = None
            tried = []  # (why) per candidate refused, for the log
            last_msg = None
            found = None
            while time.monotonic() - t0 < timeout_s:
                if self.spins:
                    rclpy.spin_once(self.node, timeout_sec=0.05)
                else:
                    time.sleep(0.02)
                if not self.ready():
                    continue
                if t_ready is None:
                    t_ready = time.monotonic()
                    self.last_timing["streams_s"] = round(t_ready - t0, 2)
                    # before any fix: is the camera where it was calibrated?
                    # A re-aimed camera put every fix 9-11 cm off, run after
                    # run, and nothing noticed (2026-09-23)
                    from rammp_box_opening.perception.scene_calib import scene_camera_moved

                    moved, why = scene_camera_moved(self.grab.color, self.tag_px)
                    self.moved_note = why
                    if moved:
                        self.last_why = why
                        return None
                if self.owl.latest is None:
                    continue
                heard = True
                # the newest BOX, not the newest message: a heartbeat that
                # lands milliseconds after it must not hide it (2026-09-23)
                m = self.owl.fresh_box()
                if m is None or m is last_msg:
                    continue
                last_msg = m
                # every candidate the OWL offered in this message, best
                # first, against the container's geometry: the first one
                # whose top is at lid height AND the container's size wins
                if "bbox_s" not in self.last_timing:
                    self.last_timing["bbox_s"] = round(time.monotonic() - t_ready, 2)
                t_cloud = time.monotonic()
                cloud = self.grab.cloud_in_color_frame(depth=self.grab.depth_median(), stride=2)
                if cloud is None:
                    self.last_why = "no scene depth cloud"
                    return None
                pc, uv = cloud
                T_base_color = self.T_base_link @ self.grab.link_to_color()
                pts_base = pc @ T_base_color[:3, :3].T + T_base_color[:3, 3]
                cands = bboxes_in_msg(m)
                got = scene_fix_from_boxes(
                    [(c[4], c[:4]) for c in cands], pts_base, uv, self.table_z, self.model
                )
                tried.extend(got.tried)
                if got.pose is not None:
                    cand = next(c for c in cands if c[:4] == got.box)
                    found = (cand, got.pose, got.top, got.n, pts_base, uv)
                if found is not None:
                    self.last_timing["lift_s"] = round(time.monotonic() - t_cloud, 2)
                    break
            if found is None:
                # three different failures wear the same "no box" face: the
                # camera is down, the OWL instance is silent (not started,
                # still loading, or getting no frames — a launch-started
                # instance was seen like that, 2026-09-16), or it is alive
                # and honestly sees no box
                self.last_why = (
                    "scene camera streams missing: %s" % ", ".join(self.grab.missing())
                    if self.grab.missing()
                    else "the scene OWL is SILENT (no heartbeat in %.1f s) — not started, still "
                    "loading, or not receiving frames: check the launch terminal" % timeout_s
                    if not heard
                    else "the scene OWL saw no box in %.1f s" % timeout_s
                    if not tried
                    else "none of the OWL's %d candidate(s) is this container: %s" % (len(tried), "; ".join(tried[-3:]))
                )
                return None
            bbox, pose, top, n, pts_base, uv = found
            table_seen = table_from_scene_points(pts_base, uv, self.grab.color.shape[0])
            return SceneFix(pose=pose, top_xyz=top, n_points=n, score=float(bbox[4]), table_z_scene=table_seen)
        finally:
            self.owl.disable()
