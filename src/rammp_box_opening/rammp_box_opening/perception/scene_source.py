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

A camera level with the lid never sees a top face (flatbot's, on a table
that high — tables change, so which view holds is not assumed): when the
top-face lift finds nothing, the nearest object in the pixel box is checked
against the table around it and placed half its width behind the face the
camera sees (box_from_side_points). With no survey, the table the box was
found on is measured for the planner too (table_extent).
"""

from dataclasses import dataclass

import numpy as np

from rammp_box_opening.perception.clear_table import MAP_R_M
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
    table_extent: object = None  # (x0, x1, y0, y1) where it shows that table, when no survey says
    partial: str = ""  # how the frame cut the box when it was finished from the part in view, else ""
    # the scene's points around the box (float32, base frame): what stands on
    # the table there, for where the lid may go (perception/clear_table)
    table_points: object = None


def robust_yaw(xy):
    """The square's orientation, mod 90 deg, as the angle whose axis-aligned
    3rd-97th percentile box is tightest. A min-area rectangle is set by the
    outermost points, so one stray sets it; a 3 % trim is not. The trim
    must stay small, though: a square's projection barely changes width
    between 10 % and 90 %, so a 10-90 box hardly knows the angle at all."""
    xy = np.asarray(xy, float)

    def best_of(th):
        # every angle's rotation in one stacked matmul and its trimmed box
        # in one percentile call (a Python loop of these was 16 ms)
        c, s = np.cos(-th), np.sin(-th)
        rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
        lo, hi = np.percentile(xy @ rot.transpose(0, 2, 1), [3, 97], axis=1)
        return th[int(np.argmin(np.prod(hi - lo, axis=-1)))]

    best = best_of(np.radians(np.arange(0.0, 90.0, 1.0)))
    best = best_of(best + np.radians(np.arange(-1.0, 1.0, 0.1)))
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


# A camera level with the lid sees the box's SIDE, never its top face (the
# flatbot scene OAK on a table this high; a lower table shows it the top).
SIDE_DEPTH_SLACK_M = 0.03  # the nearest object: this far past its own width behind its front
ABOVE_TABLE_M = 0.015  # an object point sits at least this far above the table
TABLE_RING_M = (0.08, 0.30)  # the table is read this far (horizontally) from the box's centre
TABLE_SLAB_M = 0.015  # ... from the points within this of the ring's commonest height
MIN_TABLE_POINTS = 300
TABLE_EXTENT_R_M = 0.6  # the table's extent is read this far around the box


def table_extent(pts_base, table_z, centre_xy, radius=TABLE_EXTENT_R_M):
    """(x0, x1, y0, y1): where the scene depth shows the table — the points
    within TABLE_SLAB_M of its height, within `radius` of the box — as their
    2nd-98th percentile box, so stray points do not set it. None when too
    few points say. Only what the camera sees: the table behind the box, or
    out of view, is not in it."""
    d = np.hypot(pts_base[:, 0] - centre_xy[0], pts_base[:, 1] - centre_xy[1])
    on = pts_base[(d < radius) & (np.abs(pts_base[:, 2] - table_z) < TABLE_SLAB_M)]
    if len(on) < MIN_TABLE_POINTS:
        return None
    lo, hi = np.percentile(on[:, :2], 2, axis=0), np.percentile(on[:, :2], 98, axis=0)
    return (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]))


def seen_table_extent(pts_base, table_z, centre_xy):
    """(extent, assumed): table_extent, or, when too few points say, the
    ring square table_around measured the height from (assumed True), so a
    measured height never reaches the planner as no table at all."""
    ext = table_extent(pts_base, table_z, centre_xy)
    if ext is not None:
        return ext, False
    r = TABLE_RING_M[1]
    cx, cy = float(centre_xy[0]), float(centre_xy[1])
    return (cx - r, cx + r, cy - r, cy + r), True


def table_around(pts_base, centre_xy, below_z, ring=TABLE_RING_M):
    """The table's height under `centre_xy`: the commonest height (1 cm
    bins) in a ring around it below `below_z`, refined by a plane through
    the points near that height and read at the centre. A camera level with
    the table also sees under it (flatbot 2026-10-22: streaks down to the
    floor through the gap below its front edge); those spread over every
    height while the table piles into one. nan when the ring holds too
    little."""
    d = np.hypot(pts_base[:, 0] - centre_xy[0], pts_base[:, 1] - centre_xy[1])
    ringed = pts_base[(d > ring[0]) & (d < ring[1]) & (pts_base[:, 2] < below_z)]
    if len(ringed) < MIN_TABLE_POINTS:
        return float("nan")
    hist, edges = np.histogram(ringed[:, 2], bins=np.arange(ringed[:, 2].min(), below_z + 0.01, 0.01))
    k = int(np.argmax(hist))
    slab = ringed[np.abs(ringed[:, 2] - 0.5 * (edges[k] + edges[k + 1])) < TABLE_SLAB_M]
    if len(slab) < MIN_TABLE_POINTS:
        return float("nan")
    fit = fit_plane(slab, tol_m=0.008, iters=150, max_points=3000)
    if fit is None or abs(fit[0][2]) < 0.9:
        return float(np.median(slab[:, 2]))
    n, c, _frac = fit
    return float(c[2] - (n[0] * (centre_xy[0] - c[0]) + n[1] * (centre_xy[1] - c[1])) / n[2])


def box_from_side_points(pts_base, uv, bbox, table_z, model, cam_xyz, partial=False):
    """Coarse container pose from a camera level with the lid. It sees the
    box's side, never its top face, and the wall behind a round lid falls
    inside the pixel box too (flatbot 2026-10-22: 22 of 252 points, at lid
    height, stretched the top-face lift's "top" to 621 mm). So: the nearest
    object in the box, its top checked against the table around it
    (`table_z` when surveyed), its centre half its width behind the face the
    camera sees. Returns (ContainerPose, top_xyz, n_points, table_z) or
    (None, why, n_points, table_z or nan).

    `partial`: the frame cuts the box off at its side (a round container
    only, scene_fix_from_boxes): what is in view is not centred on it and
    is narrower than it, so the centre is the lid's circle through the arc
    of its side in view (lid_circle) instead, and no width is asked."""
    x0, y0, x1, y1 = bbox
    inside = (uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)
    if inside.sum() < MIN_LID_POINTS:
        return None, "only %d depth points inside the box (need %d)" % (int(inside.sum()), MIN_LID_POINTS), int(inside.sum()), float("nan")
    cam = np.asarray(cam_xyz, float)
    q = pts_base[inside]
    rng = np.linalg.norm(q - cam, axis=1)
    half = 0.5 * min(float(model.dims[0]), float(model.dims[1]))
    near = q[rng <= np.percentile(rng, 5) + 2 * half + SIDE_DEPTH_SLACK_M]
    upper = near[near[:, 2] >= np.median(near[:, 2])]
    along = upper[:, :2].mean(axis=0) - cam[:2]  # the view across the box, horizontally
    along /= np.linalg.norm(along)
    across = np.array([-along[1], along[0]])

    def centre_of(p):
        rel = p[:, :2] - cam[:2]
        return cam[:2] + (np.percentile(rel @ along, 5) + half) * along + np.median(rel @ across) * across

    if table_z is None:
        table_z = table_around(pts_base, centre_of(upper), float(np.percentile(near[:, 2], TOP_PCTL)) - 0.03)
        if np.isnan(table_z):
            return None, "no table around it", len(near), table_z
    obj = near[near[:, 2] > table_z + ABOVE_TABLE_M]
    n = len(obj)
    if n < MIN_LID_POINTS:
        return None, "only %d depth points above the table (need %d)" % (n, MIN_LID_POINTS), n, table_z
    top_z = float(np.percentile(obj[:, 2], TOP_PCTL))
    lid_z = float(table_z) + float(model.dims[2])
    if abs(top_z - lid_z) > LID_BAND_M:
        return None, "its top is %+.0f mm from lid height (the table at %.3f + %.0f mm)" % (
            1000 * (top_z - lid_z), table_z, 1000 * model.dims[2]), n, table_z
    if partial:
        # the lid's points from one lid diameter around what the box shows, at its height
        d = float(model.lid_dims[0])
        seed = near[:, :2].mean(axis=0)
        reach = np.linalg.norm(pts_base - cam, axis=1) <= np.percentile(rng, 5) + 2 * half + SIDE_DEPTH_SLACK_M
        around = reach & (np.hypot(pts_base[:, 0] - seed[0], pts_base[:, 1] - seed[1]) <= d + PARTIAL_REACH_M)
        band = pts_base[around & (pts_base[:, 2] > top_z - float(model.lid_dims[2])) & (pts_base[:, 2] <= top_z + 0.005)]
        centre, why = lid_circle(band, model, cam)
        if centre is None:
            return None, why, n, table_z
        pose = container_pose_from_top((centre[0], centre[1], top_z), 0.0, model, table_z)  # round: yaw unused
        return pose, (centre[0], centre[1], top_z), n, float(table_z)
    lo, hi = np.percentile((obj[:, :2] - cam[:2]) @ across, [10, 90])
    width = float(hi - lo)
    need = MIN_FOOTPRINT_FRAC * min(float(model.dims[0]), float(model.dims[1]))
    if width < need:
        return None, "it is %.0f mm across, not a container (need %.0f mm)" % (1000 * width, 1000 * need), n, table_z
    too_big = MAX_FOOTPRINT_FRAC * max(float(model.dims[0]), float(model.dims[1]))
    if width > too_big:
        return None, "it is %.0f mm across, the container %.0f — another container" % (
            1000 * width, 1000 * max(model.dims[0], model.dims[1])), n, table_z
    cx, cy = centre_of(obj)
    yaw = robust_yaw(obj[:, :2] - (cx, cy)) if n >= YAW_MIN_POINTS else 0.0
    pose = container_pose_from_top((cx, cy, top_z), yaw, model, table_z)
    return pose, (float(cx), float(cy), top_z), n, float(table_z)


@dataclass(frozen=True)
class SceneBoxFix:
    pose: object  # ContainerPose, or None when no box was this container
    top: tuple  # the lid top's centre (base) — None without a pose
    n: int  # lid points behind it
    box: object  # the OWL candidate that won ([x0, y0, x1, y1]), or None
    tried: tuple  # why each refused candidate was refused, in order
    table_z: float = float("nan")  # the table the winner was measured against
    lift: str = ""  # "top" or "side": which view of it won
    clipped: tuple = ()  # (score, box, edges) of each candidate refused at the frame's edge
    partial: tuple = ()  # the edges the frame cut the winner at, when it was finished from part of it


# An OWL box this close to the frame's edge has its container running off
# the frame: the camera sees a part of it, and a lift of a part puts the box
# off by up to half of it (flatbot 2026-10-23: the canister cut by the scene
# image's edge was fixed 7 cm short of where the wrist camera then found it,
# and the staging aimed past its button)
FRAME_EDGE_PX = 3


# ... unless the container is round and the frame cuts it at its side: then
# the arc of the lid's side band in view is a circle of the lid's known
# diameter, fitted with that radius (lid_circle). Flatbot 2026-10-23: the
# canister half off the frame put 147 deg of its lid in view and the fit
# put it 3 cm from where the wrist camera found it, the old lift 7 cm.
PARTIAL_INLIER_M = 0.006  # a lid point this close to the circle is on it (the side OAK's noise at 0.3 m)
PARTIAL_MIN_ARC_DEG = 60.0  # the points on it spread over at least this much of it
# A tight OWL box on a cut lid holds a short arc, and a circle of known
# radius slides along a short arc: live 2026-10-23 the same canister fixed
# at [0.497, -0.285] in most frames and [0.445, -0.235] in the rest, where
# it stood. So the lid's points are taken from one lid diameter around what
# the box shows (PARTIAL_REACH_M), the circle is found by RANSAC, and a
# rival circle this far from the best that holds nearly as many points is
# no fix at all.
PARTIAL_REACH_M = 0.02  # beyond one lid diameter from the box's own near points
PARTIAL_RIVAL_M = 0.035
PARTIAL_RIVAL_FRAC = 0.85
PARTIAL_RANSAC_ITERS = 400


def lid_circle(band, model, cam_xyz):
    """The centre (x, y) of a round lid from the points of its side band a
    level camera sees (`band`, base frame): the circle of the lid's radius
    through the most of them (RANSAC over point pairs, the centre behind the
    pair as the camera sees it), refined on those. Returns ((cx, cy), None)
    or (None, why)."""
    r = 0.5 * float(model.lid_dims[0])
    if len(band) < MIN_LID_POINTS:
        return None, "only %d points on the lid's side in view (need %d)" % (len(band), MIN_LID_POINTS)
    P = np.asarray(band, float)[:, :2]
    cam = np.asarray(cam_xyz, float)[:2]
    rng = np.random.default_rng(0)  # the same frame, the same fix
    cands = []
    for _ in range(PARTIAL_RANSAC_ITERS):
        i, j = rng.choice(len(P), 2, replace=False)
        m, chord = 0.5 * (P[i] + P[j]), P[j] - P[i]
        dist = float(np.linalg.norm(chord))
        if dist < 0.01 or dist >= 2 * r:
            continue
        normal = np.array([-chord[1], chord[0]]) / dist
        h = float(np.sqrt(r * r - 0.25 * dist * dist))
        for c in (m + h * normal, m - h * normal):
            if np.dot(c - m, m - cam) > 0.0:  # behind the pair, from the camera
                on = np.abs(np.linalg.norm(P - c, axis=1) - r) < PARTIAL_INLIER_M
                cands.append((int(on.sum()), c))
    if not cands:
        return None, "no %.0f mm circle through the lid's side in view" % (2000 * r)
    cands.sort(key=lambda kc: -kc[0])
    best_n, c = cands[0][0], cands[0][1].copy()
    rival = next((k for k, cc in cands if np.linalg.norm(cc - c) > PARTIAL_RIVAL_M), 0)
    if rival >= PARTIAL_RIVAL_FRAC * best_n:
        return None, "the lid's arc in view fits two places %.0f+ mm apart" % (1000 * PARTIAL_RIVAL_M)
    on = np.abs(np.linalg.norm(P - c, axis=1) - r) < PARTIAL_INLIER_M
    for _it in range(30):
        dv = P[on] - c
        dist = np.maximum(np.linalg.norm(dv, axis=1), 1e-9)
        step = np.linalg.lstsq(-dv / dist[:, None], -(dist - r), rcond=None)[0]
        c = c + step
        if np.linalg.norm(step) < 1e-7:
            break
    on = np.abs(np.linalg.norm(P - c, axis=1) - r) < PARTIAL_INLIER_M
    if on.sum() < MIN_LID_POINTS:
        return None, "only %d of the lid's side points lie on a %.0f mm circle" % (int(on.sum()), 2000 * r)
    ang = np.sort(np.arctan2(P[on, 1] - c[1], P[on, 0] - c[0]))
    arc = 360.0 - float(np.degrees(np.diff(np.r_[ang, ang[0] + 2 * np.pi]).max()))
    if arc < PARTIAL_MIN_ARC_DEG:
        return None, "only %.0f deg of the lid's circle in view (need %.0f)" % (arc, PARTIAL_MIN_ARC_DEG)
    return (float(c[0]), float(c[1])), None


def frame_edges(box, image_wh, edge_px=FRAME_EDGE_PX):
    """The edges ("left", "top", "right", "bottom") of a w x h image that
    the pixel box [x0, y0, x1, y1] reaches."""
    w, h = image_wh
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    hits = (("left", x0 <= edge_px), ("top", y0 <= edge_px),
            ("right", x1 >= w - 1 - edge_px), ("bottom", y1 >= h - 1 - edge_px))
    return tuple(e for e, hit in hits if hit)


def side_cut(edges, R_base_color):
    """True when every edge in `edges` cuts the view across, not up or down:
    its image direction is level in the base frame (a camera hung level, as
    flatbot's scene OAK is, cuts a box at its left or right side)."""
    R = np.asarray(R_base_color, float)
    col = {"left": 0, "right": 0, "top": 1, "bottom": 1}
    return all(abs(R[2, col[e]]) < 0.5 for e in edges)


def off_frame_ways(edges, R_base_color):
    """The base-frame axis ("+y", ...) each image edge in `edges` lies
    toward: the image's own left and right mean nothing on a camera hung
    upside down."""
    R = np.asarray(R_base_color, float)
    out = {"right": R[:, 0], "left": -R[:, 0], "bottom": R[:, 1], "top": -R[:, 1]}
    ways = []
    for e in edges:
        d = out[e]
        k = int(np.argmax(np.abs(d)))
        ways.append(("+" if d[k] > 0 else "-") + "xyz"[k])
    return ways


def off_frame_why(score, edges, R_base_color, frame="arm_mount"):
    """What to tell the operator about a box the OWL found running off the
    frame at `edges`: which way that is in `frame` (off_frame_ways), and the
    way back."""
    ways = off_frame_ways(edges, R_base_color)
    back = ", ".join(("-" if w[0] == "+" else "+") + w[1] for w in ways)
    return ("the OWL's box (%.2f) runs off the scene camera's frame toward %s %s: it sees part of the "
            "container, and a part is not where the box is — move the box toward %s until all of it is "
            "in the camera's view" % (score, frame, " and ".join(ways), back))


# The top face is seen only from above it. A camera that stands less than
# this slope above the lid (rise over horizontal distance to the box, ~6
# deg) sees the lid's side band edge-on, and that band passes the top-face
# lift's footprint gates: flatbot 2026-10-22, camera z 0.400, lid top 0.397 —
# the top lift won or lost on a few pixels of the OWL box and moved the fix
# 13 mm and the lid 7.5 mm between runs of the same placement.
TOP_VIEW_MIN_SLOPE = 0.1


def sees_top_face(pts_base, uv, bbox, lid_z, cam_xyz):
    """Does a camera at `cam_xyz` stand high enough over a lid at `lid_z`
    to see its top face, for the depth points inside `bbox`?"""
    x0, y0, x1, y1 = bbox
    inside = (uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)
    if not inside.any():
        return True  # the lift refuses it with its own reason
    cam = np.asarray(cam_xyz, float)
    d = float(np.median(np.hypot(pts_base[inside, 0] - cam[0], pts_base[inside, 1] - cam[1])))
    return cam[2] - float(lid_z) >= TOP_VIEW_MIN_SLOPE * d


def scene_fix_from_boxes(boxes, pts_base, uv, table_z, model, cam_xyz=None, side_table_z=None, image_wh=None,
                         cam_R=None):
    """The OWL's candidates (score, [x0, y0, x1, y1]) best first: every
    candidate lifted to the lid slab (box_from_scene_points) when a table
    height is known (`table_z`) and the camera stands above the lid
    (sees_top_face), then — with `cam_xyz`, the camera's position — every
    candidate from its side (box_from_side_points), against `side_table_z`
    when surveyed or the table around it. Which view works depends on how
    high the table stands against the camera, so neither is assumed. Two
    passes, not both lifts per candidate: the side lift never measures how
    far a container runs along the view, so a tray the top-face lift just
    refused would pass it ahead of the real box. The first that is this
    container wins. What the live locate() does with every message, and
    what an offline replay of recorded frames computes the same way. With
    `image_wh` (the colour image's w, h), a candidate reaching the frame's
    edge is refused before either lift (FRAME_EDGE_PX) — unless the model
    is round and the frame cuts it at its side (`cam_R`, the camera's axes
    in the base frame, says which image edges are), when the side lift
    finishes it from the part in view (partial)."""
    tried = []
    clipped = []
    cands = []  # (score, box, the side edges a round box is finished across, or ())
    for score, box in boxes:
        edges = frame_edges(box, image_wh) if image_wh is not None else ()
        if not edges:
            cands.append((score, box, ()))
        elif model.round and cam_R is not None and side_cut(edges, cam_R):
            cands.append((score, box, edges))
        else:
            clipped.append((score, box, edges))
            tried.append("it runs off the frame at its %s edge (OWL %.2f)" % (" and ".join(edges), score))
    if table_z is not None:
        lid_z = float(table_z) + float(model.dims[2])
        for score, box, edges in cands:
            if edges:
                continue  # only the side lift finishes a box from part of it
            if cam_xyz is not None and not sees_top_face(pts_base, uv, box, lid_z, cam_xyz):
                continue
            pose, top, n = box_from_scene_points(pts_base, uv, box, table_z, model)
            if pose is not None:
                return SceneBoxFix(pose, top, n, box, tuple(tried), float(table_z), "top", tuple(clipped))
            tried.append("%s (OWL %.2f)" % (top, score))
    if cam_xyz is not None:
        for score, box, edges in cands:
            pose, top, n, tz = box_from_side_points(pts_base, uv, box, side_table_z, model, cam_xyz, partial=bool(edges))
            if pose is not None:
                return SceneBoxFix(pose, top, n, box, tuple(tried), tz, "side", tuple(clipped), edges)
            if edges:
                clipped.append((score, box, edges))
                tried.append("from the side, cut off at the frame's %s edge, %s (OWL %.2f)" % (" and ".join(edges), top, score))
            else:
                tried.append("from the side, %s (OWL %.2f)" % (top, score))
    return SceneBoxFix(None, None, 0, None, tuple(tried), clipped=tuple(clipped))


def table_from_scene_points(pts_base, uv, image_h, row_frac=0.6):
    """The table height the scene depth shows, from a plane fit over the
    lower part of the (upright) view; nan when there is no clean plane.
    Logged beside the surveyed height so a wrong survey, or a knocked
    camera, shows up in every run."""
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


# |image-down . base +z| below this: the camera looks too steeply down for
# "which way up" to mean anything
UPRIGHT_COS = 0.5


def image_upside_down(T_base_color):
    """True when the colour image's down axis points up in the base frame,
    False when it points down, None when the camera looks too steeply down
    to tell."""
    down_z = float(np.asarray(T_base_color)[2, 1])
    if abs(down_z) < UPRIGHT_COS:
        return None
    return down_z > 0


class SceneLocator:
    """The ROS side: the scene grabber, the scene OWL rung and the
    calibration, answering `locate()` with a SceneFix or None."""

    def __init__(self, node, cfg, model, table_z, T_base_link, spins=True, tag_px=None,
                 scene_cfg=None, base_frame=None):
        from rammp_box_opening.perception.owl_source import OwlRung
        from rammp_box_opening.perception.scene import SceneGrabber

        self.node = node
        # spins=False: the node is spun elsewhere (the mission's perception
        # thread); locate() then only waits, it must not spin it too
        self.spins = bool(spins)
        self.cfg = cfg
        self.model = model
        self.table_z = float(table_z) if table_z is not None else None
        # None: the robot's TF carries the camera's mount (a world with
        # external_static_tf), read as base_frame <- the camera's link
        self.T_base_link = None if T_base_link is None else np.asarray(T_base_link, dtype=float)
        self.base_frame = base_frame
        self.grab = SceneGrabber(node, keep=3, need_depth=True, scene_cfg=scene_cfg)
        self.owl = OwlRung(node, cfg, camera="scene")
        self.last_why = None
        self.last_timing = {}
        # where the calibration saw the cabinet's tag (scene_calib.scene_tag_px):
        # checked once per locate, before any fix is trusted
        self.tag_px = tag_px
        self.moved_note = None  # "tag not in view" and the like, for the log

    def base_to_link(self):
        """T(base <- the camera's link): the calibration's, or TF's."""
        if self.T_base_link is not None:
            return self.T_base_link
        return self.grab.lookup(self.base_frame, self.grab.link_frame)

    def base_to_color(self):
        T_bl, T_lc = self.base_to_link(), self.grab.link_to_color()
        return None if T_bl is None or T_lc is None else T_bl @ T_lc

    def ready(self):
        return not self.grab.missing() and self.grab.depth_to_color() is not None and self.base_to_color() is not None

    def _unplaced_why(self):
        if self.grab.depth_to_color() is None:
            return "no TF between the scene camera's depth and colour frames"
        if self.T_base_link is None:
            return "no TF from %s to the scene camera (%s) — is sheppy's static_tf node up?" % (
                self.base_frame, self.grab.link_frame)
        return "no TF from the scene camera's link to its colour frame"

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
            off_frame = None  # (score, edges, R_base_color) of the best candidate cut by the frame's edge
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
                    # the OWL needs upright frames, and nothing here turns them:
                    # a camera hung upside down rotates them on the device
                    if image_upside_down(self.base_to_color()):
                        self.last_why = (
                            "TF has the scene camera's frames upside down — the OWL needs them upright: "
                            "rotate them on the device (the OAK driver's color/depth i_rotation 180, "
                            "rammp-deployments #133) with the upright mount in TF"
                        )
                        return None
                    # is the calibration's tag where it was? Said, never
                    # refused: it rides the cabinet door (scene_calib
                    # TAG_MOVED_PX); a re-aimed camera is caught by what its
                    # fixes measure instead (press_demo.scene_trust)
                    from rammp_box_opening.perception.scene_calib import scene_camera_moved

                    _moved, why = scene_camera_moved(self.grab.color, self.tag_px)
                    self.moved_note = why
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
                T_base_color = self.base_to_color()
                if T_base_color is None:
                    continue
                pts_base = pc @ T_base_color[:3, :3].T + T_base_color[:3, 3]
                cands = bboxes_in_msg(m)
                # the top-face lift needs a table height up front: surveyed,
                # else read off the image rows (no clean plane there is no
                # refusal: the side lift reads the table around each box)
                top_table_z = self.table_z
                if top_table_z is None:
                    measured = table_from_scene_points(pts_base, uv, self.grab.color.shape[0])
                    top_table_z = None if np.isnan(measured) else measured
                got = scene_fix_from_boxes(
                    [(c[4], c[:4]) for c in cands], pts_base, uv, top_table_z, self.model,
                    cam_xyz=T_base_color[:3, 3], side_table_z=self.table_z,
                    image_wh=(self.grab.color.shape[1], self.grab.color.shape[0]), cam_R=T_base_color[:3, :3],
                )
                tried.extend(got.tried)
                for score, _box, edges in got.clipped:
                    if off_frame is None or score > off_frame[0]:
                        off_frame = (score, edges, T_base_color[:3, :3])
                if got.pose is not None:
                    cand = next(c for c in cands if c[:4] == got.box)
                    found = (cand, got.pose, got.top, got.n, pts_base, uv, got)
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
                    else self._unplaced_why()
                    if t_ready is None
                    else "the scene OWL is SILENT (no heartbeat in %.1f s) — not started, still "
                    "loading, or not receiving frames: check the launch terminal" % timeout_s
                    if not heard
                    else "the scene OWL saw no box in %.1f s" % timeout_s
                    if not tried
                    else "none of the OWL's %d candidate(s) is this container: %s" % (len(tried), "; ".join(tried[-3:]))
                )
                if off_frame is not None and heard and t_ready is not None and not self.grab.missing():
                    self.last_why = "%s; %s" % (off_frame_why(*off_frame, frame=self.base_frame or "base_link"), self.last_why)
                return None
            bbox, pose, top, n, pts_base, uv, got = found
            table_seen = (
                got.table_z
                if got.lift == "side"
                else table_from_scene_points(pts_base, uv, self.grab.color.shape[0])
            )
            # no survey: the table the box was found on is the one the
            # planner must know about (WorldStore.table_extent)
            extent = None
            if self.table_z is None:
                extent, assumed = seen_table_extent(pts_base, got.table_z, top[:2])
                if assumed:
                    print("[scene] too few table points to measure its extent; planning the %.2f m ring "
                          "square around the box at the measured height" % (2 * TABLE_RING_M[1]))
            partial = ""
            if got.partial:
                partial = "cut off by the frame toward %s %s" % (
                    self.base_frame or "base_link", " and ".join(off_frame_ways(got.partial, T_base_color[:3, :3])))
            near = np.hypot(pts_base[:, 0] - top[0], pts_base[:, 1] - top[1]) < MAP_R_M
            return SceneFix(pose=pose, top_xyz=top, n_points=n, score=float(bbox[4]), table_z_scene=table_seen,
                            table_extent=extent, partial=partial, table_points=pts_base[near].astype(np.float32))
        finally:
            self.owl.disable()
