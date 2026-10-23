"""The depth press (press_demo --depth-press): once the hand is over the
box, how far down — and whether the button went — from the wrist camera's
own depth and the arm's own joint torques.

Getting over the box is the mission's own, calibrated way: the scene camera,
the wrist's aim through TF, then the reach-out's slide in or the
straight-down staging. Flatbot 2026-10-22: that put the fingertip over the
button, but 6 in too high — the stroke went down to where TF put the lid
and met nothing. TF is good enough to get there; the height is the one
thing it got wrong. So from over the box:

  TURN    the hand straight down where it stands, turning about the
          fingertip so it stays over the button (the reach-out's tilted
          hand; at staging it already points down). If no straight-down
          hand reaches there, it stops.
  DEPTH   the wrist camera, looking down along the tool, measures how far
          below its lens the lid is: the button's own ring if it sees it,
          else the surface straight under the fingers' line. Its own depth
          in its own frame, no transform.
  TOUCH   straight down, at most that far plus travel_m — the fingertips are
          never behind the lens — stopped by the first contact the joint
          torques feel. Where it stopped says how far in front of the lens
          the fingertips are: printed, every run.
  PUSH    button_travel_m further: the torques feeling the button bottom
          out is the press. The whole travel without a stop pressed nothing.
  RISE    straight back up to where the touch started, a pull at the
          fingertip stopping it (the reach-out's unload and its checks),
  LOOK    and the camera judges the knob and whether the box moved.

What the touch measured is the box from then on: ctx.cpose is put where
the fingertips met the button (TF's lid was 76 mm high on 20:04, so every
home was planned from "inside" a box that was not there). After a contact,
a wrist camera that no longer finds the box holds the arm unless the scene
camera finds it (after_scene_check), the reach-out's own AFTER rule.
"""

import dataclasses
import math
import time
from dataclasses import dataclass

import numpy as np

from rammp_box_opening.constants import CONTACT_SPEED, FREE_LINE_SPEED, TCP_OFFSET_M
from rammp_box_opening.models.container import ContainerPose
from rammp_box_opening.runtime.kinematics import mat_to_quat_xyzw
from rammp_box_opening.primitives.core import CONTACT_SHIFT_PAD_M, _full_world, _push_guard_and_verify
from rammp_box_opening.runtime.guards import GuardSpec
from rammp_box_opening.tasks import reach_press as rp

DOWN_CONTACT_TOOL = (0.0, 0.0, TCP_OFFSET_M)  # the closed pads straight down, in tool_frame (the straight-down press's)
TURN_STEP_M = 0.005  # the turn's waypoints: this far apart ...
TURN_STEP_RAD = math.radians(2.0)  # ... and this much turn
TURN_MAX_JUMP_RAD = 0.15  # a bigger joint step between waypoints is no path to fly
# How far short of straight down the turn may stop, tried in order: exactly
# down can sit on the wrist's singularity (flatbot, the box 0.60 m out: a
# 0.26 rad jump in the last step). The stroke is vertical whatever the hand's
# lean, and the lens stays parallel to the fingers.
TURN_SHORT_DEG = (0.0, 2.0, 5.0)
LOOK_S = 2.0  # still frames are gathered this long at each look
LOOK_SETTLE_S = 0.3  # ... after the arm stopped
LOOK_HITS = 2  # sightings that agree within LOOK_AGREE_M make a look
LOOK_AGREE_M = 0.003
LENS_AHEAD_MAX_M = 0.25  # the fingertips are at most this far in front of the lens
# CENTRE: from over the box the fingers are put over the button the wrist
# camera sees — TF's aim from the tilted look landed the press 45 mm off its
# centre, on the lid's edge (flatbot 2026-10-22, run 20:04). Straight down,
# only the lens's sideways offset from the fingers matters, not where it
# sits along the tool, which is what tilted views get most wrong.
CENTRE_TOL_M = 0.003  # the button this close in world xy to the fingertip's vertical: press
CENTRE_MAX_M = 0.10  # farther than this off is not a re-centre: stop
CENTRE_MAX_MOVES = 2
CENTRE_SPEED = FREE_LINE_SPEED  # the centring move: free air, the fingertips well above the lid
BOX_MOVED_M = 0.005  # the button this far from where the camera saw it before: the box moved
LID_MOVED_M = 0.008  # ... or the lid this much nearer or farther
KNOB_UP_MM = 6.0  # press_demo.KNOB_UP_MIN_MM
# The lens offset CENTRE moves by is TF's mount, and the look after the move
# uses the same offset, so a wrong mount "converges" anyway, its whole error
# off the button (20:04: TF's lens y 0.100 against a true ~0.065 — the press
# 35 mm off, on the lid's edge). CENTRE runs only when TF's lens agrees with
# the measured one in the camera config (wrist_lens_in_hand_m) ...
MOUNT_XY_TOL_M = 0.005
MOUNT_Z_TOL_M = 0.020
# ... and the touch must meet the lid where TF's lens says the fingertips
# are in front of it, or nothing is pushed.
AHEAD_TOL_M = 0.020

# The camera's own frame with z turned up, so find_button's heights are
# minus depths below the lens: a virtual camera at the origin looking
# straight down. Nothing in it is a mount or a transform.
_LENS_DOWN = np.diag([1.0, -1.0, -1.0])

# TOUCH: the stroke flies at press_speed to TOUCH_SLOW_M above where the
# camera's depth and TF's mount put the lid, then at TOUCH_SPEED onto it.
# Flatbot 2026-10-23 16:02: all 112 mm at 0.35 met the lid moving — the
# 3 Nm touch tripped at 12.4 Nm in one sample, the push began 9 Nm loaded,
# and the box slid 6 mm. The slow stretch is the mount's own allowance
# (AHEAD_TOL_M) and some depth noise.
TOUCH_SLOW_M = AHEAD_TOL_M + 0.015
TOUCH_SPEED = CONTACT_SPEED
# A knob not yet up at the after-look (4.5 mm, then 8 mm on the next run's
# aim, 16:02-16:03) is looked at again for this long before it is judged
KNOB_RECHECK_S = 2.0
# A box the press slid this little, still standing (its lid's depth within
# LID_MOVED_M), is gripped where the camera sees it now; farther is a STOP
SLIDE_OK_M = 0.015


@dataclass
class LensSighting:
    """The button in the wrist camera's own frame (the hand's axes)."""

    x: float  # the button's centre from the lens axis, along the hand's x (m)
    y: float  # ... and its y
    lid_depth: float  # the lid ring below the lens (m), along the axis
    button_depth: float  # the button top below the lens
    up_mm: float  # the button above its own lid (refuse_open_box), or nan
    raw: object  # the find_button sighting, for the saved frame


def lid_depth_guesses(depth, n=4):
    """The nearest few flat surfaces in the frame, nearest first: the lid is
    the highest thing there, and may fill only a corner of the frame (1%)."""
    d = np.asarray(depth, float)
    d = d[np.isfinite(d) & (d > 0.08) & (d < 1.5)]
    if d.size < 200:
        return []
    hist, edges = np.histogram(d, bins=np.arange(0.08, 1.505, 0.005))
    order = [i for i in np.argsort(-hist) if hist[i] >= 0.01 * d.size][: 3 * n]
    out = []
    for i in sorted(order, key=lambda i: edges[i]):
        c = float(np.median(d[(d >= edges[i]) & (d < edges[i + 1])]))
        if all(abs(c - o) > 0.01 for o in out):
            out.append(c)
    return out[:n]


def find_in_camera(color, depth, k, model, dist=None):
    """The button in one frame from a camera looking straight down along the
    tool, in that camera's own frame: (LensSighting, None) or (None, why).
    find_button's straight-down aim, handed the lens itself as its world."""
    from rammp_box_opening.perception.button_aim import find_button

    if color is None or depth is None or k is None:
        return None, "no frame"
    whys = []
    for d0 in lid_depth_guesses(depth):
        s, why = find_button(color, depth, k, _LENS_DOWN, (0.0, 0.0, 0.0), -d0 - float(model.dims[2]), model, dist=dist)
        if s is None:
            whys.append("at %.0f mm: %s" % (1000 * d0, why))
            continue
        up = float(s.above_lid_m)
        top = -float(s.xyz[2])
        return LensSighting(
            x=float(s.xyz[0]), y=-float(s.xyz[1]), lid_depth=top + up if np.isfinite(up) else top,
            button_depth=top, up_mm=1000.0 * up, raw=s,
        ), None
    return None, "; ".join(whys) if whys else "no flat surface in depth"


def lens_in_hand(ctx, grab):
    """Where the wrist lens is from the tool frame, in the hand's own axes:
    (lens xyz (m), R_hand_cam — the camera's axes in the hand's), from the
    camera's pose at its newest frame (TF: the arm's model and the camera's
    mount) against the arm's own kinematics at the joints it holds now —
    read while the arm is still, so the two are one moment. (x, y) is how
    far the lens axis sits beside the fingers' line, z how far along the
    tool (negative: behind the tool frame). R_hand_cam is the identity on
    flatbot's OAK, 180 deg about the lens axis on the bench's D405. None
    without a camera pose."""
    from rammp_box_opening.perception.depth_source import camera_pose_at
    from rammp_box_opening.runtime import approach

    cam = camera_pose_at(grab)
    if cam is None:
        return None
    rot, trans = cam
    R, t = approach._chain().fk(ctx.client.joints())
    R = np.asarray(R, float)
    lens = [float(v) for v in R.T @ (np.asarray(trans, float) - np.asarray(t, float))]
    return lens, R.T @ np.asarray(rot, float)


def depth_under_fingers(depth, k, lens, R_hand_cam=None):
    """The depth of whatever is straight under the fingers' line, from a lens
    beside them looking parallel (m), or None. `lens`, `R_hand_cam`: the
    lens's offset from the fingers' line and its axes, in the hand
    (lens_in_hand). The line's point at depth Z is at (-x, -y, Z) from the
    lens in the hand's axes, R_hand_cam.T of that in the camera's; walked
    from far to near, the surface under the fingers is where the depth read
    at that point's pixel equals Z."""
    kk = np.asarray(k, float)
    d = np.asarray(depth, float)
    h, w = d.shape
    r = np.eye(3) if R_hand_cam is None else np.asarray(R_hand_cam, float)
    p = r.T @ np.array([-float(lens[0]), -float(lens[1]), 0.0])
    px, py = float(p[0]), float(p[1])
    prev = None
    for z_line in np.arange(1.2, 0.12, -0.002):  # far to near: the point leaves the centre for the frame's edge
        u = kk[0, 0] * px / z_line + kk[0, 2]
        v = kk[1, 1] * py / z_line + kk[1, 2]
        if not (2.0 <= u < w - 3 and 2.0 <= v < h - 3):
            break  # out of the frame, and it only goes farther out
        ui, vi = int(round(u)), int(round(v))
        win = d[vi - 2: vi + 3, ui - 2: ui + 3]
        win = win[np.isfinite(win) & (win > 0.08) & (win < 1.5)]
        if win.size < 5:
            continue
        # < 0 while the line's point is beyond the surface seen there, >= 0
        # once it is nearer than that surface
        gap = float(np.median(win)) - float(z_line)
        if prev is not None and prev[0] < 0.0 <= gap:
            g0, z0 = prev
            return z0 + (float(z_line) - z0) * (-g0) / (gap - g0)  # where the gap closes
        prev = (gap, float(z_line))
    return None


def _stamp_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def look(node, watcher, model, tag, runner=None, ctx=None, settled=False):
    """Still frames after the arm stopped, until LOOK_HITS agree within
    LOOK_AGREE_M. Returns (LensSighting or None, depth under the fingers or
    None, the lens in the hand (lens_in_hand's pair) or None, status) —
    the median of the agreeing sightings. Without `ctx` there is no offset,
    and no depth under the fingers. `settled`: an at-rest read (rp._settled,
    0.45 s still) came just before, so frames count from now, without
    LOOK_SETTLE_S more."""
    from rammp_box_opening.tasks.press_demo import _spin_detect, save_aim_frame

    g = watcher.grab
    t_end = time.monotonic() + (0.0 if settled else LOOK_SETTLE_S)
    while time.monotonic() < t_end:
        _spin_detect(node)
    after = node.get_clock().now().nanoseconds
    seen, under, last, why, frames, lens = [], [], None, "no wrist frame", 0, None
    t_end = time.monotonic() + LOOK_S
    while time.monotonic() < t_end:
        _spin_detect(node)
        if g.color is None or g.depth is None or g.k is None or g.color_stamp is None:
            continue
        st = _stamp_ns(g.color_stamp)
        if st <= after or st == last:
            continue  # shot while the arm was still moving, or seen already
        last = st
        frames += 1
        if lens is None and ctx is not None:
            lens = lens_in_hand(ctx, g)
        z = None if lens is None else depth_under_fingers(g.depth, g.k, lens[0], lens[1])
        if z is not None:
            under.append(z)
        s, why = find_in_camera(g.color, g.depth, g.k, model, dist=getattr(g, "dist", None))
        if s is None:
            continue
        seen.append(s)
        near = [o for o in seen if math.hypot(o.x - s.x, o.y - s.y) < LOOK_AGREE_M]
        if len(near) >= LOOK_HITS:
            med = LensSighting(
                x=float(np.median([o.x for o in near])), y=float(np.median([o.y for o in near])),
                lid_depth=float(np.median([o.lid_depth for o in near])),
                button_depth=float(np.median([o.button_depth for o in near])),
                up_mm=float(np.nanmedian([o.up_mm for o in near])) if any(np.isfinite(o.up_mm) for o in near) else float("nan"),
                raw=s.raw,
            )
            save_aim_frame(g, s.raw, tag)
            z_under = float(np.median(under)) if under else None
            if runner is not None:
                runner.note(tag, lens_xy_mm=[round(1000 * med.x, 1), round(1000 * med.y, 1)],
                            lid_depth_mm=round(1000 * med.lid_depth, 1),
                            under_fingers_mm=None if z_under is None else round(1000 * z_under, 1),
                            lens_in_hand_mm=None if lens is None else [round(1000 * v, 1) for v in lens[0]],
                            up_mm=None if not np.isfinite(med.up_mm) else round(med.up_mm, 1), hits=len(near), frames=frames)
            return med, z_under, lens, "%d/%d frames" % (len(near), frames)
    z_under = float(np.median(under)) if under else None
    if runner is not None:
        runner.note(tag, under_fingers_mm=None if z_under is None else round(1000 * z_under, 1), frames=frames, why=why,
                    lens_in_hand_mm=None if lens is None else [round(1000 * v, 1) for v in lens[0]])
    return None, z_under, lens, "%d frame(s), the button in none (last: %s)" % (frames, why)


def _in_hand(s, lens, R_hand_cam=None):
    """The button top from the tool frame, in the hand's axes (m): the camera
    sees it at (s.x, s.y, s.button_depth) in its own axes, from a lens at
    `lens` turned by `R_hand_cam` (lens_in_hand)."""
    r = np.eye(3) if R_hand_cam is None else np.asarray(R_hand_cam, float)
    b = r @ np.array([float(s.x), float(s.y), float(s.button_depth)])
    return np.array([float(lens[0]) + b[0], float(lens[1]) + b[1], (float(lens[2]) if len(lens) > 2 else 0.0) + b[2]])


def under_fingers(s, lens, R_hand_cam=None):
    """The button from the fingers' line, along the hand's x and y (m).
    (0, 0): the fingers' line runs through it."""
    b = _in_hand(s, lens, R_hand_cam)
    return float(b[0]), float(b[1])


def _off_in_world(ctx, s, lens, R_hand_cam):
    """The button top from the fingertip's vertical, in world xy (m): what
    the vertical stroke misses it by. Along the hand's axes it would be off
    by the depth below the fingertips times the lean when the turn stopped
    short (TURN_SHORT_DEG: 5 deg, 14 mm)."""
    from rammp_box_opening.runtime import approach

    R, t = approach._chain().fk(ctx.client.joints())
    R, t = np.asarray(R, float), np.asarray(t, float)
    b = t + R @ _in_hand(s, lens, R_hand_cam)
    tip = t + R @ np.asarray(DOWN_CONTACT_TOOL, float)
    return float(b[0] - tip[0]), float(b[1] - tip[1])


def _hand_line(ctx, dx, dy, dz, quat):
    """Joint waypoints moving the tool by (dx, dy) along the hand's own x and
    y and dz in world z, holding `quat`: ([], None) when that is no move."""
    from rammp_box_opening.runtime import approach

    R, t = approach._chain().fk(ctx.client.joints())
    d = np.asarray(R, float)[:, 0] * dx + np.asarray(R, float)[:, 1] * dy
    return rp._line(ctx, [float(t[0] + d[0]), float(t[1] + d[1]), float(t[2] + dz)], quat)


def down_quat(model, xy, short_deg=0.0):
    """The straight-down hand at `xy`, yawed as the reach-out's — leaning
    back toward the arm by `short_deg`."""
    return rp.reach_quat(model, xy, short_deg)


def turn_target(model, tip, short_deg):
    """The turn's end: (quat, tool_frame xyz) for the straight-down hand,
    `short_deg` short, with its pads (DOWN_CONTACT_TOOL) at `tip`."""
    quat = down_quat(model, (float(tip[0]), float(tip[1])), short_deg)
    return quat, np.asarray(tip, float) - rp._rot(quat) @ np.asarray(DOWN_CONTACT_TOOL)


def turn_down(ctx, runner, contact_tool, ref):
    """Turn the hand straight down where it stands, about its fingertip
    (`contact_tool`, in tool_frame), guarded. Returns (quat, moved, None) —
    moved False when it already points down — or (None, False, why)."""
    from rammp_box_opening.runtime import approach

    chain = approach._chain()
    q = ctx.client.joints()
    R, t = chain.fk(q)
    R = np.asarray(R, float)
    tip = np.asarray(t, float) + R @ np.asarray(contact_tool, float)
    if math.degrees(math.acos(float(np.clip(-R[2, 2], -1.0, 1.0)))) < 1.0:
        return mat_to_quat_xyzw(R), False, None  # straight down already
    whys = []
    for short in TURN_SHORT_DEG:
        quat, goal = turn_target(ctx.model, tip, short)
        pts, why = chain.walk(q, goal, quat, step_m=TURN_STEP_M, step_rad=TURN_STEP_RAD, max_step_rad=TURN_MAX_JUMP_RAD)
        if pts is not None:
            break
        whys.append("%.0f deg short: %s" % (short, why))
    else:
        return None, False, "no straight-down hand reaches here (%s)" % "; ".join(whys)
    if short:
        print("[press_demo] TURN: %.0f deg short of straight down (exactly down does not fly here)" % short)
    # in place, the fingertip the pivot: free air, nothing nearer the lid
    res = runner.run([rp.line_leg(ctx, "depth:turn", pts, quat, FREE_LINE_SPEED, rp.drag_guard(ref), _full_world(ctx))],
                     execute=True)
    if any(not r.ok for r in res):
        return None, False, "the turn met something (%s)" % res[-1].detail
    return quat, True, None


def walk_in_down(ctx, start_xyz, quat):
    """The reach-out's slide in and the TURN as one walk from where the arm
    stands (its look pose): to the straight-down hand whose pads stand where
    the hand at `quat` would put its fingertip (rp.CONTACT_TOOL) at the
    slide's end `start_xyz`, the tool in a straight line, the attitude
    turning on the way — TURN_SHORT_DEG tried in order, as turn_down.
    Flatbot, run 20:04's button at 35 deg: 7.7 s at the slide's speed
    against 12.0 s for the slide and the turn, the fingertip within 0.6 mm
    of the slide's line and height. Returns (points, the down quat, None)
    or (None, None, why)."""
    from rammp_box_opening.runtime import approach

    chain = approach._chain()
    q = ctx.client.joints()
    tip = np.asarray(start_xyz, float) + rp._rot(quat) @ np.asarray(rp.CONTACT_TOOL)
    whys = []
    for short in TURN_SHORT_DEG:
        down, goal = turn_target(ctx.model, tip, short)
        pts, why = chain.walk(q, goal, down, step_m=TURN_STEP_M, step_rad=TURN_STEP_RAD, max_step_rad=TURN_MAX_JUMP_RAD)
        if pts is not None:
            if short:
                print("[press_demo] TURN: %.0f deg short of straight down (exactly down does not fly here)" % short)
            return pts, down, None
        whys.append("%.0f deg short: %s" % (short, why))
    return None, None, "; ".join(whys)


def touch_leg(ctx, cfg, pts, quat, reference, name="depth:touch", speed=None, arrive_ok=False):
    """The stroke down, guarded at contact_nm on the compensated torques
    against the free-air `reference`: every trip is the first contact.
    `arrive_ok`: the stroke's fast first stretch, which is meant to arrive
    (TOUCH_SLOW_M short of the lid) and touches only a lid higher than
    expected."""
    guard = GuardSpec(touch_nm=cfg.contact_nm, trip="touch", gravity=rp._wrist_gravity(), reference=reference)

    def verify(v):
        if v.outcome == "touch":
            return True, "contact%s" % ("" if v.torque_peak is None else " at %.1f Nm" % v.torque_peak)
        if v.outcome == "arrived":
            return (True, "arrived above the lid") if arrive_ok else (False, "no contact all the way to the measured depth")
        return False, "touch %s" % v.outcome

    return rp.line_leg(ctx, name, pts, quat, cfg.press_speed if speed is None else speed, guard, _full_world(ctx), verify)


def _rise(ctx, runner, quat, top_z, ref):
    """Straight back up to `top_z`: off the button UNLOAD_CHECK_M and checked
    at rest, then the rest of the way and checked again — judged for a pull
    at the fingertip against the free-air `ref` (the reach-out's unload).
    False: the arm holds, the reason printed."""
    from rammp_box_opening.runtime import approach

    _R, t = approach._chain().fk(ctx.client.joints())
    first = min(float(t[2]) + rp.UNLOAD_CHECK_M, float(top_z))
    # the rise only after the at-rest check 15 mm up found nothing on the hand
    for name, z, strict, speed in (("depth:unload", first, False, CONTACT_SPEED),
                                   ("depth:rise", float(top_z), True, FREE_LINE_SPEED)):
        _R, t = approach._chain().fk(ctx.client.joints())
        pts, why = rp._line(ctx, [float(t[0]), float(t[1]), z], quat)
        if pts is None:
            print("[press_demo] STOP: the way up is not a straight line the arm can fly (%s) — the arm holds" % why)
            return False
        if pts:
            res = runner.run([rp.line_leg(ctx, name, pts, quat, speed,
                                          rp.unload_guard(ctx.model, ref, rp.tip_load_at(pts[0])), _full_world(ctx))],
                             execute=True)
            if any(not r.ok for r in res):
                print("[press_demo] STOP: %s" % rp._unload_stopped(res))
                return False
        why = rp._still_free(ctx, runner, ref, "%.0f mm up" % (1000 * rp.UNLOAD_CHECK_M) if not strict else "where the touch began",
                             strict)
        if why is not None:
            print("[press_demo] STOP: %s The arm holds; check the box" % why)
            return False
    return True


@dataclass
class DepthPress:
    """What the depth press came back with."""

    free: bool  # the arm ended in free air where the touch began: it may go home
    contact: bool = False  # the fingertips touched something
    felt: bool = False  # the torques felt the button bottom out
    popped: bool = False  # the camera sees the knob up after
    moved: bool = False  # the camera sees the box moved
    shift_m: float = 0.0  # how far the wrist camera saw the button move (when it saw it again)
    lid_dz_m: float = 0.0  # ... and the lid's depth change
    knob_mm: float = float("nan")  # the knob above the lid at the after-look
    why: str = None  # why it stopped short, when it did
    lost: bool = False  # after a contact the wrist camera did not find the box again
    touch_xyz: tuple = None  # where the pads met the button (the arm's own kinematics)
    ref: object = None  # the free-air (efforts, joints) reading the press was judged against


def press_down_from_here(node, ctx, cfg, runner, watcher, contact_tool, here=None, turned=None):
    """TURN, DEPTH, TOUCH, PUSH, RISE, LOOK from where the hand stands over
    the button (`contact_tool`: the fingertip's point in tool_frame at the
    hand's current attitude). `here`: the caller's at-rest reading where the
    arm stands now, nothing flown since; `turned`: the hand already points
    down at this quat (the reach-out slid in turning), no TURN. Never goes
    home itself: DepthPress.free says whether the arm may."""
    seen = {}
    result = _press_down(node, ctx, cfg, runner, watcher, contact_tool, seen, here, turned)
    return dataclasses.replace(result, ref=seen.get("ref"), touch_xyz=seen.get("touch_xyz"))


def _press_down(node, ctx, cfg, runner, watcher, contact_tool, seen, here=None, turned=None):
    from rammp_box_opening.models.container import from_container
    from rammp_box_opening.runtime import approach
    from rammp_box_opening.tasks.press_demo import refuse_open_box

    m = ctx.model
    if here is None:
        here = rp._settled(ctx, "over the box")
    if here is None:
        return DepthPress(free=True, why="no steady efforts over the box")
    if turned is not None:
        quat, moved = list(turned), False
    else:
        quat, moved, why = turn_down(ctx, runner, contact_tool, here)
        if quat is None:
            return DepthPress(free=why.startswith("no straight-down"), why=why)
    # a turn that did not move leaves the arm where `here` was read
    over = rp._settled(ctx, "over the button, pointing down") if moved else here
    if over is None:
        return DepthPress(free=True, why="no steady efforts pointing down")
    seen["ref"] = over
    _R, t0 = approach._chain().fk(ctx.client.joints())
    top_z = float(t0[2])

    s, z_under, lens_pair, status = look(node, watcher, m, "depth-look", runner, ctx, settled=True)
    if lens_pair is None:
        return DepthPress(free=True, why="no camera pose for the wrist's frames: where the lens sits from the fingers "
                                         "is unknown (%s)" % status)
    lens, r_hc = lens_pair
    measured = getattr(getattr(watcher, "grab", None), "wrist_lens_in_hand_m", None)
    if measured is None:
        return DepthPress(free=True, why="no measured wrist lens offset (wrist_lens_in_hand_m) in the camera config — "
                                         "not centring on TF's mount")
    measured = [float(v) for v in measured]
    if (math.hypot(lens[0] - measured[0], lens[1] - measured[1]) > MOUNT_XY_TOL_M
            or abs(lens[2] - measured[2]) > MOUNT_Z_TOL_M):
        runner.note("depth_mount_refused", lens_in_hand_mm=[round(1000 * v, 1) for v in lens],
                    measured_mm=[round(1000 * v, 1) for v in measured])
        return DepthPress(free=True, why="the wrist camera's mount in TF puts the lens at (%s) mm from the tool frame, "
                                         "the measured wrist_lens_in_hand_m at (%s) mm — fix the mount before centring "
                                         "on it" % (", ".join("%+.0f" % (1000 * v) for v in lens),
                                                    ", ".join("%+.0f" % (1000 * v) for v in measured)))
    if s is None:
        return DepthPress(free=True, why="the wrist camera does not see the button from over the box (%s) — not "
                                         "pressing blind" % status)
    if np.isfinite(s.up_mm):
        open_already = refuse_open_box(s.up_mm)
        if open_already is not None:
            return DepthPress(free=True, why=open_already)

    # CENTRE: the fingertip's vertical through the button the camera sees,
    # in world xy — the stroke is vertical whatever the hand's lean
    off = _off_in_world(ctx, s, lens, r_hc)
    moves = 0
    while math.hypot(*off) > CENTRE_TOL_M:
        miss = math.hypot(*off)
        if miss > CENTRE_MAX_M:
            return DepthPress(free=True, why="the button the wrist camera sees is %.0f mm from under the fingers — farther "
                                             "than a re-centre should be (check the wrist camera's mount)" % (1000 * miss))
        if moves >= CENTRE_MAX_MOVES:
            return DepthPress(free=True, why="the button is still %.0f mm from under the fingers after %d moves" % (
                1000 * miss, moves))
        print("[press_demo] CENTRE: the button %.0f mm from under the fingers (world x %+.0f, y %+.0f mm) — moving over it"
              % (1000 * miss, 1000 * off[0], 1000 * off[1]))
        _R, t = approach._chain().fk(ctx.client.joints())
        pts, why = rp._line(ctx, [float(t[0]) + off[0], float(t[1]) + off[1], float(t[2])], quat)  # level
        if pts is None:
            return DepthPress(free=True, why="the move over the button is not a straight line the arm can fly (%s)" % why)
        if pts:
            res = runner.run([rp.line_leg(ctx, "depth:centre", pts, quat, CENTRE_SPEED, rp.drag_guard(over),
                                          _full_world(ctx))], execute=True)
            if any(not r.ok for r in res):
                return DepthPress(free=False, why="the move over the button met something — the arm holds")
        moves += 1
        s, z_under, lens2, status = look(node, watcher, m, "depth-look", runner, ctx)
        lens, r_hc = lens2 or (lens, r_hc)
        if s is None:
            return DepthPress(free=True, why="the wrist camera lost the button after moving over it (%s)" % status)
        off = _off_in_world(ctx, s, lens, r_hc)
        if math.hypot(*off) > 0.5 * miss:
            return DepthPress(free=True, why="moving over the button did not bring it under the fingers (%.0f -> %.0f mm)"
                                             " — the camera's axes in TF are wrong?" % (1000 * miss, 1000 * math.hypot(*off)))
    runner.note("depth_centred", moves=moves, off_mm=[round(1000 * v, 1) for v in off],
                lens_in_hand_mm=[round(1000 * v, 1) for v in lens])
    depth = s.lid_depth
    print("[press_demo] DEPTH: the button %.1f mm from under the fingers%s; the lid %.0f mm below the wrist lens (under "
          "the fingers' line: %s)" % (1000 * math.hypot(*off), "" if not moves else " after %d move(s)" % moves,
                                      1000 * depth, "n/a" if z_under is None else "%.0f mm" % (1000 * z_under)))

    reach = float(depth) + float(cfg.travel_m)
    # fast to TOUCH_SLOW_M short of where the depth and TF's mount put the lid
    fast_m = max(0.0, float(depth) - (float(DOWN_CONTACT_TOOL[2]) - float(lens[2])) - TOUCH_SLOW_M)
    touched = []
    if fast_m > 0.0:
        fast, why = _hand_line(ctx, 0.0, 0.0, -fast_m, quat)
        if not fast:
            return DepthPress(free=True, why="the stroke down is not a straight line the arm can fly (%s)" % (why or "no length"))
        print("[press_demo] TOUCH: straight down, %.0f mm fast, then slowly, at most %.0f mm in all, until the torques "
              "feel it" % (1000 * fast_m, 1000 * reach))
        touch = touch_leg(ctx, cfg, fast, quat, over, name="depth:approach", arrive_ok=True)
        touched = runner.run([touch], execute=True)
        if any(not r.ok for r in touched):
            return DepthPress(free=_rise(ctx, runner, quat, top_z, over), why=touched[-1].detail)
    else:
        print("[press_demo] TOUCH: straight down, slowly, at most %.0f mm, until the torques feel it" % (1000 * reach))
    if not touched or touched[-1].outcome != "touch":
        stroke, why = _hand_line(ctx, 0.0, 0.0, -(reach - fast_m), quat)
        if not stroke:
            return DepthPress(free=_rise(ctx, runner, quat, top_z, over),
                              why="the stroke down is not a straight line the arm can fly (%s)" % (why or "no length"))
        touch = touch_leg(ctx, cfg, stroke, quat, over, speed=TOUCH_SPEED)  # the leg that meets the lid, for the recoil
        touched = runner.run([touch], execute=True)
    if touched[-1].outcome == "touch":
        ctx.contact_pad = CONTACT_SHIFT_PAD_M
    if any(not r.ok for r in touched):
        return DepthPress(free=_rise(ctx, runner, quat, top_z, over), why=touched[-1].detail)
    R1, t1 = approach._chain().fk(ctx.client.joints())
    down = top_z - float(t1[2])
    ahead = float(depth) - down
    ahead_tf = float(DOWN_CONTACT_TOOL[2]) - float(lens[2])  # where TF's mount puts the pads before the lens
    pad = np.asarray(t1, float) + np.asarray(R1, float) @ np.asarray(DOWN_CONTACT_TOOL)
    world_dz = float(pad[2]) - float(from_container(ctx.cpose, m.button_offset)[2])
    runner.note("depth_touch", down_mm=round(1000 * down, 1), lid_depth_mm=round(1000 * depth, 1),
                fingertips_ahead_of_lens_mm=round(1000 * ahead, 1), tf_says_mm=round(1000 * ahead_tf, 1),
                pad_xyz=[round(float(v), 4) for v in pad], world_dz_mm=round(1000 * world_dz, 1))
    print("[press_demo] TOUCHED after %.0f mm down: the fingertips are %.0f mm in front of the wrist lens (the camera's "
          "mount in TF says %.0f)" % (1000 * down, 1000 * ahead, 1000 * ahead_tf))
    if ahead > LENS_AHEAD_MAX_M or ahead < -float(cfg.travel_m):
        return DepthPress(free=_rise(ctx, runner, quat, top_z, over), contact=True,
                          why="the contact %.0f mm down is not where this hand's fingertips can meet the lid" % (1000 * down))
    if abs(ahead - ahead_tf) > AHEAD_TOL_M:  # the mount is wrong after all, or the touch met something else
        return DepthPress(free=_rise(ctx, runner, quat, top_z, over), contact=True,
                          why="the touch met something %.0f mm from where the camera saw the lid — not pushing" % (
                              1000 * abs(ahead - ahead_tf)))
    # the box is where the fingertips met its button, flush before the push:
    # every world from here on (the home's too) holds it there, not where TF
    # put it (76 mm high on 20:04, so every home started "inside" it). The
    # table is the scene camera's own measurement and is left as it is.
    seen["touch_xyz"] = tuple(round(float(v), 5) for v in pad)
    ctx.cpose = rp.cpose_for_button([float(v) for v in pad], ctx.cpose.yaw, m.button_offset)
    print("[press_demo] the box is put where the fingertips met it: %+.0f mm from where TF had its button" % (1000 * world_dz))

    pts, why = _hand_line(ctx, 0.0, 0.0, -float(cfg.button_travel_m), quat)
    if not pts:
        return DepthPress(free=_rise(ctx, runner, quat, top_z, over), contact=True,
                          why="the push is not a straight line the arm can fly (%s)" % (why or "no length"))
    # the backstop judged against the free-air reading over the box, not the
    # push's first sample: that is already loaded by whatever the touch
    # overshot (run 20:04: 5.8 Nm), and the push then rode to 13.5 Nm off
    # free air against touch_nm 10
    _g, verify = _push_guard_and_verify(m, cfg)
    guard = GuardSpec(touch_nm=float(m.touch_nm), trip="press", gravity=rp._wrist_gravity(), reference=over)
    push = rp.line_leg(ctx, "press:push", pts, quat, cfg.press_speed, guard, _full_world(ctx), verify)
    push.hold_s = float(cfg.push_hold_s)
    push.recoil_via = touch.traj  # the recoil carries on back up the touch
    res = runner.run([push], execute=True)  # held, then the runner's recoil lets go
    pushed = res[0]  # the recoil, when there was one, comes after it
    felt = pushed.ok and pushed.outcome == "touch"
    if felt:
        print("[press_demo] PRESSED: the torques felt the button bottom out%s" % (
            "" if pushed.torque_peak is None else " (%.1f Nm)" % pushed.torque_peak))
    elif pushed.ok:
        print("[press_demo] NOT PRESSED by feel: the whole %.0f mm push and no stop" % (1000 * cfg.button_travel_m))
    else:
        print("[press_demo] the push failed (%s)" % pushed.detail)
    if not _rise(ctx, runner, quat, top_z, over):
        return DepthPress(free=False, contact=True, felt=felt, why="the way up stopped")

    after, _z, _lens, status = look(node, watcher, m, "depth-after", runner, ctx, settled=True)  # after _rise's read
    popped = after is not None and np.isfinite(after.up_mm) and after.up_mm >= KNOB_UP_MM
    if after is None:
        # the reach-out's AFTER rule: a box no camera finds after a contact
        # holds the arm — the caller asks the scene camera (after_scene_check)
        print("[press_demo] AFTER: the wrist camera does not find the button again (%s)" % status)
        return DepthPress(free=False, contact=True, felt=felt, lost=True,
                          why="the wrist camera does not find the button after the press — the knob is not judged")
    shift = math.hypot(after.x - s.x, after.y - s.y)
    moved = shift > BOX_MOVED_M or abs(after.lid_depth - s.lid_depth) > LID_MOVED_M
    print("[press_demo] AFTER: the knob %s; the button %.0f mm from where it was, the lid %+.0f mm" % (
        "UP %.0f mm — the box is open" % after.up_mm if popped else (
            "%.1f mm above the lid" % after.up_mm if np.isfinite(after.up_mm) else "not judged"),
        1000 * shift, 1000 * (after.lid_depth - s.lid_depth)))
    if moved:
        # home is planned around the box where the camera sees it now: the
        # sighting's shift, turned from the camera's axes into the hand's
        # (R_hand_cam: 180 deg on the bench's D405), then into the world
        R2, _t2 = approach._chain().fk(ctx.client.joints())
        dh = _in_hand(after, lens, r_hc) - _in_hand(s, lens, r_hc)
        dh[2] = 0.0  # the lid's depth change is not a shift
        d = np.asarray(R2, float) @ dh
        c = ctx.cpose
        ctx.cpose = ContainerPose(xyz=(float(c.xyz[0] + d[0]), float(c.xyz[1] + d[1]), float(c.xyz[2])), yaw=c.yaw)
    return DepthPress(free=True, contact=True, felt=felt, popped=popped, moved=moved, shift_m=float(shift),
                      lid_dz_m=float(after.lid_depth - s.lid_depth), knob_mm=float(after.up_mm))


def knob_up_after(node, ctx, watcher, runner, result, timeout_s=KNOB_RECHECK_S):
    """A press felt to bottom out whose knob the after-look did not see up:
    look again for `timeout_s` (it may still be rising, or the first look
    read it low). Returns the DepthPress with `popped`/`knob_mm` from the
    best look."""
    if result.popped or not result.felt or result.lost or not result.free:
        return result
    t0 = time.monotonic()
    best = result.knob_mm
    while time.monotonic() - t0 < timeout_s:
        again, _z, _lens, _status = look(node, watcher, ctx.model, "depth-after", runner, ctx, settled=True)
        if again is not None and np.isfinite(again.up_mm):
            best = again.up_mm if not np.isfinite(best) else max(best, again.up_mm)
            if again.up_mm >= KNOB_UP_MM:
                break
    popped = bool(np.isfinite(best) and best >= KNOB_UP_MM)
    print("[press_demo] AFTER, again: the knob %s" % ("UP %.0f mm — the box is open" % best if popped else
                                                     "%.1f mm above the lid" % best if np.isfinite(best) else "not judged"))
    return dataclasses.replace(result, popped=popped, knob_mm=float(best))


def lid_ready(result):
    """Whether the lid may be gripped after this depth press: the arm free
    in the air where the touch began, the knob up, the box seen again by the
    wrist camera, and not moved — or only slid (SLIDE_OK_M) and still
    standing, in which case ctx.cpose already holds where it slid to."""
    if not (result.free and result.popped) or result.lost or result.why is not None:
        return False
    return not result.moved or (result.shift_m <= SLIDE_OK_M and abs(result.lid_dz_m) <= LID_MOVED_M)


def after_scene_check(ctx, runner, result, scene_fix, locator):
    """After a contact the wrist camera lost the box (DepthPress.lost): the
    scene camera looks again, as the reach-out's AFTER does. Seen, the arm
    may go home around it — shifted by what the scene camera saw it move,
    from where the arm measured it; unseen, the arm holds. Returns the
    DepthPress to finish with."""
    from rammp_box_opening.tasks.press_demo import SCENE_LOCATE_S

    if not result.lost:
        return result
    again = locator.locate(SCENE_LOCATE_S) if locator is not None and scene_fix is not None else None
    if again is None:
        runner.note("depth_after_scene", found=False)
        print("[press_demo] STOP: neither camera finds the box after the press — it may have left the table. The arm "
              "holds over where it was; check the box, then home the arm with sheppy's home_arm")
        return result
    d = np.asarray(again.top_xyz, float) - np.asarray(scene_fix.top_xyz, float)
    moved = math.hypot(d[0], d[1]) > rp.SCENE_MOVED_M or abs(d[2]) > rp.SCENE_MOVED_M
    runner.note("depth_after_scene", found=True, scene_moved_mm=round(1000 * math.hypot(d[0], d[1]), 1),
                scene_dz_mm=round(1000 * d[2], 1))
    print("[press_demo] AFTER: the scene camera puts the box %.0f mm from where it was, %+.0f mm in height" % (
        1000 * math.hypot(d[0], d[1]), 1000 * d[2]))
    if moved:
        # the scene camera's own shift: its box can stand 8-10 cm from the
        # arm's, but where it saw the box go is its own two looks' difference
        c = ctx.cpose
        ctx.cpose = ContainerPose(xyz=(float(c.xyz[0] + d[0]), float(c.xyz[1] + d[1]), float(c.xyz[2])), yaw=c.yaw)
    return dataclasses.replace(result, free=True, moved=moved)


def home_why(result):
    """What the home after a depth press says first: a moved box is a STOP
    before the arm goes anywhere."""
    return "STOP: the press MOVED the box — check it is still standing on the table" if result.moved else ""


def exit_code(result):
    """The process exit code for a DepthPress, its verdict printed: 0 the
    knob up, 1 anything else."""
    if result.why is not None:
        print("[press_demo] STOP: %s" % result.why)
        return 1
    if result.moved:
        print("[press_demo] STOP: the press MOVED the box — check it is still standing on the table")
        return 1
    if result.popped:
        print("[press_demo] the box is open")
        return 0
    print("[press_demo] STOP: %s, and the knob is not up" % ("pressed by feel" if result.felt else "not pressed"))
    return 1


def finish(ctx, result, homed):
    """The process exit code after a depth press and the home that followed
    it (`homed`: the arm got there). An arm left away from home is never a
    success. What the press established goes into the mission's record."""
    frames = getattr(ctx, "mission_frames", None)
    if frames is not None:
        facts = dict(popped=None if not result.contact or result.lost else bool(result.popped),
                     touch_xyz=None if result.touch_xyz is None else list(result.touch_xyz))
        if result.felt:
            facts["stage"] = "pressed"
        frames.note(**facts)
    code = exit_code(result)
    if not homed:
        if result.free:
            print("[press_demo] STOP: the arm did not get home — it holds; home it with sheppy's home_arm")
        code = max(code, 1)
    return code
