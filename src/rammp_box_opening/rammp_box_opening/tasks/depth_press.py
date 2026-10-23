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
"""

import math
import time
from dataclasses import dataclass

import numpy as np

from rammp_box_opening.constants import CONTACT_SPEED, TCP_OFFSET_M
from rammp_box_opening.runtime.kinematics import mat_to_quat_xyzw
from rammp_box_opening.primitives.core import CONTACT_SHIFT_PAD_M, _full_world, _push_guard_and_verify
from rammp_box_opening.runtime.guards import GuardSpec
from rammp_box_opening.tasks import reach_press as rp

# The wrist lens's axis from the fingers' line, along the hand's +y, the
# lens looking along the tool (config/robots/flatbot/transforms/cameras.yaml's
# y — the part of that mount the runs bear out; its z was 20 cm off).
LENS_FROM_FINGERS_M = 0.10
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
BOX_MOVED_M = 0.005  # the button this far from where the camera saw it before: the box moved
LID_MOVED_M = 0.008  # ... or the lid this much nearer or farther
KNOB_UP_MM = 6.0  # press_demo.KNOB_UP_MIN_MM

# The camera's own frame with z turned up, so find_button's heights are
# minus depths below the lens: a virtual camera at the origin looking
# straight down. Nothing in it is a mount or a transform.
_LENS_DOWN = np.diag([1.0, -1.0, -1.0])


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


def depth_under_fingers(depth, k, lens_from_fingers=LENS_FROM_FINGERS_M):
    """The depth of whatever is straight under the fingers' line, from a lens
    beside them looking parallel (m), or None. That line projects to the
    image column through the principal point, on the side the fingers are
    (the hand's -y: image up), its point at depth Z at the row L*fy/Z from
    the centre: the surface under the fingers is where the depth read at a
    row equals the depth that row's point would be at."""
    kk = np.asarray(k, float)
    d = np.asarray(depth, float)
    h, w = d.shape
    cx, cy, fy = kk[0, 2], kk[1, 2], kk[1, 1]
    u = int(round(cx))
    prev = None
    for v in range(int(cy) - 5, 2, -1):  # outward from the centre: the line's point comes nearer the lens
        z_line = lens_from_fingers * fy / (cy - v)
        win = d[v - 2: v + 3, max(0, u - 2): min(w, u + 3)]
        win = win[np.isfinite(win) & (win > 0.08) & (win < 1.5)]
        if win.size < 5:
            continue
        # < 0 near the centre (the line's point is far beyond the surface
        # seen there), > 0 once the line's point is nearer than the surface
        gap = float(np.median(win)) - z_line
        if prev is not None and prev[0] < 0.0 <= gap:
            g0, z0 = prev
            return z0 + (z_line - z0) * (-g0) / (gap - g0)  # where the gap closes
        prev = (gap, z_line)
    return None


def _stamp_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def look(node, watcher, model, tag, runner=None):
    """Still frames after the arm stopped, until LOOK_HITS agree within
    LOOK_AGREE_M. Returns (LensSighting or None, depth under the fingers or
    None, status) — the median of the agreeing sightings."""
    from rammp_box_opening.tasks.press_demo import _spin_detect, save_aim_frame

    g = watcher.grab
    t_end = time.monotonic() + LOOK_SETTLE_S
    while time.monotonic() < t_end:
        _spin_detect(node)
    after = node.get_clock().now().nanoseconds
    seen, under, last, why, frames = [], [], None, "no wrist frame", 0
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
        z = depth_under_fingers(g.depth, g.k)
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
                            up_mm=None if not np.isfinite(med.up_mm) else round(med.up_mm, 1), hits=len(near), frames=frames)
            return med, z_under, "%d/%d frames" % (len(near), frames)
    z_under = float(np.median(under)) if under else None
    if runner is not None:
        runner.note(tag, under_fingers_mm=None if z_under is None else round(1000 * z_under, 1), frames=frames, why=why)
    return None, z_under, "%d frame(s), the button in none (last: %s)" % (frames, why)


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


def turn_down(ctx, runner, contact_tool, ref):
    """Turn the hand straight down where it stands, about its fingertip
    (`contact_tool`, in tool_frame), slow and guarded. Returns (quat, None)
    — also when it already points down — or (None, why)."""
    from rammp_box_opening.runtime import approach

    chain = approach._chain()
    q = ctx.client.joints()
    R, t = chain.fk(q)
    R = np.asarray(R, float)
    tip = np.asarray(t, float) + R @ np.asarray(contact_tool, float)
    if math.degrees(math.acos(float(np.clip(-R[2, 2], -1.0, 1.0)))) < 1.0:
        return mat_to_quat_xyzw(R), None  # straight down already
    whys = []
    for short in TURN_SHORT_DEG:
        quat = down_quat(ctx.model, (float(tip[0]), float(tip[1])), short)
        goal = tip - rp._rot(quat) @ np.asarray(DOWN_CONTACT_TOOL)
        pts, why = chain.walk(q, goal, quat, step_m=TURN_STEP_M, step_rad=TURN_STEP_RAD, max_step_rad=TURN_MAX_JUMP_RAD)
        if pts is not None:
            break
        whys.append("%.0f deg short: %s" % (short, why))
    else:
        return None, "no straight-down hand reaches here (%s)" % "; ".join(whys)
    if short:
        print("[press_demo] TURN: %.0f deg short of straight down (exactly down does not fly here)" % short)
    res = runner.run([rp.line_leg(ctx, "depth:turn", pts, quat, CONTACT_SPEED, rp.drag_guard(ref), _full_world(ctx))],
                     execute=True)
    if any(not r.ok for r in res):
        return None, "the turn met something (%s)" % res[-1].detail
    return quat, None


def touch_leg(ctx, cfg, pts, quat, reference):
    """The stroke down, guarded at contact_nm on the compensated torques
    against the free-air `reference`: every trip is the first contact."""
    guard = GuardSpec(touch_nm=cfg.contact_nm, trip="touch", gravity=rp._wrist_gravity(), reference=reference)

    def verify(v):
        if v.outcome == "touch":
            return True, "contact%s" % ("" if v.torque_peak is None else " at %.1f Nm" % v.torque_peak)
        if v.outcome == "arrived":
            return False, "no contact all the way to the measured depth"
        return False, "touch %s" % v.outcome

    return rp.line_leg(ctx, "depth:touch", pts, quat, cfg.press_speed, guard, _full_world(ctx), verify)


def _rise(ctx, runner, quat, top_z, ref):
    """Straight back up to `top_z`: off the button UNLOAD_CHECK_M and checked
    at rest, then the rest of the way and checked again — judged for a pull
    at the fingertip against the free-air `ref` (the reach-out's unload).
    False: the arm holds, the reason printed."""
    from rammp_box_opening.runtime import approach

    _R, t = approach._chain().fk(ctx.client.joints())
    first = min(float(t[2]) + rp.UNLOAD_CHECK_M, float(top_z))
    for name, z, strict in (("depth:unload", first, False), ("depth:rise", float(top_z), True)):
        _R, t = approach._chain().fk(ctx.client.joints())
        pts, why = rp._line(ctx, [float(t[0]), float(t[1]), z], quat)
        if pts is None:
            print("[press_demo] STOP: the way up is not a straight line the arm can fly (%s) — the arm holds" % why)
            return False
        if pts:
            res = runner.run([rp.line_leg(ctx, name, pts, quat, CONTACT_SPEED,
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
    why: str = None  # why it stopped short, when it did


def press_down_from_here(node, ctx, cfg, runner, watcher, contact_tool):
    """TURN, DEPTH, TOUCH, PUSH, RISE, LOOK from where the hand stands over
    the button (`contact_tool`: the fingertip's point in tool_frame at the
    hand's current attitude). Never goes home itself: DepthPress.free says
    whether the arm may."""
    from rammp_box_opening.runtime import approach
    from rammp_box_opening.tasks.press_demo import refuse_open_box

    m = ctx.model
    here = rp._settled(ctx, "over the box")
    if here is None:
        return DepthPress(free=True, why="no steady efforts over the box")
    quat, why = turn_down(ctx, runner, contact_tool, here)
    if quat is None:
        return DepthPress(free=why.startswith("no straight-down"), why=why)
    over = rp._settled(ctx, "over the button, pointing down")
    if over is None:
        return DepthPress(free=True, why="no steady efforts pointing down")
    _R, t0 = approach._chain().fk(ctx.client.joints())
    top_z = float(t0[2])

    s, z_under, status = look(node, watcher, m, "depth-look", runner)
    if s is not None and np.isfinite(s.up_mm):
        open_already = refuse_open_box(s.up_mm)
        if open_already is not None:
            return DepthPress(free=True, why=open_already)
    depth = s.lid_depth if s is not None else z_under
    if depth is None:
        return DepthPress(free=True, why="the wrist camera measures no depth under the hand (%s)" % status)
    print("[press_demo] DEPTH: the lid %.0f mm below the wrist lens (%s; under the fingers' line: %s)" % (
        1000 * depth, "the button's ring" if s is not None else "no button in view",
        "n/a" if z_under is None else "%.0f mm" % (1000 * z_under)))

    reach = float(depth) + float(cfg.travel_m)
    stroke, why = _hand_line(ctx, 0.0, 0.0, -reach, quat)
    if not stroke:
        return DepthPress(free=True, why="the stroke down is not a straight line the arm can fly (%s)" % (why or "no length"))
    print("[press_demo] TOUCH: straight down, at most %.0f mm, until the torques feel it" % (1000 * reach))
    touched = runner.run([touch_leg(ctx, cfg, stroke, quat, over)], execute=True)
    if touched[-1].outcome == "touch":
        ctx.contact_pad = CONTACT_SHIFT_PAD_M
    if any(not r.ok for r in touched):
        return DepthPress(free=_rise(ctx, runner, quat, top_z, over), why=touched[-1].detail)
    _R, t1 = approach._chain().fk(ctx.client.joints())
    down = top_z - float(t1[2])
    ahead = float(depth) - down
    runner.note("depth_touch", down_mm=round(1000 * down, 1), lid_depth_mm=round(1000 * depth, 1),
                fingertips_ahead_of_lens_mm=round(1000 * ahead, 1))
    print("[press_demo] TOUCHED after %.0f mm down: the fingertips are %.0f mm in front of the wrist lens" % (
        1000 * down, 1000 * ahead))
    if ahead > LENS_AHEAD_MAX_M or ahead < -float(cfg.travel_m):
        return DepthPress(free=_rise(ctx, runner, quat, top_z, over), contact=True,
                          why="the contact %.0f mm down is not where this hand's fingertips can meet the lid" % (1000 * down))

    pts, why = _hand_line(ctx, 0.0, 0.0, -float(cfg.button_travel_m), quat)
    if not pts:
        return DepthPress(free=_rise(ctx, runner, quat, top_z, over), contact=True,
                          why="the push is not a straight line the arm can fly (%s)" % (why or "no length"))
    guard, verify = _push_guard_and_verify(m, cfg)
    push = rp.line_leg(ctx, "press:push", pts, quat, cfg.press_speed, guard, _full_world(ctx), verify)
    push.hold_s = float(cfg.push_hold_s)
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

    after, _z, status = look(node, watcher, m, "depth-after", runner)
    popped = after is not None and np.isfinite(after.up_mm) and after.up_mm >= KNOB_UP_MM
    moved = False
    if after is None:
        print("[press_demo] AFTER: the wrist camera does not find the button again (%s)" % status)
    else:
        shift = None if s is None else math.hypot(after.x - s.x, after.y - s.y)
        moved = s is not None and (shift > BOX_MOVED_M or abs(after.lid_depth - s.lid_depth) > LID_MOVED_M)
        print("[press_demo] AFTER: the knob %s%s" % (
            "UP %.0f mm — the box is open" % after.up_mm if popped else (
                "%.1f mm above the lid" % after.up_mm if np.isfinite(after.up_mm) else "not judged"),
            "" if s is None else "; the button %.0f mm from where it was, the lid %+.0f mm" % (
                1000 * shift, 1000 * (after.lid_depth - s.lid_depth))))
    return DepthPress(free=True, contact=True, felt=felt, popped=popped, moved=moved)


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
