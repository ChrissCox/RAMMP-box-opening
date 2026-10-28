"""The reach-out's lid off: the compliant reach-out press (tasks/reach_press)
as the whole run — press, see the knob up, grip it, lift the lid off, set it
down where the cameras see room, home. Owner 2026-10-28: every box goes
through the reach-out, and the full run is its default; --press-only stops
after the press as before (reach_press._fly_way_in).

  PRESS   the way in as one execution and the hold (reach_press.way_in_legs,
          depth_press.compliant_hold), as the press-only run flies them.
  BACK    the rise back up the stroke, then the way down backwards — the
          turn and the slide, or the walk, each reversed — to the aimed look
          pose: ONE execution, every line one the proof flew forward.
  LOOK    from there, where the wrist aimed before the press: the oblique
          aim again (reach_press._aim) reads the knob's height above its lid,
          and once more where the first read says it is not up. Flatbot's
          URDF: over boxes 0.60-0.78 m out no turned-down hand rises the 56-69
          mm above the stroke's start that would put the wrist the 245 mm off
          the lid the straight-down depth look took at staging (it rises 0-30
          mm), and at the stroke's start the button sits at the frame's top
          edge 176-189 mm from the lens, inside the OAK's stereo minimum. From
          the look pose the button is at the frame's centre, 299 mm off, 29
          deg from the lens axis — the view the press was aimed from.
  AGAIN   not up: the same press once more, its way in solved again from
          where the arm stands, and the look again. Still not up: STOP, home.
  GRIP    up: the fingers open as the hand comes down from the look pose —
          the way in once more, its stroke the grip's: straight down until the
          closed pads' reference stands GRIP_BELOW_LID_M under the lid — ONE
          execution, the carry planned while it flies. The fingers close: on
          air (outside the grip band) they open again and the hand backs out
          the way it came, home.
  LIFT    the close, then the lid straight up off the canister
          (lift_off_m) and up and back toward the arm clear of the box's
          padded volume (as the way home's clear): ONE execution.
  SPOT    there, the lid in the fingers, its one stop: where the lid goes,
          from the wrist camera's frame then — the held lid, the hand and the
          box taken out of it — pooled with the scene camera's table, the arm
          taken out of that (spot_map); the room judged as the staging run
          judged it (press_demo.clear_lid_drops), the best spot the hand
          reaches straight down from where it stands (drop_reachable). Never
          a failure (owner 2026-10-28: "it should go no matter what"):
          nothing clear, the most room seen; no map, beside the box; else
          the nearest table point the hand reaches, off the box
          (spot_candidates) — one line says which, and why.
  PLACE   the carry (planned there), down onto the table: ONE execution, home
          planned while it flies. The fingers open; back up to the carry's
          height and home: one more.

Every move flies in the driver's joint impedance under the run's one gains
spec (depth_press.COMPLIANT_GAINS, RIGID), unguarded, merged where its legs
meet: nothing the run reads stops the arm. It decides only what it must:
the knob up or not, the box where it was, the fingers on the knob or on
air, where the lid goes, the lid still in them at the table. Contact is bounded by the spring,
not stopped by a guard: the grip's pads ride on the lid where it stands
higher than estimated, and the set-down presses the lid SETDOWN_OVER_M past
where the scene camera's table says it lands.
"""

import math
import time
from dataclasses import dataclass

import numpy as np

from rammp_box_opening.constants import FREE_LINE_SPEED, GRIPPER_CMD_CLOSED, GRIPPER_CMD_OPEN, TRANSIT_SPEED
from rammp_box_opening.models.container import ContainerPose
from rammp_box_opening.perception import clear_table
from rammp_box_opening.primitives.core import (
    CONTACT_SHIFT_PAD_M,
    Place,
    PlanState,
    _full_world,
    _gripper_leg,
    band_verify,
    tcp_z,
)
from rammp_box_opening.runtime.guards import in_band
from rammp_box_opening.runtime.legs import merge_groups
from rammp_box_opening.tasks import depth_press as dp
from rammp_box_opening.tasks import reach_press as rp

# The press, then once more where the look does not see the knob up (owner
# 2026-10-28), then STOP.
PRESSES = 2
# A knob the look reads down is looked at once more before it is judged
# (depth_press.knob_up_after's recheck): pressing a knob that popped shuts
# the box again, and the oblique aim read a flush button 0.9-3.7 mm up on
# 2026-10-27/28's looks, hits apart by up to 7.5 mm.
KNOB_RECHECK_AIMS = 1
# The pink button's travel from flush to its stop: the guarded pushes of
# 2026-10-28 14:26 and 14:30 met it 7.6 and 7.4 mm below their touches. A
# compliant press that popped the knob held the pads on it there.
KNOB_STROKE_M = 0.0075
# The grip: the closed pads' reference this far under the lid as estimated
# (grip_lid_z). The 2F-85's pads stand 13.5 mm higher open than shut, and
# 2.3 mm higher closed on the 44 mm knob (flatbot's URDF: the knuckle at
# 0.41 rad of 0.8): the open fingers come down 11.5 mm above the estimate,
# and closing, the pads reach the knob at its base — riding on the lid,
# the spring giving, where it stands higher than estimated.
GRIP_BELOW_LID_M = 0.002
GRIP_SLOW_M = dp.TOUCH_SLOW_M  # the grip's last stretch down flies at TOUCH_SPEED, as the press's
# Straight up with the knob held, this much past the cap's own height: off
# the canister's rim. Flatbot's URDF lifts the hand turned down only 60-140
# mm straight up over boxes 0.60-0.78 m out (150 mm only at 0.58 m), so the
# rest of the way out is up and back toward the arm (reach_press
# .clear_targets), as the way home's clear goes.
LIFT_OFF_CLEAR_M = 0.015
# The set-down: at setdown_speed to this far above where the lid meets the
# table, then at TOUCH_SPEED to SETDOWN_OVER_M below it — the spring
# pressing the lid onto the table there (~6 N at the ~1000 N/m RIGID gives
# straight down), never a guard. The table is the scene camera's, which
# stood 7-18 mm above the touched one on 2026-10-28: a lid let go up to that
# much above the table drops onto it. Small, too, so the arm's reference
# after the release stays within the runner's drift gate (0.04 rad ~ 10
# mm) of where the retreat starts, if the hand has not yet sunk back to it.
SETDOWN_SLOW_M = 0.04
SETDOWN_OVER_M = 0.006


def lift_off_m(model):
    """How far the knob is lifted straight up: the cap off the rim."""
    return float(model.lid_dims[2]) + LIFT_OFF_CLEAR_M


def grip_lid_z(top_z, held_z=None):
    """The lid's height the grip goes by: the press's top (`top_z`, the
    higher of the aim's and the scene camera's — on 2026-10-27/28 the
    touched lid stood 1-13 mm above the scene camera's), no lower than
    where the compliant hold held the pads on the button (`held_z`: the lid
    is not below the button it held them on) and no higher than that plus
    KNOB_STROKE_M (the button at its stop, as a popped one was). Without a
    hold, `top_z`."""
    if held_z is None:
        return float(top_z)
    return min(max(float(top_z), float(held_z)), float(held_z) + KNOB_STROKE_M)


def setdown_poses(ctx, drop):
    """(attitude, tool where the lid meets the table, the carry's hover)
    for the spot `drop`: the hand straight down, turned by the spot's yaw
    (press_demo.place_quat, the attitude clear_lid_drops judged the room
    for); the lid hangs where the grip took it, GRIP_BELOW_LID_M under the
    pads' reference; the hover above it as Place.hover_for has it, high
    enough for the lid to clear the box."""
    from rammp_box_opening.tasks.press_demo import place_quat

    quat = list(place_quat(ctx.model, drop))
    contact = [float(drop.xyz[0]), float(drop.xyz[1]),
               tcp_z(float(drop.xyz[2]) + float(ctx.model.lid_dims[2]) - GRIP_BELOW_LID_M)]
    return quat, contact, [float(v) for v in Place.hover_for(ctx, contact)]


def drop_reachable(ctx, drop, q_start, seeds=()):
    """Why the hand does not set the lid down at `drop`, or None: no
    straight-down hand at the carry's hover (the arm's own IK, from
    `q_start`, else from `seeds`), or no straight line down from it to the
    set-down's end."""
    from rammp_box_opening.runtime import approach

    chain = approach._chain()
    quat, contact, hover = setdown_poses(ctx, drop)
    q = chain.nearest_solution(list(q_start), hover, quat, seeds=tuple(seeds))
    if q is None:
        return "no straight-down hand reaches its carry height %.3f" % hover[2]
    pts, why = chain.straight_line(q, [contact[0], contact[1], contact[2] - SETDOWN_OVER_M], quat)
    if pts is None:
        return "the set-down is not a straight line the arm can fly (%s)" % why
    return None


# ---- where the lid goes, chosen with the lid lifted -----------------------

# The spot look: frames after the arm stood still this long, the first new
# one within SPOT_FRAME_S (depth_press.look's settle)
SPOT_SETTLE_S = dp.LOOK_SETTLE_S
SPOT_FRAME_S = 1.0
# What of the wrist's frame is not the table: nearer the lens than this —
# the fingers and the lid in them, 116-190 mm off (the OAK's stereo drops
# most of it) — and inside the held lid's column, its radius plus this
# (the lid, and what it hides), and the box's footprint (clear_table's
# BOX_CLEAR_M past it: the box, which every spot keeps clear of anyway)
WRIST_HAND_M = 0.20
HELD_LID_CLEAR_M = 0.03
# A spot tier's candidates checked for the hand's reach at most this many,
# and the carry planned to at most CARRY_TRIES of those that pass
TIER_TRIES = {"clear": 8, "most room": 10, "beside the box": 8, "nearest": 24}
CARRY_TRIES = 4
# The last resort: table points on rings round the box, from the lid's
# disc off the box's (and LID_CLEAR_M) outward NEAREST_STEP_M apart to
# NEAREST_MAX_M, every NEAREST_DEG, nearest the box first, then nearest
# the arm (the hand reaches nearer in)
NEAREST_STEP_M = 0.025
NEAREST_MAX_M = 0.45
NEAREST_DEG = 20


def _table_z(watcher, scene_fix, cpose):
    """The table the lid goes down on: the scene camera's (main's
    adopt_scene_table puts it on the watcher), else the box's base."""
    for z in (getattr(watcher, "table_z", None), getattr(scene_fix, "table_z_scene", None)):
        if z is not None and math.isfinite(float(z)):
            return float(z)
    return float(cpose.xyz[2])


def spot_look(node, watcher):
    """The wrist camera's points in the base frame (press_demo.wrist_points)
    from its first frame shot SPOT_SETTLE_S after the arm stood still, or
    None: no camera, no node, no new frame within SPOT_FRAME_S, or no pose
    for it."""
    from rammp_box_opening.tasks.press_demo import _spin_detect, wrist_points

    g = getattr(watcher, "grab", None)
    if node is None or g is None:
        return None
    t_end = time.monotonic() + SPOT_SETTLE_S
    while time.monotonic() < t_end:
        _spin_detect(node)
    after = node.get_clock().now().nanoseconds
    t_end = time.monotonic() + SPOT_FRAME_S
    while time.monotonic() < t_end:
        _spin_detect(node)
        if g.color_stamp is not None and g.depth is not None and dp._stamp_ns(g.color_stamp) > after:
            return wrist_points(watcher)
    return None


def _off_line(pts, line, clear_m):
    """`pts` without those within `clear_m` of the polyline `line`."""
    keep = np.ones(len(pts), bool)
    line = np.asarray(line, float)
    for a, b in zip(line[:-1], line[1:]):
        ab = b - a
        t = np.clip(((pts - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        keep &= np.linalg.norm(pts - (a + t[:, None] * ab), axis=1) > clear_m
    return pts[keep]


def wrist_table(ctx, pts, q, lens_in_hand=None):
    """The wrist's points `pts` with what is not the table taken out, the
    arm at joints `q` holding the lid: nearer the lens than WRIST_HAND_M
    (the hand, the fingers, the lid in them), inside the held lid's column
    (its radius and HELD_LID_CLEAR_M round the pads: the lid, and what it
    hides) and over the box's footprint. `lens_in_hand`: the camera
    config's measured wrist_lens_in_hand_m."""
    from rammp_box_opening.runtime import approach
    from rammp_box_opening.tasks.press_demo import container_half

    pts = np.asarray(pts, float)
    if not len(pts):
        return pts
    R, t = approach._chain().fk(list(q))
    R, t = np.asarray(R, float), np.asarray(t, float)
    lens = t + R @ np.asarray(lens_in_hand if lens_in_hand is not None else (0.0, 0.065, -0.105), float)
    pads = t + R @ np.asarray(dp.DOWN_CONTACT_TOOL, float)
    lid_r = 0.5 * max(float(ctx.model.lid_dims[0]), float(ctx.model.lid_dims[1]))
    bx, by = float(ctx.cpose.xyz[0]), float(ctx.cpose.xyz[1])
    keep = np.linalg.norm(pts - lens, axis=1) > WRIST_HAND_M
    keep &= np.hypot(pts[:, 0] - pads[0], pts[:, 1] - pads[1]) > lid_r + HELD_LID_CLEAR_M
    keep &= np.hypot(pts[:, 0] - bx, pts[:, 1] - by) > container_half(ctx.model) + clear_table.BOX_CLEAR_M
    return pts[keep]


def spot_map(ctx, wrist, q, scene_fix, scene_q, table_z, lens_in_hand=None):
    """The table map the lid's spot is chosen on (clear_table.table_map, as
    press_demo.room_map builds it): the wrist's frame taken with the lid
    lifted (`wrist`, wrist_table'd at the arm's joints `q`), pooled with the
    scene camera's points of the table — moved by its error at the box (the
    aim measured where the box is), what was the arm when it looked (its
    centre line at `scene_q`, clear_table.ARM_CLEAR_M round it: flatbot
    stood at HOME) left out. None where neither shows a table."""
    from rammp_box_opening.runtime import approach

    clouds = []
    tz = getattr(scene_fix, "table_z_scene", None)
    tz = float(tz) if tz is not None and math.isfinite(float(tz)) else float(table_z)
    pts = getattr(scene_fix, "table_points", None)
    if pts is not None and len(pts):
        pts = np.asarray(pts, float) + [float(ctx.cpose.xyz[0]) - float(scene_fix.pose.xyz[0]),
                                        float(ctx.cpose.xyz[1]) - float(scene_fix.pose.xyz[1]), 0.0]
        chain = approach._chain()
        if scene_q is not None and hasattr(chain, "points"):
            pts = _off_line(pts, chain.points(list(scene_q)), clear_table.ARM_CLEAR_M)
        clouds.append((pts, tz))
    if wrist is not None and len(wrist):
        clouds.append((wrist_table(ctx, wrist, q, lens_in_hand), tz))
    return clear_table.table_map(clouds, ctx.cpose.xyz[:2]) if clouds else None


def _nearest_points(ctx, table_z):
    """spot_candidates' last resort: table points round the box, the lid's
    disc off the box's, nearest the box first, then nearest the arm —
    inside the table the scene camera saw, where it saw one."""
    from rammp_box_opening.tasks.press_demo import container_half

    m = ctx.model
    lid_r = 0.5 * max(float(m.lid_dims[0]), float(m.lid_dims[1]))
    bx, by = float(ctx.cpose.xyz[0]), float(ctx.cpose.xyz[1])
    ext = getattr(ctx.worlds, "table_extent", None)
    first = container_half(m) + lid_r + clear_table.LID_CLEAR_M
    out = []
    for r in np.arange(first, NEAREST_MAX_M + 1e-9, NEAREST_STEP_M):
        ring = []
        for a in np.radians(np.arange(0.0, 360.0, NEAREST_DEG)):
            x, y = bx + r * math.cos(a), by + r * math.sin(a)
            if ext is not None and not (ext[0] + lid_r <= x <= ext[1] - lid_r and ext[2] + lid_r <= y <= ext[3] - lid_r):
                continue
            ring.append((math.hypot(x, y), x, y))
        out += [ContainerPose(xyz=(x, y, float(table_z)), yaw=0.0) for _d, x, y in sorted(ring)]
    return out


def spot_candidates(ctx, tmap, table_z, q, seeds=()):
    """Where the lid may go, best first, each one the hand reaches straight
    down from joints `q` (drop_reachable): (spot, tier, what it says).
    Tiers, each tried at most TIER_TRIES: 'clear' — where `tmap` shows
    room for the lid and the hand (press_demo.clear_lid_drops, best first);
    'most room' — nothing clear, the spots it judged, the most room first;
    'beside the box' — the spots beside it in the set-down zone
    (press_demo.lid_drop_candidates, drop_zone); 'nearest' — the table
    points round the box (_nearest_points). A generator: only as many as
    are asked for are checked."""
    from rammp_box_opening.tasks import press_demo as pd

    m = ctx.model
    tried = []

    def fresh(d):
        return all(math.hypot(d.xyz[0] - t.xyz[0], d.xyz[1] - t.xyz[1]) > 1e-3 for t in tried)

    tiers = []
    if tmap is not None:
        judged = []
        drops, spots, why = pd.clear_lid_drops(m, ctx.cpose, tmap, table_z, judged_out=judged)
        tiers.append(("clear", [(d, "where the cameras see room for it and the hand (%.0f mm to spare%s)" % (
            1000 * s.room_m, "" if s.what == "nothing" else ", %s nearest anything" % s.what))
            for d, s in zip(drops, spots)]))
        most = sorted(judged, key=lambda s: -s.room_m)
        tiers.append(("most room", [(ContainerPose(xyz=(s.xy[0], s.xy[1], float(table_z)), yaw=s.turn),
                                     "nothing reads clear (%s): the most room seen, %s %.0f mm short" % (
                                         why or "no spot judged", s.what, -1000 * s.room_m))
                                    for s in most]))
    zone = pd.drop_zone(m, ctx.worlds)
    said = "no camera shows the table" if tmap is None else "the hand reaches no spot the cameras judged"
    tiers.append(("beside the box", [(ContainerPose(xyz=tuple(float(v) for v in xyz), yaw=0.0),
                                      "%s: beside the box, in the set-down zone" % said)
                                     for xyz, _side in pd.lid_drop_candidates(m, ctx.cpose, table_z, zone=zone)]))
    tiers.append(("nearest", [(d, "%s, nor beside the box in the set-down zone: the nearest table point the hand "
                                  "reaches, off the box" % said) for d in _nearest_points(ctx, table_z)]))
    for tier, cands in tiers:
        n = 0
        for d, says in cands:
            if n >= TIER_TRIES[tier]:
                break
            if not fresh(d):
                continue
            n += 1
            tried.append(d)
            why = drop_reachable(ctx, d, q, seeds)
            if why is None:
                yield d, tier, says
# ---- the way back to the look pose, and the look ----------------------------


def _look_joints(way):
    """Where `way` (a reach_press.WayIn) starts its way down: the aimed
    look pose."""
    if way.move:
        return [float(v) for v in way.move[-1]]
    first = way.down.pts if way.down.walked else way.slide
    return [float(v) for v in first[0]]


def _down_backwards(ctx, way, world):
    """The way down backwards, to the look pose it starts at: the turn,
    then the slide — or the walk — each reversed (a proven line flown
    back is the same joints, the runner's gates alike)."""
    lines = []
    if way.down.pts is not None and len(way.down.pts) > 1:
        lines.append(("reach:slide-out" if way.down.walked else "depth:turn-back", list(way.down.pts)[::-1]))
    if not way.down.walked and way.slide is not None and len(way.slide) > 1:
        lines.append(("reach:slide-out", list(way.slide)[::-1]))
    return [dp.compliant_line(ctx, name, [[float(v) for v in q] for q in pts], way.plan.quat, FREE_LINE_SPEED, world)
            for name, pts in lines]


def way_back_legs(ctx, way, q_held):
    """From the hold, the arm held at joints `q_held`, back to the aimed
    look pose: the rise up the stroke (reach_press._rise_legs), then the way
    down backwards (_down_backwards) — one execution where they meet. A rise
    that is not the stroke's own (a straight line to its start) is followed
    by a walk from its end to the look pose, the one the runner flies.
    Returns (legs, None, None); (None, 'up', why) where no way up flies; or
    (the rise's legs, 'look', why) where no way from its end to the look
    pose does."""
    from rammp_box_opening.runtime import approach
    from rammp_box_opening.runtime.kinematics import mat_to_quat_xyzw

    world = _full_world(ctx)
    up, on_stroke, why = rp._rise_legs(ctx, way.stroke, q_held, world)
    if up is None:
        return None, "up", why
    if on_stroke:
        return up + _down_backwards(ctx, way, world), None, None
    chain = approach._chain()
    R, t = chain.fk(_look_joints(way))
    quat = mat_to_quat_xyzw(np.asarray(R, float))
    q = up[-1].goal_joints if up else [float(v) for v in q_held]
    pts, why = chain.walk(q, [float(v) for v in t], quat, step_m=dp.TURN_STEP_M, step_rad=dp.TURN_STEP_RAD,
                          max_step_rad=dp.TURN_MAX_JUMP_RAD)
    if pts is not None:
        why = dp.line_refusal(chain, pts)
    if pts is None or why is not None:
        return up, "look", "no way from the stroke's start back to the look pose flies (%s)" % why
    return up + [dp.compliant_line(ctx, "reach:slide-out", pts, quat, FREE_LINE_SPEED, world)], None, None


@dataclass
class KnobLook:
    """What the look after a press saw (knob_look)."""

    found: bool
    up_mm: float = None  # the knob above its lid: the highest any look read
    shift_m: float = None  # the button from where the press was aimed
    lid_dz_m: float = None  # the lid, by the wrist's own two readings
    status: str = ""
    stop: str = None  # why the run stops here: the box not found, or moved

    @property
    def popped(self):
        from rammp_box_opening.tasks.press_demo import knob_is_up

        return self.found and knob_is_up(self.up_mm)


def knob_look(node, ctx, runner, watcher, button, yaw, lid_before, press):
    """From the look pose after press number `press`: the oblique aim at
    the aimed `button` (reach_press._aim, the knob's height above its own
    lid), once more where it reads the knob down or finds no button
    (KNOB_RECHECK_AIMS) — the highest reading the verdict, as
    depth_press.knob_up_after's. A box the press slid farther than the
    staging grip takes (depth_press.SLIDE_OK_M) or whose lid the wrist
    reads BOX_LIFTED_M off its reading before (`lid_before`) moved: STOP."""
    from rammp_box_opening.tasks.press_demo import knob_is_up

    best, last, status, looks = None, None, "", 0
    for _k in range(1 + KNOB_RECHECK_AIMS):
        got, status, up, measured = rp._aim(node, watcher, yaw, button[2], button[:2], runner, "reach-after")
        looks += 1
        if got is None:
            continue
        last = (got, up, measured)
        if up is not None and (best is None or up > best):
            best = float(up)
        if knob_is_up(up):
            break
    if last is None:
        runner.note("reach_knob", press=press, found=False, looks=looks, why=status)
        print("[press_demo] AFTER PRESS %d: the wrist does not find the button from the look pose (%s)" % (press, status))
        return KnobLook(False, status=status, stop="the wrist does not find the box after the press (%s) — the knob is "
                                                    "not judged" % status)
    got, up, measured = last
    shift = rp.judge_box_moved(button, got[0])
    lid_after = rp._lid_z(measured[2], up)
    dz = None if lid_before is None or lid_after is None else lid_after - lid_before
    look = KnobLook(True, best, shift, dz, status)
    runner.note("reach_knob", press=press, found=True, looks=looks, knob_up_mm=None if best is None else round(best, 1),
                popped=look.popped, shift_mm=round(1000 * shift, 1), lid_dz_mm=None if dz is None else round(1000 * dz, 1))
    print("[press_demo] AFTER PRESS %d: knob %s (%d look%s); the button %.0f mm from where the press was aimed%s" % (
        press, "UP %.1f mm — the box is open" % best if look.popped else (
            "%.1f mm above the lid" % best if best is not None else "not judged"), looks, "" if looks == 1 else "s",
        1000 * shift, "" if dz is None else ", its lid %+.0f mm" % (1000 * dz)))
    if shift > dp.SLIDE_OK_M:
        look.stop = "the press MOVED the box %.0f mm — check it is still standing on the table" % (1000 * shift)
    elif dz is not None and abs(dz) > rp.BOX_LIFTED_M:
        look.stop = "the press moved the box: its lid %+.0f mm — check it is still standing on the table" % (1000 * dz)
    return look


# ---- the grip ---------------------------------------------------------------


@dataclass
class GripDescent:
    """The grip's stroke from the hand turned down at `q0` (grip_descent):
    `legs` [(name, joint points, speed)] as a press stroke's
    (reach_press.way_in_legs flies them), 'grip:approach' at grip_speed to
    GRIP_SLOW_M above where it ends, then 'grip:down' at TOUCH_SPEED, the
    closed pads' reference ending at `pad_z`."""

    legs: list
    quat: list
    q0: list
    pad_z: float


def grip_descent(chain, cfg, q0, quat, lid_z):
    """Straight down from the hand turned down at joints `q0`, holding
    `quat`, until the closed pads' reference (depth_press.DOWN_CONTACT_TOOL)
    stands GRIP_BELOW_LID_M under the lid at `lid_z`. Returns (GripDescent,
    None) or (None, why)."""
    q0 = [float(v) for v in q0]
    R0, t0 = chain.fk(q0)
    R0, t0 = np.asarray(R0, float), np.asarray(t0, float)
    pad0 = t0 + R0 @ np.asarray(dp.DOWN_CONTACT_TOOL, float)
    end = float(lid_z) - GRIP_BELOW_LID_M
    if float(pad0[2]) < end + GRIP_SLOW_M:
        return None, "the pads already stand %.0f mm above where the grip ends" % (1000 * (float(pad0[2]) - end))

    def down_to(q, z):
        tool_z = float(t0[2]) + (z - float(pad0[2]))  # the attitude held: the tool drops as the pads do
        pts, why = chain.straight_line(q, [float(t0[0]), float(t0[1]), tool_z], quat)
        return pts, None if pts is not None else "the grip's way down is not a straight line the arm can fly (%s)" % why

    approach, why = down_to(q0, end + GRIP_SLOW_M)
    if approach is None:
        return None, why
    down, why = down_to(approach[-1], end)
    if down is None:
        return None, why
    return GripDescent([("grip:approach", approach, cfg.grip_speed), ("grip:down", down, dp.TOUCH_SPEED)],
                       list(quat), q0, end), None


def solve_grip_way(ctx, cfg, button, tilt, q_from, lid_z):
    """The way from joints `q_from` (the look pose) down to the knob of the
    `button` the press was aimed at, at `tilt`: the press's way in
    (reach_press.reach_from_here's move to the look pose and slide, then
    walk_or_turn) with the grip's stroke (grip_descent onto the lid at
    `lid_z`) where the press's was — every line one the runner flies
    (depth_press.line_refusal). Returns (reach_press.WayIn, None) or (None,
    why)."""
    from rammp_box_opening.runtime import approach

    chain = approach._chain()
    move, plan, why = rp.reach_from_here(ctx, cfg, button, tilt, need_stroke=False, q=q_from)
    if plan is None:
        return None, why
    why = dp.line_refusal(chain, move)
    if why is not None:
        return None, "the move to its look pose (%s)" % why
    down, why = rp.walk_or_turn(ctx, move[-1] if move else q_from, plan.slide_pts, plan.start_xyz, plan.quat,
                                lid_z)
    if down is None:
        return None, "the way down (%s)" % why
    if down.slide is not None:  # it turns lower down, as the press did: the slide carries on to there
        plan.slide_pts, plan.start_xyz = down.slide, down.start_xyz
    grip, why = grip_descent(chain, cfg, down.pts[-1] if down.pts else plan.slide_pts[-1], down.quat, lid_z)
    if grip is None:
        return None, why
    lines = ([] if down.walked else [("the slide in", plan.slide_pts)]) + [(n, p) for n, p, _s in grip.legs]
    for name, line in lines:
        why = dp.line_refusal(chain, line)
        if why is not None:
            return None, "%s: %s" % (name, why)
    return rp.WayIn(move, plan, down, None if down.walked else plan.slide_pts, grip), None


def _grip_backwards(ctx, way, world):
    """The grip's way down backwards, from the knob to the look pose: up
    its stroke, then _down_backwards."""
    pts = []
    for _name, p, _speed in way.stroke.legs:
        pts += [[float(v) for v in q] for q in (p[1:] if pts else p)]
    legs = [dp.compliant_line(ctx, "grip:up", pts[::-1], way.stroke.quat, FREE_LINE_SPEED, world)] if len(pts) > 1 else []
    return legs + _down_backwards(ctx, way, world)


# ---- the lid's way off the box, and from its spot to the table -------------


@dataclass
class LiftOff:
    """From the knob held to the spot look (plan_lift): 'lid:lift'
    straight up and 'lid:clear' up and back toward the arm, `back_m`
    back — one execution after the close."""

    legs: list
    back_m: float = 0.0


def _clear_after_lift(ctx, chain, q, quat):
    """The clear from the lifted lid at joints `q`: to the first of
    reach_press.clear_targets whose straight line flies and leaves every
    joint room for the planner to start from (reach_press._near_a_limit),
    else the first that flies, as plan_way_out picks. Returns (points, how
    far back, None) or (None, 0, why)."""
    R, t = chain.fk(q)
    chosen, whys = None, []
    for target in rp.clear_targets(ctx, np.asarray(R, float), t):
        back = math.hypot(target[0] - float(t[0]), target[1] - float(t[1]))
        pts, why = chain.straight_line(q, target, quat)
        if pts is not None:
            why = dp.line_refusal(chain, pts)
        if pts is None or why is not None:
            whys.append("%.0f mm back: %s" % (1000 * back, why))
            continue
        pts = [[float(v) for v in p] for p in pts]
        near = rp._near_a_limit(ctx, pts[-1])
        if not near:
            return pts, back, None
        whys.append("%.0f mm back: %s at its end" % (1000 * back, near))
        chosen = chosen or (pts, back)
    if chosen is not None:
        return chosen[0], chosen[1], None
    return None, 0.0, "no way up and back from the lifted lid flies (%s)" % "; ".join(whys)


def plan_lift(ctx, cfg, q_grip, quat):
    """The lid's way off the box from the knob held at joints `q_grip`, the
    hand at `quat` (LiftOff): straight up lift_off_m, then the clear
    (_clear_after_lift). The grip's lookahead, while the hand comes down to
    the knob. Returns (LiftOff, None) or (None, why)."""
    from rammp_box_opening.runtime import approach

    chain = approach._chain()
    world = _full_world(ctx)
    q_grip = [float(v) for v in q_grip]
    _R, t = chain.fk(q_grip)
    lift, why = chain.straight_line(q_grip, [float(t[0]), float(t[1]), float(t[2]) + lift_off_m(ctx.model)], quat)
    if lift is not None:
        why = dp.line_refusal(chain, lift)
    if lift is None or why is not None:
        return None, "the lift off the canister is not a straight line the arm can fly (%s)" % why
    lift = [[float(v) for v in p] for p in lift]
    clear, back, why = _clear_after_lift(ctx, chain, lift[-1], quat)
    if clear is None:
        return None, why
    return LiftOff([dp.compliant_line(ctx, "lid:lift", lift, quat, cfg.lift_speed, world),
                    dp.compliant_line(ctx, "lid:clear", clear, quat, FREE_LINE_SPEED, world)], back), None


@dataclass
class Carry:
    """From the spot look to the lid on the table (plan_carry): `legs`
    'lid:carry' (the planner's) to the hover over `drop`, 'place:down' and
    'place:touch' — one execution — and `retreat`, the joint points back up
    to the hover after the release, at `quat`."""

    legs: list
    retreat: list
    quat: list
    drop: object


def plan_carry(ctx, cfg, q_from, drop):
    """The carry from joints `q_from` (where the spot look stands) to the
    spot `drop`, in the full world — the box and what the cameras see
    standing (press_demo.adopt_seen) — then the set-down's straight lines
    from its end and the retreat's back up. Returns (Carry, None) or (None,
    why)."""
    from rammp_box_opening.runtime import approach

    chain = approach._chain()
    world = _full_world(ctx)
    q_from = [float(v) for v in q_from]
    quat, contact, hover = setdown_poses(ctx, drop)
    try:
        carry, _st = rp._plan_motion(ctx, PlanState(joints=q_from, chain=0), "lid:carry", ("pose", hover, quat),
                                     world, TRANSIT_SPEED)
    except RuntimeError as e:
        return None, "the carry (%s)" % rp._short(e)
    if carry.traj is not None:
        carry.traj = approach.on_branch_of(carry.traj, q_from)  # continues the clear, not a turn away
        carry.goal_joints = [float(v) for v in carry.traj.points[-1].positions]
    carry.gains = dp.COMPLIANT_GAINS
    lines = []
    q = carry.goal_joints
    for name, z in (("place:down", contact[2] + SETDOWN_SLOW_M), ("place:touch", contact[2] - SETDOWN_OVER_M),
                    ("place:retreat", hover[2])):
        pts, why = chain.straight_line(q, [hover[0], hover[1], z], quat)
        if pts is not None:
            why = dp.line_refusal(chain, pts)
        if pts is None or why is not None:
            return None, "%s (%s)" % (name, why)
        lines.append([[float(v) for v in p] for p in pts])
        q = lines[-1][-1]
    legs = [carry, dp.compliant_line(ctx, "place:down", lines[0], quat, cfg.setdown_speed, world),
            dp.compliant_line(ctx, "place:touch", lines[1], quat, dp.TOUCH_SPEED, world)]
    return Carry(legs, lines[2], quat, drop), None


def plan_home_after(ctx, cfg, carry):
    """The retreat back up from the set-down and home planned from its
    end, with the lid lying at the spot (ctx.lid_at, for this plan only):
    the legs after the release, one gains spec, or (None, why). The carry's
    lookahead."""
    from rammp_box_opening.runtime import approach

    world = _full_world(ctx)
    q = carry.retreat[-1]
    was = ctx.lid_at
    ctx.lid_at = carry.drop  # home is planned around the lid where it will lie
    try:
        home = rp._home_legs(ctx, cfg, start=q)
    except RuntimeError as e:
        return None, "home after the set-down (%s)" % rp._short(e)
    finally:
        ctx.lid_at = was
    for leg in home:
        if leg.traj is not None:
            leg.traj = approach.on_branch_of(leg.traj, q)
            leg.goal_joints = [float(v) for v in leg.traj.points[-1].positions]
            q = leg.goal_joints
        if leg.guard is None:
            leg.gains = dp.COMPLIANT_GAINS  # one gains spec: the retreat and home merge
    return [dp.compliant_line(ctx, "place:retreat", carry.retreat, carry.quat, FREE_LINE_SPEED, world), *home], None


# ---- the runs ---------------------------------------------------------------


def _started(last):
    return last.flown_traj is not None and bool(last.progress)


def _stop_home(ctx, cfg, runner, why):
    """STOP from the look pose, in free air: home (reach_press._go_home,
    which lets go of a home stopped part-way). Returns 1."""
    rp._go_home(ctx, cfg, runner, "STOP: %s" % why)
    return 1


def _note_frames(ctx, **facts):
    frames = getattr(ctx, "mission_frames", None)
    if frames is not None:
        frames.note(**facts)


def _press_and_back(ctx, cfg, runner, way, legs, top_z, press):
    """Press number `press`: the way in `legs` (way_in_legs of `way`) as
    one execution, the hold on the button whose top is at `top_z`, and the
    way back to the look pose (way_back_legs) as one more. Returns (None,
    where the hold held the pads when it felt the button, else None) at the
    look pose, or (the exit code, None) where the run stopped: home from the
    look pose where the way in did not start, the arm let go and held where
    a move stopped part-way, up off the button and home from over the box
    where the way back did not start."""
    names = [lg.name for lg in legs]
    if way.down.short_deg:
        print("[press_demo] TURN: %.0f deg short of straight down (no hand nearer straight down flies here)"
              % way.down.short_deg)
    print("[press_demo] %sONE MOTION from the look pose to below the button: %s" % (
        "PRESS %d of %d: " % (press, PRESSES) if press > 1 else "", ", ".join(names)))
    print("[press_demo] %s" % dp.press_text(way.stroke, cfg))
    res = runner.run(legs, execute=True)
    last = res[-1]
    flown = all(r.ok for r in res)
    retimed = rp._note_flight(runner, "reach_way_in_flight", names, last) if last.flown_traj is not None else {}
    if not flown and not _started(last):
        return _stop_home(ctx, cfg, runner, "the way in did not start (%s) — nothing was pressed%s" % (
            last.detail or last.outcome, "" if press == 1 else " again")), None
    if not flown:
        print("[press_demo] STOP: the way in stopped during %s (%s) — the arm holds" % (
            rp._phase_at(names, retimed.get("seg_start_fracs") or [0.0], last.progress), last.detail or last.outcome))
        rp.let_go(ctx)
        return 1, None
    runner.note("reach_slid_in", turned_down=way.down.walked, residual_nm=None)
    seen = {}
    held = dp.compliant_hold(ctx, cfg, runner, way.stroke, top_z, seen)
    if held.felt:
        _note_frames(ctx, stage="pressed", touch_xyz=list(seen["touch_xyz"]) if seen.get("touch_xyz") else None)
    back, kind, why = way_back_legs(ctx, way, seen.get("held_joints") or ctx.client.joints())
    if kind == "up":
        print("[press_demo] STOP: the way up is not a straight line the arm can fly (%s) — the arm holds" % why)
        rp.let_go(ctx)
        return 1, None
    if kind == "look":
        print("[press_demo] STOP: %s — up off the button, then home from over the box; the knob is not judged" % why)
        result, homed = rp._rise_then_home(ctx, cfg, runner, back, held, None, None)
        if not result.free or not homed:
            rp.let_go(ctx)
        return 1, None
    groups = merge_groups(back)
    print("[press_demo] %s from the button back to the look pose: %s" % (
        "ONE MOTION" if len(groups) == 1 else "in %d goals" % len(groups), ", ".join(lg.name for lg in back)))
    res = runner.run(back, execute=True)
    last = res[-1]
    if len(res) == 1 and last.flown_traj is not None:
        rp._note_flight(runner, "reach_way_back_flight", [lg.name for lg in groups[0]], last)
    if all(r.ok for r in res):
        return None, (float(seen["touch_xyz"][2]) if held.felt and seen.get("touch_xyz") else None)
    if _started(last):
        print("[press_demo] STOP: the way back to the look pose stopped during %s (%s) — the arm holds" % (
            rp._in_flight(back, last, runner.last_retime), last.detail or last.outcome))
        rp.let_go(ctx)
        return 1, None
    if len(res) == 1:
        # nothing moved, the pads still on the button: up on its own,
        # solved again from where the arm stands, then home from over the box
        up, _on_stroke, why = rp._rise_legs(ctx, way.stroke, ctx.client.joints(), _full_world(ctx))
        if up is None:
            print("[press_demo] STOP: the way up is not a straight line the arm can fly (%s) — the arm holds" % why)
            rp.let_go(ctx)
            return 1, None
        print("[press_demo] STOP: the way back to the look pose did not start (%s) — up off the button, then home "
              "from over the box" % (last.detail or last.outcome))
        result, homed = rp._rise_then_home(ctx, cfg, runner, up, held, None, None)
    else:
        print("[press_demo] STOP: the way back's %s did not start (%s) — home from over the box" % (
            rp._in_flight(back, last, runner.last_retime), last.detail or last.outcome))
        result, homed = held, rp._home_from_over_box(ctx, cfg, runner, held)
    if not result.free or not homed:
        rp.let_go(ctx)
    return 1, None


def run_full(ctx, cfg, runner, eyes, way, legs, button, yaw, lid_before):
    """The full reach-out run from the aimed look pose, its way in `way`
    proven and flown as `legs` (reach_press._reach_press's, one execution):
    the press and the way back (_press_and_back), the look (knob_look), the
    press once more where the knob is not up, then the lid off
    (take_lid_off) at the aimed `button` (its top the press's), the grip
    going by the lid where the last hold held the pads (grip_lid_z).
    `eyes` (node, watcher, scene_fix, the joints the scene camera looked
    at the arm in): the cameras. Returns the process exit code: 0 only with
    the lid set down and the arm home. From the way in's start on, anything raised lets go (as
    _fly_way_in): a Ctrl+C in the hold's read lands with no goal in flight,
    the springs pushing on into the box."""
    node, watcher = eyes[0], eyes[1]
    tilt = way.plan.tilt_deg
    top_z = float(button[2])
    ctx.contact_pad = CONTACT_SHIFT_PAD_M  # the stroke meets the box by design: every world pads it from here
    held_z = None
    try:
        for press in range(1, PRESSES + 1):
            code, z = _press_and_back(ctx, cfg, runner, way, legs, top_z, press)
            if code is not None:
                return code
            held_z = z if z is not None else held_z
            knob = knob_look(node, ctx, runner, watcher, button, yaw, lid_before, press)
            _note_frames(ctx, popped=bool(knob.popped) if knob.found else None)
            if knob.stop is not None:
                return _stop_home(ctx, cfg, runner, knob.stop)
            if knob.popped:
                break
            if press == PRESSES:
                return _stop_home(ctx, cfg, runner, "pressed %d times and the knob is not up (%s) — the lid is not "
                                                    "taken off" % (PRESSES, "%.1f mm above the lid" % knob.up_mm
                                                                   if knob.up_mm is not None else "not judged"))
            print("[press_demo] the knob is not up (%s): pressing once more — the same press, from the look pose" % (
                "%.1f mm above the lid" % knob.up_mm if knob.up_mm is not None else "not judged"))
            way, why, _kind = rp.solve_way_in(ctx, cfg, button, tilt, ctx.client.joints())
            if way is not None:
                legs = rp.way_in_legs(ctx, cfg, way)
                why = rp._not_one_execution(ctx, legs)
            if why is not None:
                return _stop_home(ctx, cfg, runner, "the press once more does not fly (%s)" % why)
            runner.note("reach_way_in", press=press + 1, tilt_deg=way.plan.tilt_deg, walked=way.down.walked,
                        short_deg=way.down.short_deg, k=round(way.stroke.k, 1), end_z=round(way.stroke.end_z, 4))
        lid_z = grip_lid_z(top_z, held_z)
        return take_lid_off(ctx, cfg, runner, eyes, button, tilt, lid_z, "the lid at %.3f (the press's top %.3f%s)" % (
            lid_z, top_z, "" if held_z is None else ", the pads held at %.3f" % held_z))
    except BaseException:
        rp.let_go(ctx)
        raise


def _back_out(ctx, cfg, runner, way, why, opened=True):
    """From the knob, the fingers on nothing they keep: open (`opened`
    False: they are open already), the grip's way down backwards to the look
    pose (_grip_backwards: one execution), then home from there. Returns 1."""
    world = _full_world(ctx)
    if opened:
        res = runner.run([_gripper_leg(ctx, PlanState(joints=None, chain=0), "grip:open", GRIPPER_CMD_OPEN, world)],
                         execute=True)
        if not all(r.ok for r in res):
            print("[press_demo] STOP: %s, and the fingers do not open (%s) — the arm holds" % (why, res[-1].detail))
            rp.let_go(ctx)
            return 1
    back = _grip_backwards(ctx, way, world)
    print("[press_demo] backing out the way the hand came down: %s" % ", ".join(lg.name for lg in back))
    res = runner.run(back, execute=True)
    if not all(r.ok for r in res):
        print("[press_demo] STOP: %s, and the way back out %s (%s) — the arm holds" % (
            why, "stopped" if _started(res[-1]) else "did not start", res[-1].detail or res[-1].outcome))
        rp.let_go(ctx)
        return 1
    return _stop_home(ctx, cfg, runner, why)


def take_lid_off(ctx, cfg, runner, eyes, button, tilt, lid_z, how):
    """From the look pose, the knob up: the lid off, set down, home. The
    grip's way down (solve_grip_way, from where the arm stands), the
    fingers opening as it flies, ONE execution, the lift planned while it
    does (plan_lift); the close, judged by the grip band; the lift and the
    clear, ONE execution; there the spot look (choose_and_carry); the carry
    and the set-down, ONE execution, home planned while it flies; the
    release; the retreat and home, ONE execution. `eyes` (node, watcher,
    scene_fix, the joints the scene camera looked at the arm in) the
    cameras; `how` says what the grip's lid height `lid_z` is. Returns the
    exit code: 0 with the lid set down and the arm home."""
    world = _full_world(ctx)
    ctx.contact_pad = CONTACT_SHIFT_PAD_M  # the grip meets the box by design
    way, why = solve_grip_way(ctx, cfg, button, tilt, ctx.client.joints(), lid_z)
    if way is None:
        return _stop_home(ctx, cfg, runner, "the hand does not come down to the knob (%s) — the lid is not taken off" % why)
    legs = rp.way_in_legs(ctx, cfg, way)
    names = [lg.name for lg in legs]
    runner.note("reach_grip", lid_z=round(lid_z, 4), pads_z=round(way.stroke.pad_z, 4), tilt_deg=tilt,
                walked=way.down.walked, short_deg=way.down.short_deg, how=how)
    print("[press_demo] GRIP: the fingers open on the way down; ONE MOTION from the look pose to the knob: %s — the "
          "closed pads %.0f mm under %s%s" % (", ".join(names), 1000 * GRIP_BELOW_LID_M, how,
                                             ", the hand %.0f deg short of straight down" % way.down.short_deg
                                             if way.down.short_deg else ""))
    opened = _gripper_leg(ctx, PlanState(joints=None, chain=0), "grip:open", GRIPPER_CMD_OPEN, world, defer_join=True)
    res = runner.run([opened, *legs], execute=True, lookahead=lambda end: plan_lift(ctx, cfg, end, way.stroke.quat))
    ahead = runner.lookahead_result
    last = res[-1]
    if not all(r.ok for r in res):
        if not _started(last):
            return _stop_home(ctx, cfg, runner, "the way down to the knob did not start (%s) — the lid is not taken "
                                                "off" % (last.detail or last.outcome))
        print("[press_demo] STOP: the way down to the knob stopped during %s (%s) — the arm holds" % (
            rp._in_flight(legs, last, runner.last_retime), last.detail or last.outcome))
        rp.let_go(ctx)
        return 1
    lift, why = ahead if ahead is not None else (None, "the lookahead plan failed")
    if lift is None:
        lift, why = plan_lift(ctx, cfg, ctx.client.joints(), way.stroke.quat)
    if lift is None:
        return _back_out(ctx, cfg, runner, way, "the lid has no way off the box (%s) — not gripping it" % why,
                         opened=False)
    close = _gripper_leg(ctx, PlanState(joints=None, chain=0), "grip:close", GRIPPER_CMD_CLOSED, world,
                         verify=band_verify(cfg.grip_band))
    print("[press_demo] CLOSE, then ONE MOTION: %s — %.0f mm straight up, then %.0f mm back toward the arm" % (
        ", ".join(lg.name for lg in lift.legs), 1000 * lift_off_m(ctx.model), 1000 * lift.back_m))
    res = runner.run([close, *lift.legs], execute=True)
    grip = [r for r in res if r.leg_name in ("grip:open", "grip:close")]
    if not grip or not grip[-1].ok or grip[-1].leg_name != "grip:close":
        detail = grip[-1].detail if grip else res[-1].detail
        runner.note("reach_grip_result", ok=False, detail=detail)
        print("[press_demo] GRIP FAILED — %s: the fingers closed on air or on something else, so the knob is not up "
              "or the grip missed it" % detail)
        return _back_out(ctx, cfg, runner, way, "the grip failed (%s) — the lid is not taken off" % detail)
    runner.note("reach_grip_result", ok=True, detail=grip[-1].detail)
    print("[press_demo] LID GRIPPED — %s" % grip[-1].detail)
    moves = [r for r in res if r.leg_name not in ("grip:open", "grip:close")]
    last = moves[-1] if moves else res[-1]
    if not moves or not all(r.ok for r in moves):
        if len(moves) <= 1 and not _started(last):
            # nothing moved: the knob in the fingers, the lid still on the canister
            return _back_out(ctx, cfg, runner, way, "the lift did not start (%s) — the lid is let go where it lies, "
                                                   "on the box" % (last.detail or last.outcome))
        print("[press_demo] STOP: the lift stopped during %s (%s) — the arm holds the lid" % (
            rp._in_flight(lift.legs, last, runner.last_retime), last.detail or last.outcome))
        rp.let_go(ctx)
        return 1
    return choose_and_carry(ctx, cfg, runner, eyes)


def choose_and_carry(ctx, cfg, runner, eyes):
    """The lid lifted and clear of the box, in the fingers: its one stop.
    The wrist's frame (spot_look) and the scene camera's table make the map
    (spot_map) the planner's world is made from too (press_demo.adopt_seen,
    the spot kept clear); the first of spot_candidates the carry plans to
    (plan_carry, at most CARRY_TRIES) is where the lid goes, one line saying
    which tier and why. Then the carry and the set-down, ONE execution,
    home planned while it flies (plan_home_after); the knuckle read (the
    lid slipped?); the release; the retreat and home, ONE execution.
    Returns the exit code: 0 with the lid set down and the arm home."""
    from rammp_box_opening.primitives.look import look_joints
    from rammp_box_opening.tasks import press_demo as pd

    node, watcher, scene_fix, scene_q = eyes
    world = _full_world(ctx)
    q = [float(v) for v in ctx.client.joints()]
    table_z = _table_z(watcher, scene_fix, ctx.cpose)
    wrist = spot_look(node, watcher)
    tmap = spot_map(ctx, wrist, q, scene_fix, scene_q, table_z,
                    getattr(getattr(watcher, "grab", None), "wrist_lens_in_hand_m", None))
    rest = list(pd.rest_joints(cfg))
    carry, refused, chosen = None, [], None
    for drop, tier, says in spot_candidates(ctx, tmap, table_z, q, seeds=(look_joints(rest), rest)):
        if len(refused) >= CARRY_TRIES:
            break
        pd.adopt_seen(ctx, runner, tmap, ctx.cpose.xyz[:2], table_z, keep_clear=[drop.xyz],
                      when="wrist and scene camera, the lid lifted")
        carry, why = plan_carry(ctx, cfg, q, drop)
        if carry is not None:
            chosen = (drop, tier, says)
            break
        refused.append("[%.2f, %.2f]: %s" % (drop.xyz[0], drop.xyz[1], why))
    runner.note("lid_spot", xyz=None if chosen is None else [round(v, 4) for v in chosen[0].xyz],
                tier=None if chosen is None else chosen[1], wrist_points=0 if wrist is None else len(wrist),
                mapped=tmap is not None, refused="; ".join(refused) or None)
    if carry is None:
        print("[press_demo] STOP: no carry to a spot the lid could go to plans (%s) — the arm holds the lid"
              % ("; ".join(refused) or "no spot the hand reaches"))
        return 1
    drop, tier, says = chosen
    ctx.lid_drop = drop
    print("[press_demo] lid goes to [%.2f, %.2f, %.3f] — %.0f mm from the box, %s%s" % (
        drop.xyz[0], drop.xyz[1], drop.xyz[2], 1000 * math.hypot(drop.xyz[0] - ctx.cpose.xyz[0],
                                                                 drop.xyz[1] - ctx.cpose.xyz[1]),
        says, "; the planner refused %s" % "; ".join(refused) if refused else ""))
    print("[press_demo] ONE MOTION: %s — to [%.2f, %.2f] and down onto the table" % (
        ", ".join(lg.name for lg in carry.legs), drop.xyz[0], drop.xyz[1]))
    res = runner.run(carry.legs, execute=True, lookahead=lambda _end: plan_home_after(ctx, cfg, carry))
    ahead = runner.lookahead_result
    last = res[-1]
    if len(res) == 1 and last.flown_traj is not None:
        rp._note_flight(runner, "reach_carry_flight", [lg.name for lg in merge_groups(carry.legs)[0]], last)
    if not all(r.ok for r in res):
        print("[press_demo] STOP: the lid's way to the table %s during %s (%s) — the arm holds the lid" % (
            "stopped" if _started(last) else "did not start", rp._in_flight(carry.legs, last, runner.last_retime),
            last.detail or last.outcome))
        if _started(last):
            rp.let_go(ctx)
        return 1
    home, why = ahead if ahead is not None else (None, "the lookahead plan failed")
    if home is None:
        home, why = plan_home_after(ctx, cfg, carry)
    ok_q, pos, _stalled = ctx.client.gripper_cmd(None)  # query only: still on the knob?
    kept = not ok_q or pos is None or in_band(pos, cfg.grip_band)
    runner.note("reach_setdown", knuckle=None if pos is None else round(float(pos), 3), kept=kept)
    if not kept:
        print("[press_demo] the fingers read %.3f at the table, outside the grip band %s: the lid slipped on the way"
              % (pos, list(cfg.grip_band)))
    release = _gripper_leg(ctx, PlanState(joints=None, chain=0), "place:open", GRIPPER_CMD_OPEN, world)
    ctx.lid_at = drop  # every world from here holds the lid where it was set down
    retreat = home if home is not None else [
        dp.compliant_line(ctx, "place:retreat", carry.retreat, carry.quat, FREE_LINE_SPEED, world)]
    print("[press_demo] RELEASE, then ONE MOTION: %s" % ", ".join(lg.name for lg in retreat))
    res = runner.run([release, *retreat], execute=True)
    if not res[0].ok:
        print("[press_demo] STOP: the fingers did not open at the table (%s) — the arm holds" % res[0].detail)
        rp.let_go(ctx)  # the spring pressing the lid onto the table
        return 1
    if not all(r.ok for r in res):
        last = res[-1]
        if _started(last):
            print("[press_demo] STOP: the way home from the lid stopped during %s (%s) — the arm holds" % (
                rp._in_flight(retreat, last, runner.last_retime), last.detail or last.outcome))
            rp.let_go(ctx)
            return 1
        up, _why = rp._line(ctx, retreat[0].target[1], carry.quat)
        if up:
            res = runner.run([dp.compliant_line(ctx, "place:retreat", up, carry.quat, FREE_LINE_SPEED, world)],
                             execute=True)
            if not all(r.ok for r in res):
                print("[press_demo] STOP: the way up from the lid stopped (%s) — the arm holds" % res[-1].detail)
                rp.let_go_part_way(ctx, res)
                return 1
        home = None  # planned from where the arm stands now
    if home is None and not rp._go_home(ctx, cfg, runner, "home from the lid (%s)" % why if why else ""):
        return 1
    runner.finish()
    if not kept:
        print("[press_demo] STOP: the arm is home, but the lid slipped out of the fingers on the way to [%.2f, %.2f]"
              % (drop.xyz[0], drop.xyz[1]))
        return 1
    _note_frames(ctx, stage="done", popped=True)
    print("[press_demo] DONE — box open, lid set down at [%.2f, %.2f] (%s), arm home" % (drop.xyz[0], drop.xyz[1], tier))
    return 0
