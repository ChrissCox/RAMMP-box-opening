"""The reach-out's lid off: the compliant reach-out press (tasks/reach_press)
as the whole run — press, see the knob up, grip it, lift the lid off, set it
down where the cameras see room, home. Owner 2026-10-28: every box goes
through the reach-out, and the full run is its default; --press-only stops
after the press as before (reach_press._fly_way_in).

  SPOT    before anything moves: where the lid goes, among what the scene
          camera sees standing on the table (press_demo.clear_lid_drops, the
          staging run's own rules), the first spot the hand reaches straight
          down (drop_reachable). None: refused, nothing moved (exit 4).
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
  LIFT    the lid straight up off the canister (lift_off_m), up and back
          toward the arm clear of the box's padded volume (as the way home's
          clear), the carry to the spot, down onto the table: ONE execution.
          The fingers open; back up to the carry's height and home: one more.

Every move flies in the driver's joint impedance under the run's one gains
spec (depth_press.COMPLIANT_GAINS, RIGID), unguarded, merged where its legs
meet: nothing the run reads stops the arm. It decides only what it must:
the knob up or not, the box where it was, the fingers on the knob or on
air, the lid still in them at the table. Contact is bounded by the spring,
not stopped by a guard: the grip's pads ride on the lid where it stands
higher than estimated, and the set-down presses the lid SETDOWN_OVER_M past
where the scene camera's table says it lands.
"""

import math
from dataclasses import dataclass

import numpy as np

from rammp_box_opening.constants import FREE_LINE_SPEED, GRIPPER_CMD_CLOSED, GRIPPER_CMD_OPEN, TRANSIT_SPEED
from rammp_box_opening.models.container import ContainerPose
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
# (as the staging run's after-look was): pressing a knob that popped shuts
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


def drop_reachable(ctx, drop, q_start):
    """Why the hand does not set the lid down at `drop`, or None: no
    straight-down hand at the carry's hover (the arm's own IK, from
    `q_start` and the hand turned down from there), or no straight line
    down from it to the set-down's end."""
    from rammp_box_opening.primitives.look import look_joints
    from rammp_box_opening.runtime import approach

    chain = approach._chain()
    quat, contact, hover = setdown_poses(ctx, drop)
    q = chain.nearest_solution(list(q_start), hover, quat, seeds=(look_joints(list(q_start)),))
    if q is None:
        return "no straight-down hand reaches its carry height %.3f" % hover[2]
    pts, why = chain.straight_line(q, [contact[0], contact[1], contact[2] - SETDOWN_OVER_M], quat)
    if pts is None:
        return "the set-down is not a straight line the arm can fly (%s)" % why
    return None


def resolve_lid_drop(ctx, cfg, runner, watcher, scene_fix):
    """Where the lid goes, resolved before the reach-out's first motion by
    the staging run's rules (press_demo main before 2026-10-28): among what
    the scene camera sees standing on the table (room_map,
    clear_lid_drops), what it sees kept clear of every motion planned
    (adopt_seen); without its points, beside the box in the set-down zone
    (lid_drop_candidates). Nothing in the environment is configured. Of
    those, the first the hand reaches straight down (drop_reachable) is
    ctx.lid_drop, and it and the spots after it ctx.lid_drops. False — and
    why, printed; nothing has moved — where there is none."""
    from rammp_box_opening.runtime.branches import planner_branch
    from rammp_box_opening.tasks import press_demo as pd

    m = ctx.model
    cpose = scene_fix.pose
    table_z = getattr(watcher, "table_z", None)
    if table_z is None or not math.isfinite(float(table_z)):
        table_z = getattr(scene_fix, "table_z_scene", float("nan"))
    if table_z is None or not math.isfinite(float(table_z)):
        table_z = float(cpose.xyz[2])  # the box stands on the table
    table_z = float(table_z)
    tmap = pd.room_map(watcher, scene_fix, cpose, at_staging=False)
    spots = None
    if tmap is not None:
        drops, spots, why = pd.clear_lid_drops(m, cpose, tmap, table_z)
        # the planner's worlds from the same map, the lid's spots kept clear
        pd.adopt_seen(ctx, runner, tmap, cpose.xyz[:2], table_z, keep_clear=[d.xyz for d in drops],
                      when="scene camera")
        if not drops:
            print("[press_demo] STOP: nowhere to set the lid down: %s — move the box, or clear the table around it; "
                  "nothing moved" % why)
            return False
    else:
        zone = pd.drop_zone(m, ctx.worlds)
        # every spot beside the box, best first, as the staging run's place
        # fell through them
        drops = [ContainerPose(xyz=tuple(float(v) for v in xyz), yaw=0.0)  # the hand unturned
                 for xyz, _side in pd.lid_drop_candidates(m, cpose, table_z, zone=zone)]
        if not drops:
            print("[press_demo] STOP: no lid drop spot beside the box at [%.2f, %.2f] falls inside the set-down zone "
                  "(need %.0f mm clear, x %s y %s%s) — move the box; nothing moved" % (
                      cpose.xyz[0], cpose.xyz[1], 1000 * pd.lid_place_min_clear(m), [round(v, 2) for v in zone[:2]],
                      [round(v, 2) for v in zone[2:]],
                      ": the table the scene camera saw" if ctx.worlds.table_extent is not None else ""))
            return False
    rest = planner_branch(pd.rest_joints(cfg))
    passed = []
    for i, d in enumerate(drops):
        why = drop_reachable(ctx, d, rest)
        if why is None:
            break
        passed.append("[%.2f, %.2f]: %s" % (d.xyz[0], d.xyz[1], why))
    else:
        runner.note("lid_spot", xyz=None, spots=len(drops), passed="; ".join(passed))
        print("[press_demo] STOP: the hand reaches none of the %d spot(s) the lid could go to (%s) — move the box "
              "nearer the arm; nothing moved" % (len(drops), "; ".join(passed)))
        return False
    ctx.lid_drop, ctx.lid_drops = drops[i], drops[i:]
    d = drops[i]
    far = 1000 * math.hypot(d.xyz[0] - cpose.xyz[0], d.xyz[1] - cpose.xyz[1])
    if spots is None:
        runner.note("lid_spot", xyz=[round(v, 4) for v in d.xyz], spots=len(drops), wrist=False,
                    passed="; ".join(passed) or None)
        print("[press_demo] lid goes to [%.2f, %.2f, %.3f] — %.0f mm from the box, beside it (no room map: nothing "
              "checked what stands there)%s" % (*d.xyz, far, "; passed over, out of the hand's reach: %s"
                                                % "; ".join(passed) if passed else ""))
        return True
    best = spots[i]
    runner.note("lid_spot", xyz=[round(v, 4) for v in d.xyz], turn_deg=round(math.degrees(best.turn)),
                room_mm=round(1000 * best.room_m, 1), nearest=best.what, spots=len(spots), wrist=False,
                passed="; ".join(passed) or None)
    print("[press_demo] lid goes to [%.2f, %.2f, %.3f] — %.0f mm from the box, where the cameras see room for it and "
          "the hand (%.0f mm to spare%s)%s%s" % (
              *d.xyz, far, 1000 * best.room_m, "" if best.what == "nothing" else ", %s nearest anything" % best.what,
              "; the hand turns %+.0f deg to set it down" % math.degrees(best.turn) if best.turn else "",
              "; passed over, out of the hand's reach: %s" % "; ".join(passed) if passed else ""))
    return True


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
    (KNOB_RECHECK_AIMS) — the highest reading the verdict, as the staging
    run's after-look's was. A box the press slid farther than the staging
    grip took (depth_press.SLIDE_OK_M) or whose lid the wrist
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


# ---- the lid's way to the table, planned while the hand comes down to it ---


@dataclass
class Tail:
    """From the knob held to home (plan_tail): `carry` the legs of one
    execution — 'lid:lift' straight up, 'lid:clear' up and back toward the
    arm, 'lid:carry' (the planner's) to the hover over `drop`, then
    'place:down' and 'place:touch' onto the table — and `home` the legs
    after the release: 'place:retreat' back up to the hover, then home.
    `passed`: why nearer spots were passed over, or None."""

    carry: list
    home: list
    drop: object
    back_m: float = 0.0
    passed: str = None


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


def _carry_to(ctx, cfg, chain, q_clear, drop, world):
    """The carry from joints `q_clear` to the spot `drop` (the planner's,
    in the full world: the box and what the cameras see standing), the
    set-down from its end, the retreat back up and home planned from there
    with the lid on the table (ctx.lid_at, for that plan only). Returns
    ((carry legs, home legs), None) or (None, why)."""
    from rammp_box_opening.runtime import approach

    quat, contact, hover = setdown_poses(ctx, drop)
    try:
        carry, _st = rp._plan_motion(ctx, PlanState(joints=list(q_clear), chain=0), "lid:carry",
                                     ("pose", hover, quat), world, TRANSIT_SPEED)
    except RuntimeError as e:
        return None, "the carry (%s)" % rp._short(e)
    if carry.traj is not None:
        carry.traj = approach.on_branch_of(carry.traj, q_clear)  # continues the clear, not a turn away
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
        lines.append((name, [[float(v) for v in p] for p in pts]))
        q = lines[-1][1][-1]
    was = ctx.lid_at
    ctx.lid_at = drop  # home is planned around the lid where it will lie
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
    speeds = {"place:down": cfg.setdown_speed, "place:touch": dp.TOUCH_SPEED, "place:retreat": FREE_LINE_SPEED}
    legs = [dp.compliant_line(ctx, name, pts, quat, speeds[name], world) for name, pts in lines]
    return ([carry, *legs[:2]], [legs[2], *home]), None


def plan_tail(ctx, cfg, q_grip, quat):
    """The lid's way from the knob held at joints `q_grip` (the hand at
    `quat`) to the table, and home after it (Tail): the lift straight up
    lift_off_m, the clear up and back (_clear_after_lift), then the carry,
    the set-down and home to ctx.lid_drop (_carry_to) — the other spots the
    cameras saw room at in turn where the planner refuses one. Run as the
    grip's lookahead, while the hand comes down to the knob. Returns (Tail,
    None) or (None, why)."""
    from rammp_box_opening.runtime import approach

    if ctx.lid_drop is None:
        return None, "no spot to set the lid down was found this run"
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
    first = ctx.lid_drop
    drops = [first] + [d for d in (ctx.lid_drops or [])
                       if math.hypot(d.xyz[0] - first.xyz[0], d.xyz[1] - first.xyz[1]) > 1e-3]
    passed = []
    for drop in drops:
        legs, why = _carry_to(ctx, cfg, chain, clear[-1], drop, world)
        if legs is None:
            passed.append("[%.2f, %.2f]: %s" % (drop.xyz[0], drop.xyz[1], why))
            continue
        head = [dp.compliant_line(ctx, "lid:lift", lift, quat, cfg.lift_speed, world),
                dp.compliant_line(ctx, "lid:clear", clear, quat, FREE_LINE_SPEED, world)]
        return Tail(head + legs[0], legs[1], drop, back, "; ".join(passed) or None), None
    return None, "no carry to a spot the lid could go to plans (%s)" % "; ".join(passed)


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


def run_full(node, ctx, cfg, runner, watcher, way, legs, button, yaw, lid_before):
    """The full reach-out run from the aimed look pose, its way in `way`
    proven and flown as `legs` (reach_press._reach_press's, one execution):
    the press and the way back (_press_and_back), the look (knob_look), the
    press once more where the knob is not up, then the lid off
    (take_lid_off) at the aimed `button` (its top the press's), the grip
    going by the lid where the last hold held the pads (grip_lid_z).
    Returns the process exit code: 0 only with the lid set down and the arm
    home. From the way in's start on, anything raised lets go (as
    _fly_way_in): a Ctrl+C in the hold's read lands with no goal in flight,
    the springs pushing on into the box."""
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
        return take_lid_off(ctx, cfg, runner, button, tilt, lid_z, "the lid at %.3f (the press's top %.3f%s)" % (
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


def take_lid_off(ctx, cfg, runner, button, tilt, lid_z, how):
    """From the look pose, the knob up: the lid off and onto ctx.lid_drop,
    then home. The grip's way down (solve_grip_way, from where the arm
    stands), the fingers opening as it flies, ONE execution, the lid's way
    to the table (plan_tail) planned while it does; the close, judged by the
    grip band; the lift, the clear, the carry and the set-down, ONE
    execution; the release; the retreat and home, ONE execution. `how`
    says what the grip's lid height `lid_z` is. Returns the exit code: 0
    with the lid set down where it was carried and the arm home."""
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
    res = runner.run([opened, *legs], execute=True, lookahead=lambda end: plan_tail(ctx, cfg, end, way.stroke.quat))
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
    tail, why = ahead if ahead is not None else (None, "the lookahead plan failed")
    if tail is None:
        print("[press_demo] the lid's way to the table was not planned on the way down (%s) — planning it here" % why)
        tail, why = plan_tail(ctx, cfg, ctx.client.joints(), way.stroke.quat)
    runner.note("reach_tail", planned=tail is not None, why=why if tail is None else tail.passed,
                drop=None if tail is None else [round(v, 4) for v in tail.drop.xyz],
                legs=None if tail is None else [lg.name for lg in tail.carry + tail.home])
    if tail is None:
        return _back_out(ctx, cfg, runner, way, "the lid would have nowhere to go: %s — not gripping it" % why,
                         opened=False)
    if tail.passed:
        print("[press_demo] lid spot(s) refused by the planner (%s) — setting the lid down at [%.2f, %.2f] instead"
              % (tail.passed, tail.drop.xyz[0], tail.drop.xyz[1]))
        ctx.lid_drop = tail.drop
    close = _gripper_leg(ctx, PlanState(joints=None, chain=0), "grip:close", GRIPPER_CMD_CLOSED, world,
                         verify=band_verify(cfg.grip_band))
    carry = [lg.name for lg in tail.carry]
    print("[press_demo] CLOSE, then ONE MOTION: %s — %.0f mm straight up, %.0f mm back toward the arm, to [%.2f, %.2f] "
          "and down onto the table" % (", ".join(carry), 1000 * lift_off_m(ctx.model), 1000 * tail.back_m,
                                       tail.drop.xyz[0], tail.drop.xyz[1]))
    res = runner.run([close, *tail.carry], execute=True)
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
    if len(moves) == 1 and last.flown_traj is not None:
        rp._note_flight(runner, "reach_carry_flight", [lg.name for lg in merge_groups(tail.carry)[0]], last)
    if not moves or not all(r.ok for r in moves):
        if len(moves) <= 1 and not _started(last):
            # nothing moved: the knob in the fingers, the lid still on the canister
            return _back_out(ctx, cfg, runner, way, "the lift did not start (%s) — the lid is let go where it lies, "
                                                   "on the box" % (last.detail or last.outcome))
        print("[press_demo] STOP: the lid's way to the table stopped during %s (%s) — the arm holds the lid" % (
            rp._in_flight(tail.carry, last, runner.last_retime), last.detail or last.outcome))
        rp.let_go(ctx)
        return 1
    ok_q, pos, _stalled = ctx.client.gripper_cmd(None)  # query only: still on the knob?
    kept = not ok_q or pos is None or in_band(pos, cfg.grip_band)
    runner.note("reach_setdown", knuckle=None if pos is None else round(float(pos), 3), kept=kept)
    if not kept:
        print("[press_demo] the fingers read %.3f at the table, outside the grip band %s: the lid slipped on the way"
              % (pos, list(cfg.grip_band)))
    release = _gripper_leg(ctx, PlanState(joints=None, chain=0), "place:open", GRIPPER_CMD_OPEN, world)
    ctx.lid_at = tail.drop  # every world from here holds the lid where it was set down
    print("[press_demo] RELEASE, then ONE MOTION: %s" % ", ".join(lg.name for lg in tail.home))
    res = runner.run([release, *tail.home], execute=True)
    if not res[0].ok:
        print("[press_demo] STOP: the fingers did not open at the table (%s) — the arm holds" % res[0].detail)
        rp.let_go(ctx)  # the spring pressing the lid onto the table
        return 1
    if not all(r.ok for r in res):
        last = res[-1]
        if _started(last):
            print("[press_demo] STOP: the way home from the lid stopped during %s (%s) — the arm holds" % (
                rp._in_flight(tail.home, last, runner.last_retime), last.detail or last.outcome))
            rp.let_go(ctx)
            return 1
        up, why = rp._line(ctx, tail.home[0].target[1], tail.home[0].target[2])
        if up:
            res = runner.run([dp.compliant_line(ctx, "place:retreat", up, tail.home[0].target[2], FREE_LINE_SPEED,
                                                world)], execute=True)
            if not all(r.ok for r in res):
                print("[press_demo] STOP: the way up from the lid stopped (%s) — the arm holds" % res[-1].detail)
                rp.let_go_part_way(ctx, res)
                return 1
        rp._go_home(ctx, cfg, runner, "the way home from the lid did not start (%s)" % (last.detail or last.outcome))
        return 1
    runner.finish()
    if not kept:
        print("[press_demo] STOP: the arm is home, but the lid slipped out of the fingers on the way to [%.2f, %.2f]"
              % (tail.drop.xyz[0], tail.drop.xyz[1]))
        return 1
    _note_frames(ctx, stage="done", popped=True)
    print("[press_demo] DONE — box open, lid set down at [%.2f, %.2f], arm home" % (tail.drop.xyz[0], tail.drop.xyz[1]))
    return 0
