"""The reach-out press: for a box the hand cannot stand straight down over.

Flatbot, 2026-10-22: a table 0.30 m above the arm's mount and the box
0.80-0.89 m out — no straight-down pose reaches the button there at any
wrist yaw (planner probe, 12 yaws x 3 heights). It is pressed with the hand
tilted back toward the arm, ONE FINGERTIP coming STRAIGHT DOWN onto the
button, and only when the arm's own kinematics proves straight down out of
reach (press_demo.staging_out_of_reach):

  LOOK    the hand pulled back toward the arm (REACH_PULLBACK_M) and up
          (REACH_LOOK_RISE_M) clear of the box's padded cuboid, already at
          the press attitude. The wrist
          camera looks down at the lid from there and aims (the oblique aim,
          perception/button_aim). The box is put where the aim found it and
          the hand moves to THAT button's look pose along a straight line,
          guarded in a guarded run — the scene camera's box can be 8-10 cm
          off.
  SLIDE   a plan-free straight line in and down, holding the attitude, to
          REACH_START_M above the aimed button. Guarded at DRAG_NM in a
          guarded run.
  TOUCH   a plan-free straight line DOWN in world z to travel_m below it,
          guarded at contact_nm like the straight-down touch: the force is
          vertical, nothing shoves the box sideways.
  PUSH    reach_push_m further down from the contact — no deeper than the
          gripper clears the lid at this tilt — backstopped, held, then the
          runner's recoil reverses it.
  UNLOAD  straight back up the stroke, judged for a PULL at the fingertip
          (HANG_N) against the free-air reading taken there before the
          touch: leaving the button reads as a push let go, never as a pull.
          It stops UNLOAD_CHECK_M up and checks at rest that nothing came
          with the finger — a latched box rises that far at most — then
          rises to the stroke's start, above the highest popped knob.
  CHECK   at rest at the stroke's start, against the same pose read before
          the touch: anything on the hand and the arm holds.
  SLIDE   back out to the look pose, guarded at DRAG_NM and HANG_N against
          that same free-air reading. No guard here can feel a box slid
          along the table (a 0.3-0.5 kg canister slides at 1-2.5 N, and the
          recorded residual reads as 2.4-2.6 N sideways): what keeps the
          box on the table is the hand starting this leg 35 mm or more above
          anything on the lid, with nothing on it.
  AFTER   the wrist re-aims from the look pose and the scene camera looks
          again: did the knob pop, did the box move? Judged BEFORE the arm
          goes anywhere; neither camera finding the box holds the arm.

Every guard here judges what the arm's own weight does not explain
(runtime/guards.CompensatedGuard with kinematics.GravityModel): along a
15 cm reach at 0.8-0.9 m joint_4's gravity torque moves 1.1-1.9 Nm — as
much as a canister hanging off a finger — so a raw guard either tripped on
the reach itself or missed the canister (review 2026-10-22). The model is
checked against the arm before any guarded line flies (gravity_mismatch).

The tilt is the smallest of REACH_TILTS_DEG that plans. It stops at 45 deg
because of the gripper's shape: past that its knuckles, then its palm, come
down onto the flush lid before the fingertip reaches the button (the URDF's
collision meshes with the measured 131 mm fingers: 8 mm of lid clearance at
30 deg, 5.6 at 45, under 3 past 60). The press ends the run: the grip, the
lift and the set-down are straight-down primitives that reach no further.
"""

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from rammp_box_opening.constants import (
    COMPLIANT_RUN_GAINS,
    CONTACT_SPEED,
    FREE_LINE_SPEED,
    IMPEDANCE_KQ,
    IMPEDANCE_NAMES,
    TIP_TO_TOOL_M,
    TRANSIT_SPEED,
)
from rammp_box_opening.geometry import rotate_about_world_axis
from rammp_box_opening.models.container import ContainerPose, attitude_quat
from rammp_box_opening.primitives.core import (
    CLEAR_MARGIN_MAX_STEP_RAD,
    CONTACT_SHIFT_PAD_M,
    ClearMarginFree,
    PlanState,
    _full_world,
    _interaction_world,
    _plan_motion,
    _push_guard_and_verify,
)
from rammp_box_opening.runtime.guards import GuardSpec
from rammp_box_opening.runtime.legs import Kind, Leg, can_merge, merge_groups

REACH_TILTS_DEG = (30, 35, 40, 45)
# How far the rest of the gripper stands above the flush lid when the
# fingertip touches the button's centre, by tilt (URDF collision meshes with
# the measured 131 mm fingers; 30 and 45 computed, 35 and 40 interpolated).
# The push must never go deeper than this, or the knuckles press the lid.
# Off centre across the bearing it is less — 3.2 mm at 45 deg 4 mm off (the
# other fingertip's corner): such a push meets the lid, trips its backstop
# and the knob does not pop, which the run reports as a failure (exit 1).
REACH_LID_CLEAR_M = {30: 0.0082, 35: 0.0073, 40: 0.0064, 45: 0.0056}
PUSH_MARGIN_M = 0.0015
# CONTACT_TOOL and REACH_LID_CLEAR_M were measured for this attitude (the
# fingers side by side across the line to the box): another roll puts a
# different part of the hand lowest
REACH_ATTITUDE_RPY_DEG = (180.0, 0.0, 90.0)


def reach_push_m(cfg, tilt_deg):
    """The push's depth at this tilt: the configured bound, or less, so the
    gripper stays PUSH_MARGIN_M clear of the lid."""
    return min(float(cfg.button_travel_m), REACH_LID_CLEAR_M[int(tilt_deg)] - PUSH_MARGIN_M)


# the stroke starts this far above the button (a straight-down press's final
# approach) — and the unload ends there: above a popped knob's 10-25 mm
REACH_START_M = 0.06
# Before planning home from over the box, the tool this far above the box's
# padded cuboid (worlds.planned_top_z). The planner's lowest collision
# spheres belong to the gripper's base link, not the fingers (rammp_curobo
# configs/robot_gen3_2f85.yaml: arm_robotiq_85_base_link centres
# (+-0.045, 0, 0.145) and (0, +-0.045, 0.145), r 0.018 + its
# collision_sphere_buffer 0.005), 25 mm past tool_frame and
# HAND_SPHERE_OFF_AXIS_M off its axis. Their lowest point is
# 0.025 cos(lean) + 0.045 sin(lean) + 0.023 below tool_frame: 48 mm with the
# hand straight down, which this leaves 12 mm clear; 67 mm at 30 deg, which
# it leaves 7 mm INSIDE the padded cuboid (2026-10-27 16:23: home refused
# INVALID_START_STATE_WORLD_COLLISION after the clear, then a 15 cm
# back-off). The four are symmetric, so the sin term bounds every lean.
HOME_CLEAR_M = 0.06
# A compliant run's clear (clear_targets) stands that, plus what the hand's
# lean hangs those spheres lower (HAND_SPHERE_OFF_AXIS_M: the robot's body,
# not the scene's), plus CLEAR_EXTRA_M over the padded cuboid, and pulls
# back toward the arm CLEAR_BACK_M as it rises, else REACH_PULLBACK_M.
# Straight up, a hand turned down drives joint_6 into its limit (15:19:
# 0.104 rad, a 1.44 s clear-margin step; 14:33: no line up flew, home was
# refused START_IN_LIMIT_MARGIN and the arm held over the box).
HAND_SPHERE_OFF_AXIS_M = 0.045
CLEAR_EXTRA_M = 0.005
CLEAR_BACK_M = 0.05
# The look pose, pulled back toward the arm from the stroke's start: 15-20 cm
# plans in the full world at both spots, 10 cm does not — the gripper meets
# the padded container cuboid (planner probe 2026-10-22).
REACH_PULLBACK_M = 0.15
# ... and this far above the stroke's start. Level with it the wrist camera
# stood ~0.245 m from the lid (flatbot 2026-10-27 14:04, the scene camera
# placed by its calibration) and the glossy lid and button held depth in
# patches; the looks that had depth on the lid stood ~0.30 m off it (12:58,
# when the scene camera put the lid 9 cm high). 6.5 cm up is ~0.30 m, and
# the slide in, from above the old one, comes no nearer the box.
REACH_LOOK_RISE_M = 0.065
# The closed 2F-85's lowest point at these tilts, in tool_frame, with the
# reach attitude's roll (fingers side by side across the line to the box):
# one fingertip's end, 11 mm past tool_frame like TCP_OFFSET_M straight down.
CONTACT_TOOL = (0.0, -0.009, 0.011)
TIP_MID_TOOL = (0.0, 0.0, -TIP_TO_TOOL_M)  # the finger-tip links' midpoint (client.contact_xyz)
# Every guard below judges the gravity-compensated residual. On flatbot's
# recording (kinematics.GravityModel) it stayed under 0.76 Nm on every wrist
# joint through 22 s of motion; the reach lines' own model error is ~0.15.
DRAG_NM = 1.5
# A pull straight down at the fingertip, read off the residual (guards
# CompensatedGuard `hang`): a 0.3-0.5 kg canister (estimated — weigh it) is
# 2.9-4.9 N; the recording's worst residual reads as 1.6-1.7 N moving, and
# 0.5-0.6 N between two readings at rest
HANG_N = 2.5
HANG_STATIC_N = 1.0
STATIC_NM = 0.75  # ... and at rest at the stroke's start, any wrist joint this far off its free-air reading
UNLOAD_CHECK_M = 0.015  # the first stop on the way up: past the button's travel
# The model against the arm, between the rest pose and the look pose: off by
# more than this on any wrist joint (and a sign error is off by twice the
# change) and no guard here can be trusted
GRAVITY_CHECK_NM = 1.0
GRAVITY_CHECK_FRAC = 0.3
LINE_MIN_M = 0.001  # a line shorter than this is not flown: the arm is there
BOX_MOVED_M = 0.005  # the re-aimed button this far from the first aim: the press moved the box
BOX_LIFTED_M = 0.008  # ... or its lid this much higher or lower
SCENE_MOVED_M = 0.02  # the scene camera's box this far from where it was before the press


def reach_quat(model, xy, tilt_deg):
    """The press attitude turned to face `xy` — the wrist camera rides above
    the hand, the fingers side by side across the line to the box — with its
    down axis tilted `tilt_deg` away from the arm: the hand leans back."""
    d = math.hypot(xy[0], xy[1])
    q = attitude_quat(model.press_attitude_rpy_deg, math.atan2(xy[1], xy[0]))
    return list(rotate_about_world_axis(q, (xy[1] / d, -xy[0] / d, 0.0), math.radians(tilt_deg)))


def _rot(quat):
    from rammp_box_opening.perception.d405 import quat_to_mat

    return np.asarray(quat_to_mat(*quat), float)


def tool_for_contact(contact_xyz, quat):
    """The tool_frame position that puts the fingertip's contact point at
    `contact_xyz` with the hand at `quat`."""
    return [float(v) for v in np.asarray(contact_xyz, float) - _rot(quat) @ np.asarray(CONTACT_TOOL)]


def contact_from_tips(tip_mid_xyz, quat):
    """Where the fingertip's contact point is, from the finger-tip links'
    midpoint the arm measures at a trip (client.contact_xyz)."""
    v = _rot(quat) @ (np.asarray(CONTACT_TOOL) - np.asarray(TIP_MID_TOOL))
    return [float(c) for c in np.asarray(tip_mid_xyz, float) + v]


def cpose_for_button(button, yaw, offset):
    """The container pose whose button (from_container with `offset`) is at
    `button`: every world from the aim on holds the box there."""
    c, s = math.cos(yaw), math.sin(yaw)
    ox, oy, oz = (float(v) for v in offset)
    return ContainerPose(
        xyz=(float(button[0]) - (c * ox - s * oy), float(button[1]) - (s * ox + c * oy), float(button[2]) - oz), yaw=yaw
    )


def unsupported(model):
    """Why this container cannot be pressed reaching out, or None."""
    rpy = tuple(float(v) for v in model.press_attitude_rpy_deg)
    if any(abs(a - b) > 1e-6 for a, b in zip(rpy, REACH_ATTITUDE_RPY_DEG)):
        return (
            "its press attitude is %s; the reach-out's fingertip contact and lid clearance were measured for %s"
            % (list(rpy), list(REACH_ATTITUDE_RPY_DEG))
        )
    return None


@dataclass
class ReachPlan:
    tilt_deg: int
    quat: list
    button: tuple  # the button top it is aimed at (x, y, z)
    look_xyz: list
    start_xyz: list
    bottom_xyz: list
    look_leg: object  # planned from the joints asked, full world
    slide_pts: list  # look -> start, the attitude held
    stroke_pts: list  # start -> bottom, straight down; None for a depth plan, which never flies it
    # a depth plan's (plan_reach depth=True, solve_way_in): how the hand turns
    # down from the look pose (depth_press.WayDown) and the compliant press
    # stroke from there (depth_press.StrokePlan)
    down: object = None
    stroke: object = None


def _short(e):
    s = str(e)
    return s.split(": ", 1)[-1][:90]


def reach_geometry(model, button, tilt, travel_m):
    """(quat, start, bottom, look) for pressing `button` at `tilt`: the
    hand's attitude, the stroke's start REACH_START_M above the button and
    its bottom travel_m below it (tool_frame, for the fingertip's contact
    point), and the look pose REACH_PULLBACK_M back toward the arm from the
    start and REACH_LOOK_RISE_M above it."""
    bx, by, bz = (float(v) for v in button)
    d = math.hypot(bx, by)
    quat = reach_quat(model, (bx, by), tilt)
    start = tool_for_contact((bx, by, bz + REACH_START_M), quat)
    bottom = tool_for_contact((bx, by, bz - travel_m), quat)
    look = [start[0] - REACH_PULLBACK_M * bx / d, start[1] - REACH_PULLBACK_M * by / d, start[2] + REACH_LOOK_RISE_M]
    return quat, start, bottom, look


def way_down(ctx, cfg, q_look, slide_pts, start_xyz, quat, top_z):
    """How the compliant depth press gets from the look pose's joints
    `q_look` to pressing the button whose top is at `top_z`, solved by the
    arm's own kinematics before anything flies: one walk in from the look
    pose to the hand turned straight down over the slide's end `start_xyz`
    (depth_press.walk_in_down; WayDown walked, its end the stroke's start),
    else the slide `slide_pts` held at `quat`, then the TURN about the
    fingertip (CONTACT_TOOL) where it ends (depth_press.turn_down), up to
    15 deg short of straight down as the compliant press turns
    (depth_press.COMPLIANT_TURN_SHORT_DEG: way_down serves only that
    press); then the press stroke from the hand turned down
    (depth_press.compliant_stroke). The aimed button's walk or turn and
    stroke fly as solved, in one execution with the move and the slide
    before them (way_in_legs); only where those legs do not make one
    (_not_one_execution) is each solved again from where the arm stands,
    trying the same leans. Returns (WayDown, StrokePlan, None) or (None,
    None, why)."""
    from rammp_box_opening.runtime import approach
    from rammp_box_opening.tasks import depth_press as dp  # imports this module: not at the top

    chain = approach._chain()
    pts, down, short, why_walk = dp.walk_down_path(chain, ctx.model, q_look, start_xyz, quat)
    if pts is not None:
        way = dp.WayDown(pts, down, short, True)
    else:
        pts, down, short, why_turn = dp.turn_down_path(chain, ctx.model, slide_pts[-1], CONTACT_TOOL,
                                                       dp.COMPLIANT_TURN_SHORT_DEG)
        if pts is None:
            return None, None, "walk in: %s; turn after the slide: %s" % (why_walk, why_turn)
        way = dp.WayDown(pts, down, short, False)
    stroke, why = dp.compliant_stroke(chain, cfg, way.pts[-1] if way.pts else slide_pts[-1], way.quat, top_z)
    if stroke is None:
        return None, None, why
    return way, stroke, None


def plan_reach(ctx, cfg, button, q_from, tilts=REACH_TILTS_DEG, depth=False, looks=None):
    """The smallest tilt whose look pose plans (full world, from `q_from`)
    and whose slide and stroke the arm's own kinematics can fly as straight
    lines. The way back out retraces both. `depth` (the compliant depth
    press): no tilted stroke — that press never flies it — but the hand
    must turn down from the look pose over the slide and stroke from there
    (way_down); a tilt whose look pose and slide solve but whose way down
    does not is passed over, and appended to `looks` (a list) with no
    `down`. Returns (ReachPlan, None) or (None, why)."""
    from rammp_box_opening.runtime import approach

    chain = approach._chain()
    bx, by, bz = (float(v) for v in button)
    refused = []
    for tilt in tilts:
        quat, start, bottom, look = reach_geometry(ctx.model, button, tilt, cfg.travel_m)
        try:
            leg, _st = _plan_motion(
                ctx, PlanState(joints=list(q_from), chain=0), "reach:look", ("pose", look, quat), _full_world(ctx), TRANSIT_SPEED
            )
        except RuntimeError as e:
            refused.append("%d deg: the look pose (%s)" % (tilt, _short(e)))
            continue
        slide, why = chain.straight_line(leg.goal_joints, start, quat)
        if slide is None:
            refused.append("%d deg: the slide in (%s)" % (tilt, why))
            continue
        if depth:
            down, stroke, why = way_down(ctx, cfg, leg.goal_joints, slide, start, quat, bz)
            if down is None:
                refused.append("%d deg: the way down (%s)" % (tilt, why))
                if looks is not None:
                    looks.append(ReachPlan(tilt, quat, (bx, by, bz), look, start, bottom, leg, slide, None))
                continue
            return ReachPlan(tilt, quat, (bx, by, bz), look, start, bottom, leg, slide, None, down, stroke), None
        stroke, why = chain.straight_line(slide[-1], bottom, quat)
        if stroke is None:
            refused.append("%d deg: the stroke (%s)" % (tilt, why))
            continue
        return ReachPlan(tilt, quat, (bx, by, bz), look, start, bottom, leg, slide, stroke), None
    return None, "; ".join(refused)


# Where the compliant depth press does not reach the scene camera's button,
# the look still flies when it reaches the button this far off it — toward
# the arm, or to either side (_shifted_buttons): the aim, not the scene
# camera, puts the button the press goes to. On 2026-10-27 the four aims
# put the button 12-23 mm from the scene camera's (14:33, 15:19, 15:34,
# 16:23), each 11-22 mm to its right and up to 11 mm nearer the arm.
# Refusing on the scene button alone at the edge of reach refuses boxes the
# aim brings into reach; two such were found offline (flatbot's URDF): the
# scene's button at az -22.0 deg, r 0.700 m with the aim's at -23.6, 0.692,
# and at r 0.766, az -15 with the aim's at 0.757, -16.6.
PROOF_SHIFT_M = 0.03
SHIFTED = ("nearer the arm", "to its left", "to its right")  # _shifted_buttons' order, seen from the arm


def _shifted_buttons(button, m):
    """Three buttons at `button`'s height: `m` toward the arm along its
    bearing, and `m` to either side across it (SHIFTED's order)."""
    bx, by, bz = (float(v) for v in button)
    d = math.hypot(bx, by)
    ux, uy = bx / d, by / d
    return [(bx - m * ux, by - m * uy, bz), (bx - m * uy, by + m * ux, bz), (bx + m * uy, by - m * ux, bz)]


def shifted_way_down(ctx, cfg, look, button, m=PROOF_SHIFT_M):
    """For a scene button no tilt presses (plan_reach depth=True): whether
    the press reaches one of _shifted_buttons from `look` — the smallest
    tilt whose look pose and slide solved (plan_reach's `looks`) — the slide
    from its look joints straight to that button's start at its tilt, then
    way_down. Returns (SHIFTED's name, None) for the first that does, or
    (None, why)."""
    from rammp_box_opening.runtime import approach

    chain = approach._chain()
    q = look.look_leg.goal_joints
    whys = []
    for name, b in zip(SHIFTED, _shifted_buttons(button, m)):
        quat, start, _bottom, _look = reach_geometry(ctx.model, b, look.tilt_deg, cfg.travel_m)
        slide, why = chain.straight_line(q, start, quat)
        if slide is None:
            whys.append("%.0f mm %s: the slide in (%s)" % (1000 * m, name, why))
            continue
        down, _stroke, why = way_down(ctx, cfg, q, slide, start, quat, b[2])
        if down is None:
            whys.append("%.0f mm %s: the way down (%s)" % (1000 * m, name, why))
            continue
        return name, None
    return None, "; ".join(whys)


@dataclass
class WayIn:
    """The compliant depth press's way in for the aimed button, from where
    the arm stands (solve_way_in): the MOVE to its look pose ([] when there),
    reach_from_here's ReachPlan, how the hand turns down (WayDown), the
    slide before the turn (None when the way down is walked) and the press
    stroke (StrokePlan)."""

    move: list
    plan: ReachPlan
    down: object
    slide: list
    stroke: object


def solve_way_in(ctx, cfg, button, tilt, q_from):
    """The aimed button's whole way in at `tilt` from joints `q_from`, before
    the arm moves toward it: reach_from_here's lines (the move to the look
    pose, the slide), then way_down from the look pose. Returns (WayIn, None,
    None) or (None, why, kind): kind 'lines' when the lines do not fly (the
    planner may plan the move instead), 'down' when the way down does not."""
    move, plan, why = reach_from_here(ctx, cfg, button, tilt, need_stroke=False, q=q_from)
    if plan is None:
        return None, why, "lines"
    down, stroke, why = way_down(ctx, cfg, move[-1] if move else q_from, plan.slide_pts, plan.start_xyz, plan.quat,
                                 float(button[2]))
    if down is None:
        return None, "the way down (%s)" % why, "down"
    plan.down, plan.stroke = down, stroke
    return WayIn(move, plan, down, None if down.walked else plan.slide_pts, stroke), None, None


def reach_from_here(ctx, cfg, button, tilt, need_stroke=True, q=None):
    """The reach for `button` at `tilt` as plan-free lines from where the arm
    stands (`q`, else the arm's joints): a MOVE to its look pose at its
    attitude, then the slide and the stroke (not solved without
    `need_stroke`: the depth press flies its own). Returns (move points —
    [] when already there —, ReachPlan with no look_leg, None) or (None,
    None, why)."""
    from rammp_box_opening.runtime import approach

    chain = approach._chain()
    quat, start, bottom, look = reach_geometry(ctx.model, button, tilt, cfg.travel_m)
    q = ctx.client.joints() if q is None else [float(v) for v in q]
    _R, t = chain.fk(q)
    if float(np.linalg.norm(np.asarray(look, float) - np.asarray(t, float))) < LINE_MIN_M:
        move, q_look = [], q
    else:
        move, why = chain.straight_line(q, look, quat)
        if move is None:
            return None, None, "the move to its look pose (%s)" % why
        q_look = move[-1]
    slide, why = chain.straight_line(q_look, start, quat)
    if slide is None:
        return None, None, "the slide in (%s)" % why
    stroke = None
    if need_stroke:
        stroke, why = chain.straight_line(slide[-1], bottom, quat)
        if stroke is None:
            return None, None, "the stroke (%s)" % why
    return move, ReachPlan(int(tilt), quat, tuple(float(v) for v in button), look, start, bottom, None, slide, stroke), None


def line_leg(ctx, name, pts, quat, speed, guard, world, verify=None, chain=0):
    """A plan-free straight-line leg through joint waypoints `pts` (the arm's
    own kinematics, runtime/kinematics.straight_line). The runner never
    re-plans it (Leg.plan_free)."""
    from rammp_box_opening.runtime import approach

    _R, t = approach._chain().fk(pts[-1])
    world_name, world_path = world
    return Leg(
        name=name,
        kind=Kind.MOTION,
        traj=approach.line_trajectory(ctx.client.joint_names, pts),
        speed=float(speed),
        guard=guard,
        world=world_name,
        world_path=str(world_path),
        chain=chain,
        target=("pose", [float(v) for v in t], list(quat)),
        goal_joints=[float(v) for v in pts[-1]],
        verify=verify,
        plan_s=0.0,
        plan_free=True,
    )


def way_in_legs(ctx, cfg, way):
    """The compliant depth press's way in (`way`, solve_way_in's) as the
    legs of ONE execution, in this order, the empty ones left out:
    'reach:re-aim' (way.move, at the plan's attitude); 'reach:slide-in', the
    walk turning down when the way down is walked, else the slide at the
    plan's attitude, then 'depth:turn' in place; then the press stroke
    (way.stroke.legs: 'depth:approach' at press_speed, 'depth:press' at
    TOUCH_SPEED). The free-air lines fly at FREE_LINE_SPEED. Each starts at
    the last waypoint of the one before — solve_way_in chains them — and
    every one is unguarded under depth_press.COMPLIANT_GAINS
    (compliant_line): the one gains spec legs.can_merge asks of one
    execution, and the spring the press is modelled on. `cfg` is the press
    config the stroke was solved with (its press_speed is in way.stroke)."""
    from rammp_box_opening.tasks import depth_press as dp

    # Their world is a label: a plan-free line is never planned, re-planned
    # or collision-checked against it (line_leg builds it, Runner._replan_group
    # refuses plan-free legs, push_name only writes a YAML). "full" only
    # satisfies Runner._refusal's spec §6 rule that an unguarded leg faster
    # than CONTACT_SPEED be labelled full or bench.
    world = _full_world(ctx)
    lines = [("reach:re-aim", way.move, way.plan.quat, FREE_LINE_SPEED)]
    if way.down.walked:
        lines.append(("reach:slide-in", way.down.pts, way.down.quat, FREE_LINE_SPEED))
    else:
        lines += [("reach:slide-in", way.slide, way.plan.quat, FREE_LINE_SPEED),
                  ("depth:turn", way.down.pts, way.down.quat, FREE_LINE_SPEED)]
    lines += [(name, pts, way.stroke.quat, speed) for name, pts, speed in way.stroke.legs]
    return [dp.compliant_line(ctx, name, pts, quat, speed, world) for name, pts, quat, speed in lines
            if pts is not None and len(pts) > 0]


def _wrist_gravity():
    from rammp_box_opening.runtime import approach

    return approach._gravity().wrist_holding


def tip_load_at(q):
    """The wrist joints' holding torque per newton pulled straight down at
    the fingertip's contact point, with the arm at `q`."""
    from rammp_box_opening.runtime import approach

    return [float(v) for v in approach._gravity().tip_load(q, CONTACT_TOOL)]


def touch_leg(ctx, cfg, pts, quat, world, reference):
    """The TOUCH as a straight line down: stops at first contact
    (cfg.contact_nm on the compensated residual, against the free-air
    `reference` read at the stroke's start). Every trip is a surface found
    — how far above the button is touch_verdict's to judge from where the
    fingertip stopped; no trip found no surface."""
    guard = GuardSpec(touch_nm=cfg.contact_nm, trip="touch", gravity=_wrist_gravity(), reference=reference)

    def verify(v):
        if v.outcome == "touch":
            peak = "" if v.torque_peak is None else " at %.1f Nm" % v.torque_peak
            return True, "surface found%s" % peak
        if v.outcome == "arrived":
            return False, "no surface within %.0f mm below the aimed button" % (cfg.travel_m * 1000)
        return False, "touch %s" % v.outcome

    return line_leg(ctx, "press:down", pts, quat, cfg.press_speed, guard, world, verify)


def drag_guard(reference, hang_k=None):
    """Free air, nothing to meet: DRAG_NM of compensated residual on any
    wrist joint is a strike, and (with `hang_k`) HANG_N of pull at the
    fingertip is the box coming along."""
    return GuardSpec(
        touch_nm=DRAG_NM, trip="obstruction", gravity=_wrist_gravity(), reference=reference,
        hang=None if hang_k is None else (list(hang_k), HANG_N),
    )


def unload_guard(model, reference, hang_k):
    """Leaving the button: the leg starts loaded (the touch's contact), so
    only the push's own backstop bounds the residual — but any PULL at the
    fingertip past HANG_N, against the free-air `reference`, stops it."""
    return GuardSpec(
        touch_nm=float(model.touch_nm), trip="obstruction", gravity=_wrist_gravity(), reference=reference,
        hang=(list(hang_k), HANG_N),
    )


def compliant(ctx):
    """A compliant run (PlannerClient.compliant): impedance, nothing stops it."""
    return bool(getattr(ctx.client, "compliant", False))


def gains_text(profile):
    """A driver gains profile as the console names it, e.g. "STIFF (Kq 160/60
    N·m/rad on joints 1-4/5-7)"."""
    p = int(profile)
    kq = IMPEDANCE_KQ.get(p)
    name = IMPEDANCE_NAMES.get(p, "gains profile %d" % p)
    return name if kq is None else "%s (Kq %g/%g N·m/rad on joints 1-4/5-7)" % (name, kq[0], kq[4])


def free_guard(ctx, reference, hang_k=None):
    """drag_guard for a free-air leg — none in a compliant run, which stops
    for nothing it reads."""
    return None if compliant(ctx) else drag_guard(reference, hang_k)


def residual(gravity, ref, now):
    """What gravity does not explain between two (efforts, joints) readings."""
    return [
        (float(e1) - float(e0)) - (float(h1) - float(h0))
        for e0, e1, h0, h1 in zip(ref[0], now[0], gravity(ref[1]), gravity(now[1]))
    ]


def down_force(k, r):
    """The residual `r` read as a pull straight down at the fingertip (N)."""
    kk = sum(float(v) * float(v) for v in k)
    return 0.0 if kk <= 0.0 else sum(float(a) * float(b) for a, b in zip(k, r)) / kk


def gravity_mismatch(gravity, a, b):
    """Why the gravity model does not describe this arm, or None: the wrist
    joints' measured change between two rest readings `a` -> `b` against the
    model's. A sign error is off by twice the change, a wrong URDF or a
    payload by its own size."""
    r = residual(gravity, a, b)
    dh = [float(h1) - float(h0) for h0, h1 in zip(gravity(a[1]), gravity(b[1]))]
    for j, (rj, dj) in enumerate(zip(r, dh)):
        if abs(rj) > GRAVITY_CHECK_NM + GRAVITY_CHECK_FRAC * abs(dj):
            return "joint_%d's effort changed %+.2f Nm where the arm's gravity model says %+.2f" % (j + 4, dj + rj, dj)
    return None


def judge_box_moved(first_xy, again_xy):
    """The press moved the box by this much (m), from two aims of its button."""
    return math.hypot(float(again_xy[0]) - float(first_xy[0]), float(again_xy[1]) - float(first_xy[1]))


def oblique_aim_available():
    """Whether this install's wrist aim takes the reach-out's priors (the
    oblique aim, perception/button_aim): without it the look pose is flown
    for nothing."""
    import inspect

    from rammp_box_opening.perception.button_aim import ButtonAimer

    params = inspect.signature(ButtonAimer).parameters
    return "lid_z" in params and "prior_xy" in params


def _aim(node, watcher, yaw, lid_z, prior_xy, runner, tag):
    from rammp_box_opening.tasks.press_demo import aim_button_at_staging

    seen = {}
    got, status = aim_button_at_staging(node, watcher, yaw, runner=runner, tag=tag, out=seen, lid_z=lid_z, prior_xy=prior_xy)
    return got, status, seen.get("button_up_mm")


def _lid_z(button_z, up_mm):
    return None if up_mm is None else float(button_z) - float(up_mm) / 1000.0


def _stopped_by(res):
    """What stopped a guarded free-air leg, for its STOP line: a trip is
    something met; anything else says itself (15:34 2026-10-27: "met
    something" printed for a guard that only lost its stream)."""
    last = res[-1]
    if last.outcome == "touch":
        return "met something"
    return "stopped (%s)" % (last.detail or last.outcome)


def _settled(ctx, where):
    got = ctx.client.settled_efforts()
    if got is None:
        print("[press_demo] STOP: no steady joint efforts at %s — the reach-out's guards cannot be referenced" % where)
    return got


def _line(ctx, xyz, quat):
    """Joint waypoints for a straight line from where the arm is, or
    ([], None) when it is already there."""
    from rammp_box_opening.runtime import approach

    chain = approach._chain()
    q = ctx.client.joints()
    _R, t = chain.fk(q)
    if float(np.linalg.norm(np.asarray(xyz, float) - np.asarray(t, float))) < LINE_MIN_M:
        return [], None
    return chain.straight_line(q, xyz, quat)


REUSE_RAD = 1e-3  # the arm this close (every joint) to where a solved line starts: it is flown as solved


def _line_from(ctx, pts, xyz, quat):
    """`pts` (a line solved earlier, ending at `xyz`) when the arm stands at
    its start, else _line from where it stands. The runner refuses a
    plan-free line that no longer starts where the arm is all the same."""
    if pts and float(np.abs(np.asarray(ctx.client.joints(), float) - np.asarray(pts[0], float)).max()) < REUSE_RAD:
        return pts, None
    return _line(ctx, xyz, quat)


def _home_legs(ctx, cfg, start=None):
    """Home to the rest pose from where the arm stands — or from joints
    `start`, where it will stand (plan_way_out: the clear's end, planned
    while the way in flies) — in the full world.
    A joint too near its limit for the planner to start from gets home_arm's
    plan-free step along the joint line toward the rest pose first, the
    shortest one home plans from (primitives.ClearMarginFree): flatbot
    2026-10-27 14:33, the depth press ended with arm_joint_6 0.043 rad from
    its limit, home was refused START_IN_LIMIT_MARGIN, and the arm held over
    the box until home_arm stepped it clear. Raises the last refusal."""
    from rammp_box_opening.tasks.press_demo import rest_joints

    rest = list(rest_joints(cfg))
    q = ctx.client.joints() if start is None else start
    state = PlanState(joints=[float(v) for v in q], chain=0)
    clear = ClearMarginFree(rest)
    near = clear.near(ctx, state.joints)
    if not near:
        leg, _ = _plan_motion(ctx, state, "home", ("joints", rest), _full_world(ctx), TRANSIT_SPEED)
        return [leg]
    print("[press_demo] %s: too near for the planner to start from — trying plan-free steps along the joint line "
          "toward the rest pose, shortest first" % ", ".join("%s is %.3f rad from its limit" % jr for jr in near))
    last = None
    for step, at, info in clear.steps(ctx, state):
        try:
            leg, _ = _plan_motion(ctx, at, "home", ("joints", rest), _full_world(ctx), TRANSIT_SPEED)
        except RuntimeError as e:
            last = e
            continue
        if step:
            d = info["tool_move"]
            print("[press_demo] a plan-free step %.0f%% of the way to the rest pose first (the largest joint %.3f rad, "
                  "the tool %+.0f mm up and %.0f mm across; obstruction-guarded)" % (
                      100 * info["fraction"], info["largest_rad"], 1000 * d[2], 1000 * math.hypot(d[0], d[1])))
        return step + [leg]
    raise RuntimeError("no plan-free step toward the rest pose of up to %.2f rad per joint lets the planner plan home "
                       "(last: %s) — jog the arm clear by hand" % (CLEAR_MARGIN_MAX_STEP_RAD, last))


def _back_off(ctx, runner, ref=None, why=None):
    """Straight back toward the arm by REACH_PULLBACK_M, level, at the
    hand's own attitude — guarded, like the slide out: out of the
    padded cuboid of a box that is not where the arm's pose was planned
    around, so the planner takes the start again. Guarded against `ref`, a
    free-air reading the caller already holds, else one read here. `why`:
    the planner's refusal of home from here, printed as it said it — not
    always the box's padded volume (INVALID_START_STATE_JOINT_LIMITS takes
    the back-off too). False when it cannot be flown or met something."""
    from rammp_box_opening.runtime import approach
    from rammp_box_opening.runtime.kinematics import mat_to_quat_xyzw

    chain = approach._chain()
    q = ctx.client.joints()
    R, t = chain.fk(q)
    bx, by = float(ctx.cpose.xyz[0]), float(ctx.cpose.xyz[1])
    d = math.hypot(bx, by)
    target = [float(t[0]) - REACH_PULLBACK_M * bx / d, float(t[1]) - REACH_PULLBACK_M * by / d, float(t[2])]
    quat = mat_to_quat_xyzw(R)
    pts, why_line = chain.straight_line(q, target, quat)
    if pts is None:
        print("[press_demo] the way back off toward the arm is not a straight line the arm can fly (%s)" % why_line)
        return False
    if not compliant(ctx):
        ref = ref or _settled(ctx, "where the arm stands")
        if ref is None:
            return False
    said = ("the planner sees the arm inside the box's padded volume" if why is None
            else "the planner refused home from here (%s)" % _short(why))
    print("[press_demo] %s — backing %.0f cm straight off toward the arm first" % (said, 100 * REACH_PULLBACK_M))
    # from free air only (_go_home's callers), level and away from the box
    res = runner.run([line_leg(ctx, "reach:back-off", pts, quat, FREE_LINE_SPEED, free_guard(ctx, ref),
                               _full_world(ctx))], execute=True)
    return all(r.ok for r in res)


def _go_home(ctx, cfg, runner, why, ref=None):
    """Home from free air (the look pose, or where the run started), planned
    in the full world around the box where it now is. A start the planner
    refuses INVALID_START — inside the box's padded cuboid (the box not
    where the pose was planned around: the scene camera 85 mm off,
    2026-10-22), or INVALID_START_STATE_JOINT_LIMITS, which the same test
    takes — backs straight off toward the arm first (guarded against `ref`
    when given), the refusal printed. Every refusal is in the run log.
    False — the arm holds — when the plan is refused or a move fails."""
    if why:
        print("[press_demo] %s — returning home" % why)
    try:
        legs = _home_legs(ctx, cfg)
    except RuntimeError as e:
        runner.note("home_refused", why=str(e)[:300])
        if "INVALID_START" not in str(e) or not _back_off(ctx, runner, ref, why=str(e)):
            print("[press_demo] home plan refused (%s) — the arm holds; home it with sheppy's home_arm" % _short(e))
            return False
        try:
            legs = _home_legs(ctx, cfg)
        except RuntimeError as e2:
            runner.note("home_refused", why=str(e2)[:300], after="reach:back-off")
            print("[press_demo] home plan refused (%s) — the arm holds; home it with sheppy's home_arm" % _short(e2))
            return False
    res = runner.run(legs, execute=True)
    return all(r.ok for r in res)


def hand_lean(R):
    """How far the hand at attitude `R` (tool_frame in the world) leans off
    straight down (rad), as depth_press.turn_down measures it."""
    return math.acos(float(np.clip(-float(np.asarray(R, float)[2, 2]), -1.0, 1.0)))


def clear_targets(ctx, R, t):
    """Where a compliant run's clear takes the tool from (R, t) over the
    box, in the order they are tried: HOME_CLEAR_M + CLEAR_EXTRA_M over the
    box's padded cuboid, higher by what the hand's lean hangs the planner's
    lowest gripper spheres lower (HAND_SPHERE_OFF_AXIS_M sin(lean)) — so
    they clear it by 17 mm or more at any lean — and never lower than the
    hand; and back toward the arm along the box's bearing, CLEAR_BACK_M,
    then REACH_PULLBACK_M: where 16:23's planner took home after its 15 cm
    back-off."""
    from rammp_box_opening.worlds import planned_top_z

    want = (planned_top_z(ctx.model, ctx.cpose) + HOME_CLEAR_M + HAND_SPHERE_OFF_AXIS_M * math.sin(hand_lean(R))
            + CLEAR_EXTRA_M)
    z = max(float(t[2]), want)
    bx, by = float(ctx.cpose.xyz[0]), float(ctx.cpose.xyz[1])
    d = math.hypot(bx, by)
    return [[float(t[0]) - b * bx / d, float(t[1]) - b * by / d, z] for b in (CLEAR_BACK_M, REACH_PULLBACK_M)]


def _near_a_limit(ctx, q):
    """The joints `q` holds too near a limit for home to plan from without
    _home_legs' clear-margin step (primitives.ClearMarginFree.near, whose
    goal does not enter it), as text; "" when none is."""
    return ", ".join("%s %.3f rad from its limit" % jr for jr in ClearMarginFree(q).near(ctx, q))


def _clear_up_and_back(ctx, runner, R, t):
    """_clear_over_box for a compliant run: up and back toward the arm to
    the first of clear_targets whose straight line flies and leaves every
    joint room for home to plan from (_near_a_limit), else the first whose
    line flies — unguarded, at FREE_LINE_SPEED. Flatbot's URDF, 15:19's box
    pressed from the scene camera's top (0.3111), turned down: the 50 mm
    line ends with arm_joint_6 0.160 rad from its limit, the 150 mm one
    0.368. True when there (or already high enough), None when no line
    flies (the back-off is the fallback), False — the arm holds — when the
    leg failed."""
    from rammp_box_opening.runtime.kinematics import mat_to_quat_xyzw

    targets = clear_targets(ctx, R, t)
    if float(t[2]) >= targets[0][2]:
        return True
    quat = mat_to_quat_xyzw(R)
    up, lean = targets[0][2] - float(t[2]), math.degrees(hand_lean(R))
    chosen, whys = None, []
    for target in targets:
        back = math.hypot(target[0] - float(t[0]), target[1] - float(t[1]))
        pts, why = _line(ctx, target, quat)
        if pts is None:
            whys.append("%.0f mm back: %s" % (1000 * back, why))
            continue
        if not pts:
            return True
        near = _near_a_limit(ctx, pts[-1])
        if chosen is None or not near:
            chosen = (pts, back)
        if not near:
            break
        whys.append("%.0f mm back: %s at its end" % (1000 * back, near))
    if chosen is None:
        runner.note("depth_clear", flown=False, up_mm=round(1000 * up, 1), lean_deg=round(lean, 1), why="; ".join(whys))
        return None
    pts, back = chosen
    # the console says why a nearer target was passed over, as the run log does
    print("[press_demo] up %.0f mm and %.0f mm back toward the arm (the hand leaning %.0f deg), clear of the box's "
          "padded volume, before home%s" % (1000 * up, 1000 * back, lean, " (%s)" % "; ".join(whys) if whys else ""))
    # up and away from the box, toward the look pose the hand came in from
    res = runner.run([line_leg(ctx, "depth:clear", pts, quat, FREE_LINE_SPEED, None, _full_world(ctx))], execute=True)
    if any(not r.ok for r in res):
        print("[press_demo] STOP: the way up from over the box %s — the arm holds; check the box" % _stopped_by(res))
        return False
    runner.note("depth_clear", flown=True, up_mm=round(1000 * up, 1), back_mm=round(1000 * back, 1),
                lean_deg=round(lean, 1), why="; ".join(whys) or None)
    return True


@dataclass
class WayOut:
    """A compliant reach-out's way home from the stroke's start, planned
    while its way in flies (plan_way_out): the clear — joint waypoints up
    and back toward the arm to one of clear_targets, `back_m` back, the
    hand's attitude `quat` held — and the home legs planned from its end
    (_home_legs). `passed`: why a nearer target was passed over, or None."""

    clear: list
    quat: list
    home: list
    back_m: float = 0.0
    passed: str = None


def plan_way_out(ctx, cfg, q_from, quat):
    """The way home from joints `q_from` — the stroke's start, where the
    rise back up from the hold ends (depth_press.rise_back_up) — planned
    before the arm is there: _fly_way_in hands it to the runner as the way
    in's lookahead, so the planner is asked while the way in flies and
    never after the hold (16:23 2026-10-27: home refused INVALID_START after
    the clear, then a 3.54 s back-off). The clear goes to the first of
    clear_targets(fk(q_from)) whose straight line flies and leaves every
    joint room for home to plan from, else the first whose line flies, as
    _clear_up_and_back picks (15:19's box turned down: the 50 mm line ends
    with arm_joint_6 0.160 rad from its limit, the 150 mm one 0.368); home
    is planned from its end (_home_legs, its clear-margin step included),
    and a target whose home does not plan is passed over. Returns (WayOut,
    None) or (None, why)."""
    from rammp_box_opening.runtime import approach

    chain = approach._chain()
    q_from = [float(v) for v in q_from]
    R, t = chain.fk(q_from)
    whys, near_limit = [], []

    def home_from(pts, back):
        try:
            home = _home_legs(ctx, cfg, start=pts[-1])
        except RuntimeError as e:
            whys.append("%.0f mm back: home refused (%s)" % (1000 * back, _short(e)))
            return None
        return WayOut(pts, list(quat), home, back, "; ".join(whys) or None)

    for target in clear_targets(ctx, R, t):
        back = math.hypot(target[0] - float(t[0]), target[1] - float(t[1]))
        pts, why = chain.straight_line(q_from, target, quat)
        if pts is None:
            whys.append("%.0f mm back: %s" % (1000 * back, why))
            continue
        pts = [[float(v) for v in q] for q in pts]
        near = _near_a_limit(ctx, pts[-1])
        if near:
            whys.append("%.0f mm back: %s at its end" % (1000 * back, near))
            near_limit.append((pts, back))
            continue
        way = home_from(pts, back)
        if way is not None:
            return way, None
    for pts, back in near_limit:
        way = home_from(pts, back)
        if way is not None:
            return way, None
    return None, "; ".join(whys) or "no clear target"


def _clear_over_box(ctx, runner, ref):
    """From over the box, before its home is planned: straight up until the
    tool stands HOME_CLEAR_M above the box's cuboid as the planner pads it
    (worlds.planned_top_z) — the gripper base's collision spheres reach 48
    mm below tool_frame straight down, and a start among them is refused
    (20:04: INVALID_START twice, the arm held over the box). Slow, drag- and
    hang-guarded against the free-air `ref`: the hang's reference,
    tip_load_at the line's start, holds only along a short vertical line.
    A compliant run goes up as far as the hand's lean needs and back toward
    the arm instead (_clear_up_and_back). True when there (or already
    clear), None when no straight line flies from here (the back-off is the
    fallback), False — the arm holds — when it met something."""
    from rammp_box_opening.runtime import approach
    from rammp_box_opening.runtime.kinematics import mat_to_quat_xyzw
    from rammp_box_opening.worlds import planned_top_z

    R, t = approach._chain().fk(ctx.client.joints())
    if compliant(ctx):
        return _clear_up_and_back(ctx, runner, R, t)
    want = planned_top_z(ctx.model, ctx.cpose) + HOME_CLEAR_M
    if float(t[2]) >= want:
        return True
    quat = mat_to_quat_xyzw(R)
    pts, why = _line(ctx, [float(t[0]), float(t[1]), want], quat)
    if pts is None:
        runner.note("depth_clear", flown=False, up_mm=round(1000 * (want - float(t[2])), 1), why=why)
        return None
    if not pts:
        return True
    print("[press_demo] up %.0f mm, clear of the box's padded volume, before home" % (1000 * (want - float(t[2]))))
    res = runner.run([line_leg(ctx, "depth:clear", pts, quat, CONTACT_SPEED, drag_guard(ref, tip_load_at(pts[0])),
                               _full_world(ctx))], execute=True)
    if any(not r.ok for r in res):
        print("[press_demo] STOP: the way up from over the box %s — the arm holds; check the box" % _stopped_by(res))
        return False
    return True


def _home_from_over_box(ctx, cfg, runner, result):
    """Home after a free depth press (DepthPress.free): clear of the box's
    padded volume first, then planned once, the back-off left as the
    fallback. False — the arm holds — when it did not get there."""
    ref = result.ref or (None if compliant(ctx) else _settled(ctx, "over the box"))
    if ref is not None or compliant(ctx):
        clear = _clear_over_box(ctx, runner, ref)
        if clear is False:
            return False
    from rammp_box_opening.tasks.depth_press import home_why

    return _go_home(ctx, cfg, runner, home_why(result), ref=ref)


def _unload_stopped(res):
    """What a stopped unload means, from its guard's report."""
    trip = res[-1].trip or {}
    if trip.get("why") == "hang":
        return "something pulled down on the hand (%.1f N — the box caught on the fingers?) — the arm holds; check the box" % (
            trip.get("down_n") or 0.0)
    if trip.get("why") == "torque":
        return "the hand is still pressing on the box (%.1f Nm on the way up) — the arm holds ON it" % (trip.get("dev_nm") or 0.0)
    return "the way up stopped (%s) — the arm holds" % (res[-1].detail or res[-1].outcome)


def _still_free(ctx, runner, ref_start, where, strict):
    """At rest, nothing on the hand against the free-air reading at the
    stroke's start: a pull under HANG_STATIC_N, and (`strict`, above the
    knob) no wrist joint off by STATIC_NM. None when free, else why."""
    now = _settled(ctx, where)
    if now is None:
        return "no steady efforts at %s" % where
    r = residual(_wrist_gravity(), ref_start, now)
    pull = down_force(tip_load_at(now[1]), r)
    worst = max(range(len(r)), key=lambda j: abs(r[j]))
    runner.note("reach_free_check", where=where, residual_nm=[round(v, 3) for v in r], pull_n=round(pull, 2))
    if pull > HANG_STATIC_N:
        return "the hand carries %.1f N pulling down at the fingertip at %s — the box came with the hand?" % (pull, where)
    if strict and abs(r[worst]) > STATIC_NM:
        return "joint_%d is %.1f Nm off its free-air reading at %s — something is on the hand" % (worst + 4, abs(r[worst]), where)
    return None


def _back_out(ctx, runner, plan, world, ref_start):
    """Off the button and back to the look pose, never pulling the box:
    UNLOAD UNLOAD_CHECK_M straight up the stroke, CHECK at rest that nothing
    came with the finger, RISE to the stroke's start, CHECK again, SLIDE back
    out along the slide in. Every leg and check is judged against the
    free-air reading at the stroke's start (`ref_start`). False — the arm
    holds where it stopped, the reason printed — when anything was felt or
    a line cannot be flown."""
    from rammp_box_opening.runtime import approach

    m = ctx.model
    _R, t = approach._chain().fk(ctx.client.joints())
    first = [float(t[0]), float(t[1]), min(float(t[2]) + UNLOAD_CHECK_M, float(plan.start_xyz[2]))]
    # the rise only after the at-rest check 15 mm up found nothing on the hand
    for name, target, strict, speed in (("reach:unload", first, False, CONTACT_SPEED),
                                        ("reach:rise", plan.start_xyz, True, FREE_LINE_SPEED)):
        up, why = _line(ctx, target, plan.quat)
        if up is None:
            print("[press_demo] STOP: the way up is not a straight line the arm can fly (%s) — the arm holds%s"
                  % (why, " with the fingertip still on the button" if name == "reach:unload" else ""))
            return False
        if up:
            res = runner.run([line_leg(ctx, name, up, plan.quat, speed,
                                       unload_guard(m, ref_start, tip_load_at(up[0])), world)], execute=True)
            if any(not r.ok for r in res):
                print("[press_demo] STOP: %s" % _unload_stopped(res))
                return False
        why = _still_free(ctx, runner, ref_start, "%.0f mm up" % (1000 * UNLOAD_CHECK_M) if not strict else "the stroke's start",
                          strict)
        if why is not None:
            print("[press_demo] STOP: %s The arm holds; check the box" % why)
            return False
    out, why = _line(ctx, plan.look_xyz, plan.quat)
    if out is None:
        print("[press_demo] STOP: the slide out is not a straight line the arm can fly (%s) — the arm holds" % why)
        return False
    if out:
        # after the strict check at the stroke's start: away from the box, above it
        res = runner.run([line_leg(ctx, "reach:slide-out", out, plan.quat, FREE_LINE_SPEED,
                                   drag_guard(ref_start, tip_load_at(out[0])), world)], execute=True)
        if any(not r.ok for r in res):
            print("[press_demo] STOP: something pulled on the way out (the box caught on the fingers?) — the arm "
                  "holds; check the box")
            return False
    return True


def _after_check(node, ctx, runner, watcher, scene_fix, locator, button, lid_before, yaw):
    """From the look pose, before the arm goes anywhere: the wrist re-aims
    (did the knob pop, did the box move or lift?) and the scene camera looks
    again. Moves ctx.cpose to where the box is now seen. Returns
    (found, moved, popped, up_after)."""
    from rammp_box_opening.tasks.press_demo import SCENE_LOCATE_S, knob_is_up

    m = ctx.model
    if locator is not None:
        # the first scene bbox takes ~1.2 s (tick + inference): let it come
        # while the wrist re-aims (no OWL, no GPU, the arm still at the look
        # pose); locate() enables again (idempotent) and disables in its finally
        locator.owl.enable()
    try:
        again, status, up_after = _aim(node, watcher, yaw, button[2], button[:2], runner, "reach-after")
    except BaseException:
        if locator is not None:
            locator.owl.disable()  # never leave the latched enable on
        raise
    moved = None if again is None else judge_box_moved(button, again[0])
    lid_after = None if again is None else _lid_z(again[0][2], up_after)
    lifted = None if lid_before is None or lid_after is None else lid_after - lid_before
    popped = knob_is_up(up_after)
    seen_again = locator.locate(SCENE_LOCATE_S) if locator is not None else None
    scene_moved = None if seen_again is None else judge_box_moved(scene_fix.top_xyz, seen_again.top_xyz)
    scene_dz = None if seen_again is None else float(seen_again.top_xyz[2]) - float(scene_fix.top_xyz[2])
    runner.note("reach_after", knob_up_mm=None if up_after is None else round(up_after, 1),
                moved_mm=None if moved is None else round(1000 * moved, 1),
                lid_dz_mm=None if lifted is None else round(1000 * lifted, 1),
                scene_moved_mm=None if scene_moved is None else round(1000 * scene_moved, 1),
                scene_dz_mm=None if scene_dz is None else round(1000 * scene_dz, 1))
    print(
        "[press_demo] AFTER THE PRESS: knob %s; the button %s; the scene camera %s"
        % (
            "UP %.1f mm — the box is open" % up_after if popped else ("%.1f mm above the lid" % up_after if up_after is not None else "not judged"),
            "not re-found (%s)" % status if moved is None else "%.0f mm from where it was%s" % (
                1000 * moved, "" if lifted is None else ", its lid %+.0f mm" % (1000 * lifted)),
            "did not find the box" if scene_moved is None else "puts the box %.0f mm from where it was, %+.0f mm in height"
            % (1000 * scene_moved, 1000 * scene_dz),
        )
    )
    found = again is not None or seen_again is not None
    box_moved = (
        (moved is not None and moved > BOX_MOVED_M)
        or (lifted is not None and abs(lifted) > BOX_LIFTED_M)
        or (scene_moved is not None and (scene_moved > SCENE_MOVED_M or abs(scene_dz) > SCENE_MOVED_M))
    )
    # home is planned around the box where it is seen now; a popped knob's
    # top is not the box's height
    if again is not None:
        x, y, z = (float(v) for v in again[0])
        ctx.cpose = cpose_for_button((x, y, button[2] if popped else z), yaw, m.button_offset)
    elif seen_again is not None:
        ctx.cpose = seen_again.pose
    return found, box_moved, popped, up_after


def _home_after_contact(node, ctx, cfg, runner, watcher, scene_fix, locator, button, lid_before, yaw):
    """The look pose reached after a contact: judge the box, then home —
    or hold when neither camera finds it. Returns (homed, popped, up_after)."""
    found, box_moved, popped, up_after = _after_check(node, ctx, runner, watcher, scene_fix, locator, button, lid_before, yaw)
    if not found:
        print("[press_demo] STOP: neither camera finds the box after the press — it may have left the table. The arm "
              "holds at the look pose; check the box, then home the arm with sheppy's home_arm")
        return False, popped, up_after
    if box_moved:
        _go_home(ctx, cfg, runner, "STOP: the press MOVED the box — check it is still standing on the table")
        return False, popped, up_after
    return _go_home(ctx, cfg, runner, ""), popped, up_after


# _predicted_drift samples every this many points plus the last: it is a
# printed estimate, and evaluating the gravity model at every point (and the
# start again for each) cost ~0.2 s at rest before the look leg (2026-10-22)
DRIFT_SAMPLE_EVERY = 4


def _predicted_drift(plan):
    """joint_4's gravity torque change along the slide and the stroke (Nm):
    what a raw guard would have read as a strike. A display estimate only
    (gravity_mismatch is the gate): sampled every DRIFT_SAMPLE_EVERY points
    plus the last. A depth plan has no tilted stroke: 0 down it."""
    g = _wrist_gravity()

    def drift(pts):
        if pts is None or len(pts) == 0:
            return 0.0
        g0 = float(g(pts[0])[0])
        idx = sorted(set(range(DRIFT_SAMPLE_EVERY, len(pts), DRIFT_SAMPLE_EVERY)) | {len(pts) - 1})
        return max(abs(float(g(pts[i])[0]) - g0) for i in idx)

    return drift(plan.slide_pts), drift(plan.stroke_pts)


def _not_one_execution(ctx, legs):
    """Why the way in's legs (way_in_legs) do not fly as one execution, or
    None: they do not merge into one group (legs.merge_groups — solve_way_in
    chains them and way_in_legs names one gains spec on every one, so it
    should not happen), or the first does not start where the arm stands
    (REUSE_RAD, as _line_from reuses a solved line). After the planner's
    look leg the way in starts at that leg's planned end, which an
    impedance arrival leaves a few mm off; the driver would jump to it, and
    the runner refuses it from DRIFT_REPLAN_RAD on. Leg by leg, each is
    solved again from where the arm stands."""
    if not legs:
        return "no line to fly"
    if len(merge_groups(legs)) != 1:
        return "; ".join("%s does not continue %s as one goal" % (b.name, a.name)
                         for a, b in zip(legs, legs[1:]) if not can_merge(a, b))
    start = np.asarray(legs[0].traj.points[0].positions, float)
    off = float(np.abs(np.asarray(ctx.client.joints(), float) - start).max())
    if off >= REUSE_RAD:
        return "the arm stands %.3f rad from where %s starts" % (off, legs[0].name)
    return None


def _phase_at(names, fracs, progress):
    """The member of a merged execution flying at `progress`, a time
    fraction of it: the last of `names` whose start fraction (retime_group's
    seg_start_fracs) is at or before it."""
    at = names[0]
    for name, frac in zip(names, fracs):
        if float(frac) <= float(progress):
            at = name
    return at


def _note_flight(runner, kind, names, last):
    """The run record's row `kind` for one execution the runner re-timed
    and flew, its members `names` and its result `last`: a merged group
    logs every leg with the group's t_wall, and where each began, as a
    fraction of the execution's time, keeps them apart — with how fast it
    took each junction and how many it stopped at. Returns the runner's
    report of it (last_retime)."""
    retimed = runner.last_retime or {}
    runner.note(kind, legs=list(names), outcome=last.outcome,
                t_wall=None if last.t_wall is None else round(float(last.t_wall), 3),
                duration_s=round(float(retimed.get("duration_s") or 0.0), 3),
                seg_start_fracs=[round(float(v), 4) for v in retimed.get("seg_start_fracs") or []],
                junction_speeds=[round(float(v), 3) for v in retimed.get("junction_speeds") or []],
                stops=retimed.get("stops"))
    return retimed


def _fly_way_in(ctx, cfg, runner, way, legs, top_z, scene_fix, locator):
    """A compliant depth press's way in (`way`, flown as `legs`,
    way_in_legs') from the aimed look pose to the stroke's end below the
    button's bottom: ONE runner.run, re-timed into one execution, nothing
    read between its legs — the way home planned while it flies, from the
    stroke's start (plan_way_out, the runner's lookahead). Then the hold on
    the button whose top is at `top_z` (depth_press.compliant_hold) and the
    way home from it (_way_home): the rise back up the stroke, the clear
    and home, one execution where they meet. A way in that did not start
    goes home from the look pose; one that stopped part-way holds the arm
    there, its STOP naming the leg in flight. Returns the process exit
    code."""
    from rammp_box_opening.tasks import depth_press as dp

    names = [lg.name for lg in legs]
    if way.down.short_deg:
        print("[press_demo] TURN: %.0f deg short of straight down (no hand nearer straight down flies here)"
              % way.down.short_deg)
    print("[press_demo] ONE MOTION from the look pose to below the button: %s" % ", ".join(names))
    print("[press_demo] %s" % dp.press_text(way.stroke, cfg))
    # The stroke meets the box by design: from here every world pads it for
    # a shove, as depth_press.press_compliant's does — home's too, planned
    # while the way in flies, so it is padded before it flies
    pad = ctx.contact_pad
    ctx.contact_pad = CONTACT_SHIFT_PAD_M
    # The way home from the stroke's start, where the rise back up ends. The
    # end the lookahead is handed, the stroke's bottom inside the padded box,
    # is no start for it, and is left unused
    res = runner.run(legs, execute=True, lookahead=lambda _end: plan_way_out(ctx, cfg, way.stroke.q0, way.stroke.quat))
    ahead = runner.lookahead_result  # every run() resets it
    last = res[-1]
    flown = all(r.ok for r in res)
    # The runner's report of the group it re-timed (last_retime) is this
    # execution's only where the runner went on to fly it, and so handed
    # back what flew (LegResult.flown_traj): a refusal before that — its
    # gates, a plan-free start the arm has left — leaves the report of the
    # last group that flew, the look's.
    tried = last.flown_traj is not None
    retimed = _note_flight(runner, "reach_way_in_flight", names, last) if tried else {}
    # nothing moved where the runner refused the execution, nor where the
    # client reports no progress for it: the driver rejected the goal, or
    # the client's own gates refused it before sending
    moved = flown or (tried and bool(last.progress))
    if not moved:
        ctx.contact_pad = pad  # nothing came near the box: no shove to pad it for
    if not flown:
        if not moved:
            # from the look pose, in free air: nothing came near the box
            _go_home(ctx, cfg, runner, "STOP: the way in did not start (%s) — nothing was pressed"
                     % (last.detail or last.outcome))
            return 1
        print("[press_demo] STOP: the way in stopped during %s (%s) — the arm holds"
              % (_phase_at(names, retimed.get("seg_start_fracs") or [0.0], last.progress), last.detail or last.outcome))
        return 1
    runner.note("reach_slid_in", turned_down=way.down.walked, residual_nm=None)
    seen = {}
    held = dp.compliant_hold(ctx, cfg, runner, way.stroke, top_z, seen)
    result, homed = _way_home(ctx, cfg, runner, way.stroke, held, seen.get("held_joints") or ctx.client.joints(), ahead,
                              scene_fix, locator)
    runner.finish()
    return dp.finish(ctx, dataclasses.replace(result, touch_xyz=seen.get("touch_xyz")), homed)


def _in_flight(legs, last, retimed):
    """The leg of `legs` (one runner.run's) in flight when the execution
    whose result is `last` stopped: by where each member of its merge group
    began (_phase_at), where the runner re-timed and flew that group
    (`retimed`, its last_retime), else the group's first. A merged group's
    result carries its last member's name."""
    group = next((g for g in merge_groups(legs) if g[-1].name == last.leg_name), None)
    names = [lg.name for lg in group] if group else [last.leg_name]
    fracs = (retimed or {}).get("seg_start_fracs") or []
    if len(names) == 1 or last.flown_traj is None or len(fracs) != len(names):
        return names[0]
    return _phase_at(names, fracs, last.progress or 0.0)


def _rise_legs(ctx, stroke, q, world):
    """The way up from a compliant hold, the arm held at joints `q` by the
    button under `stroke`, as the legs that fly it, labelled `world`:
    'depth:rise' back up the stroke to its start (depth_press.rise_back_up)
    — else, as compliant_rise flies it, a straight line from where the arm
    stands to there; none when it stands there. Returns (legs, True where
    it goes back up the stroke, None), or (None, False, why) where no way
    up flies."""
    from rammp_box_opening.runtime import approach
    from rammp_box_opening.tasks import depth_press as dp

    pts = dp.rise_back_up(approach._chain(), stroke, q)
    on_stroke = pts is not None
    if pts is None:
        pts, why = _line(ctx, [float(v) for v in stroke.t0], stroke.quat)
        if pts is None:
            return None, False, why
    legs = [dp.compliant_line(ctx, "depth:rise", pts, stroke.quat, FREE_LINE_SPEED, world)] if pts else []
    return legs, on_stroke, None


def _no_way_up(held, why):
    """(DepthPress, homed) where no way up from the hold flies: the hold's
    verdict `held`, free False — the arm holds."""
    why = "the way up is not a straight line the arm can fly (%s) — the arm holds" % why
    return dataclasses.replace(held, free=False, why=why), False


def _rise_then_home(ctx, cfg, runner, up, held, scene_fix, locator):
    """The way home from the hold as before the one execution: the rise
    `up` (its legs; none where the arm stands at the stroke's start) on its
    own, then home from over the box (_home_from_over_box: the clear up and
    back, then home planned, the back-off its fallback). Returns
    (DepthPress, homed): free False with why where the rise did not fly."""
    from rammp_box_opening.tasks import depth_press as dp

    if up:
        res = runner.run(up, execute=True)
        if any(not r.ok for r in res):
            return dataclasses.replace(held, free=False, why="the way up stopped (%s) — the arm holds" % res[-1].detail), False
    result = dp.after_scene_check(ctx, runner, held, scene_fix, locator)
    return result, result.free and _home_from_over_box(ctx, cfg, runner, result)


def _way_home(ctx, cfg, runner, stroke, held, q_held, ahead, scene_fix, locator):
    """From a compliant hold to home, the arm held at `q_held` by the
    button under `stroke` (the hold's verdict `held`): the rise back up the
    stroke to its start (depth_press.rise_back_up; else, as compliant_rise
    flies it, a straight line from where the arm stands), then the clear and
    home that plan_way_out planned while the way in flew (`ahead`, its
    (WayOut, why) as the runner's lookahead result), each home leg written
    on the branch of the end of the leg before it (approach.on_branch_of) —
    ONE runner.run, no planner call after the hold. All under
    COMPLIANT_GAINS, so they merge into one execution: the home transit
    flies STIFF too. Legs that do not meet (a home plan off the clear's end,
    home's clear-margin step, guarded) the runner flies as goals of their
    own. Where nothing was planned, or the runner refused the one execution
    before anything moved: the rise alone — never the pads left on the
    button — then home from over the box (_rise_then_home); after a refusal
    that rise is solved again from where the arm stands. Stopped part-way:
    the arm holds, the STOP naming the leg in flight. One execution — or
    the first, where it was not one and that one stopped — has its row in
    the run record (reach_way_out_flight), as the way in's has. Returns
    (DepthPress, homed): free False with why where the arm holds."""
    from rammp_box_opening.runtime import approach
    from rammp_box_opening.tasks import depth_press as dp

    world = _full_world(ctx)
    up, on_stroke, why = _rise_legs(ctx, stroke, q_held, world)
    if up is None:
        return _no_way_up(held, why)
    if not held.free:
        return _rise_then_home(ctx, cfg, runner, up, held, scene_fix, locator)  # off the button, and no further
    # None: the lookahead raised (the runner's "lookahead plan failed" line says what)
    way, why = ahead if ahead is not None else (None, "the lookahead plan failed")
    if way is None:
        runner.note("reach_way_out", planned=False, why=why, clear_back_mm=None, legs=[lg.name for lg in up],
                    rise_on_stroke=on_stroke)
        print("[press_demo] no way home was planned while the way in flew (%s) — up off the button, then home from "
              "over the box" % why)
        return _rise_then_home(ctx, cfg, runner, up, held, scene_fix, locator)
    legs = up + [dp.compliant_line(ctx, "depth:clear", way.clear, way.quat, FREE_LINE_SPEED, world)]
    for leg in way.home:
        if leg.traj is not None:
            leg.traj = approach.on_branch_of(leg.traj, legs[-1].goal_joints)
            leg.goal_joints = [float(v) for v in leg.traj.points[-1].positions]
        if leg.guard is None:
            leg.gains = dp.COMPLIANT_GAINS  # one gains spec: the home transit merges, and flies STIFF
        legs.append(leg)
    names = [lg.name for lg in legs]
    groups = merge_groups(legs)
    runner.note("reach_way_out", planned=True, why=way.passed, clear_back_mm=round(1000 * way.back_m, 1), legs=names,
                rise_on_stroke=on_stroke)
    print("[press_demo] %s from the button to home: %s — up to the stroke's start, then up and %.0f mm back toward "
          "the arm, clear of the box's padded volume" % (
              "ONE MOTION" if len(groups) == 1 else "in %d goals" % len(groups), ", ".join(names), 1000 * way.back_m))
    res = runner.run(legs, execute=True)
    last = res[-1]
    if len(res) == 1 and last.flown_traj is not None:
        # the first execution, the rise in it, re-timed and sent: the
        # runner's report is its own (a later goal's would replace it)
        _note_flight(runner, "reach_way_out_flight", [lg.name for lg in groups[0]], last)
    if all(r.ok for r in res):
        return held, True
    started = last.flown_traj is not None and bool(last.progress)
    if len(res) == 1 and not started:
        # nothing moved: the runner's gates refused it, or the driver its
        # first goal. The pads are still on the button. One gate is the
        # arm no longer standing where a plan-free line starts, and the
        # held reading's rise would be refused again: it is solved anew
        # from where the arm stands
        print("[press_demo] the way home did not start (%s) — up off the button on its own, then home from over the box"
              % (last.detail or last.outcome))
        up, _on_stroke, why = _rise_legs(ctx, stroke, ctx.client.joints(), world)
        if up is None:
            return _no_way_up(held, why)
        return _rise_then_home(ctx, cfg, runner, up, held, scene_fix, locator)
    if not started:
        # a goal after the first did not start: the first, the rise in it,
        # flew, and the arm stands at its end in free air off the button
        print("[press_demo] the way home's %s did not start (%s) — home from where the arm stands, over the box"
              % (_in_flight(legs, last, runner.last_retime), last.detail or last.outcome))
        return held, _home_from_over_box(ctx, cfg, runner, held)
    return dataclasses.replace(held, free=False, why="the way home stopped during %s (%s) — the arm holds" % (
        _in_flight(legs, last, runner.last_retime), last.detail or last.outcome)), False


def run_reach_press(node, ctx, cfg, runner, watcher, args, scene_fix, locator):
    """The whole reach-out press from the scene camera's fix. Returns the
    process exit code: 0 pressed and the knob popped, 1 a failure, the box
    moved or the knob not up, 2 no button.

    COMPLIANT (args.compliant with --depth-press, press_demo's default
    since 2026-10-27): every leg in the driver's joint impedance, under the
    client's compliant_gains (STIFF) where a leg names none, and
    nothing the run reads stops the arm — no torque guard, no at-rest read
    or check, no look after the press (depth_press.press_compliant). Its
    free-air lines (the re-aim, the slide in, the clear before home) fly at
    FREE_LINE_SPEED: CONTACT_SPEED is there so a guard can stop a line in
    time, and these carry none (16:23 crawled 10.5 s of them at 0.15). The
    box's height is the higher of the aim's and the scene camera's (15:19:
    the touch met the button at 0.312, the scene said 0.311, the wrist
    camera's aim 0.288). 0 then means pressed, by the spring's hold. Its
    press is proven by the arm's own kinematics before the arm moves
    toward the box: from the look pose the hand turns down and strokes —
    for the scene camera's button, or one PROOF_SHIFT_M off it, before the
    look (plan_reach depth=True, shifted_way_down), and for the aimed
    button before the re-aim (solve_way_in). Where it does not, nothing
    moves, or the arm goes home from the look pose. Where it does, the way
    in flies as proven, as ONE execution from the aimed look pose to the
    stroke's end — the re-aim, the slide in or the walk, the turn and the
    stroke (way_in_legs, _fly_way_in) — then the hold, and from it the rise,
    the clear and home as ONE execution, planned while the way in flew
    (plan_way_out, _way_home); leg by leg, each solved again from where the
    arm stands, only where the way in's legs do not make one execution
    (_not_one_execution)."""
    on = bool(getattr(args, "compliant", False)) and bool(getattr(args, "depth_press", False))
    was = getattr(ctx.client, "compliant", False)
    if on:
        ctx.client.compliant = True
        print("[press_demo] COMPLIANT: every move in the arm's joint impedance, %s, and nothing the run reads stops "
              "it — no torque guard, no at-rest check, no look after the press. The e-stop and Ctrl+C stop it"
              % gains_text(getattr(ctx.client, "compliant_gains", COMPLIANT_RUN_GAINS)))
    try:
        return _reach_press(node, ctx, cfg, runner, watcher, args, scene_fix, locator)
    finally:
        ctx.client.compliant = was


def _reach_press(node, ctx, cfg, runner, watcher, args, scene_fix, locator):
    from rammp_box_opening.models.container import from_container
    from rammp_box_opening.runtime import approach
    from rammp_box_opening.tasks.press_demo import dump_wrist_frame, refuse_open_box, touch_verdict

    m = ctx.model
    why = unsupported(m)
    if why is None and not oblique_aim_available():
        why = "this install's wrist aim cannot look at the lid obliquely (perception/button_aim takes no lid_z/prior_xy)"
    if why is not None:
        print("[press_demo] STOP: no reach-out press — %s; nothing moved" % why)
        return 1
    yaw = scene_fix.pose.yaw
    prior = from_container(scene_fix.pose, m.button_offset)
    # the scene camera's own lid top: on flatbot it read 99-102 mm above the
    # table against the model's 121 (2026-10-22), so the aim measures it
    prior_button = (float(prior[0]), float(prior[1]), float(scene_fix.top_xyz[2]))
    print(
        "[press_demo] REACH-OUT: the hand cannot stand straight down over this box — pressing it with the hand tilted "
        "back toward the arm, one fingertip straight down onto the button"
    )
    soft = compliant(ctx)  # no guard references to read, no checks to make
    depth = bool(getattr(args, "depth_press", False))
    # A compliant depth press is proven before the arm moves toward the box:
    # from the look pose the hand turns down and strokes (way_down). 16:23
    # 2026-10-27 flew the look, aimed, flew the re-aim and a 7.4 s slide in
    # to a turn no IK reaches ("no straight-down hand reaches here"), then
    # the clear, a back-off and home: 20.2 s from the look to home, nothing
    # pressed. This is planning before the arm moves, not a check that stops it.
    proof = soft and depth
    looks = [] if proof else None
    # One read: the hold that primes the joint impedance and the look's plan
    # start from the same joints. The first compliant goal after a POSITION
    # one waits out the driver's switch (mode_settle_s 0.25 s) and starts on
    # a spring still fading in (gain_ramp_s 0.5 s): 16:23 2026-10-27's look
    # took 1.857 s against its 1.60 s preview, and no compliant leg after it
    # ran more than 3 ms over its own. The hold (client.prime_compliant) pays
    # both while the reach is planned. A preview and a guarded run send none.
    q0 = ctx.client.joints()
    if args.execute and soft:
        prime = getattr(ctx.client, "prime_compliant", None)
        if prime is not None and prime(q0):
            print("[press_demo] the driver enters its joint impedance now, while the reach is planned")
    plan, why = plan_reach(ctx, cfg, prior_button, q0, depth=proof, looks=looks)
    proven = "scene" if proof and plan is not None else None
    if plan is None and proof and looks:
        near, why_near = shifted_way_down(ctx, cfg, looks[0], prior_button)
        if near is None:
            print("[press_demo] STOP: no reach-out press for the scene camera's box — no tilt turns the hand down and "
                  "strokes there, nor with the button %.0f mm nearer the arm or to either side (%s; at %d deg: %s); "
                  "nothing moved" % (1000 * PROOF_SHIFT_M, why, looks[0].tilt_deg, why_near))
            return 1
        plan, proven = looks[0], "shifted"
        print("[press_demo] REACH-OUT: the press does not reach the scene camera's button itself (%s), but at %d deg "
              "it does with the button %.0f mm %s — inside what the aim corrects — so the look flies and the aimed "
              "button decides" % (why, plan.tilt_deg, 1000 * PROOF_SHIFT_M, near))
    if plan is None:
        print("[press_demo] STOP: no reach-out press plans for the scene camera's box (%s); nothing moved" % why)
        return 1
    runner.note("reach_plan", tilt_deg=plan.tilt_deg, look_xyz=[round(v, 4) for v in plan.look_xyz],
                button=list(prior_button), proof=proven,
                down=None if plan.down is None else {"walked": plan.down.walked, "short_deg": plan.down.short_deg})
    try:
        slide_nm, stroke_nm = _predicted_drift(plan)
        drift = "; gravity alone moves joint_4 %.2f Nm along the slide and %.2f down the stroke (taken out)" % (slide_nm, stroke_nm)
    except Exception as e:  # noqa: BLE001 - a display line only; the check below is the gate
        drift = " (no gravity model: %s)" % _short(e)
    print("[press_demo] REACH-OUT: hand tilted %d deg; looking from %.2f m back toward the arm%s"
          % (plan.tilt_deg, REACH_PULLBACK_M, drift))
    if not args.execute:
        runner.run([plan.look_leg], execute=False)
        print("[press_demo] preview ends at the look pose: the aim, the slide, the press and the checks need --execute")
        return 0
    at_rest = None if soft else _settled(ctx, "the rest pose")
    if at_rest is None and not soft:
        return 1
    res = runner.run([plan.look_leg], execute=True)
    if any(not r.ok for r in res):
        return 1
    at_look = None
    if not soft:
        at_look = _settled(ctx, "the look pose")
        if at_look is None:
            return 1
        mismatch = gravity_mismatch(_wrist_gravity(), at_rest, at_look)
        runner.note("reach_gravity_check", ok=mismatch is None,
                    residual_nm=[round(v, 3) for v in residual(_wrist_gravity(), at_rest, at_look)])
        if mismatch is not None:
            _go_home(ctx, cfg, runner, "STOP: %s — the reach-out's guards would be judging the wrong arm; "
                                       "nothing was pressed" % mismatch)
            return 1

    got, status, up_mm = _aim(node, watcher, yaw, prior_button[2], prior_button[:2], runner, "reach-aim")
    if got is None:
        dump_wrist_frame(watcher, "reach-look")
        _go_home(ctx, cfg, runner, "STOP: no button from the look pose (%s) — nothing was pressed" % status)
        return 2
    open_already = refuse_open_box(up_mm)
    if open_already is not None:
        runner.note("open_box", button_above_lid_mm=round(up_mm, 1))
        _go_home(ctx, cfg, runner, "STOP: " + open_already)
        return 1
    button = tuple(float(v) for v in got[0])
    lid_before = _lid_z(button[2], up_mm)
    print(
        "[press_demo] AIMED from the look pose: button at [%.3f, %.3f, %.3f], %.0f mm from the scene camera's"
        % (button[0], button[1], button[2], 1000 * judge_box_moved(prior_button, button))
    )
    if soft and float(prior_button[2]) > button[2]:
        # the press goes by the higher top, the stroke reaching on past it
        print("[press_demo] the scene camera puts the lid %.0f mm higher than the wrist's aim: pressing from that "
              "height" % (1000 * (float(prior_button[2]) - button[2])))
        button = (button[0], button[1], float(prior_button[2]))
    # the box is where its button is, not where the scene camera put it: every
    # world from here on holds its cuboid there
    ctx.cpose = cpose_for_button(button, yaw, m.button_offset)
    if ctx.mission_frames is not None:
        # a "fix", not an "aim": the oblique aim goes through the wrist
        # camera's mount in TF, which is known to be off (76 mm high, 20:04)
        ctx.mission_frames.located(button, "fix")
    world = _interaction_world(ctx, button[2], cfg.travel_m, "reach")
    # A compliant run flies the re-aim and the slide in at FREE_LINE_SPEED:
    # CONTACT_SPEED is there so a guard can stop a line in time, and these
    # carry none (16:23 2026-10-27 crawled them 9.9 s at 0.15). Their world
    # is a label: a plan-free line is never planned, re-planned or
    # collision-checked against it (line_leg builds it, Runner._replan_group
    # refuses plan-free legs, push_name only writes a YAML). "full" only
    # satisfies Runner._refusal's spec §6 rule that an unguarded leg faster
    # than CONTACT_SPEED be labelled full or bench, as depth:turn, approach,
    # press and rise already are; the speed without it is refused, and the
    # arm holds at the look pose with nothing pressed.
    line_speed, line_world = (FREE_LINE_SPEED, _full_world(ctx)) if soft else (CONTACT_SPEED, world)
    # To the look pose for the AIMED button. Where the scene camera's box was
    # wrong, the arm may already stand inside the padded cuboid of the box
    # where it really is, and the planner refuses every start there
    # (2026-10-22: the box 85 mm nearer the arm). So the arm moves there the
    # way it slides — a straight line at the tilt it holds, slow and guarded
    # in a guarded run — and only when no such line flies does the planner
    # plan the move.
    if proof:
        # the aimed button's whole way in — the move, the slide or the walk,
        # the turn and the stroke — proven from the look pose before the arm
        # moves toward it, then flown as it was proven, as one execution
        way, why_way, kind = solve_way_in(ctx, cfg, button, plan.tilt_deg, ctx.client.joints())
        if way is None and kind == "lines":
            plan2, why = plan_reach(ctx, cfg, button, ctx.client.joints(), depth=True)
            if plan2 is None:
                why_way = "as straight lines: %s; planned: %s" % (why_way, why)
            else:
                res = runner.run([plan2.look_leg], execute=True)
                if any(not r.ok for r in res):
                    return 1
                # plan_reach proved the way down from the leg's end: that proof
                # is the way in. Solved again from the live joints, an
                # impedance arrival a few mm off the look pose would only add a
                # re-aim, which the planned move never had. It starts at the
                # leg's planned end, though, and an impedance arrival read the
                # moment its goal ends stands mm off it, past REUSE_RAD: this
                # way in then flies leg by leg (_not_one_execution)
                way = WayIn([], plan2, plan2.down, None if plan2.down.walked else plan2.slide_pts, plan2.stroke)
        if way is None:
            # from the look pose, in free air: nothing slid into the box's
            # padded volume, so no clear and no back-off before home
            _go_home(ctx, cfg, runner, "STOP: no press reaches the aimed button (%s) — nothing was pressed" % why_way)
            return 1
        move, plan2 = way.move, way.plan
        runner.note("reach_way_in", tilt_deg=plan2.tilt_deg, walked=way.down.walked, short_deg=way.down.short_deg,
                    k=round(way.stroke.k, 1), end_z=round(way.stroke.end_z, 4))
        # Nothing is read between the re-aim, the slide in, the turn and the
        # stroke here, yet each flew as a goal of its own, ending at rest
        # (16:23 2026-10-27: 0.18 s between its re-aim and its slide in), and
        # each new goal's first waypoint — the joints measured where the last
        # one stopped — became the driver's impedance reference. As one
        # execution the retimer flows through every junction but a reversal
        # in joint space (into the stroke), taken at rest inside the goal.
        legs = way_in_legs(ctx, cfg, way)
        why_not = _not_one_execution(ctx, legs)
        if why_not is None:
            return _fly_way_in(ctx, cfg, runner, way, legs, button[2], scene_fix, locator)
        print("[press_demo] the way in does not fly as one execution (%s) — leg by leg, each solved again from where "
              "the arm stands" % why_not)
    else:
        move, plan2, why_line = reach_from_here(ctx, cfg, button, plan.tilt_deg, need_stroke=not depth)
    if plan2 is not None:
        if move:
            res = runner.run([line_leg(ctx, "reach:re-aim", move, plan2.quat, line_speed, free_guard(ctx, at_look),
                                       line_world)], execute=True)
            if any(not r.ok for r in res):
                print("[press_demo] STOP: the move to the aimed button's look pose %s — the arm holds; "
                      "nothing was pressed" % _stopped_by(res))
                return 1
    else:
        plan2, why = plan_reach(ctx, cfg, button, ctx.client.joints())
        if plan2 is None:
            _go_home(ctx, cfg, runner, "STOP: no reach-out press for the aimed button (as straight lines: %s; planned: %s)"
                     % (why_line, why))
            return 1
        res = runner.run([plan2.look_leg], execute=True)
        if any(not r.ok for r in res):
            return 1
    if not soft:
        at_look = _settled(ctx, "the look pose")
        if at_look is None:
            return 1
    # With the depth press the slide in also turns the hand straight down,
    # its pads ending where the tilted fingertip would: one walk, flown as
    # the slide is (line_speed, the slide's guard), never lower than it
    # (tasks/depth_press walk_in_down). Where no such walk flies, the slide,
    # then the TURN.
    walk, turned = None, None
    if depth:
        from rammp_box_opening.tasks.depth_press import walk_in_down

        walk, turned, why_walk = walk_in_down(ctx, plan2.start_xyz, plan2.quat)
        if walk is None:
            print("[press_demo] the slide in does not turn the hand down on the way (%s) — sliding, then turning" % why_walk)
    if walk is not None:
        slide, quat_in = walk, turned
    else:
        slide, why = _line_from(ctx, plan2.slide_pts, plan2.start_xyz, plan2.quat)
        if not slide:
            _go_home(ctx, cfg, runner, "STOP: the slide in to the button is not a straight line the arm can fly (%s)" % (why or "no length"))
            return 1
        quat_in = plan2.quat
    res = runner.run([line_leg(ctx, "reach:slide-in", slide, quat_in, line_speed, free_guard(ctx, at_look),
                               line_world)], execute=True)
    if any(not r.ok for r in res):
        print("[press_demo] STOP: the slide in %s before the button — the arm holds; nothing was pressed"
              % _stopped_by(res))
        return 1
    at_start = None if soft else _settled(ctx, "the stroke's start" if walk is None
                                          else "over the button, pointing down")
    if at_start is None and not soft:
        return 1
    runner.note("reach_slid_in", turned_down=walk is not None, residual_nm=None if soft else [
        round(v, 3) for v in residual(_wrist_gravity(), at_look, at_start)])
    if depth:
        # over the box: from here the height is the wrist camera's depth and
        # the press the torques' (tasks/depth_press), not TF's lid — the
        # look's aim where the close-up does not confirm the button
        from rammp_box_opening.tasks import depth_press

        result = depth_press.press_down_from_here(node, ctx, cfg, runner, watcher, CONTACT_TOOL, here=at_start,
                                                  turned=turned, knob_up_mm=up_mm, aimed=button, compliant=soft,
                                                  top_z=button[2])
        result = depth_press.after_scene_check(ctx, runner, result, scene_fix, locator)
        homed = result.free and _home_from_over_box(ctx, cfg, runner, result)
        runner.finish()
        return depth_press.finish(ctx, result, homed)

    def stop_after_back_out(why, code=1):
        print("[press_demo] STOP: %s" % why)
        if not _back_out(ctx, runner, plan2, world, at_start):
            return code
        if ctx.contact_pad > 0.0:  # something was touched: judge the box before going anywhere
            _home_after_contact(node, ctx, cfg, runner, watcher, scene_fix, locator, button, lid_before, yaw)
        else:
            _go_home(ctx, cfg, runner, "")
        return code

    stroke, why = _line_from(ctx, plan2.stroke_pts, plan2.bottom_xyz, plan2.quat)
    if not stroke:
        return stop_after_back_out("the stroke down is not a straight line the arm can fly (%s) — nothing was pressed" % (why or "no length"))
    touch = touch_leg(ctx, cfg, stroke, plan2.quat, world, at_start)
    touched = runner.run([touch], execute=True)
    if touched[-1].outcome == "touch":
        # contact happened: from here every world pads the box for a shove
        ctx.contact_pad = CONTACT_SHIFT_PAD_M
    if any(not r.ok for r in touched) or touched[-1].contact_xyz is None:
        return stop_after_back_out("the touch did not find the button (%s)" % touched[-1].detail)
    pad = contact_from_tips(touched[-1].contact_xyz, plan2.quat)
    off = pad[2] - button[2]
    verdict = touch_verdict(off)
    runner.note("touch", offset_mm=round(1000 * off, 1), verdict=verdict, reach_tilt_deg=plan2.tilt_deg,
                landed_mm=round(1000 * judge_box_moved(button, pad), 1))
    print(
        "[press_demo] TOUCH: the fingertip met the surface %+.1f mm from the aimed button top, %.1f mm from its centre"
        % (1000 * off, 1000 * judge_box_moved(button, pad))
    )
    if verdict == "foreign":
        return stop_after_back_out("the fingertip met a surface %.0f mm above the button — not this box's button" % (1000 * off))
    pressed = verdict == "press"
    if pressed:
        depth = reach_push_m(cfg, plan2.tilt_deg)
        print("[press_demo] PUSH %.1f mm (the gripper clears the lid by %.1f mm at %d deg)"
              % (1000 * depth, 1000 * REACH_LID_CLEAR_M[plan2.tilt_deg], plan2.tilt_deg))
        _R, t = approach._chain().fk(ctx.client.joints())
        down, why = _line(ctx, [float(t[0]), float(t[1]), float(t[2]) - depth], plan2.quat)
        if not down:
            return stop_after_back_out("the push is not a straight line the arm can fly (%s) — nothing was pressed" % (why or "no length"))
        # the backstop judged against the free-air reading at the stroke's
        # start: the push's first sample already carries the touch's
        # overshoot, and a backstop riding on it is no bound on the total
        _g, verify = _push_guard_and_verify(m, cfg, depth_m=depth)
        guard = GuardSpec(touch_nm=float(m.touch_nm), trip="press", gravity=_wrist_gravity(), reference=at_start)
        push = line_leg(ctx, "press:push", down, plan2.quat, cfg.press_speed, guard, world, verify)
        push.hold_s = float(cfg.push_hold_s)
        push.recoil_via = touch.traj  # the recoil carries on back up the touch
        res = runner.run([push], execute=True)  # held, then the runner's recoil reverses it
        if any(not r.ok for r in res):
            return stop_after_back_out("the push failed (%s)" % res[-1].detail)
        print("[press_demo] PRESSED — %s" % res[0].detail)
        if ctx.mission_frames is not None:
            ctx.mission_frames.note(stage="pressed")
    else:
        print("[press_demo] the knob is ALREADY up — the fingertip met it %.1f mm above the lid; not pushing it down" % (1000 * off))
    if not _back_out(ctx, runner, plan2, world, at_start):
        return 1

    # did the knob pop, did the box move — judged before the arm goes anywhere
    homed, popped, up_after = _home_after_contact(node, ctx, cfg, runner, watcher, scene_fix, locator, button, lid_before, yaw)
    if ctx.mission_frames is not None and pressed:
        ctx.mission_frames.note(popped=None if up_after is None else bool(popped))
    if not homed:
        return 1
    runner.finish()
    if not pressed:
        print("[press_demo] nothing was pressed: the knob was already up when the fingertip met it")
        return 1
    if not popped:
        print("[press_demo] STOP: pressed, but the knob did not pop%s" % (
            "" if up_after is not None else " (the wrist could not judge it)"))
        return 1
    if not args.press_only:
        print("[press_demo] the reach-out run ends after the press: the grip and the lift reach no further than straight down did")
    return 0
