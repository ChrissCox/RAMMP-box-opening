"""The primitives (spec §5): plan/execute split, guarded descents shared.

Each primitive chain-plans from `state.joints` (tour_demo chaining). A
contact leg invalidates downstream pre-plans: it increments the chain, so
nothing merges across it, and what follows it is LAZY — planned by the
Runner from the live arm once the guard has stopped it.

Ctx.last_pose / last_world track the most recent commanded tool pose and
the world it was planned against, so pose-relative primitives (Lift,
Retreat) need no pose argument of their own.
"""

import time
from dataclasses import dataclass

from rammp_box_opening.constants import (
    JOINTS,
    JOINT_ARC_PER_M,
    JOINT_VMAX,
    TCP_OFFSET_M,
    TIP_TO_TOOL_M,
    CONTACT_SPEED,
    GRIPPER_CMD_OPEN,
    HOME,
    TRANSIT_SPEED,
)
from rammp_box_opening.models.container import from_container
from rammp_box_opening.runtime.guards import (
    GuardSpec,
    in_band,
    time_fraction_at_path_fraction,
)
from rammp_box_opening.runtime.legs import Kind, Leg

# a set-down's success IS the guard trip — overdrive the commanded depth
# past nominal surface contact so the table is always felt (err-TALL world
# modeling can otherwise leave an exact-height target arriving untouched)
SETDOWN_OVERDRIVE_M = 0.005

# clearance between the CARRIED lid's underside and the container top
# during the place transit — the planner cannot model a held object
CARRY_CLEAR_M = 0.04

# extra container xy half-extent in FULL worlds planned after a contact
# leg: a press can scoot the box off its detected pose (field 2026-09-01,
# ~2 cm at 8.1 Nm) and a transit must not thread the needle beside a
# cuboid the box may no longer be inside
CONTACT_SHIFT_PAD_M = 0.03


@dataclass
class Ctx:
    model: object
    cpose: object
    client: object
    worlds: object
    lid_at: object = None  # set after Place(lid): later worlds carry the lid
    lid_drop: object = None  # runtime-resolved drop spot (adapts to the box)
    config_path: str = None
    last_pose: tuple = None  # (xyz, quat_xyzw) of the last commanded pose
    last_world: tuple = None  # (name, path) of the last interaction world
    contact_pad: float = 0.0  # container xy padding once contact has happened
    mission_frames: object = None  # this run's detection record (detection_set.MissionFrames), --execute missions


@dataclass
class PlanState:
    joints: list  # predicted joints the next leg plans from; None = lazy
    chain: int  # bumped by every contact leg: nothing merges across it


def tcp_z(z):
    """Commanded tool_frame z for a FINGERTIP target z.

    The fingertips reach TCP_OFFSET_M beyond the frame the planner is
    commanded in, so a target meant for them is commanded that much
    higher. Everything fingertip-referenced goes through here: the press
    stroke, the grip descent, the lid set-down and the carry hover — they
    must shift TOGETHER, or the geometry between grabbing the lid and
    putting it down stops matching."""
    return float(z) + TCP_OFFSET_M


def band_verify(band):
    """Grip-band check with an honest fallback when no gripper state exists."""

    def verify(ctx):
        if ctx.gripper_pos is None:
            return True, "band unchecked — no gripper state"
        ok = in_band(ctx.gripper_pos, band)
        return ok, "grip %.3f vs band %s" % (ctx.gripper_pos, list(band))

    return verify


def _full_world(ctx, tag=""):
    tag = tag or ("lid" if ctx.lid_at is not None else "")
    return ctx.worlds.push_name(
        "full",
        model=ctx.model,
        cpose=ctx.cpose,
        lid_at=ctx.lid_at,
        tag=tag,
        container_pad_xy=ctx.contact_pad,
    )


def _interaction_world(ctx, contact_z, depth_max, tag):
    return ctx.worlds.push_name(
        "interaction",
        model=ctx.model,
        cpose=ctx.cpose,
        contact_z=contact_z,
        depth_max=depth_max,
        lid_at=ctx.lid_at,
        tag=tag,
    )


def descent_index(plan):
    """Index into a plan's trajectory where its vertical final descent
    begins (its approach waypoint), or None for a plan without one."""
    wp = getattr(plan, "waypoint", None)
    return None if wp is None else int(wp[0])


def _plan_motion(
    ctx,
    state,
    name,
    target,
    world,
    speed,
    guard=None,
    invalidates=False,
    verify=None,
    lazy=False,
):
    world_name, world_path = world
    kind, *rest = target
    if lazy or state.joints is None:
        # LAZY: this leg follows an expected touch, so its start is unknown
        # until the guard stops the arm. Pre-planning it was pure waste —
        # the post-touch replan discarded it every run (audit 2026-09-02).
        # The Runner plans it from live joints, once, in the leg's world.
        # Everything chained after it is lazy too (its end is unknown).
        if kind == "pose":
            ctx.last_pose = (list(rest[0]), list(rest[1]))
        leg = Leg(
            name=name,
            kind=Kind.MOTION,
            traj=None,
            speed=speed,
            guard=guard,
            world=world_name,
            world_path=str(world_path),
            chain=state.chain,
            target=target,
            goal_joints=None,
            verify=verify,
        )
        next_chain = state.chain + 1 if invalidates else state.chain
        return leg, PlanState(joints=None, chain=next_chain)
    # Worlds are a PLAN-time concern (spec §6): the planner must hold this
    # leg's world BEFORE the plan is requested — SetWorld only at execution
    # time means every trajectory was actually planned against the previous
    # world (2026-08-24 review, critical). The client deduplicates pushes of
    # the world it already holds (one tracker, review 2026-09-02).
    ok, msg = ctx.client.set_world(world_path)
    if not ok:
        raise RuntimeError("set_world before planning %s failed: %s" % (name, msg))
    t_plan = time.monotonic()

    if kind == "pose":
        offset = float(rest[2]) if len(rest) > 2 else 0.0
        plan = ctx.client.plan_to_pose(
            rest[0], rest[1], state.joints, approach_offset_m=offset
        )
        ctx.last_pose = (list(rest[0]), list(rest[1]))
    else:
        plan = ctx.client.plan_to_joints(rest[0], state.joints)
    plan_s = time.monotonic() - t_plan
    if plan is None or not plan.success:
        raise RuntimeError(
            "planning failed for %s: %s"
            % (name, getattr(plan, "message", "no response"))
        )
    if "PLANNER descent" in str(getattr(plan, "message", "")):
        # the straight final stretch was refused and the planner's own
        # (bowing) plan flies instead — say so, a press may land off-centre
        print("  NOTE %s: %s" % (name, plan.message))
    end = list(plan.trajectory.points[-1].positions)
    leg = Leg(
        name=name,
        kind=Kind.MOTION,
        traj=plan.trajectory,
        speed=speed,
        guard=guard,
        world=world_name,
        world_path=str(world_path),
        chain=state.chain,
        target=target,
        goal_joints=end,
        verify=verify,
        plan_s=plan_s,
        plan_server_s=getattr(plan, "planning_time", None),
        waypoint=getattr(plan, "waypoint", None),
        guard_from=descent_index(plan),
    )
    if guard is not None:
        # from here on the box may not be exactly where it was detected —
        # every later FULL world allows for a contact-shifted container
        ctx.contact_pad = CONTACT_SHIFT_PAD_M
    next_chain = state.chain + 1 if invalidates else state.chain
    return leg, PlanState(joints=end, chain=next_chain)


def _gripper_leg(
    ctx,
    state,
    name,
    cmd,
    world,
    verify=None,
    defer_join=False,
    join_before_motion=False,
    send_with_previous_motion=False,
):
    world_name, world_path = world
    return Leg(
        name=name,
        kind=Kind.GRIPPER,
        traj=None,
        speed=0.0,
        guard=None,
        world=world_name,
        world_path=str(world_path),
        chain=state.chain,
        target=None,
        goal_joints=None,
        gripper_cmd=cmd,
        verify=verify,
        defer_join=defer_join,
        join_before_motion=join_before_motion,
        send_with_previous_motion=send_with_previous_motion,
    )


class Lift:
    """Planned ascent by dz from the last commanded pose. No grip-band check
    here: a verify would close the lift's merge group and cost a dead stop
    before the carry — the slip check sits at the set-down (Place)."""

    def __init__(self, dz, name="lift", speed=CONTACT_SPEED):
        self.dz = float(dz)
        self.name = name
        self.speed = float(speed)

    def plan(self, ctx, state):
        if ctx.last_pose is None:
            raise RuntimeError("lift needs a preceding pose-directed leg")
        xyz, quat = ctx.last_pose
        target = [xyz[0], xyz[1], xyz[2] + self.dz]
        world = ctx.last_world or _full_world(ctx)
        leg, state = _plan_motion(
            ctx, state, self.name, ("pose", target, list(quat)), world, self.speed
        )
        return [leg], state


class Place:
    """Transit above the pose (+ lid-height margin — the planner cannot
    model a held object), guarded descent where a trip = set-down, then
    open the gripper (spec §5, §6)."""

    def __init__(self, target_xyz, quat, name="place", speed=None, touch_nm=None, band=None):
        self.target_xyz = list(target_xyz)
        self.quat = list(quat)
        self.name = name
        # grip band to re-check at the bottom, right before the release:
        # the LAST moment at which still holding the thing is checkable,
        # and the moment it matters. The lift used to carry this check,
        # which cost a dead stop between the lift and the carry every run
        # (they are otherwise one continuous motion).
        self.band = None if band is None else tuple(band)
        # descent speed; None keeps the conservative contact default for
        # callers that predate the config knob (tests, isolated CLI use)
        self.speed = CONTACT_SPEED if speed is None else float(speed)
        # set-down trip threshold; None keeps the model's press threshold.
        # A lid touching a table loads the wrist far less than a press
        # pops a seal — at 7.0 the guard stayed blind through a 10 mm
        # crunch at the slid drop spot (field 2026-09-02).
        self.touch_nm = touch_nm

    @staticmethod
    def hover_for(ctx, target_xyz):
        """The carry pose above a set-down target: hover standoff plus a
        lid height, raised to the carry floor — the carried lid hangs a
        lid-height below the fingertips and the planner cannot see it, so
        the carry must clear the container body even directly overhead."""
        m = ctx.model
        carry_floor = ctx.cpose.xyz[2] + m.dims[2] + m.lid_dims[2] + CARRY_CLEAR_M
        z = max(target_xyz[2] + m.hover_standoff + m.lid_dims[2], tcp_z(carry_floor))
        return [target_xyz[0], target_xyz[1], z]

    def plan(self, ctx, state):
        m = ctx.model
        hover = self.hover_for(ctx, self.target_xyz)
        full = _full_world(ctx)
        transit, state = _plan_motion(
            ctx,
            state,
            self.name + ":transit",
            ("pose", hover, self.quat),
            full,
            TRANSIT_SPEED,
        )
        world = _interaction_world(
            ctx, self.target_xyz[2], SETDOWN_OVERDRIVE_M, self.name
        )
        ctx.last_world = world
        guard = GuardSpec(
            touch_nm=m.touch_nm if self.touch_nm is None else float(self.touch_nm),
            trip="setdown",
            # contact is only possible at the stroke's very end — the
            # fast segment's dynamics must not trip the gentler set-down
            # threshold (lid released 110 mm up, field 2026-09-02)
            arm_after=0.5,
        )

        def verify(v):
            slipped = (
                self.band is not None
                and v.gripper_pos is not None
                and not in_band(v.gripper_pos, self.band)
            )
            if v.outcome == "touch":
                if v.progress is not None and v.progress < 0.5:
                    return False, (
                        "guard tripped at %.0f%% of the descent — struck "
                        "something on the way down, set-down NOT confirmed"
                        % (v.progress * 100)
                    )
                if slipped:
                    return False, (
                        "felt a surface but the fingers read %.3f, outside "
                        "%s — the lid slipped on the way here and this is "
                        "the table, not the lid" % (v.gripper_pos, list(self.band))
                    )
                peak = "" if v.torque_peak is None else " at %.1f Nm" % v.torque_peak
                return True, "surface felt%s — set down" % peak
            if v.outcome == "arrived":
                if slipped:
                    return False, (
                        "never felt the surface and the fingers read %.3f, "
                        "outside %s — the lid slipped"
                        % (v.gripper_pos, list(self.band))
                    )
                return False, (
                    "full stroke with no trip — never felt the surface, "
                    "set-down NOT confirmed"
                )
            return False, "set-down %s" % v.outcome
        # a set-down SUCCEEDS only on the touch: target exactly at surface
        # height can 'arrive' without ever feeling the table — command a
        # hair below so the guard verdict is deterministic
        down_xyz = [
            self.target_xyz[0],
            self.target_xyz[1],
            self.target_xyz[2] - SETDOWN_OVERDRIVE_M,
        ]
        descend, state = _plan_motion(
            ctx,
            state,
            self.name + ":down",
            # vertical final 50 mm: every free plan bows a little (the
            # "small arch", field 2026-09-01); a set-down comes straight
            # down onto its spot
            ("pose", down_xyz, self.quat, 0.05),
            world,
            self.speed,
            guard=guard,
            invalidates=True,
            verify=verify,
        )
        release = _gripper_leg(
            ctx,
            state,
            self.name + ":open",
            GRIPPER_CMD_OPEN,
            world,
            # dispatched at once; the retreat's post-touch replan runs while
            # the fingers open and the join lands right before that retreat
            # executes — a release still completes before the arm moves
            # away (audit 2026-09-02)
            defer_join=True,
            join_before_motion=True,
        )
        return [transit, descend, release], state


class LiftFree:
    """A plan-free STRAIGHT vertical move from the live joints: the arm's own
    kinematics (runtime/kinematics), no planner. The recovery the bringup
    guide used to call "jog clear by hand first": a pose the planner
    refuses as a start (fingers at the table after an abort or a jog) is
    lifted straight up — away from the only thing it can be in — until the
    planner accepts it. Guarded as an obstruction trip: anything met on
    the way UP is a failure, never pushed through. The attitude is held."""

    def __init__(self, dz, name="lift-free", speed=None, touch_nm=4.0):
        self.dz = float(dz)
        self.name = name
        self.speed = CONTACT_SPEED if speed is None else float(speed)
        self.touch_nm = float(touch_nm)

    def plan(self, ctx, state):
        from rammp_box_opening.runtime.approach import _chain, line_trajectory
        from rammp_box_opening.runtime.kinematics import mat_to_quat_xyzw

        chain = _chain()
        q0 = list(state.joints if state.joints is not None else ctx.client.joints())
        R, t = chain.fk(q0)
        quat = mat_to_quat_xyzw(R)
        end_xyz = [float(t[0]), float(t[1]), float(t[2]) + self.dz]
        pts, why = chain.straight_line(q0, end_xyz, quat)
        if pts is None:
            raise RuntimeError("plan-free lift refused: %s" % why)
        world_name, world_path = ctx.worlds.push_name("bench", model=None, tag="bare")
        guard = GuardSpec(touch_nm=self.touch_nm, trip="obstruction")
        leg = Leg(
            name=self.name,
            kind=Kind.MOTION,
            traj=line_trajectory(JOINTS, pts),
            speed=self.speed,
            guard=guard,
            world=world_name,
            world_path=str(world_path),
            chain=state.chain,
            target=("pose", end_xyz, list(quat)),
            goal_joints=[float(v) for v in pts[-1]],
            plan_s=0.0,
        )
        ctx.last_pose = (end_xyz, list(quat))
        return [leg], PlanState(joints=list(leg.goal_joints), chain=state.chain + 1)


class Retreat:
    """Vertical disengage by dz from the last commanded pose. Planned
    against the interaction world (a full-world plan would start inside
    the container cuboid after contact). Lazy after a touch: the Runner
    plans it from live once the guard has stopped the arm, and a failed
    post-touch replan stops the mission with the arm holding — there is
    no plan-free fallback."""

    def __init__(self, dz, name="retreat", speed=CONTACT_SPEED, lazy=False):
        self.dz = float(dz)
        self.name = name
        self.speed = float(speed)
        self.lazy = lazy  # follows an expected touch: planned at execution

    def plan(self, ctx, state):
        if ctx.last_pose is None:
            raise RuntimeError("retreat needs a preceding pose-directed leg")
        xyz, quat = ctx.last_pose
        target = [xyz[0], xyz[1], xyz[2] + self.dz]
        world = ctx.last_world or _full_world(ctx)
        leg, state = _plan_motion(
            ctx,
            state,
            self.name,
            ("pose", target, list(quat)),
            world,
            self.speed,
            lazy=self.lazy,
        )
        return [leg], state


# The last stretch of every press comes STRAIGHT down this far: a diagonal
# descent touches the button before its lateral convergence finishes (10 mm
# off-centre at 20 mm height from a 199 mm start — the edge presses of
# 2026-09-01). runtime/approach.py builds it as the arm's own straight line.
PRESS_APPROACH_M = 0.06


def press_stroke(ctx, state, cfg, name="press:down"):
    """The TOUCH: one guarded stroke from staging toward travel_m below the
    button, stopping at first contact (cfg.contact_nm). Returns (leg, state).

    A single hard stroke pressed the button flush and then compressed the
    whole container until the torque built up (video, 2026-09-03): a force
    the button can only produce by bottoming out must not be what ends the
    stroke. So this stage only FINDS the surface — the arm's fingertip TF
    at the trip is the button's true top — and press_push completes the
    press a bounded distance from there.

    A trip counts only near where contact is EXPECTED: one well above the
    button struck something else, and full travel with no trip found no
    surface. Both are failures. Contact is expected after
    staging/(staging+travel) of the stroke's PATH, but execution progress
    is a TIME fraction and the two differ on any real velocity profile —
    so the expectation lives in leg.retime, re-evaluated on whatever
    trajectory actually flies (a replan, a merged group's profile)."""
    m = ctx.model
    button = from_container(ctx.cpose, m.button_offset)
    quat = m.press_quat(button)
    world = _interaction_world(ctx, button[2], cfg.travel_m, "button")
    ctx.last_world = world
    guard = GuardSpec(
        touch_nm=cfg.contact_nm,
        trip="touch",
        # a light threshold must not judge the launch transient; a warped
        # caller raises this to its slow-zone rebaseline (_apply_warp)
        arm_after=0.25,
    )
    expect = {"frac": 1.0}  # TIME fraction; set by retime below

    def verify(v):
        expected = expect["frac"]
        if v.outcome == "touch":
            if v.progress is not None and v.progress < expected - 0.15:
                return False, (
                    "guard tripped EARLY at %.0f%% of the stroke (contact "
                    "expected ~%.0f%%) — struck something above the button"
                    % (v.progress * 100, expected * 100)
                )
            peak = "" if v.torque_peak is None else " at %.1f Nm" % v.torque_peak
            return True, "surface found%s" % peak
        if v.outcome == "arrived":
            return False, (
                "no surface within %.0f mm below the estimated button — the "
                "fix is high or the box moved" % (cfg.travel_m * 1000)
            )
        return False, "touch %s" % v.outcome

    # press_offset_xy: a base-frame trim for where the closed pads actually
    # meet the lid relative to the tool axis (a jammed-and-freed gripper can
    # leave a few mm of offset — the 2026-09-03 miss and hit were the same
    # stroke aimed 1.5 mm apart). Zero by default; dialled at the bench.
    target = [
        button[0] + cfg.press_offset_xy[0],
        button[1] + cfg.press_offset_xy[1],
        tcp_z(button[2] - cfg.travel_m),
    ]
    press, state = _plan_motion(
        ctx,
        state,
        name,
        ("pose", target, quat, PRESS_APPROACH_M),
        world,
        cfg.press_speed,
        guard=guard,
        invalidates=True,
        verify=verify,
    )

    def retime(traj):
        # a replan swaps the trajectory — the expected-contact fraction
        # must follow the one actually flown (review 2026-09-02)
        expect["frac"] = time_fraction_at_path_fraction(traj, press.contact_path_frac)

    press.contact_path_frac = cfg.staging_m / (cfg.staging_m + cfg.travel_m)
    press.retime = retime
    retime(press.traj)
    return press, state


def press_push(ctx, state, cfg, contact_xyz, world, name="press:push"):
    """The PUSH stage: from the measured first contact, drive the
    fingertips button_travel_m further — position-bounded — with the
    old trip threshold left only as a backstop.

    contact_xyz is the fingertip TF the runner read when the touch
    tripped: the arm's own measurement of where the button top is, in
    the frame it is commanded in (via TIP_TO_TOOL_M) — no camera, no
    dims.z, no TCP constant in this chain. Arriving at the bound IS the
    press (the button latches inside its travel); a trip means the push
    met a stop early, which is also a press.
    """
    m = ctx.model
    button = from_container(ctx.cpose, m.button_offset)
    quat = m.press_quat(button)
    tip_z = float(contact_xyz[2])
    tool_at_contact = tip_z - TIP_TO_TOOL_M
    bottom = tool_at_contact - cfg.button_travel_m
    xy = ctx.last_pose[0][:2] if ctx.last_pose else [button[0], button[1]]
    guard, verify = _push_guard_and_verify(m, cfg)

    return _plan_motion(
        ctx,
        state,
        name,
        ("pose", [float(xy[0]), float(xy[1]), float(bottom)], quat, 0.0),
        world,
        cfg.press_speed,
        guard=guard,
        invalidates=True,
        verify=verify,
    )


def _push_guard_and_verify(m, cfg):
    guard = GuardSpec(touch_nm=max(0.5, m.touch_nm - cfg.contact_nm), trip="press")

    def verify(v):
        mm = cfg.button_travel_m * 1000
        if v.outcome == "touch":
            peak = "" if v.torque_peak is None else " at %.1f Nm" % v.torque_peak
            return True, "pressed — met a stop%s inside the %.0f mm push" % (peak, mm)
        if v.outcome == "arrived":
            return True, "pressed — full %.0f mm push from the measured contact" % mm
        return False, "push %s" % v.outcome

    return guard, verify


def press_push_from_touch(ctx, state, cfg, touch_leg, live, contact_xyz, world):
    """The PUSH cut from the touch's own trajectory: its unexecuted
    continuation past the stop is a validated path straight on through
    the contact, so the push needs no planner call while the fingers sit
    on the button (the planner round trip was ~0.5 s of pressing at
    contact_nm, 2026-09-04). Re-timed like any motion at press_speed.
    Returns (leg, state) or None when too little of the stroke remains —
    the caller then plans it (press_push)."""
    from rammp_box_opening.runtime.retime import (
        RetimeParams,
        forward_tail,
        positions_to_traj,
        retime_group,
    )

    m = ctx.model
    if touch_leg.traj is None:
        return None
    path = forward_tail(touch_leg.traj, live, cfg.button_travel_m * JOINT_ARC_PER_M)
    if path is None:
        return None
    traj, _ = retime_group(
        [positions_to_traj(touch_leg.traj.joint_names, path)],
        [cfg.press_speed],
        JOINT_VMAX,
        RetimeParams(),
    )
    tool_at_contact = float(contact_xyz[2]) - TIP_TO_TOOL_M
    guard, verify = _push_guard_and_verify(m, cfg)
    button = from_container(ctx.cpose, m.button_offset)
    quat = m.press_quat(button)
    xy = ctx.last_pose[0][:2] if ctx.last_pose else [button[0], button[1]]
    world_name, world_path = world
    leg = Leg(
        name="press:push",
        kind=Kind.MOTION,
        traj=traj,
        speed=1.0,  # the profile is baked in
        guard=guard,
        world=world_name,
        world_path=str(world_path),
        chain=state.chain,
        # a replan (drift at execution) falls back to the planned push
        target=("pose", [float(xy[0]), float(xy[1]), tool_at_contact - cfg.button_travel_m], quat, 0.0),
        goal_joints=[float(v) for v in path[-1]],
        verify=verify,
    )
    ctx.last_pose = ([float(xy[0]), float(xy[1]), tool_at_contact - cfg.button_travel_m], list(quat))
    return leg, PlanState(joints=list(path[-1]), chain=state.chain + 1)


class Home:
    """Return to the rest joints (factory HOME, or the look pose when the
    mission rests tool-down) via plan_to_joints (spec §5). Lazy when it
    follows a lazy retreat (its start is unknown until then)."""

    def __init__(self, joints=None):
        self.joints = list(HOME if joints is None else joints)

    def plan(self, ctx, state):
        world = _full_world(ctx)
        leg, state = _plan_motion(
            ctx, state, "home", ("joints", list(self.joints)), world, TRANSIT_SPEED
        )
        return [leg], state
