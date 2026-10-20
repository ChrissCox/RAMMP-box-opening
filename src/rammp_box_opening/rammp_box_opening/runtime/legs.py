"""Legs (the unit of planning/execution) and the merge rules (spec §5, §6).

MOTION legs merge into one re-timed execution (runtime/retime.py) only when
dynamically valid: same planning chain (B planned from A's predicted
endpoint), and nothing after a contact — a guarded leg may END a group,
never sit inside one. `world` is a checked precondition, never a merge key.
"""

from dataclasses import dataclass, field
from enum import Enum


class Kind(Enum):
    MOTION = "motion"
    GRIPPER = "gripper"


@dataclass
class VerifyCtx:
    outcome: str
    gripper_pos: float = None
    progress: float = None
    torque_peak: float = None


@dataclass
class Leg:
    name: str
    kind: Kind
    # JointTrajectory, or None for a LAZY motion leg: one that follows an
    # expected touch, whose start is unknown until the guard stops the arm
    # — the Runner plans it from live joints exactly once, at execution
    traj: object
    speed: float
    guard: object  # GuardSpec | None
    world: str  # world name this leg was PLANNED against
    chain: int
    target: tuple  # ("pose", xyz, quat_xyzw) | ("joints", q7) | None
    goal_joints: list  # predicted end joints (MOTION), None for GRIPPER
    verify: object = None  # Callable[[VerifyCtx], tuple[bool, str]] | None
    gripper_cmd: float = None
    # GRIPPER legs only: start the command and carry on, joining before
    # anything that needs the fingers to have arrived. Set it where the
    # overlap is SAFE — a close during a transit — never on a release,
    # which must complete before the arm moves away from what it dropped.
    defer_join: bool = False
    # GRIPPER legs only, with defer_join: a RELEASE may be dispatched now
    # and the next motion's replan may proceed, but the arm must not move
    # away from what it dropped before the fingers have settled — the
    # Runner joins it right before that motion executes
    join_before_motion: bool = False
    # GRIPPER legs only, with defer_join: dispatch the command as soon as
    # the PRECEDING motion group starts flying, not after it arrives — the
    # fingers move while the arm moves. Only for a command that is safe at
    # every point of that motion (an open during a retreat that starts
    # clear of the box); the usual join still gates the next guarded leg.
    send_with_previous_motion: bool = False
    world_path: str = None  # generated world YAML to push (SetWorld wants a path)
    # MOTION legs planned with a vertical approach: (index into traj.points,
    # joints) of the waypoint the final straight line starts from, at rest.
    # A leg that has one can be re-fitted to a corrected target without a
    # planner round trip (runtime/approach.refit_descent), which moves it
    # to above the new target.
    waypoint: tuple = None
    # MOTION legs with a guard: index into traj.points where the final
    # straight DESCENT begins (the waypoint's). Flown on its own, the leg's
    # guard takes its baseline and arms there plus a settle
    # (runner._run_motion): everything before it is free-air approach,
    # whose braking into the waypoint read as a 3 Nm "touch" three inches
    # above the box (bench 2026-09-21).
    guard_from: int = None
    # MOTION legs with a "press" guard: after a good push (stop met or bound
    # run), hold still on the button this long before the recoil lets go —
    # a latch may need a moment held down (2026-09-23: a push that met its
    # stop and let go at once left the box shut)
    hold_s: float = 0.0
    # MOTION legs only: polled while the leg flies; the first True cancels it
    # and the leg reads as 'stopped' — a SEARCH, which exists to find
    # something rather than to arrive (the look and the sweep).
    stop_when: object = None
    # Planning cost, for the preview table. plan_s is the client's round
    # trip; plan_server_s is what the planner reports it spent solving.
    # The gap between them is action/transport overhead — worth watching:
    # off-bench the solve measures ~0.21 s while live runs showed ~1.0 s
    # per plan, and only these two numbers side by side say which half.
    plan_s: float = field(default=None, compare=False)
    plan_server_s: float = field(default=None, compare=False)
    # (fast_scale, slow_scale, slow_path_fraction) when a guarded descent
    # was time-warped; display only — leg.speed is 1.0 once it is baked in
    warp: tuple = field(default=None, compare=False)
    # A touch stroke expects contact at this fraction of its own PATH, and
    # judges the trip against the matching TIME fraction of whatever
    # trajectory actually flies: `retime(traj)` recomputes that, and is
    # called on every replan, warp and merge (primitives.core.press_stroke)
    contact_path_frac: float = field(default=None, compare=False)
    retime: object = field(default=None, compare=False)


def can_merge(a, b):
    """May `b` join the execution `a` ends?

    A guarded leg may be the TAIL of a group: the approach and the descent
    it leads into fly as one re-timed trajectory, with the guard armed from
    the point the descent begins (runtime/runner.py _run_motion). That
    removes the controller round trip and the full stop between them — the
    pause a person does not make when reaching for something. Nothing ever
    merges AFTER a guarded leg: a trip must not strand queued motion."""
    return (
        a.kind is Kind.MOTION
        and b.kind is Kind.MOTION
        and a.chain == b.chain
        # speeds may differ: the re-timer builds ONE profile for the group,
        # each leg's speed becoming its cruise fraction (retime.py)
        # a planned lead never merges with a lazy tail (or vice versa):
        # a group profile cannot chain a trajectory that does not exist yet
        and (a.traj is None) == (b.traj is None)
        # a contact ENDS its execution: never merge past one
        and a.guard is None
        and a.verify is None  # a verify CLOSES its merge group
        # ...and a verify cannot be ABSORBED into one either: only the
        # execution's OWNING member is verified, so appending a
        # verify-carrying leg as a non-lead member silently discarded its
        # check (review 2026-08-28). A guarded tail IS the owner
        # (_run_motion picks it), so its verify still runs.
        and (b.verify is None or b.guard is not None)
        # a leg whose timing is already baked in (a warped descent: its
        # fast-then-slow profile and speed 1.0) cannot join a group
        # profile — re-timing it would discard the warp and cruise it into
        # contact at full speed
        and b.warp is None
        and a.warp is None
    )


def merge_groups(legs):
    groups = []
    for leg in legs:
        if groups and can_merge(groups[-1][-1], leg):
            groups[-1].append(leg)
        else:
            groups.append([leg])
    return groups
