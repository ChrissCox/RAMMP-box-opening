"""The vertical final approach: a planner waypoint, then the arm's own line.

cuRobo's approach_offset_m (RAMMP-CuRobo feat/perceived-world 21b1ba0)
never reached a release: the v1.0.0 PlanToPose carries a target and
start_joints and nothing else. Its job stands — the last approach_offset_m
metres of every contact descent must come straight down onto the goal,
because a diagonal descent touches the surface before its lateral
convergence finishes (edge presses, field 2026-09-01).

Two plans were the first answer (one to a waypoint approach_offset_m above
the goal, one from there to the goal), and they missed: a plan between
two vertically aligned poses is vertical only at its ends. Its joint-space
path bowed 5-7 mm radially outward, widest 26-29 mm above the goal, which
is exactly where the guard trips (travel_m + TCP_OFFSET_M above it); every
press since the migration landed 5.6-6.6 mm off the button, and the
2026-09-16 probe on the live planner reproduced each miss to a millimetre.

So the waypoint stays a planner result (collision-checked, exact to
0.01 mm) and the stretch below it is built from the arm's own kinematics
(runtime/kinematics.py): a straight line in 2 mm steps, the attitude held,
0.01 mm from vertical. cuRobo accepts the line's end joints as a valid
target (checked live). Should the line be refused (a joint limit, a
singularity), the planner's own plan flies instead and the message says so.

They fly as ONE trajectory with a rest at the waypoint (every cuRobo plan
starts and ends at rest). An approach action in cuRobo v1.1.0 would
remove the stop.

The line carries a REAL velocity profile (line_trajectory). It used to
carry placeholder stamps — 0.05 s a step, zero velocity at every waypoint
— on the belief that the Runner re-times every leg. It re-times unguarded
and merged groups; a guarded descent flown on its own (the press from
staging, grip:down, the plan-free lift) went to the driver as it was,
and the arm stop-started every 2 mm: 2.98 s for the 45 mm grip descent
and 3 Nm of torque ripple in free air on the live run of 2026-09-17.
"""

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rammp_box_opening.constants import JOINT_VMAX
from rammp_box_opening.runtime.retime import RetimeParams, positions_to_traj, retime_group
from rammp_box_opening.runtime.stamps import secs, set_stamp

_CHAIN = None


def _chain():
    """The arm's kinematic chain, parsed once from the driver's URDF."""
    global _CHAIN
    if _CHAIN is None:
        from rammp_box_opening.runtime.kinematics import ArmChain

        _CHAIN = ArmChain()
    return _CHAIN


def line_trajectory(joint_names, points):
    """A JointTrajectory through joint-space `points` with a rest-to-rest
    velocity profile at full cruise (positions untouched, runtime/retime):
    exactly what a planner trajectory is, so a leg's speed dilates it, a
    warp reshapes it and a merged group re-profiles it like any other."""
    traj, _info = retime_group(
        [positions_to_traj(joint_names, points)], [1.0], JOINT_VMAX, RetimeParams()
    )
    return traj


class ChainedPlan:
    """A PlanToPose-shaped result for the two plans together. `waypoint` is
    (index into trajectory.points, joints) where the straight line begins."""

    def __init__(self, success, message, trajectory=None, planning_time=0.0, waypoint=None):
        self.success = success
        self.message = message
        self.trajectory = trajectory
        self.planning_time = planning_time
        self.waypoint = waypoint


def _point(p, t):
    q = JointTrajectoryPoint()
    q.positions = list(p.positions)
    q.velocities = list(p.velocities)
    q.accelerations = list(p.accelerations)
    set_stamp(q.time_from_start, t)
    return q


def chain(first, second):
    """`first` then `second` as one trajectory. `second` opens at `first`'s
    final configuration, so its opening point is dropped: time stays
    strictly increasing through the waypoint."""
    out = JointTrajectory()
    out.joint_names = list(first.joint_names)
    for p in first.points:
        out.points.append(_point(p, secs(p.time_from_start)))
    offset = secs(first.points[-1].time_from_start)
    for p in second.points[1:]:
        out.points.append(_point(p, secs(p.time_from_start) + offset))
    return out


def plan_with_vertical_approach(plan_pose, xyz, quat_xyzw, start_joints, offset_m, chain_model=None):
    """Plan to `xyz`/`quat_xyzw` from `start_joints`, arriving from a
    waypoint `offset_m` straight above the goal along a straight line.

    `plan_pose(xyz, quat_xyzw, start_joints)` is one PlanToPose round trip
    returning a result (success, message, trajectory, planning_time) or
    None when the planner did not answer. `chain_model` (an ArmChain, or
    anything with straight_line(q, xyz, quat) -> (points, why)) defaults
    to the driver URDF's chain."""
    if offset_m <= 0.0:
        return plan_pose(xyz, quat_xyzw, start_joints)
    above = [xyz[0], xyz[1], xyz[2] + float(offset_m)]
    first = plan_pose(above, quat_xyzw, start_joints)
    if first is None:
        return None
    if not first.success:
        return ChainedPlan(
            False,
            "approach waypoint %.3f m above the goal: %s" % (offset_m, first.message),
        )
    end = list(first.trajectory.points[-1].positions)
    kin = chain_model if chain_model is not None else _chain()
    pts, why = kin.straight_line(end, xyz, quat_xyzw)
    if pts is not None:
        return ChainedPlan(
            True,
            "straight descent: %d points over %.0f mm" % (len(pts), 1000.0 * offset_m),
            chain(first.trajectory, line_trajectory(first.trajectory.joint_names, pts)),
            float(first.planning_time or 0.0),
            waypoint=(len(first.trajectory.points) - 1, list(end)),
        )
    # the planner's own plan below the waypoint — it bows (see above), so
    # the message carries the reason the line was refused
    second = plan_pose(xyz, quat_xyzw, end)
    if second is None:
        return None
    if not second.success:
        return ChainedPlan(False, "descent from the approach waypoint: %s" % second.message)
    return ChainedPlan(
        True,
        "PLANNER descent (may bow — the straight line was refused: %s)" % why,
        chain(first.trajectory, second.trajectory),
        float(first.planning_time or 0.0) + float(second.planning_time or 0.0),
        # the descent begins at the same junction either way: a guard flown
        # with this leg arms there, and a re-fit may still replace the
        # planner's bowing descent with the straight line
        waypoint=(len(first.trajectory.points) - 1, list(end)),
    )


REFIT_MAX_M = 0.03  # a corrected target further than this from the planned one is re-planned


def refit_descent(leg, xyz, quat_xyzw, chain_model=None):
    """Re-fit a planned-with-vertical-approach leg to a corrected target
    `xyz` WITHOUT the planner, from the arm's own kinematics: a short level
    move at the height the leg STARTS from, to above the new target; one
    straight line down to the approach offset above it; and from there the
    same final line a planned leg ends with, from rest — the guarded part
    keeps the dynamics it has on every other press. Returns True and
    rewrites leg.traj / target / goal_joints / waypoint / guard_from, or
    False (no waypoint on the leg, the start not within REFIT_MAX_M of
    above the new target, a line refused) — the caller then re-plans.

    None of it is collision-checked: it stays within REFIT_MAX_M of the
    path the planner did check, in the air above the container.

    The correction used to be made at the WAYPOINT, keeping the planner's
    descent over the old target down to there: the arm came down over the
    wrong spot and swerved onto the button 56 mm above it — "it shifted to
    the right as it was pressing, almost missing it" (bench 2026-09-21; the
    touch itself landed 0.4 mm from the aim). Where the arm stands still
    after aiming is where a correction belongs.

    The pre-planned descent used to be discarded whenever the wrist's aim
    moved the target by more than 5 mm (the scene camera is good to about
    that), costing a planner round trip (0.4 s) at the one moment the arm
    is stopped above the box (2026-09-17)."""
    if leg.waypoint is None or leg.traj is None:
        return False
    kin = chain_model if chain_model is not None else _chain()
    _idx, q_wp = leg.waypoint
    q0 = [float(v) for v in leg.traj.points[0].positions]
    _R, t0 = kin.fk(q0)
    _R, t_wp = kin.fk(q_wp)
    _R, t_end = kin.fk(leg.traj.points[-1].positions)
    xyz = [float(v) for v in xyz]
    above = [xyz[0], xyz[1], xyz[2] + float(t_wp[2] - t_end[2])]  # the approach offset the leg was planned with
    top = [xyz[0], xyz[1], float(t0[2])]
    d = ((top[0] - t0[0]) ** 2 + (top[1] - t0[1]) ** 2) ** 0.5
    if d > REFIT_MAX_M or top[2] <= above[2]:
        return False  # not a descent from (nearly) above the new target
    pts = [q0]
    if d > 1e-4:
        pts, _why = kin.straight_line(q0, top, quat_xyzw)
        if pts is None:
            return False
    upper, _why = kin.straight_line(pts[-1], above, quat_xyzw)
    if upper is None:
        return False
    pts = list(pts) + list(upper[1:])
    lower, _why = kin.straight_line(pts[-1], xyz, quat_xyzw)
    if lower is None:
        return False
    names = list(leg.traj.joint_names)
    leg.traj = chain(line_trajectory(names, pts), line_trajectory(names, lower))
    leg.waypoint = (len(pts) - 1, [float(v) for v in pts[-1]])
    leg.guard_from = leg.waypoint[0]
    leg.goal_joints = [float(v) for v in lower[-1]]
    if leg.target and leg.target[0] == "pose":
        leg.target = ("pose", xyz, list(quat_xyzw)) + tuple(leg.target[3:])
    return True
