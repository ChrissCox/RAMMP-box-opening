"""The vertical final approach: a planner waypoint, then the arm's own
straight line (the v1.0.0 planner's PlanToPose carries no approach_offset_m,
and a plan between two aligned poses bows sideways in between)."""

import math

import numpy as np
import pytest
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rammp_box_opening.runtime.approach import plan_with_vertical_approach

QUAT = [1.0, 0.0, 0.0, 0.0]
NAMES = ["joint_%d" % i for i in range(1, 8)]
GOAL = [0.45, 0.0, 0.09]
START = [0.0] * 7


def _traj(rows, times):
    t = JointTrajectory()
    t.joint_names = list(NAMES)
    for row, tt in zip(rows, times):
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in row]
        p.velocities = [0.0] * 7
        p.accelerations = [0.0] * 7
        p.time_from_start.sec = int(tt)
        p.time_from_start.nanosec = int(round((tt - int(tt)) * 1e9))
        t.points.append(p)
    return t


def _secs(p):
    return p.time_from_start.sec + p.time_from_start.nanosec * 1e-9


class Plan:
    """A PlanToPose result: success, message, trajectory, planning_time."""

    def __init__(self, success, trajectory=None, planning_time=0.0, message="ok"):
        self.success = success
        self.message = message
        self.trajectory = trajectory
        self.planning_time = planning_time


class Planner:
    """Stands in for the PlanToPose round trip: records every request and
    answers from a script, in order."""

    def __init__(self, answers):
        self.calls = []
        self.answers = list(answers)

    def __call__(self, xyz, quat, start_joints):
        self.calls.append((list(xyz), list(quat), list(start_joints)))
        return self.answers.pop(0)


def _above():
    return Plan(True, _traj([[0.0] * 7, [0.1] * 7, [0.2] * 7], [0.0, 1.0, 2.0]), 0.2)


def _down():
    return Plan(True, _traj([[0.2] * 7, [0.25] * 7, [0.3] * 7], [0.0, 0.5, 1.0]), 0.3)


class Line:
    """Stands in for the arm chain: records the request, answers a script."""

    def __init__(self, points=None, why=None):
        self.points, self.why, self.calls = points, why, []

    def straight_line(self, q, xyz, quat):
        self.calls.append((list(q), list(xyz), list(quat)))
        return (None, self.why) if self.points is None else (self.points, None)


LINE = [[0.2] * 7, [0.25] * 7, [0.3] * 7]


def test_without_an_offset_the_goal_is_planned_directly():
    only = Plan(True, _traj([[0.0] * 7, [0.1] * 7], [0.0, 1.0]))
    planner = Planner([only])
    assert plan_with_vertical_approach(planner, GOAL, QUAT, START, 0.0) is only
    assert planner.calls == [(GOAL, QUAT, START)]


def test_an_offset_plans_to_a_waypoint_above_then_flies_the_line_from_its_end():
    planner = Planner([_above()])
    line = Line(LINE)
    got = plan_with_vertical_approach(planner, GOAL, QUAT, START, 0.06, chain_model=line)
    assert planner.calls == [([0.45, 0.0, 0.15], QUAT, START)]  # ONE planner call
    # the line starts at the waypoint plan's end joints, aimed at the goal
    assert line.calls == [([0.2] * 7, GOAL, QUAT)]
    assert got.success and "straight descent" in got.message
    assert got.planning_time == pytest.approx(0.2)


def test_the_plan_and_the_line_fly_as_one_trajectory_with_strictly_increasing_time():
    got = plan_with_vertical_approach(Planner([_above()]), GOAL, QUAT, START, 0.06, chain_model=Line(LINE))
    pts = got.trajectory.points
    assert [p.positions[0] for p in pts] == pytest.approx([0.0, 0.1, 0.2, 0.25, 0.3])
    ts = [_secs(p) for p in pts]
    assert ts[:3] == pytest.approx([0.0, 1.0, 2.0]) and all(b > a for a, b in zip(ts, ts[1:]))
    assert list(got.trajectory.joint_names) == NAMES
    assert all(len(p.velocities) == 7 and len(p.accelerations) == 7 for p in pts)  # the driver's gate


def test_a_refused_line_falls_back_to_the_planners_descent_and_says_so():
    planner = Planner([_above(), _down()])
    got = plan_with_vertical_approach(planner, GOAL, QUAT, START, 0.06, chain_model=Line(None, "a joint limit 40 mm down"))
    assert planner.calls[1][0] == GOAL and planner.calls[1][2] == [0.2] * 7
    assert got.success and "PLANNER" in got.message and "joint limit" in got.message
    assert [p.positions[0] for p in got.trajectory.points] == pytest.approx([0.0, 0.1, 0.2, 0.25, 0.3])
    assert got.planning_time == pytest.approx(0.5)


def test_a_refused_waypoint_refuses_the_approach_without_planning_the_descent():
    planner = Planner([Plan(False, message="IK_FAIL")])
    got = plan_with_vertical_approach(planner, GOAL, QUAT, START, 0.06)
    assert not got.success and "IK_FAIL" in got.message
    assert len(planner.calls) == 1


def test_a_refused_descent_refuses_the_approach():
    planner = Planner([_above(), Plan(False, message="INVALID_START")])
    got = plan_with_vertical_approach(planner, GOAL, QUAT, START, 0.06, chain_model=Line(None, "singular"))
    assert not got.success and "INVALID_START" in got.message


def test_the_real_chain_flies_straight_below_a_real_waypoint():
    """The planner's own waypoint joints for the bearing -15 deg box (live
    probe 2026-09-16): below them the trajectory must stay on the vertical
    to 0.1 mm — the planner's plan for the same stretch bowed 7 mm."""
    from rammp_box_opening.models.container import attitude_quat
    from rammp_box_opening.runtime.kinematics import ArmChain

    chain = ArmChain()
    q_ab = [0.1208, 0.5375, 3.3522, -1.8057, -0.1478, -0.8136, 0.1407]
    goal = [0.4264, -0.1151, 0.0815]
    quat = attitude_quat([180.0, 0.0, 0.0], math.atan2(goal[1], goal[0]))
    first = Plan(True, _traj([[0.0] * 7, q_ab], [0.0, 1.0]), 0.2)
    got = plan_with_vertical_approach(Planner([first]), goal, quat, [0.0] * 7, 0.06)  # the default chain
    assert got.success and "straight descent" in got.message
    tail = got.trajectory.points[2:]
    assert len(tail) >= 29
    lat = [np.hypot(*(chain.fk(p.positions)[1][:2] - np.array(goal[:2]))) for p in tail]
    assert max(lat) < 1e-4
    _R, t_end = chain.fk(tail[-1].positions)
    assert np.allclose(t_end, goal, atol=1e-4)


def test_no_answer_from_the_planner_is_no_plan():
    assert plan_with_vertical_approach(Planner([None]), GOAL, QUAT, START, 0.06) is None


def test_a_refit_corrects_sideways_at_the_top_then_descends_in_one_straight_line():
    """The aim moves the target 10 mm. The correction happens FIRST, at the
    height the leg starts from — where the arm stands still after aiming —
    and the whole descent is then one straight line onto the new target; no
    planner call (2026-09-17).

    It used to keep the planner's descent over the OLD target down to the
    waypoint and jog sideways there, 56 mm above the button: the arm came
    down over the wrong spot and swerved onto the button at the last
    moment — "shifted to the right as it was pressing, almost missing it"
    (bench 2026-09-21; the press itself landed 0.4 mm from the aim)."""
    from rammp_box_opening.models.container import attitude_quat
    from rammp_box_opening.runtime.approach import REFIT_MAX_M, refit_descent
    from rammp_box_opening.runtime.kinematics import ArmChain
    from rammp_box_opening.runtime.legs import Kind, Leg

    chain = ArmChain()
    q_wp = [0.1208, 0.5375, 3.3522, -1.8057, -0.1478, -0.8136, 0.1407]  # a live planner waypoint
    goal = [0.4264, -0.1151, 0.0815]
    quat = attitude_quat([180.0, 0.0, 0.0], math.atan2(goal[1], goal[0]))
    _R, t_wp = chain.fk(q_wp)
    up, why = chain.straight_line(q_wp, [t_wp[0], t_wp[1], t_wp[2] + 0.064], quat)
    assert up is not None, why
    q_staging = [float(v) for v in up[-1]]  # staging: 64 mm above the waypoint
    first = Plan(True, _traj([q_staging, q_wp], [0.0, 1.0]), 0.2)
    plan = plan_with_vertical_approach(Planner([first]), goal, quat, q_staging, 0.06)
    assert plan.waypoint == (1, q_wp)
    leg = Leg(name="press:down", kind=Kind.MOTION, traj=plan.trajectory, speed=0.35, guard=None, world="w",
              chain=0, target=("pose", goal, list(quat), 0.06), goal_joints=list(plan.trajectory.points[-1].positions),
              waypoint=plan.waypoint, guard_from=plan.waypoint[0])
    new_goal = [goal[0] + 0.007, goal[1] - 0.007, goal[2]]
    assert refit_descent(leg, new_goal, quat)

    tool = [chain.fk(p.positions)[1] for p in leg.traj.points]
    assert list(leg.traj.points[0].positions) == pytest.approx(q_staging)  # starts where the arm stands
    z0 = tool[0][2]
    level = [i for i, t in enumerate(tool) if abs(t[2] - z0) < 1e-4]
    jog = level[-1]
    assert level == list(range(jog + 1)) and jog >= 3  # ~10 mm in 2 mm steps, before anything else
    # the sideways correction ends over the NEW target ...
    assert np.hypot(*(tool[jog][:2] - np.array(new_goal[:2]))) < 1e-4
    # ... and from there it is straight down onto it, all the way
    assert max(np.hypot(*(t[:2] - np.array(new_goal[:2]))) for t in tool[jog:]) < 1e-4
    zs = [t[2] for t in tool[jog:]]
    assert all(b < a for a, b in zip(zs, zs[1:]))
    assert np.allclose(tool[-1], new_goal, atol=1e-4)
    assert np.allclose(chain.fk(leg.goal_joints)[1], new_goal, atol=1e-4)
    assert leg.target[1] == pytest.approx(new_goal)
    # the final descent is the one every planned leg ends with: from REST,
    # the approach offset above the (new) target — and the guard arms from there
    idx, q_above = leg.waypoint
    assert leg.guard_from == idx > jog
    assert tool[idx][2] == pytest.approx(new_goal[2] + (t_wp[2] - goal[2]), abs=1e-4)
    assert t_wp[2] - goal[2] == pytest.approx(0.06, abs=1e-3)
    assert list(leg.traj.points[idx].positions) == pytest.approx(list(q_above))
    assert max(abs(v) for v in leg.traj.points[idx].velocities) < 1e-9
    ts = [_secs(p) for p in leg.traj.points]
    assert all(b > a for a, b in zip(ts, ts[1:]))
    # too far: not refitted, the caller re-plans
    assert not refit_descent(leg, [goal[0] + REFIT_MAX_M + 0.01, goal[1], goal[2]], quat)
    leg.waypoint = None
    assert not refit_descent(leg, new_goal, quat)


def test_the_line_carries_a_real_velocity_profile_not_placeholder_stamps():
    """A guarded descent flown on its own reaches the driver as planned (the
    Runner re-times only unguarded and merged groups). With placeholder
    stamps and zero velocity at every 2 mm waypoint the arm stop-started
    down the whole line: 2.98 s for the 45 mm grip descent and 3 Nm of
    free-air torque ripple, live 2026-09-17. The line is a rest-to-rest
    profile now, and passes the gates in front of the driver as it is."""
    from rammp_box_opening.constants import JOINT_VMAX
    from rammp_box_opening.runtime import driver
    from rammp_box_opening.runtime.approach import line_trajectory

    pts = [[0.002 * 4.0 * k] * 7 for k in range(31)]  # 60 mm of line at ~4 rad/m
    line = line_trajectory(NAMES, pts)
    assert [list(p.positions) for p in line.points] == pts  # positions untouched
    speeds = [max(abs(v) for v in p.velocities) for p in line.points]
    assert speeds[0] == 0.0 and speeds[-1] == 0.0  # rest to rest
    assert all(v > 0.0 for v in speeds[1:-1])  # and nowhere in between
    assert driver.refusal(line, pts[0], JOINT_VMAX) is None
    # chained behind a planner trajectory, the whole still passes, slowed
    got = plan_with_vertical_approach(Planner([_above()]), GOAL, QUAT, START, 0.06, chain_model=Line(pts))
    flown = driver.dilate(got.trajectory, 0.35)
    ts = [_secs(p) for p in flown.points]
    assert all(b > a for a, b in zip(ts, ts[1:]))


def test_every_vertical_approach_says_where_its_descent_begins():
    """The guard of a descent flown on its own arms where the descent
    begins (runner._run_motion). Both ways of building one say where that
    is — the straight line, and the planner's own descent when the line is
    refused — and a re-fit moves it past the lateral jog it inserts."""
    got = plan_with_vertical_approach(Planner([_above()]), GOAL, QUAT, START, 0.06, chain_model=Line(LINE))
    assert got.waypoint == (2, [0.2] * 7)
    fallback = plan_with_vertical_approach(
        Planner([_above(), _down()]), GOAL, QUAT, START, 0.06, chain_model=Line(None, why="a joint limit")
    )
    assert fallback.success and "PLANNER descent" in fallback.message
    assert fallback.waypoint == (2, [0.2] * 7)
