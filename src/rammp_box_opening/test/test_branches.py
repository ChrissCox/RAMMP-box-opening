"""Which way round a continuous joint is written (runtime/branches).

Bench 2026-09-25: the arm turned joint_3 a full circle (4.7 s) on its way
to a box on the left. At HOME joint_3 sits at 3.142, a hair past +pi, and
the transport reports it wrapped: -3.1411 that day. The planner, which
bounds the joint at +/-6.0 rad and plans between the numbers it is given,
went from -3.14 to the +3.x its IK solution sat on — the long way round."""

import math

import numpy as np
import pytest

from rammp_box_opening.constants import HOME
from rammp_box_opening.runtime.branches import on_branch, planner_branch

HOME_AS_REPORTED = [-1.17e-05, 0.2621, -3.1411412, -2.2691, -3.2e-05, 0.95997, 1.5710226]  # /joint_states, 2026-09-25


def test_the_planner_is_handed_home_on_its_own_side_of_the_wrap():
    q = planner_branch(HOME_AS_REPORTED)
    assert q[2] == pytest.approx(HOME[2], abs=2e-3)  # -3.1411 -> +3.1421: one angle, the planner's way round
    assert q[0] == pytest.approx(HOME_AS_REPORTED[0]) and q[6] == pytest.approx(HOME_AS_REPORTED[6])
    assert q[1] == HOME_AS_REPORTED[1] and q[3] == HOME_AS_REPORTED[3]  # bounded joints are never touched
    for a, b in zip(q, HOME_AS_REPORTED):  # the same physical angles
        assert math.isclose(math.cos(a), math.cos(b), abs_tol=1e-9) and math.isclose(math.sin(a), math.sin(b), abs_tol=1e-9)
    # a reading already on HOME's side is left alone
    assert planner_branch(HOME) == pytest.approx(HOME)
    # a branch the planner's bounds (+/-6.0) cannot hold is left as reported
    far = list(HOME)
    far[2] = -0.1  # joint_3 near 0: HOME's branch would be 6.18
    assert planner_branch(far)[2] == pytest.approx(-0.1)


def test_a_live_reading_lines_up_with_a_planned_path_past_pi():
    path_q3 = 3.25  # a press that took joint_3 past +pi
    reported = path_q3 - 2 * math.pi  # -3.033, as the transport reports it
    ref = list(HOME)
    ref[2] = 3.2
    live = list(HOME)
    live[2] = reported
    assert on_branch(live, ref)[2] == pytest.approx(path_q3)


def _traj(rows):
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    t = JointTrajectory()
    t.joint_names = ["joint_%d" % i for i in range(1, 8)]
    for k, row in enumerate(rows):
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in row]
        p.time_from_start.sec, p.time_from_start.nanosec = 0, int(2e7 * (k + 1))
        t.points.append(p)
    return t


def test_the_push_and_the_recoil_start_where_the_arm_is_on_a_path_past_pi():
    """The push is cut from the touch's own path and the recoil retraces it,
    both from the NEAREST sample to the live stop. With joint_3 past +pi
    the reading is 2*pi from its own path: the push began with a full-turn
    jump in it (the driver's continuity gate refuses that), the recoil at
    the wrong end of the stroke."""
    from rammp_box_opening.runtime.retime import forward_tail, reverse_tail

    down = [list(HOME) for _ in range(50)]
    for k, q in enumerate(down):
        q[2] = 3.2 + 0.001 * k  # joint_3 past +pi all the way down
        q[5] = 0.9 + 0.01 * k  # the stroke itself
    stop = list(down[30])
    stop[2] -= 2 * math.pi  # as the transport reports it
    push = forward_tail(_traj(down), stop, arc_rad=0.05)
    assert push is not None
    assert push[0][2] == pytest.approx(down[30][2])  # on the path's side of the wrap
    assert max(np.abs(np.diff(push, axis=0)).max(axis=0)) < 0.05  # no jump anywhere
    back = reverse_tail(_traj(down), progress=0.62, live=stop, arc_rad=0.05)
    assert back is not None and back[0][2] == pytest.approx(down[30][2])
    assert max(np.abs(np.diff(back, axis=0)).max(axis=0)) < 0.05
    assert [q[5] for q in back] == sorted([q[5] for q in back], reverse=True)  # only back out


def test_the_client_hands_the_planner_its_start_on_homes_side():
    from rammp_box_opening.runtime.client import PlannerClient

    c = PlannerClient.__new__(PlannerClient)
    c._plan_pose = c._plan_joints = None
    c._call = lambda client, goal: goal  # the goal as it would go out
    g = c._plan_pose_once([0.42, 0.22, 0.21], [1.0, 0.0, 0.0, 0.0], HOME_AS_REPORTED)
    assert list(g.start_joints)[2] == pytest.approx(HOME[2], abs=2e-3)
    g = c.plan_to_joints(HOME, HOME_AS_REPORTED)
    assert list(g.start_joints)[2] == pytest.approx(HOME[2], abs=2e-3)
