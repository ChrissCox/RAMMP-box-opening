"""Stand-in for the RAMMP-CuRobo v1.0.0 planner container, for the
stub-isolated e2e harnesses.

Serves the planner's whole surface — /rammp_curobo/plan_to_pose,
plan_to_joints and set_world — minus the physics: plans are straight-line
joint interpolations stamped like cuRobo's (point k at (k + 1) * dt), and
like v1.0.0 a goal without start_joints is refused. Nothing here moves;
execution is scripts/stub_arm.py. Run ONLY on an isolated ROS_DOMAIN_ID
(the harnesses enforce this) — it serves the real names.

STUB_PLAN_S env shortens the fake plan duration (default 6 s) so flow
harnesses finish quickly while abort harnesses keep long strokes.
"""

import os
from pathlib import Path

import rclpy
from rclpy.action import ActionServer, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rammp_curobo_interfaces.action import PlanToJoints, PlanToPose
from rammp_curobo_interfaces.srv import SetWorld

NAMES = ["joint_%d" % i for i in range(1, 8)]
PLAN_S = float(os.environ.get("STUB_PLAN_S", "6.0"))


def interp_traj(q0, q1, n=40, dur=PLAN_S):
    t = JointTrajectory()
    t.joint_names = list(NAMES)
    dt = dur / n
    for i in range(n):
        a = i / (n - 1)
        p = JointTrajectoryPoint()
        p.positions = [x + a * (y - x) for x, y in zip(q0, q1)]
        p.velocities = [0.0] * 7
        p.accelerations = [0.0] * 7
        tt = (i + 1) * dt
        p.time_from_start.sec = int(tt)
        p.time_from_start.nanosec = int((tt - int(tt)) * 1e9)
        t.points.append(p)
    return t


class StubPlanner(Node):
    """The /rammp_curobo surface box-opening uses, minus the physics."""

    def __init__(self):
        super().__init__("rammp_curobo")
        cb = ReentrantCallbackGroup()
        self.plan_goals = 0
        ActionServer(
            self,
            PlanToJoints,
            "/rammp_curobo/plan_to_joints",
            execute_callback=self._plan_joints,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            callback_group=cb,
        )
        ActionServer(
            self,
            PlanToPose,
            "/rammp_curobo/plan_to_pose",
            execute_callback=self._plan_pose,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            callback_group=cb,
        )
        self.create_service(
            SetWorld, "/rammp_curobo/set_world", self._set_world, callback_group=cb
        )
        print("STUB PLANNER READY", flush=True)

    def _set_world(self, req, resp):
        # a path the planner cannot read is a failed push, as in the container
        known = "/" not in req.world or Path(req.world).is_file()
        print("SET_WORLD %s%s" % (req.world, "" if known else " (MISSING)"), flush=True)
        resp.success = known
        resp.message = "stub" if known else "world file not found"
        return resp

    def _no_start(self, gh, result_type):
        print("PLAN REFUSED: no start_joints", flush=True)
        res = result_type.Result()
        res.success = False
        res.message = "start_joints is required (RAMMP-CuRobo v1.0.0)"
        gh.succeed()
        return res

    def _plan_joints(self, gh):
        self.plan_goals += 1
        print("PLAN GOAL #%d (joints)" % self.plan_goals, flush=True)
        if len(gh.request.start_joints) != 7:
            return self._no_start(gh, PlanToJoints)
        res = PlanToJoints.Result()
        res.success = True
        res.message = "stub plan"
        res.trajectory = interp_traj(
            list(gh.request.start_joints), list(gh.request.target_joints)
        )
        res.planning_time = 0.01
        res.goal_mismatch_rad = 0.0
        gh.succeed()
        return res

    def _plan_pose(self, gh):
        self.plan_goals += 1
        print("PLAN GOAL #%d (pose)" % self.plan_goals, flush=True)
        if len(gh.request.start_joints) != 7:
            return self._no_start(gh, PlanToPose)
        q0 = list(gh.request.start_joints)
        q1 = list(q0)
        # bounded oscillation, not accumulation: an unconditional +0.3 rad
        # per plan walked joint_1 ~3 rad across a mission, which no real
        # plan does — and the runner's joint-swing cap (a REAL safety gate,
        # field 2026-09-01) rightly refused the fake home
        q1[0] += 0.3 if q0[0] < 1.0 else -0.3
        res = PlanToPose.Result()
        res.success = True
        res.message = "stub plan"
        res.trajectory = interp_traj(q0, q1)
        res.planning_time = 0.01
        gh.succeed()
        return res


def main():
    rclpy.init()
    ex = MultiThreadedExecutor()
    ex.add_node(StubPlanner())
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
