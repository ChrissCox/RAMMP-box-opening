"""PlannerClient against the kinova-gen3-ros2 driver's contract: what reaches
/execute_joint_trajectory and /setpoint/gripper, and what never does.

The client is built without a ROS graph (its __init__ opens real action
clients); the doubles below stand in for rclpy's ActionClient and
publisher, shaped like the real ones (a goal handle with .accepted and
get_result_async(), a result wrapper with .result)."""

import time
from types import SimpleNamespace

import pytest
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rammp_arm_interfaces.action import ExecuteJointTrajectory

from rammp_box_opening.runtime import client as client_mod

NAMES = ["joint_%d" % i for i in range(1, 8)]
LIVE = [0.0, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571]


def _traj(start, end, times=(0.02, 1.0)):
    t = JointTrajectory()
    t.joint_names = list(NAMES)
    for row, tt in zip((start, end), times):
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


class _Future:
    def __init__(self, result):
        self._result = result

    def done(self):
        return True

    def result(self):
        return self._result


class _GoalHandle:
    accepted = True

    def __init__(self, error_code, error_string):
        self._wrapped = SimpleNamespace(
            status=4,
            result=ExecuteJointTrajectory.Result(
                error_code=error_code, error_string=error_string
            ),
        )

    def get_result_async(self):
        return _Future(self._wrapped)

    def cancel_goal_async(self):
        return _Future(object())


class _Driver:
    """The execute_joint_trajectory action as the client sees it: records
    each goal, replays scripted feedback, answers with one result."""

    def __init__(self, error_code=0, error_string="", feedback=()):
        self.goals = []
        self._answer = (error_code, error_string)
        self._feedback = list(feedback)

    def wait_for_server(self, timeout_sec=None):
        return True

    def send_goal_async(self, goal, feedback_callback=None):
        self.goals.append(goal)
        for fraction in self._feedback:
            fb = ExecuteJointTrajectory.Feedback(fraction_complete=fraction)
            feedback_callback(SimpleNamespace(feedback=fb))
        return _Future(_GoalHandle(*self._answer))


class _RunningFuture:
    """A goal result that is outstanding until the driver settles it."""

    def __init__(self, result):
        self._result = result
        self._done = False

    def settle(self):
        self._done = True

    def done(self):
        return self._done

    def result(self):
        return self._result if self._done else None


class _RunningHandle:
    """A goal that is still flying: its result settles when cancelled."""

    accepted = True

    def __init__(self, error_code=-6):
        self._fut = _RunningFuture(
            SimpleNamespace(
                status=5,
                result=ExecuteJointTrajectory.Result(error_code=error_code),
            )
        )
        self.cancels = 0

    def get_result_async(self):
        return self._fut

    def cancel_goal_async(self):
        self.cancels += 1
        self._fut.settle()  # the driver settles the goal on a cancel
        return _Future(object())


class _FlyingDriver(_Driver):
    """Accepts a goal that keeps running until something cancels it."""

    def __init__(self):
        super().__init__()
        self.handle = _RunningHandle()

    def send_goal_async(self, goal, feedback_callback=None):
        self.goals.append(goal)
        return _Future(self.handle)


class _Publisher:
    def __init__(self):
        self.sent = []

    def publish(self, msg):
        self.sent.append(msg)


class _Guard:
    def __init__(self):
        self.progress = []
        self.peak = 0.0

    def on_progress(self, p):
        self.progress.append(p)

    def on_efforts(self, eff):
        return False


@pytest.fixture(autouse=True)
def _no_spin(monkeypatch):
    monkeypatch.setattr(client_mod.rclpy, "spin_once", lambda node, timeout_sec=0.0: None)


def _client(driver=None, armed=True):
    c = client_mod.PlannerClient.__new__(client_mod.PlannerClient)
    c.node = object()
    c._abort = None
    c._armed = armed
    c._execute = driver if driver is not None else _Driver()
    c._q = list(LIVE)
    c._eff = [0.0] * 7
    c._eff_at = time.monotonic()
    c._gripper_pos = 0.0
    c._gripper_target = None
    c._gripper_pub = _Publisher()
    return c


def test_nothing_reaches_the_driver_while_the_client_is_not_armed():
    driver = _Driver()
    outcome, info = _client(driver, armed=False).execute(_traj(LIVE, LIVE), 1.0)
    assert outcome == "failed" and driver.goals == []


def test_a_trajectory_that_starts_away_from_the_arm_is_never_sent():
    driver = _Driver()
    away = [LIVE[0] + 0.1] + LIVE[1:]
    outcome, info = _client(driver).execute(_traj(away, away), 1.0)
    assert outcome == "failed" and driver.goals == []


def test_a_slowed_leg_reaches_the_driver_dilated_in_position_mode():
    driver = _Driver()
    end = [v + 0.05 for v in LIVE]
    outcome, _info = _client(driver).execute(_traj(LIVE, end), 0.5)
    assert outcome == "arrived"
    (goal,) = driver.goals
    assert _secs(goal.trajectory.points[-1]) == pytest.approx(2.0)
    assert goal.control_mode == 0  # POSITION: impedance with default gains is not a press
    assert goal.preemption == 0  # QUEUE: a new goal never displaces one still settling


def test_a_driver_failure_fails_the_leg_with_its_reason():
    driver = _Driver(error_code=-9, error_string="halted: emergency stop")
    end = [v + 0.05 for v in LIVE]
    outcome, info = _client(driver).execute(_traj(LIVE, end), 1.0)
    assert outcome == "failed"
    assert "HALTED" in info["message"] and "emergency stop" in info["message"]


def test_execution_progress_arms_the_guard():
    driver = _Driver(feedback=(0.0, 0.3))
    guard = _Guard()
    end = [v + 0.05 for v in LIVE]
    outcome, info = _client(driver).execute(_traj(LIVE, end), 1.0, guard=guard)
    assert guard.progress == [0.0, pytest.approx(0.3)]
    assert info["progress"] == pytest.approx(0.3)


def test_a_gripper_command_goes_out_as_the_drivers_normalized_setpoint():
    c = _client()
    handle = c.gripper_send(0.8)
    assert handle is not None
    (msg,) = c._gripper_pub.sent
    assert msg.position == pytest.approx(1.0)


def test_a_gripper_command_pending_through_a_long_motion_is_still_joined():
    """press:close is dispatched early and joined after the scan flight, so the
    fingers close while the arm moves. In slow mode that flight alone outlasts
    the join timeout — the join must still read the gripper, not declare a
    failure without ever looking at it (field 2026-09-14, --speed-scale 0.3:
    "STOP: leg press:close -> failed (gripper at 0.793)", a gripper that had
    in fact closed)."""
    c = _client()
    handle = c.gripper_send(0.8)
    # the arm flew for longer than the whole gripper timeout
    stale = handle._replace(t0=handle.t0 - (client_mod.PlannerClient.GRIPPER_TIMEOUT_S + 3.0))
    c._gripper_pos = 0.793  # the fingers closed while it flew
    c._eff_at = time.monotonic()
    assert c.gripper_join(stale) == (True, 0.793, False)


def test_a_gripper_that_never_moved_still_fails_after_a_long_motion():
    """The other half: pending a long time is not itself success."""
    c = _client()
    handle = c.gripper_send(0.8)
    stale = handle._replace(t0=handle.t0 - (client_mod.PlannerClient.GRIPPER_TIMEOUT_S + 3.0))
    c._eff_at = time.monotonic()  # still reading 0.0: nothing moved
    ok, pos, _stalled = c.gripper_join(stale)
    assert ok is False and pos == pytest.approx(0.0)


def test_a_motion_stops_the_moment_the_search_finds_something():
    """The sweep flies until the detector commits a coarse fix. The goal is
    cancelled then and there, and the arm holds where it stopped — which is
    where the precise fix is taken from."""
    driver = _FlyingDriver()
    c = _client(driver)
    polls = {"n": 0}

    def found_it(progress=None):
        polls["n"] += 1
        return polls["n"] >= 2  # nothing on the first look, then a fix

    end = [v + 0.05 for v in LIVE]
    outcome, _info = c.execute(_traj(LIVE, end), 1.0, stop_when=found_it)
    assert outcome == "stopped"
    assert driver.handle.cancels == 1


def test_a_motion_that_finds_nothing_runs_to_its_end():
    driver = _Driver()
    end = [v + 0.05 for v in LIVE]
    outcome, _info = _client(driver).execute(
        _traj(LIVE, end), 1.0, stop_when=lambda progress=None: False
    )
    assert outcome == "arrived"


def test_no_gripper_command_goes_out_while_the_client_is_not_armed():
    c = _client(armed=False)
    assert c.gripper_send(0.8) is None
    assert c._gripper_pub.sent == []
