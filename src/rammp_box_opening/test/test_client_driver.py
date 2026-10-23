"""PlannerClient against the kinova-gen3-ros2 driver's contract: what reaches
/execute_joint_trajectory and /setpoint/gripper, and what never does.

The client is built without a ROS graph (its __init__ opens real action
clients); the doubles below stand in for rclpy's ActionClient and
publisher, shaped like the real ones (a goal handle with .accepted and
get_result_async(), a result wrapper with .result)."""

import time
from types import SimpleNamespace

import pytest
from action_msgs.srv import CancelGoal
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
        return _Future(SimpleNamespace(return_code=0))  # ERROR_NONE: taken


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
        return _Future(SimpleNamespace(return_code=0))  # ERROR_NONE: taken


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


def test_a_compensated_guard_is_fed_the_joints_of_the_same_joint_states_message():
    """The reach-out's guards take gravity out at the joints the efforts
    were read at (runtime/guards.CompensatedGuard): the client hands both."""
    driver = _FlyingDriver()
    c = _client(driver)
    c._eff = [9.0, 9.0, 9.0, 1.0, 2.0, 3.0, 4.0]
    seen = []

    class Guard(_Guard):
        wants_joints = True

        def on_efforts(self, eff, joints=None):
            seen.append((eff, joints))
            return True  # a trip cancels the goal

    end = [v + 0.05 for v in LIVE]
    outcome, _info = c.execute(_traj(LIVE, end), 1.0, guard=Guard())
    assert outcome == "touch" and driver.handle.cancels == 1
    assert seen == [([1.0, 2.0, 3.0, 4.0], list(LIVE))]
    # a plain guard is called as it always was
    c2 = _client(_FlyingDriver())
    calls = []

    class Plain(_Guard):
        def on_efforts(self, eff):
            calls.append(eff)
            return True

    assert c2.execute(_traj(LIVE, end), 1.0, guard=Plain())[0] == "touch" and calls == [[0.0] * 4]


def test_settled_efforts_average_fresh_readings_and_refuse_a_frozen_stream(monkeypatch):
    c = _client()
    ticks = {"n": 0}

    def spin(node, timeout_sec=0.0):  # a /joint_states message per spin
        ticks["n"] += 1
        c._eff = [0.0, 0.0, 0.0, float(ticks["n"] % 2), 0.0, 0.0, 0.0]
        c._eff_at = time.monotonic()

    monkeypatch.setattr(client_mod.rclpy, "spin_once", spin)
    eff, q = c.settled_efforts(duration_s=0.05, settle_s=0.0)
    assert eff[0] == pytest.approx(0.5, abs=0.2) and q == pytest.approx(LIVE)
    monkeypatch.setattr(client_mod.rclpy, "spin_once", lambda node, timeout_sec=0.0: None)
    c._eff_at = time.monotonic() - 5.0  # the stream stopped
    assert c.settled_efforts(duration_s=0.05, settle_s=0.0, timeout_s=0.2) is None


def test_settled_joints_are_averaged_on_one_branch_across_the_wrap(monkeypatch):
    """joint_3 at HOME sits 4.6 mrad from +-pi and is reported either side
    of it: the average is pi, not 0."""
    import math

    c = _client()
    flip = {"n": 0}

    def spin(node, timeout_sec=0.0):
        flip["n"] += 1
        q = list(LIVE)
        q[2] = math.pi - 0.002 if flip["n"] % 2 else -math.pi + 0.002
        c._q = q
        c._eff_at = time.monotonic()

    monkeypatch.setattr(client_mod.rclpy, "spin_once", spin)
    _eff, q = c.settled_efforts(duration_s=0.05, settle_s=0.0)
    assert abs(abs(q[2]) - math.pi) < 0.003


# -- a stop is claimed only when the driver confirmed it (review 2026-10-22) --


class _Pending:
    """A future the scripted driver settles (or never does)."""

    def __init__(self, result=None, done=False):
        self._r, self._d = result, done

    def done(self):
        return self._d

    def result(self):
        return self._r if self._d else None

    def settle(self, r=None):
        if r is not None:
            self._r = r
        self._d = True


def _ended(status, error_code=0):
    return SimpleNamespace(status=status, result=ExecuteJointTrajectory.Result(error_code=error_code))


class _ScriptedHandle:
    """A flying goal whose cancel answer and result are scripted: `cancel`
    is the CancelGoal response (None: it never comes), `ended` the result
    the driver settles on the cancel (None: never)."""

    accepted = True

    def __init__(self, cancel, ended):
        self._cancel = cancel
        self._ended = ended
        self.result_fut = _Pending()
        self.cancels = 0

    def get_result_async(self):
        return self.result_fut

    def cancel_goal_async(self):
        self.cancels += 1
        if self._ended is not None:
            self.result_fut.settle(self._ended)
        return _Pending(self._cancel, done=self._cancel is not None)


class _ScriptedDriver(_Driver):
    def __init__(self, handle):
        super().__init__()
        self.handle = handle

    def send_goal_async(self, goal, feedback_callback=None):
        self.goals.append(goal)
        return _Future(self.handle)


class _TripGuard(_Guard):
    def on_efforts(self, eff):
        return eff is not None  # trips on the first reading


def _cresp(code):
    return CancelGoal.Response(return_code=code)


def _no_waiting(monkeypatch):
    # the confirmation waits (3 s cancel, 10 s result) shortened to a tick
    real = client_mod.spin_until_done
    monkeypatch.setattr(
        client_mod, "spin_until_done", lambda node, fut, timeout_s, abort=None: real(node, fut, 0.01, abort)
    )


def _run(handle, **kw):
    end = [v + 0.05 for v in LIVE]
    return _client(_ScriptedDriver(handle)).execute(_traj(LIVE, end), 1.0, **kw)


def test_a_trip_nobody_confirmed_stopping_is_a_failure_not_a_touch(monkeypatch):
    """The driver neither answers the cancel nor ends the goal (it dies when
    one cyclic exchange goes 3 s unanswered): reading a contact height off
    that arm and pushing on top of it is how a box gets pressed through."""
    _no_waiting(monkeypatch)
    outcome, info = _run(_ScriptedHandle(None, None), guard=_TripGuard())
    assert outcome == "failed" and "did not confirm" in info["message"]
    assert info["torque_peak"] == 0.0


def test_a_trip_whose_cancel_was_refused_and_stroke_completed_is_a_failure(monkeypatch):
    _no_waiting(monkeypatch)
    h = _ScriptedHandle(_cresp(CancelGoal.Response.ERROR_REJECTED), _ended(4, 0))  # SUCCESSFUL
    outcome, info = _run(h, guard=_TripGuard())
    assert outcome == "failed" and "did not confirm" in info["message"]


@pytest.mark.parametrize(
    "cancel, ended",
    [
        (_cresp(CancelGoal.Response.ERROR_NONE), _ended(5, -6)),  # taken, CANCELED
        (None, _ended(5, -6)),  # the answer lost, but the goal reads CANCELED
        (_cresp(CancelGoal.Response.ERROR_GOAL_TERMINATED), _ended(4, 0)),  # ended on its own just then
    ],
)
def test_a_trip_the_driver_confirmed_stopping_is_a_touch(monkeypatch, cancel, ended):
    _no_waiting(monkeypatch)
    outcome, info = _run(_ScriptedHandle(cancel, ended), guard=_TripGuard())
    assert outcome == "touch" and info["message"] == "torque guard trip"


@pytest.mark.parametrize(
    "cancel, ended",
    [(None, None), (_cresp(CancelGoal.Response.ERROR_REJECTED), _ended(4, 0))],
)
def test_a_stop_when_nobody_confirmed_is_a_failure_not_a_stop(monkeypatch, cancel, ended):
    _no_waiting(monkeypatch)
    outcome, info = _run(_ScriptedHandle(cancel, ended), stop_when=lambda progress=None: True)
    assert outcome == "failed" and "did not confirm" in info["message"]


def test_a_guarded_stroke_is_cancelled_within_a_fifth_of_a_second_of_the_stream_stopping(monkeypatch):
    """/joint_states stops as the stroke starts while the driver keeps
    flying it: 0.5 s staleness plus a 1 s grace was 1.5 s blind, ~126 mm
    of the depth touch's 84 mm/s stroke."""
    monkeypatch.setattr(client_mod.rclpy, "spin_once", lambda node, timeout_sec=0.0: time.sleep(0.005))
    h = _ScriptedHandle(_cresp(CancelGoal.Response.ERROR_NONE), _ended(5, -6))
    c = _client(_ScriptedDriver(h))
    t0 = time.monotonic()
    c._eff_at = t0  # the last message, now
    outcome, info = c.execute(_traj(LIVE, [v + 0.05 for v in LIVE]), 1.0, guard=_Guard())
    blind = time.monotonic() - t0
    assert outcome == "failed" and "effort stream lost" in info["message"]
    assert h.cancels == 1
    assert blind < client_mod.PlannerClient.GUARD_STALE_S + 0.05 < 0.3



def test_a_slow_stop_when_does_not_read_as_a_lost_stream(monkeypatch):
    """An in-flight confirm runs the wrist's depth fix in stop_when on the
    guarded stroke: its own time must not age the stream the guard judges."""
    h = _ScriptedHandle(_cresp(CancelGoal.Response.ERROR_NONE), _ended(5, -6))
    c = _client(_ScriptedDriver(h))
    spins = {"n": 0}

    def spin(node, timeout_sec=0.0):  # a /joint_states message per spin
        spins["n"] += 1
        c._eff_at = time.monotonic()
        if spins["n"] == 3:
            h.result_fut.settle(_ended(4, 0))

    def slow(progress=None):
        time.sleep(client_mod.PlannerClient.GUARD_STALE_S + 0.05)
        return False

    monkeypatch.setattr(client_mod.rclpy, "spin_once", spin)
    outcome, info = c.execute(_traj(LIVE, [v + 0.05 for v in LIVE]), 1.0, guard=_Guard(), stop_when=slow)
    assert outcome == "arrived" and h.cancels == 0

def _abort_client(handle):
    from rammp_box_opening.runtime.abort import AbortFlag, sigint_handler

    flag = AbortFlag()
    c = _client(_ScriptedDriver(handle))
    c._abort = flag
    return c, flag, sigint_handler(flag)


def test_a_ctrl_c_in_the_spin_that_delivers_the_result_is_not_swallowed(monkeypatch):
    h = _ScriptedHandle(None, None)
    c, flag, handler = _abort_client(h)

    def spin(node, timeout_sec=0.0):
        handler(2, None)  # Ctrl+C ...
        h.result_fut.settle(_ended(4, 0))  # ... while the result callback runs

    monkeypatch.setattr(client_mod.rclpy, "spin_once", spin)
    with pytest.raises(KeyboardInterrupt):
        c.execute(_traj(LIVE, [v + 0.05 for v in LIVE]), 1.0)
    assert flag.goal_in_flight is False and h.cancels == 0  # over already: nothing to cancel
    with pytest.raises(KeyboardInterrupt):
        c.gripper_send(0.8)  # what the runner does next never goes out
    assert c._gripper_pub.sent == []


def test_a_ctrl_c_while_the_goal_is_rejected_is_not_swallowed(monkeypatch):
    class Rejected:
        accepted = False

    c, flag, handler = _abort_client(None)
    c._execute.send_goal_async = lambda goal, feedback_callback=None: (handler(2, None), _Future(Rejected()))[1]
    with pytest.raises(KeyboardInterrupt):
        c.execute(_traj(LIVE, [v + 0.05 for v in LIVE]), 1.0)




def _live_stream(c, monkeypatch, fresh_after_settle=None, settle_s=0.15):
    """A /joint_states message every ~20 ms (49 Hz live); with
    `fresh_after_settle` the stream freezes after that many fresh messages
    past the settle. Returns the count of fresh messages delivered after it."""
    t0 = time.monotonic()
    seen = {"after": 0}

    def spin(node, timeout_sec=0.0):
        time.sleep(0.02)
        late = time.monotonic() - t0 >= settle_s
        if late and fresh_after_settle is not None and seen["after"] >= fresh_after_settle:
            return  # frozen: _eff_at stops moving, then goes stale
        c._eff = [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4]
        c._eff_at = time.monotonic()
        if late:
            seen["after"] += 1

    monkeypatch.setattr(client_mod.rclpy, "spin_once", spin)
    return seen


def test_settled_efforts_default_read_holds_the_arm_about_0_3_s(monkeypatch):
    """Review 2026-10-22: the default 0.15 s settle + 0.30 s mean held the
    arm ~0.47 s per read, ~9 reads a depth run. 0.15 s + 0.15 s (~7
    messages at 49 Hz) is the whole read."""
    c = _client()
    seen = _live_stream(c, monkeypatch)
    t0 = time.monotonic()
    got = c.settled_efforts()
    elapsed = time.monotonic() - t0
    assert got is not None
    eff, q = got
    assert eff == pytest.approx([0.1, 0.2, 0.3, 0.4]) and q == pytest.approx(LIVE)
    assert 0.30 <= elapsed < 0.40
    assert seen["after"] >= 3


def test_settled_efforts_still_refuse_fewer_than_three_fresh_readings(monkeypatch):
    """Two fresh messages after the settle, then a frozen stream: no
    reference, at the default window as at any other."""
    c = _client()
    _live_stream(c, monkeypatch, fresh_after_settle=2)
    t0 = time.monotonic()
    assert c.settled_efforts(timeout_s=0.8) is None
    assert time.monotonic() - t0 < 1.0


def test_settled_efforts_defaults_keep_the_settle():
    import inspect

    params = inspect.signature(client_mod.PlannerClient.settled_efforts).parameters
    assert params["settle_s"].default == 0.15
    assert params["duration_s"].default == 0.15
