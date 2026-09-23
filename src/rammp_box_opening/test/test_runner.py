import time

import pytest
from trajectory_msgs.msg import JointTrajectoryPoint

from conftest import Q0, Q1, Q2, FakeClient, leg, runner, traj

from rammp_box_opening.runtime.guards import GuardSpec
from rammp_box_opening.runtime.legs import Kind


def test_dry_run_executes_nothing(tmp_path):
    c = FakeClient()
    res = runner(c, tmp_path).run([leg("a")], execute=False)
    assert [r.outcome for r in res] == ["skipped"]
    assert c.executed == [] and c.worlds_pushed == []


def test_merged_group_is_one_execution(tmp_path):
    c = FakeClient()
    legs = [leg("a", Q0, Q1, chain=0), leg("b", Q1, Q2, chain=0)]
    res = runner(c, tmp_path).run(legs, execute=True)
    assert all(r.ok for r in res)
    assert len(c.executed) == 1  # merged: one goal


def test_unguarded_leg_requires_full_world(tmp_path):
    c = FakeClient()
    res = runner(c, tmp_path).run(
        [leg("a", world="interaction_button")], execute=True
    )
    assert res[0].outcome == "refused" and c.executed == []


def test_nothing_moves_while_the_client_is_not_armed(tmp_path):
    """The driver has no dry-run gate of its own: the client's motion latch
    (set only by --execute) stands between a wiring bug and the arm."""
    c = FakeClient()
    c.armed = False
    res = runner(c, tmp_path).run([leg("a")], execute=True)
    assert res[0].outcome == "refused" and c.executed == []
    gl = leg("close", kind=Kind.GRIPPER, cmd=0.8)
    res = runner(c, tmp_path).run([gl], execute=True)
    assert res[0].outcome == "refused" and c.gripper_sent == []


def test_a_search_leg_that_stops_early_has_done_its_job(tmp_path):
    """The look and the sweep exist to find the box, not to reach their goal:
    stopping part-way because a fix committed is success, and the mission
    carries on from where the arm stopped."""
    c = FakeClient()
    c.exec_script = [("stopped", {"message": "a box came into view", "progress": 0.4})]
    lg = leg("look", world="bench")
    lg.stop_when = lambda progress=None: True
    res = runner(c, tmp_path).run([lg], execute=True)
    assert res[0].outcome == "stopped" and res[0].ok


def test_an_ordinary_leg_stopping_early_is_still_a_failure(tmp_path):
    c = FakeClient()
    c.exec_script = [("stopped", {"message": "?", "progress": 0.4})]
    res = runner(c, tmp_path).run([leg("a")], execute=True)
    assert res[0].outcome == "stopped" and not res[0].ok


def test_guarded_leg_refused_without_efforts(tmp_path):
    c = FakeClient()
    c.efforts = False
    g = GuardSpec(
        touch_nm=3.0, trip="press"
    )
    res = runner(c, tmp_path).run(
        [leg("press", guard=g, world="interaction_b")], execute=True
    )
    assert res[0].outcome == "refused" and c.executed == []


def test_start_drift_triggers_replan(tmp_path):
    c = FakeClient()
    c.live = [0.06] + [0.0] * 6  # 0.06 > 0.04 threshold
    r = runner(c, tmp_path)
    res = r.run([leg("a", Q0, Q1)], execute=True)
    assert res[0].ok
    # re-planned from live: the executed trajectory starts at the live joints
    assert c.exec_starts and c.exec_starts[0][0] == 0.06


def test_other_failure_stops_without_retry(tmp_path):
    c = FakeClient()
    c.exec_script = [
        (
            "failed",
            {"message": "controller rejected", "progress": 0.4, "torque_peak": None},
        )
    ]
    res = runner(c, tmp_path).run(
        [leg("a"), leg("b", Q1, Q2)], execute=True
    )
    assert res[0].outcome == "failed"
    assert len(res) == 1 and len(c.executed) == 1  # stopped, b never ran


def test_contact_invalidates_downstream(tmp_path):
    c = FakeClient()
    g = GuardSpec(touch_nm=3.0, trip="setdown")
    c.exec_script = [
        ("touch", {"message": "contact", "progress": 0.5, "torque_peak": 4.0})
    ]
    legs = [
        leg("descend", guard=g, world="interaction_x", chain=0),
        leg("after", Q1, Q2, chain=0),
    ]
    res = runner(c, tmp_path).run(legs, execute=True)
    assert res[0].outcome == "touch" and res[0].ok  # setdown: trip = success
    assert res[1].ok
    assert len(c.executed) == 2  # never merged with contact


def test_jsonl_log_written(tmp_path):
    import json

    c = FakeClient()
    runner(c, tmp_path).run([leg("a")], execute=True)
    logs = list(tmp_path.glob("run-*.jsonl"))
    assert len(logs) == 1
    row = json.loads(logs[0].read_text().splitlines()[0])
    assert row["leg"] == "a" and row["outcome"] == "arrived"


def test_transit_gate_accepts_bench_world_pre_detection(tmp_path):
    c = FakeClient()
    r = runner(c, tmp_path)
    assert r._refusal(leg("scan", world="bench")) is None
    assert "full or bench" in r._refusal(leg("weird", world="interaction_button"))


def test_replanned_trajectories_pass_the_sanity_gate(tmp_path):
    c = FakeClient()
    r = runner(c, tmp_path)
    wandering = traj(Q0, Q1)
    mid = JointTrajectoryPoint()
    mid.positions = [1.5] + [0.05] * 6  # joint_1 wanders way out and back
    mid.time_from_start.sec = 1
    wandering.points.insert(1, mid)
    wandering.points[-1].time_from_start.sec = 2

    class R:
        success = True
        message = "ok"
        trajectory = wandering

    c.plans = [R]
    group, chain = r._replan_group([leg("a")], next_chain=5)
    assert group is None  # wandering replan refused, same gate as pre-built


def test_fast_retreat_allowed_in_interaction_world(tmp_path):
    c = FakeClient()
    r = runner(c, tmp_path)
    ok_leg = leg("retreat", speed=0.35, world="interaction_button")
    assert r._refusal(ok_leg) is None  # ascends its own corridor
    other = leg("wander", speed=0.35, world="interaction_button")
    assert r._refusal(other) is not None  # only retreats get the pass


def test_no_replan_when_contact_left_arm_on_plan(tmp_path):
    c = FakeClient()
    g = GuardSpec(touch_nm=3.0, trip="setdown")
    c.exec_script = [
        ("touch", {"message": "contact", "progress": 0.97, "torque_peak": 4.0})
    ]
    legs = [
        leg("descend", guard=g, world="interaction_x", chain=0),
        leg("retreat", Q1, Q2, chain=0, speed=0.15, world="interaction_x"),
    ]
    r = runner(c, tmp_path)
    res = r.run(legs, execute=True)
    assert all(x.ok for x in res)
    # live == planned start (FakeClient tracks to traj end): NO replan —
    # the pre-planned retreat executed as built (no pause at the bottom)
    assert c.exec_starts[1] == Q1


def test_lift_may_run_faster_than_contact_in_an_interaction_world(tmp_path):
    """Lift ascends out of the corridor it just descended, exactly like
    retreat — both are exempt from the transit-speed world gate."""
    c = FakeClient()
    res = runner(c, tmp_path).run(
        [leg("lift", speed=0.35, world="interaction_button")],
        execute=True,
    )
    assert res[0].ok and res[0].outcome != "refused"


def test_unguarded_fast_leg_in_an_interaction_world_is_still_refused(tmp_path):
    """The exemption is name-scoped — it must not open the gate widely."""
    c = FakeClient()
    res = runner(c, tmp_path).run(
        [leg("place:lid:transit", speed=0.35, world="interaction_place")],
        execute=True,
    )
    assert res[0].outcome == "refused"


def test_merged_group_takes_its_guard_from_the_guarded_member(tmp_path):
    """can_merge forbids this today; the lookup is what keeps a future
    relaxation from running a guarded stroke with the lead's (absent)
    guard, the lead's speed and the lead's verify."""
    c = FakeClient()
    g = GuardSpec(touch_nm=3.0, trip="setdown")
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9})]
    group = [leg("fast", Q0, Q1, chain=0), leg("descend", Q1, Q2, chain=0, guard=g)]
    r = runner(c, tmp_path)
    res = r._run_motion(group)
    # setdown semantics come from the guarded member: a trip is SUCCESS
    assert res.outcome == "touch" and res.ok


def test_merged_group_refuses_a_guard_that_is_not_last(tmp_path):
    c = FakeClient()
    g = GuardSpec(touch_nm=3.0, trip="setdown")
    group = [leg("descend", Q0, Q1, chain=0, guard=g), leg("after", Q1, Q2, chain=0)]
    import pytest as _pytest

    with _pytest.raises(RuntimeError, match="not last"):
        runner(c, tmp_path)._run_motion(group)


def test_replans_push_the_legs_own_world(tmp_path):
    """Worlds are pushed at plan time and by the replan path per leg —
    execution never consults the collision world, so the runner no longer
    pushes per group (one tracker lives in the client, audit 2026-09-02).
    A replanned leg must reach the planner with ITS world (content-hashed
    path), not whatever was loaded."""
    c = FakeClient()
    a = leg("a", Q0, Q1, world="full", chain=0)
    a.world_path = "/w/full-aaaa.yaml"
    b = leg("b", Q1, Q2, world="full", chain=1)
    b.world_path = "/w/full-bbbb.yaml"  # same name, new contents
    c.live = [0.06] + [0.0] * 6  # drift: a replans, then b chains clean
    runner(c, tmp_path).run([a, b], execute=True)
    assert c.worlds_pushed == ["/w/full-aaaa.yaml"]


def test_lazy_leg_is_planned_once_from_live_at_execution(tmp_path):
    """A lazy leg (traj None) after a touch is planned from live joints
    exactly when it executes, in its own world."""
    c = FakeClient()
    g = GuardSpec(touch_nm=3.0, trip="setdown")
    c.exec_script = [
        ("touch", {"message": "contact", "progress": 0.9, "torque_peak": 4.0})
    ]
    lazy = leg("retreat", Q1, Q2, chain=1, world="interaction_x")
    lazy.traj = None
    lazy.goal_joints = None
    lazy.world_path = "/w/interaction_x-1.yaml"
    legs = [leg("down", guard=g, world="interaction_x", chain=0), lazy]
    res = runner(c, tmp_path).run(legs, execute=True)
    assert [r.outcome for r in res] == ["touch", "arrived"]
    assert c.worlds_pushed == ["/w/interaction_x-1.yaml"]
    assert len(c.executed) == 2


class _AsyncGripClient(FakeClient):
    """FakeClient that records send/join ordering against motion."""

    def __init__(self):
        super().__init__()
        self.events = []

    def gripper_send(self, position):
        self.events.append("send")
        return ("handle", position)

    def gripper_join(self, handle):
        self.events.append("join")
        return True, float(handle[1]), False

    def gripper_cmd(self, position):
        # the real client's blocking path is send + join; mirror it so the
        # ordering assertions mean the same thing on both paths
        if position is None:
            return super().gripper_cmd(position)
        return self.gripper_join(self.gripper_send(position))

    def execute(self, traj, speed, guard=None, while_running=None, stop_when=None):
        self.events.append("execute")
        return super().execute(traj, speed, guard=guard, while_running=while_running)


def test_deferred_gripper_close_overlaps_the_next_transit(tmp_path):
    """The owner's own example: fingers shut WHILE the arm moves."""
    c = _AsyncGripClient()
    close = leg("press:close", kind=Kind.GRIPPER, cmd=0.8)
    close.defer_join = True
    legs = [close, leg("approach", Q0, Q1, chain=0)]
    r = runner(c, tmp_path)
    res = r.run(legs, execute=True)
    assert all(r_.ok for r_ in res)
    # sent, THEN the transit ran; the join is lazy — it outlives the run
    # to overlap whatever planning follows, and finish() collects it
    assert c.events == ["send", "execute"]
    assert r.finish().ok
    assert c.events == ["send", "execute", "join"]


def test_deferred_gripper_is_joined_before_any_guarded_leg(tmp_path):
    """A press descends with the fingers closed — the overlap must not
    let a guarded leg start while they are still moving."""
    c = _AsyncGripClient()
    g = GuardSpec(touch_nm=3.0, trip="press")
    close = leg("press:close", kind=Kind.GRIPPER, cmd=0.8)
    close.defer_join = True
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9})]
    legs = [close, leg("press:down", Q0, Q1, chain=0, guard=g, world="interaction_b")]
    runner(c, tmp_path).run(legs, execute=True)
    assert c.events.index("join") < c.events.index("execute")


def test_a_release_is_never_deferred(tmp_path):
    """defer_join is opt-in; an un-flagged gripper leg still blocks."""
    c = _AsyncGripClient()
    legs = [leg("place:lid:open", kind=Kind.GRIPPER, cmd=0.0), leg("retreat", Q0, Q1)]
    runner(c, tmp_path).run(legs, execute=True)
    assert c.events[:2] == ["send", "join"], "release settles before the arm moves"


def test_touch_forces_replan_even_under_the_drift_gate(tmp_path):
    """After a guard trip the predicted start is wrong BY DESIGN, and a
    sub-gate joint delta is still a multi-mm Cartesian shove into the
    thing just touched: the pre-planned retreat re-pressed the button at
    full speed, guardless (field 2026-09-02). Post-touch, the next
    motion replans from live unconditionally."""
    class TouchStopsShort(FakeClient):
        # a real trip halts the arm shy of the endpoint; the plain fake
        # teleports to it, which would hide exactly the hazard under test
        def execute(self, traj, speed, guard=None, while_running=None, stop_when=None):
            outcome, info = super().execute(traj, speed, guard, while_running=while_running)
            if outcome == "touch":
                self.live = [self.live[0] + 0.01] + self.live[1:]
            return outcome, info

    c = TouchStopsShort()
    g = GuardSpec(touch_nm=3.0, trip="setdown")
    c.exec_script = [
        ("touch", {"message": "contact", "progress": 0.9, "torque_peak": 4.0})
    ]
    legs = [
        leg("descend", guard=g, world="interaction_x", chain=0),
        leg("retreat", Q1, Q2, chain=1),
    ]
    res = runner(c, tmp_path).run(legs, execute=True)
    assert res[0].outcome == "touch" and res[1].ok
    # the trip left the arm 0.01 rad from the predicted end — well under
    # the 0.04 free-air drift gate, which must NOT matter after a touch
    assert c.exec_starts[1][0] == 0.11  # replanned from live, not Q1


def test_arrived_leg_still_uses_the_drift_gate(tmp_path):
    """Free-air chaining keeps the owner's fewer-pauses rule: no touch,
    sub-gate drift, the pre-planned leg runs as planned. (The drift must
    appear AFTER leg a runs — the fake teleports live to each executed
    endpoint, so a pre-set offset only ever tests leg a's own gate.)"""

    class DriftsAfterArrive(FakeClient):
        def execute(self, traj, speed, guard=None, while_running=None, stop_when=None):
            outcome, info = super().execute(traj, speed, guard, while_running=while_running)
            if outcome == "arrived" and len(self.executed) == 1:
                self.live = [self.live[0] + 0.01] + self.live[1:]
            return outcome, info

    c = DriftsAfterArrive()
    legs = [leg("a", Q0, Q1, chain=0), leg("b", Q1, Q2, chain=1)]
    res = runner(c, tmp_path).run(legs, execute=True)
    assert all(r.ok for r in res)
    # 0.01 under the 0.04 gate, no touch: pre-planned start kept — an
    # always-replan mutation would execute from live (0.11) instead
    assert c.exec_starts[1][0] == 0.1


def test_replanned_warped_leg_keeps_its_execution_profile(tmp_path):
    """_apply_warp bakes fast-then-slow into the trajectory and sets
    speed=1.0 as a do-not-dilate sentinel. A replan swaps in a fresh
    UNWARPED trajectory — inheriting the sentinel would descend into
    contact at full speed (review 2026-09-02). The profile is re-warped,
    or the leg honestly downgrades to the slow contact speed."""
    from rammp_box_opening.runtime.runner import _restore_execution_profile

    g = GuardSpec(touch_nm=3.0, trip="setdown", rebaseline_after=0.7)

    # a trajectory long enough to warp: profile re-applied, sentinel kept
    many = [[0.0 + 0.01 * i] * 7 for i in range(40)]
    t = traj(many[0], many[-1])
    t.points = []
    from trajectory_msgs.msg import JointTrajectoryPoint

    for i, row in enumerate(many):
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in row]
        pt.velocities = [0.0] * 7
        pt.accelerations = [0.0] * 7
        pt.time_from_start.sec = i
        t.points.append(pt)
    lg = leg("down", guard=g, world="interaction_x")
    lg.speed = 1.0
    lg.warp = (0.5, 0.35, 0.3)
    lg.traj = t
    _restore_execution_profile(lg)
    assert lg.speed == 1.0  # profile baked in again
    assert lg.guard.rebaseline_after is not None
    end = lg.traj.points[-1].time_from_start
    assert end.sec + end.nanosec * 1e-9 > 39.0  # slower than the raw plan

    # a degenerate trajectory that cannot be warped: honest downgrade
    lg2 = leg("down2", guard=g, world="interaction_x")
    lg2.speed = 1.0
    lg2.warp = (0.35, 0.35, 0.3)  # fast==slow -> warp declines
    _restore_execution_profile(lg2)
    assert lg2.speed == 0.35  # the slow contact speed, never the sentinel
    assert lg2.guard.rebaseline_after is None

    # the retime hook follows the trajectory actually flown
    seen = {}
    lg3 = leg("press", guard=g, world="interaction_x")
    lg3.retime = lambda traj: seen.__setitem__("traj", traj)
    _restore_execution_profile(lg3)
    assert seen["traj"] is lg3.traj


def test_guard_observes_but_cannot_trip_before_arm_after():
    """The fast warp segment's dynamics tripped the gentle set-down
    threshold 64 ms into the descent and the lid was released 110 mm up
    (field 2026-09-02). Before arm_after the guard watches but never
    trips; after it, the same deviation trips."""
    from rammp_box_opening.runtime.guards import TorqueGuard

    g = TorqueGuard(4.0, arm_after=0.5)
    g.on_progress(0.05)
    assert g.on_efforts([0.0] * 4) is False  # baseline
    assert g.on_efforts([9.0, 0.0, 0.0, 0.0]) is False  # fast-zone jolt held
    assert g.peak == 9.0  # still observed
    g.on_progress(0.6)
    assert g.on_efforts([9.0, 0.0, 0.0, 0.0]) is True  # same dev now trips


def test_pending_gripper_outlives_the_run_and_joins_before_a_guarded_leg(tmp_path):
    """grip:open is dispatched on arrival at the hop (last leg of the press
    phase) and must overlap the NEXT phase's planning — so a run() ends
    without joining it, and the following run joins it before its guarded
    descent (audit 2026-09-02)."""
    c = _AsyncGripClient()
    r = runner(c, tmp_path)
    open_leg = leg("grip:open", kind=Kind.GRIPPER, cmd=0.0)
    open_leg.defer_join = True
    r.run([leg("retreat", Q0, Q1), open_leg], execute=True)
    assert c.events == ["execute", "send"]  # NOT joined at the end of run()
    g = GuardSpec(touch_nm=3.0, trip="obstruction")
    r.run([leg("grip:down", Q1, Q2, guard=g, world="interaction_b")], execute=True)
    assert c.events == ["execute", "send", "join", "execute"]


def test_start_gripper_dispatches_now_and_joins_lazily(tmp_path):
    c = _AsyncGripClient()
    r = runner(c, tmp_path)
    assert r.start_gripper("press:close", 0.8, execute=True)
    assert c.events == ["send"]
    g = GuardSpec(touch_nm=3.0, trip="press")
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9})]
    r.run([leg("press:down", Q0, Q1, guard=g, world="interaction_b")], execute=True)
    # the trailing execute is the reflex recoil off the button
    assert c.events == ["send", "join", "execute", "execute"]
    assert r.finish() is None  # nothing left pending


def test_release_overlaps_the_replan_but_never_the_motion(tmp_path):
    """place:lid:open is dispatched at once; the retreat's post-touch replan
    proceeds while the fingers open; the join lands before the retreat
    EXECUTES (a release completes before the arm moves away)."""
    c = _AsyncGripClient()
    g = GuardSpec(touch_nm=3.0, trip="setdown")
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9})]
    release = leg("place:lid:open", kind=Kind.GRIPPER, cmd=0.0)
    release.defer_join = True
    release.join_before_motion = True

    class Spy(_AsyncGripClient):
        def plan_to_pose(self, *a, **k):
            self.events.append("plan")
            return super().plan_to_pose(*a, **k)

    c = Spy()
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9})]
    lazy = leg("retreat", Q1, Q2, chain=1, world="interaction_x")
    lazy.traj = None
    lazy.target = ("pose", [0.5, 0.0, 0.2], [0.0, 1.0, 0.0, 0.0])
    legs = [
        leg("down", guard=g, world="interaction_x", chain=0),
        release,
        lazy,
    ]
    res = runner(c, tmp_path).run(legs, execute=True)
    assert all(r.ok for r in res)
    assert c.events == ["execute", "send", "plan", "join", "execute"]


def test_lookahead_runs_during_the_last_unguarded_motion(tmp_path):
    """The next phase is planned while this run's last unguarded motion
    flies, from that motion's predicted end joints; a guarded stroke never
    hosts it (audit 2026-09-02)."""
    c = FakeClient()
    r = runner(c, tmp_path)
    seen = []
    res = r.run(
        [leg("a", Q0, Q1, chain=0), leg("b", Q1, Q2, chain=1)],
        execute=True,
        lookahead=lambda q: seen.append(list(q)) or ["next-legs"],
    )
    assert all(x.ok for x in res)
    assert seen == [Q2]  # hosted on b, from b's predicted end
    assert r.lookahead_result == ["next-legs"]

    # a failing lookahead is not a failed leg: the caller builds afterwards
    def boom(q):
        raise RuntimeError("planner said no")

    res = r.run([leg("c", Q2, Q1, chain=0)], execute=True, lookahead=boom)
    assert res[0].ok and r.lookahead_result is None

    # guarded strokes never host it
    g = GuardSpec(touch_nm=3.0, trip="press")
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9})]
    seen.clear()
    r.run([leg("p", Q1, Q2, guard=g, world="interaction_b")], execute=True, lookahead=lambda q: seen.append(q))
    assert seen == [] and r.lookahead_result is None


def test_planned_lead_never_merges_with_a_lazy_tail():
    from rammp_box_opening.runtime.legs import merge_groups

    a = leg("a", Q0, Q1, chain=1)
    b = leg("b", Q1, Q2, chain=1)
    b.traj = None
    assert [len(g) for g in merge_groups([a, b])] == [1, 1]
    c = leg("c", Q1, Q2, chain=1)
    c.traj = None
    assert [len(g) for g in merge_groups([b, c])] == [2]  # both lazy: one group


def test_restore_profile_moves_arm_after_with_the_rebaseline():
    """A replanned warped set-down gets a new time base; the guard must not
    be armed before the NEW slow zone or it judges slow-zone efforts
    against the fast baseline (review 2026-09-02)."""
    from rammp_box_opening.runtime.runner import _restore_execution_profile
    from trajectory_msgs.msg import JointTrajectoryPoint

    many = [[0.0 + 0.01 * i] * 7 for i in range(40)]
    t = traj(many[0], many[-1])
    t.points = []
    for i, row in enumerate(many):
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in row]
        pt.velocities = [0.0] * 7
        pt.accelerations = [0.0] * 7
        pt.time_from_start.sec = i
        t.points.append(pt)
    g = GuardSpec(touch_nm=4.0, trip="setdown", rebaseline_after=0.3, arm_after=0.5)
    lg = leg("down", guard=g, world="interaction_x")
    lg.speed = 1.0
    lg.warp = (0.5, 0.35, 0.3)
    lg.traj = t
    _restore_execution_profile(lg)
    assert lg.guard.rebaseline_after is not None
    assert lg.guard.arm_after >= lg.guard.rebaseline_after
    assert lg.guard.arm_after >= 0.5


def test_ctrl_c_during_a_hosted_lookahead_still_cancels_the_goal():
    """The hosted plan runs INSIDE execute's cancel backstop: a Ctrl+C
    raised from it (second press) reaches cancel_goal_async, and a first
    press that only set the abort flag while the motion finished under
    the plan is delivered as a confirmed cancel, not swallowed (review
    2026-09-02)."""
    import pytest

    from rammp_box_opening.runtime import client as client_mod

    class Fut:
        def __init__(self, result=None, done=True):
            self._r, self._d = result, done

        def done(self):
            return self._d

        def result(self):
            return self._r

    class Send:
        accepted = True

        def __init__(self, events):
            self.events = events

        def get_result_async(self):
            return Fut(done=False)  # the motion is flying

        def cancel_goal_async(self):
            self.events.append("cancel")
            return Fut(result=object())

    class Exec:
        def __init__(self, events):
            self.events = events

        def wait_for_server(self, timeout_sec=None):
            return True

        def send_goal_async(self, goal, feedback_callback=None):
            return Fut(result=Send(self.events))

    class Abort:
        requested = False
        goal_in_flight = False

    class Node:
        pass

    events = []
    c = client_mod.PlannerClient.__new__(client_mod.PlannerClient)
    c.node = Node()
    c._execute = Exec(events)
    c._abort = Abort()
    # armed, with a fresh live state at the trajectory's start: the client
    # refuses anything else before a goal is ever sent
    c._armed = True
    c._eff, c._eff_at, c._q = [0.0] * 7, time.monotonic(), list(Q0)
    c.wrist_efforts = lambda: None
    c._cancel_confirm = lambda send, fut: events.append("cancel-confirm")
    # a planned trajectory's first point is at dt, never t=0 (cuRobo stamps
    # (k + 1) * dt) — the client refuses zero-stamped timing
    flown = traj(Q0, Q1)
    flown.points[0].time_from_start.nanosec = 20000000
    orig_spin = client_mod.rclpy.spin_once
    client_mod.rclpy.spin_once = lambda node, timeout_sec=0.0: None
    try:
        # second Ctrl+C: raised from inside the hosted plan
        def boom():
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            c.execute(flown, 0.5, guard=None, while_running=boom)
        assert "cancel" in events  # the backstop cancel was sent
        events.clear()

        # first Ctrl+C: the flag is set while the plan runs; the motion may
        # even have finished meanwhile — it must still be delivered
        def flag():
            c._abort.requested = True
            return "legs"

        with pytest.raises(KeyboardInterrupt):
            c.execute(flown, 0.5, guard=None, while_running=flag)
        assert events == ["cancel-confirm"]
        assert c._abort.goal_in_flight is False
    finally:
        client_mod.rclpy.spin_once = orig_spin


def test_unguarded_groups_fly_the_retimed_profile_at_the_sentinel(tmp_path):
    """Every unguarded motion group is re-timed (positions untouched) and
    executed at the 1.0 sentinel; guarded strokes keep their own speed."""
    c = FakeClient()
    r = runner(c, tmp_path)
    res = r.run([leg("a", Q0, Q1, chain=0), leg("b", Q1, Q2, chain=0)], execute=True)
    assert all(x.ok for x in res)
    assert c.executed[-1][1] == 1.0  # the profile is baked in
    assert r.last_retime is not None and r.last_retime["n_points"] == 3  # junction de-duplicated
    g = GuardSpec(touch_nm=3.0, trip="press")
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9})]
    r.run([leg("p", Q1, Q2, guard=g, world="interaction_b", speed=0.35)], execute=True)
    assert c.executed[-1][1] == 0.35  # guarded: not re-timed here


def test_speed_scale_dilates_guarded_and_unguarded_alike(tmp_path):
    c = FakeClient()
    r = runner(c, tmp_path)
    r.time_scale = 0.5
    r.run([leg("a", Q0, Q1)], execute=True)
    assert c.executed[-1][1] == 1.0  # profile baked in (dilated inside it)
    assert r.last_retime is not None
    g = GuardSpec(touch_nm=3.0, trip="press")
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9})]
    r.run([leg("p", Q1, Q2, guard=g, world="interaction_b", speed=0.35)], execute=True)
    # [-1] is the recoil (re-timed, 1.0 sentinel); the guarded stroke is [-2]
    assert c.executed[-2][1] == pytest.approx(0.175)  # guarded: speed x scale
    assert c.executed[-1][1] == 1.0


def _press_leg(**kw):
    g = GuardSpec(touch_nm=7.0, trip="press")
    return leg("press:down", Q0, Q1, guard=g, world="interaction_button", **kw)


def test_press_trip_recoils_before_the_next_leg_is_planned(tmp_path):
    """A guard trip leaves the arm on the button; the recoil reverses the
    descent's own path at once and the considered leg is planned from the
    recoil's end while it flies (2026-09-03)."""
    c = FakeClient()
    c.exec_script = [("touch", {"message": "contact", "progress": 0.8, "torque_peak": 7.1})]
    res = runner(c, tmp_path).run(
        [_press_leg(chain=0), leg("retreat", Q1, Q2, chain=1, world="interaction_button")],
        execute=True,
    )
    assert [r.leg_name for r in res] == ["press:down", "recoil", "retreat"]
    assert all(r.ok for r in res)
    assert len(c.executed) == 3  # press, recoil, retreat
    # the retreat was planned ONCE, from the recoil's end — not replanned
    # again from live afterwards
    assert len(c.joint_starts) == 1


def test_a_set_down_trip_never_recoils(tmp_path):
    """The trip IS the success there and the lid must be released while it
    rests on the table — recoiling would drop it from 20 mm up."""
    c = FakeClient()
    g = GuardSpec(touch_nm=4.0, trip="setdown", arm_after=0.5)
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9, "torque_peak": 4.2})]
    res = runner(c, tmp_path).run(
        [leg("place:lid:down", Q0, Q1, guard=g, world="interaction_place", chain=0)],
        execute=True,
    )
    assert [r.leg_name for r in res] == ["place:lid:down"]
    assert len(c.executed) == 1


def test_a_failed_press_trip_holds_where_it_struck(tmp_path):
    """An early trip means the stroke hit something above the button: the
    arm holds so the operator can see it, no recoil."""
    c = FakeClient()
    c.exec_script = [("touch", {"message": "contact", "progress": 0.1, "torque_peak": 7.0})]
    press = _press_leg(chain=0)
    press.verify = lambda v: (False, "guard tripped EARLY at 10% of the stroke")
    res = runner(c, tmp_path).run([press], execute=True)
    assert [r.leg_name for r in res] == ["press:down"] and not res[0].ok
    assert len(c.executed) == 1


def test_a_trip_records_where_the_fingertips_were(tmp_path):
    """The arm measures the surface it touched: at a trip the fingertip TF
    is the contact height, independent of the camera and of every model
    constant (2026-09-03)."""
    c = FakeClient()
    c.contact_at = [0.44, -0.16, 0.0837]
    g = GuardSpec(touch_nm=7.0, trip="press")
    c.exec_script = [("touch", {"message": "contact", "progress": 0.85})]
    res = runner(c, tmp_path).run(
        [leg("press:down", Q0, Q1, guard=g, world="interaction_button")],
        execute=True,
    )
    assert res[0].contact_xyz == [0.44, -0.16, 0.0837]
    # an untripped leg has nothing to report
    c.exec_script = []
    res2 = runner(c, tmp_path).run([leg("a", Q0, Q1)], execute=True)
    assert res2[0].contact_xyz is None


def test_flagged_gripper_leg_is_sent_under_the_previous_motion(tmp_path):
    """grip:open rides the retreat: sent the moment the retreat's goal is
    accepted, joined before the guarded grip:down as ever."""
    c = _AsyncGripClient()
    r = runner(c, tmp_path)
    retreat = leg("retreat", Q0, Q1, chain=1, world="interaction_b")
    open_leg = leg("grip:open", kind=Kind.GRIPPER, cmd=0.0, chain=1, world="interaction_b")
    open_leg.defer_join = True
    open_leg.send_with_previous_motion = True
    g = GuardSpec(touch_nm=7.0, trip="obstruction")
    down = leg("grip:down", Q1, Q2, guard=g, world="interaction_b", chain=2)
    res = r.run([retreat, open_leg, down], execute=True)
    assert [x.leg_name for x in res] == ["retreat", "grip:open", "grip:down"]
    # the send happened INSIDE the retreat's execute, and the join before grip:down
    assert c.events == ["execute", "send", "join", "execute"]


def test_flagged_gripper_leg_is_not_sent_under_a_guarded_motion(tmp_path):
    c = _AsyncGripClient()
    r = runner(c, tmp_path)
    g = GuardSpec(touch_nm=7.0, trip="obstruction")
    down = leg("grip:down", Q0, Q1, guard=g, world="interaction_b", chain=1)
    open_leg = leg("grip:open", kind=Kind.GRIPPER, cmd=0.0, chain=1, world="interaction_b")
    open_leg.defer_join = True
    open_leg.send_with_previous_motion = True
    r.run([down, open_leg], execute=True)
    assert c.events[:2] == ["execute", "send"]  # sent after, in order, not during
    r.finish()


def test_recoil_follows_a_push_that_ran_its_bound(tmp_path):
    """The arm is on the button whether the push met a stop or arrived at
    its bound: both recoil before the retreat."""
    c = FakeClient()
    g = GuardSpec(touch_nm=4.0, trip="press")
    c.exec_script = [("arrived", {"message": "ok", "progress": 1.0, "torque_peak": 1.2})]
    push = leg("press:push", Q0, Q1, guard=g, world="interaction_button", chain=0)
    push.verify = lambda v: (v.outcome in ("touch", "arrived"), "pressed")  # as press_push's
    res = runner(c, tmp_path).run(
        [push, leg("retreat", Q1, Q2, chain=1, world="interaction_button")],
        execute=True,
    )
    assert [x.leg_name for x in res] == ["press:push", "recoil", "retreat"]


def test_an_approach_and_its_guarded_descent_fly_as_one_goal(tmp_path):
    """The reach and the touch are ONE motion. Two goals meant a controller
    round trip and a full stop mid-reach — the pause a person does not make.
    One re-timed profile flows through the junction, and the torque guard
    arms where the descent begins instead of watching the approach's own
    dynamics swing the wrist."""
    c = FakeClient()
    c.exec_script = [("touch", {"message": "torque guard trip", "progress": 0.8})]
    g = GuardSpec(touch_nm=3.0, trip="press")
    legs = [
        leg("approach", Q0, Q1, chain=0, speed=0.75),
        leg("press:down", Q1, Q2, chain=0, speed=0.15, guard=g),
    ]
    res = runner(c, tmp_path).run(legs, execute=True)
    assert all(r.ok for r in res)
    assert len(c.executed) == 1  # one goal
    assert c.executed[0][1] == 1.0  # the profile is baked in, not dilated
    armed = c.guards[0].arm_after
    assert armed is not None and 0.0 < armed < 1.0
    # the baseline is taken AT the moment the guard becomes able to act, not
    # earlier: a baseline captured at the junction and compared a tenth of a
    # second later tripped a 3 Nm touch threshold on nothing, at exactly the
    # first armed instant (field 2026-09-15, peak 5.9 Nm 87 mm above the
    # button). Taking both at the same point absorbs whatever steady offset
    # the corner left behind.
    assert c.guards[0].rebaseline_after == pytest.approx(armed)


def _line_leg(name, a, b, n=30, **kw):
    """A leg whose trajectory has enough samples to re-time realistically."""
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    t = JointTrajectory()
    t.joint_names = ["joint_%d" % i for i in range(1, 8)]
    for k in range(n):
        f = k / (n - 1)
        pt = JointTrajectoryPoint()
        pt.positions = [x + (y - x) * f for x, y in zip(a, b)]
        pt.velocities = [0.0] * 7
        pt.accelerations = [0.0] * 7
        pt.time_from_start.sec = int(k * 0.05)
        pt.time_from_start.nanosec = int(((k * 0.05) % 1.0) * 1e9)
        t.points.append(pt)
    lg = leg(name, a, b, **kw)
    lg.traj = t
    return lg


def test_a_merged_guard_waits_a_real_settle_after_the_junction(tmp_path):
    """The settle is a TIME, not a fraction. A fraction of a short trajectory
    is a shorter wait than the same fraction of a long one, and what the arm
    needs after a corner does not scale with how far it is going."""
    from rammp_box_opening.runtime.guards import GROUP_SETTLE_S

    c = FakeClient()
    c.exec_script = [("touch", {"message": "trip", "progress": 0.95})]
    g = GuardSpec(touch_nm=3.0, trip="press")
    far = [0.6] * 7
    legs = [
        _line_leg("approach", [0.0] * 7, far, chain=0, speed=0.75),
        _line_leg("press:down", far, [0.75] * 7, chain=0, speed=0.15, guard=g),
    ]
    r = runner(c, tmp_path)
    r.run(legs, execute=True)
    dur = r.last_retime["duration_s"]
    junction = r.last_retime["seg_start_fracs"][1]
    assert 0.0 < junction < 0.9  # the descent really is the tail of the group
    assert c.guards[0].arm_after == pytest.approx(junction + GROUP_SETTLE_S / dur)
    assert c.guards[0].rebaseline_after == pytest.approx(c.guards[0].arm_after)


def test_slow_mode_arms_a_merged_guard_at_the_same_place_along_the_path(tmp_path):
    """Bench 2026-09-21 13:10, --speed-scale 0.5: the chained approach +
    press tripped its 3 Nm touch guard at the first armed instant, 87 mm
    above the button, on nothing. The same motion had found the button four
    runs out of four at full speed (09-16, 09-17), where its guard arms
    0.25 s after the junction: 70 mm above the button. Slow mode stretches
    everything the settle exists to wait out, and the settle stayed 0.25 s
    of wall time — half as far along the path, back in the corner's wake,
    where the 09-15 false trip had been. The operator's slow mode is for
    watching the SAME run slowly: it must arm the guard at the same point
    of the path."""
    g = GuardSpec(touch_nm=3.0, trip="press")
    far = [0.6] * 7
    armed = {}
    for scale in (1.0, 0.5):
        c = FakeClient()
        c.exec_script = [("touch", {"message": "trip", "progress": 0.95})]
        legs = [
            _line_leg("approach", [0.0] * 7, far, chain=0, speed=0.75),
            _line_leg("press:down", far, [0.75] * 7, chain=0, speed=0.15, guard=g),
        ]
        r = runner(c, tmp_path)
        r.time_scale = scale
        r.run(legs, execute=True)
        armed[scale] = (c.guards[0].arm_after, r.last_retime["duration_s"])
    # twice as long (to the 20 ms lead-in before point 0, which is not motion) ...
    assert armed[0.5][1] == pytest.approx(2 * armed[1.0][1], rel=1e-2)
    assert armed[0.5][0] == pytest.approx(armed[1.0][0], abs=2e-3)  # ... armed at the same fraction of it


def test_a_lone_guarded_stroke_keeps_its_own_timing(tmp_path):
    """Nothing in front of it: the stroke flies exactly as planned and
    warped, dilated by its own speed — the contact profile is not the
    re-timer's business."""
    c = FakeClient()
    c.exec_script = [("touch", {"message": "trip", "progress": 0.9})]
    g = GuardSpec(touch_nm=3.0, trip="press")
    legs = [leg("press:down", Q0, Q1, chain=0, speed=0.15, guard=g)]
    runner(c, tmp_path).run(legs, execute=True)
    assert c.executed[0][1] == pytest.approx(0.15)
    assert c.guards[0].arm_after is None


def test_a_merged_stroke_expects_contact_in_the_groups_time_base(tmp_path):
    """press:down's expected-contact fraction is measured along its OWN
    stroke. Flown as the tail of a merged group the execution's progress
    counts the approach too, so the expectation is rescaled to the path
    actually flown — otherwise an honest contact reads as an early strike."""
    from rammp_box_opening.runtime.guards import time_fraction_at_path_fraction

    seen = {}
    c = FakeClient()
    c.exec_script = [("touch", {"message": "trip", "progress": 0.9})]
    g = GuardSpec(touch_nm=3.0, trip="press")
    approach = leg("approach", Q0, Q1, chain=0, speed=0.75)
    down = leg("press:down", Q1, Q2, chain=0, speed=0.15, guard=g)
    down.contact_path_frac = 0.8
    down.retime = lambda traj: seen.update(
        frac=time_fraction_at_path_fraction(traj, down.contact_path_frac)
    )
    runner(c, tmp_path).run([approach, down], execute=True)
    # the approach is half the path, so 0.8 along the stroke is 0.9 along
    # the group — and later in time than 0.8 of it
    assert seen["frac"] > 0.8
    # the leg's own expectation is left as it was: a replan re-times it
    # against its own fresh trajectory before the group is re-scaled again
    assert down.contact_path_frac == 0.8


def _two_part_traj(n_head=11, n_line=31, head_s=1.5, line_s=2.0):
    """A press from staging as it reaches the driver: the planner's part to
    the waypoint (head_s), then the straight line down (line_s)."""
    from rammp_box_opening.runtime.stamps import set_stamp
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    t = JointTrajectory()
    t.joint_names = ["joint_%d" % i for i in range(1, 8)]
    n = n_head + n_line - 1
    for k in range(n):
        p = JointTrajectoryPoint()
        p.positions = [0.004 * k] * 7
        p.velocities = [0.0] * 7
        p.accelerations = [0.0] * 7
        tt = head_s * k / (n_head - 1) if k < n_head else head_s + line_s * (k - n_head + 1) / (n_line - 1)
        set_stamp(p.time_from_start, tt)
        t.points.append(p)
    return t, n_head - 1


def test_a_standalone_descent_arms_where_its_straight_line_begins(tmp_path):
    """Bench 2026-09-21: a press flown from staging on its own tripped its
    3 Nm touch guard at 40 % of the stroke, three inches above the box, on
    nothing — 3.08 Nm while the arm was still braking into the waypoint.
    The guard armed at a fixed 25 % of the leg, which used to cover the
    whole free-air part only because the line below crawled. A standalone
    descent now arms, and takes its baseline, where its straight line
    begins plus a settle — the rule the merged press always had."""
    from rammp_box_opening.runtime.guards import GROUP_SETTLE_S

    c = FakeClient()
    traj_, wp = _two_part_traj()  # the line begins at 1.5 s of 3.5 s = 0.4286
    g = GuardSpec(touch_nm=3.0, trip="touch", arm_after=0.25)
    press = leg("press:down", guard=g, world="interaction_button", speed=1.0)
    press.traj = traj_
    press.goal_joints = list(traj_.points[-1].positions)
    press.guard_from = wp
    c.live = list(traj_.points[0].positions)
    c.exec_script = [("touch", {"message": "contact", "progress": 0.88, "torque_peak": 3.2})]
    runner(c, tmp_path).run([press], execute=True)
    guard = c.guards[-1]
    at = 1.5 / 3.5
    assert guard.arm_after == pytest.approx(at + GROUP_SETTLE_S / 3.5, abs=1e-6)
    assert guard.rebaseline_after == pytest.approx(guard.arm_after)  # baseline AT the arming point
    # today's failure, replayed against that guard
    guard.on_progress(0.01)
    assert guard.on_efforts([1.0, 1.0, 1.0, 1.0]) is False  # first reading: baseline
    guard.on_progress(0.404)
    assert guard.on_efforts([4.08, 1.0, 1.0, 1.0]) is False  # braking into the waypoint: not a touch
    guard.on_progress(guard.arm_after + 0.01)
    assert guard.on_efforts([2.0, 1.0, 1.0, 1.0]) is False  # the baseline is re-taken here
    assert guard.on_efforts([2.5, 1.0, 1.0, 1.0]) is False
    assert guard.on_efforts([5.2, 1.0, 1.0, 1.0]) is True  # THIS is a touch


def test_slow_mode_arms_a_standalone_descent_at_the_same_place_along_the_path(tmp_path):
    """The settle is wall time AT FULL OPERATOR SPEED: --speed-scale dilates
    it with the motion, so the guard arms at the same point of the path
    however slowly the run is watched. (This test first asserted the
    opposite — "the same 0.25 s is half the fraction" — and the chained
    press false-tripped in slow mode the same day, 2026-09-21.)"""
    armed = {}
    for scale in (1.0, 0.5):
        c = FakeClient()
        traj_, wp = _two_part_traj()
        # 3.5 s at full operator speed: the 0.25 s settle (7 % of it) is
        # what arms the guard, not the 5 % floor
        press = leg("press:down", guard=GuardSpec(touch_nm=3.0, trip="touch", arm_after=0.25), world="interaction_button", speed=1.0)
        press.traj, press.guard_from = traj_, wp
        press.goal_joints = list(traj_.points[-1].positions)
        c.live = list(traj_.points[0].positions)
        c.exec_script = [("touch", {"message": "contact", "progress": 0.88})]
        r = runner(c, tmp_path)
        r.time_scale = scale
        r.run([press], execute=True)
        armed[scale] = c.guards[-1].arm_after
    from rammp_box_opening.runtime.guards import GROUP_SETTLE_S

    assert armed[1.0] == pytest.approx(1.5 / 3.5 + GROUP_SETTLE_S / 3.5, abs=1e-6)  # 3.5 s leg: the settle, not the floor
    assert armed[0.5] == pytest.approx(armed[1.0], abs=1e-6)


def test_a_replan_refreshes_where_the_descent_begins(tmp_path):
    """A replanned leg flies a NEW trajectory: an index into the old one
    would arm the guard at an arbitrary point of it."""
    c = FakeClient()
    c.live = [0.06] + [0.0] * 6  # drifted: forces the replan
    press = leg("press:down", guard=GuardSpec(touch_nm=3.0, trip="touch", arm_after=0.25), world="interaction_button")
    press.guard_from = 7
    press.waypoint = (7, [0.0] * 7)
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9})]
    runner(c, tmp_path).run([press], execute=True)
    assert press.guard_from is None and press.waypoint is None  # the fake plan carries no waypoint
