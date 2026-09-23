from rammp_box_opening.models.container import from_container
from rammp_box_opening.primitives.core import Home, Lift, Place, PlanState, tcp_z
from rammp_box_opening.runtime.legs import Kind, VerifyCtx

CFG = "src/rammp_box_opening/config/containers/oxo_pop.yaml"


def state():
    return PlanState(joints=[0.0] * 7, chain=0)


def test_chaining_start_joints_flow(ctx):
    c = ctx
    c.last_pose = ([0.45, 0.0, 0.10], [0.0, 1.0, 0.0, 0.0])
    st = state()
    legs_a, st = Lift(0.10).plan(c, st)
    legs_b, st = Home().plan(c, st)
    # same chain (no contact between them) and b planned from a's end
    assert legs_a[0].chain == legs_b[0].chain
    assert list(legs_b[0].traj.points[0].positions) == list(legs_a[0].goal_joints)


def test_place_sequence_and_release(ctx):
    import pytest

    from rammp_box_opening.primitives.core import (
    tcp_z,
        CARRY_CLEAR_M,
        SETDOWN_OVERDRIVE_M,
    )

    c = ctx
    target_z = -0.07 + c.model.lid_dims[2]
    legs, _ = Place([0.45, -0.25, target_z], [0.5, 0.5, 0.5, 0.5]).plan(c, state())
    kinds = [leg.kind for leg in legs]
    assert kinds == [Kind.MOTION, Kind.MOTION, Kind.GRIPPER]
    transit, descend, open_ = legs
    assert descend.guard.trip == "setdown"
    assert open_.chain == descend.chain + 1  # contact breaks the chain
    assert open_.gripper_cmd == 0.0
    # the set-down plans in an interaction world capped under ITS contact
    interaction = [kw for k, kw in c.worlds.pushes if k == "interaction"]
    assert interaction and interaction[-1]["contact_z"] == pytest.approx(target_z)
    # the carried lid hangs below the fingertips, invisible to the
    # planner: the transit hover must clear the container top by a
    # lid-height plus margin (field 2026-08-26: lid clipped the box line)
    carry_floor = c.cpose.xyz[2] + c.model.dims[2] + c.model.lid_dims[2] + CARRY_CLEAR_M
    assert transit.target[1][2] == pytest.approx(tcp_z(carry_floor))
    # success is the TOUCH: the stroke overdrives past nominal surface
    # contact so an exact-height 'arrived' can't slip through untripped
    assert descend.target[1][2] == pytest.approx(target_z - SETDOWN_OVERDRIVE_M)


def test_lift_rises_from_the_last_commanded_pose_and_carries_no_verify(ctx):
    """A verify would close the lift's merge group (a dead stop before the
    carry); the slip check sits at the set-down. With no pose commanded
    yet there is nothing to rise from, and it says so."""
    import pytest

    c = ctx
    with pytest.raises(RuntimeError, match="preceding pose"):
        Lift(0.10).plan(c, state())
    c.last_pose = ([0.45, 0.0, 0.10], [0.0, 1.0, 0.0, 0.0])
    legs, _ = Lift(0.10).plan(c, state())
    assert len(legs) == 1 and legs[0].verify is None
    assert legs[0].target[1] == pytest.approx([0.45, 0.0, 0.20])


def test_press_stroke_is_one_touch_stroke_from_staging(ctx):
    import pytest

    from rammp_box_opening.models.container import load_press_demo
    from rammp_box_opening.primitives.core import PRESS_APPROACH_M, press_stroke

    c = ctx
    cfg = load_press_demo(CFG)
    press, st = press_stroke(c, state(), cfg)
    assert press.name == "press:down" and press.target[3] == PRESS_APPROACH_M  # vertical final
    button = from_container(c.cpose, c.model.button_offset)
    assert press.kind is Kind.MOTION and press.world.startswith("interaction")
    # the TOUCH stage: a light threshold that finds the surface; the push
    # from the measured contact is press_push (2026-09-03)
    assert press.guard is not None and press.guard.trip == "touch"
    assert press.guard.touch_nm == pytest.approx(cfg.contact_nm)
    assert press.speed == pytest.approx(cfg.press_speed)
    assert press.target[1][2] == pytest.approx(tcp_z(button[2] - cfg.travel_m))
    assert st.chain == press.chain + 1  # contact breaks the chain


def test_press_stroke_verify_expected_contact_semantics(ctx):
    from rammp_box_opening.models.container import load_press_demo
    from rammp_box_opening.primitives.core import press_stroke

    cfg = load_press_demo(CFG)
    press, _ = press_stroke(ctx, state(), cfg)
    expected = cfg.staging_m / (cfg.staging_m + cfg.travel_m)
    # trip near the expected contact depth = pressed
    ok, detail = press.verify(VerifyCtx(outcome="touch", progress=expected))
    assert ok and "surface found" in detail
    # trip far ABOVE the button = struck something else, honest failure
    ok, detail = press.verify(VerifyCtx(outcome="touch", progress=0.3))
    assert not ok and "EARLY" in detail
    # full travel with no trip: the touch found NOTHING — the fix is high
    # or the box moved. Not a press.
    ok, detail = press.verify(VerifyCtx(outcome="arrived"))
    assert not ok and "no surface" in detail
    ok, _ = press.verify(VerifyCtx(outcome="failed"))
    assert not ok


def test_worlds_are_pushed_at_plan_time(ctx):
    # spec §6: the planner must hold the leg's world BEFORE the plan is
    # requested — execution-time pushes alone mean every trajectory was
    # planned against the previous world (2026-08-24 review, critical)
    from rammp_box_opening.models.container import load_press_demo
    from rammp_box_opening.primitives.core import press_stroke

    c = ctx
    Home().plan(c, state())
    assert c.client.worlds_pushed == ["full.yaml"]
    press_stroke(c, state(), load_press_demo(CFG))
    assert c.client.worlds_pushed == ["full.yaml", "interaction_button.yaml"]


def test_contact_sets_the_container_pad_for_later_full_worlds(ctx):
    """Any guarded plan marks the mission contact-tainted: every later
    FULL world allows for a scooted container."""
    from rammp_box_opening.primitives.core import (
        CONTACT_SHIFT_PAD_M,
        _full_world,
        press_stroke,
    )

    c, st = ctx, state()
    assert c.contact_pad == 0.0
    _full_world(c)
    assert c.worlds.pushes[-1][1].get("container_pad_xy", 0.0) == 0.0
    from rammp_box_opening.models.container import load_press_demo

    cfg = load_press_demo(CFG)
    press_stroke(c, st, cfg)
    assert c.contact_pad == CONTACT_SHIFT_PAD_M
    _full_world(c)
    assert c.worlds.pushes[-1][1]["container_pad_xy"] == CONTACT_SHIFT_PAD_M


def test_press_push_is_bounded_from_the_measured_contact(ctx):
    """The push drives button_travel_m past the contact the fingertip TF
    measured, in the frame the planner is commanded in; arriving at the
    bound IS the press, and the old threshold is only a backstop."""
    import pytest

    from rammp_box_opening.constants import TIP_TO_TOOL_M
    from rammp_box_opening.models.container import load_press_demo
    from rammp_box_opening.primitives.core import press_push

    c = ctx
    cfg = load_press_demo(CFG)
    tip = [0.45, -0.15, 0.1024]  # what the run of 2026-09-03 15:05 read
    c.last_pose = ([0.451, -0.151, 0.09], [0.0, 1.0, 0.0, 0.0])
    push, st = press_push(c, state(), cfg, tip, ("interaction_button", "/tmp/interaction_button.yaml"))
    assert push.name == "press:push" and push.kind is Kind.MOTION
    assert push.target[1][2] == pytest.approx(tip[2] - TIP_TO_TOOL_M - cfg.button_travel_m)
    assert push.target[1][:2] == pytest.approx([0.451, -0.151])  # the touch's own xy
    assert push.guard.trip == "press"
    assert push.guard.touch_nm == pytest.approx(c.model.touch_nm - cfg.contact_nm)
    assert push.speed == pytest.approx(cfg.press_speed)
    ok, d = push.verify(VerifyCtx(outcome="arrived"))
    assert ok and "full" in d
    ok, d = push.verify(VerifyCtx(outcome="touch", torque_peak=4.1))
    assert ok and "stop" in d
    ok, _ = push.verify(VerifyCtx(outcome="failed"))
    assert not ok


def test_press_push_is_cut_from_the_touch_stroke_when_enough_remains(ctx):
    """No planner call while the fingers press: the push continues the
    touch's own trajectory from the live stop, re-timed at press_speed,
    with the same guard and verify as the planned push."""
    import numpy as np
    import pytest
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    from rammp_box_opening.constants import JOINT_ARC_PER_M, TIP_TO_TOOL_M
    from rammp_box_opening.models.container import load_press_demo
    from rammp_box_opening.primitives.core import press_push_from_touch
    from rammp_box_opening.runtime.legs import Leg

    c = ctx
    cfg = load_press_demo(CFG)
    # a touch stroke: joint_7 descends 0.5 rad over 60 samples
    msg = JointTrajectory()
    msg.joint_names = ["joint_%d" % i for i in range(1, 8)]
    for k in range(60):
        pt = JointTrajectoryPoint()
        pt.positions = [0.0] * 6 + [0.5 * k / 59]
        pt.velocities = [0.0] * 7
        pt.accelerations = [0.0] * 7
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = int((k + 1) * 0.02 * 1e9)
        msg.points.append(pt)
    touch = Leg(name="press:down", kind=Kind.MOTION, traj=msg, speed=1.0, guard=None,
                world="interaction_button", chain=0, target=None, goal_joints=list(msg.points[-1].positions))
    live = [0.0] * 6 + [0.5 * 30 / 59]  # the guard stopped at sample 30
    tip = [0.45, -0.15, 0.1024]
    c.last_pose = ([0.451, -0.151, 0.09], [0.0, 1.0, 0.0, 0.0])
    out = press_push_from_touch(c, state(), cfg, touch, live, tip, ("interaction_button", "/tmp/w.yaml"))
    assert out is not None
    push, st = out
    q = np.asarray([p.positions for p in push.traj.points])
    assert np.allclose(q[0], live)  # starts at the live stop
    arc = float(np.abs(np.diff(q[:, 6])).sum())
    assert arc == pytest.approx(cfg.button_travel_m * JOINT_ARC_PER_M, rel=0.35)
    assert push.name == "press:push" and push.guard.trip == "press" and push.speed == 1.0
    assert push.target[1][2] == pytest.approx(tip[2] - TIP_TO_TOOL_M - cfg.button_travel_m)
    ok, d = push.verify(VerifyCtx(outcome="arrived"))
    assert ok and "full" in d
    # stopped at the very end of the stroke: nothing to cut -> the caller plans
    assert press_push_from_touch(c, state(), cfg, touch, list(msg.points[-1].positions), tip,
                                 ("interaction_button", "/tmp/w.yaml")) is None
