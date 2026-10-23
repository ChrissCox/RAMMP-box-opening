import pytest

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


PINK = "src/rammp_box_opening/config/containers/ankou_pink.yaml"
FLATBOT_URDF = "src/rammp_box_opening/test/data/arm_flatbot_20261021.urdf"
# where the pink canister's button stood under the press: flatbot (arm_mount,
# run 2026-10-22 20:04's contact) and the bench (base_link)
ARMS = {"flatbot": (FLATBOT_URDF, "arm_mount", [0.4824, -0.1572, 0.3313]),
        "bench": (None, "base_link", [0.45, 0.0, 0.20])}


def _press_at(ctx, monkeypatch, arm, tool_contact_off_m, braked_m):
    """The arm (a real URDF) stopped straight down over the button with the
    tool `braked_m` below where the fingertip TF read the contact, which was
    `tool_contact_off_m` from the arm's nominal contact height."""
    from rammp_box_opening.constants import HOME, TIP_TO_TOOL_M
    from rammp_box_opening.models.container import ContainerModel, attitude_quat
    from rammp_box_opening.runtime import approach
    from rammp_box_opening.runtime.kinematics import ArmChain
    from rammp_box_opening.runtime.legs import Leg

    path, root, (x, y, z) = ARMS[arm]
    chain = ArmChain(path=path, root=root)
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    ctx.model = ContainerModel.load(PINK)
    quat = attitude_quat([180.0, 0.0, 90.0], 0.0)
    tool_at_contact = z + tool_contact_off_m
    live = chain.nearest_solution(HOME, [x, y, tool_at_contact - braked_m], quat)
    assert live is not None
    ctx.client.live = [float(v) for v in live]
    ctx.last_pose = ([x, y, tool_at_contact - 0.015], list(quat))
    touch = Leg(name="press:down", kind=Kind.MOTION, traj=approach.line_trajectory(ctx.client.joint_names, [live, live]),
                speed=0.35, guard=None, world="interaction_button", chain=0, target=None, goal_joints=list(live))
    tip = [x, y, tool_at_contact + TIP_TO_TOOL_M]
    return chain, touch, ctx.client.joints(), tip


@pytest.mark.parametrize("arm", ["flatbot", "bench"])
@pytest.mark.parametrize("contact_mm", [-3.7, 0.0, 6.2])  # the bench's logged contact spread
@pytest.mark.parametrize("braked_mm", [0.0, 1.5])
def test_the_push_is_button_travel_m_straight_down_from_the_measured_contact(ctx, monkeypatch, arm, contact_mm, braked_mm):
    """The push is the arm's own straight line to exactly button_travel_m
    below the MEASURED contact: wherever the contact came in the touch
    stroke and however far the arm braked past it. Cut from the touch's own
    continuation by a bench joint-arc constant it flew 7.5-11 mm on flatbot
    and still reported a full 12 mm push."""
    import numpy as np

    from rammp_box_opening.constants import TIP_TO_TOOL_M
    from rammp_box_opening.models.container import load_press_demo
    from rammp_box_opening.primitives.core import press_push_from_touch

    cfg = load_press_demo(PINK)
    chain, touch, live, tip = _press_at(ctx, monkeypatch, arm, contact_mm / 1000.0, braked_mm / 1000.0)
    out = press_push_from_touch(ctx, state(), cfg, touch, live, tip, ("interaction_button", "/tmp/w.yaml"))
    assert out is not None
    push, st = out
    q = np.asarray([p.positions for p in push.traj.points])
    assert np.allclose(q[0], live)  # starts at the live stop
    bound = tip[2] - TIP_TO_TOOL_M - cfg.button_travel_m
    xyz = np.asarray([chain.fk(v)[1] for v in q])
    assert xyz[-1][2] == pytest.approx(bound, abs=5e-4)  # exactly the bound, from the contact
    assert np.abs(xyz[:, :2] - xyz[0][:2]).max() < 5e-4  # straight down: the press does not slide
    assert (np.diff(xyz[:, 2]) <= 1e-6).all()
    assert push.name == "press:push" and push.guard.trip == "press" and push.speed == 1.0 and push.plan_free
    assert push.guard.touch_nm == pytest.approx(ctx.model.touch_nm - cfg.contact_nm)
    assert push.target[1][2] == pytest.approx(bound) and st.joints == pytest.approx(list(q[-1]))
    ok, d = push.verify(VerifyCtx(outcome="arrived"))
    assert ok and "full" in d


def test_no_fk_push_when_the_arm_is_at_its_bound_or_far_from_the_contact(ctx, monkeypatch):
    """At or past the bound there is nothing to fly; a live tool far above
    the contact TF reads means the chain and TF disagree, and a line from it
    would not be button_travel_m: the caller plans the push instead."""
    from rammp_box_opening.models.container import load_press_demo
    from rammp_box_opening.primitives.core import PUSH_FK_SLACK_M, press_push_from_touch

    cfg = load_press_demo(PINK)
    w = ("interaction_button", "/tmp/w.yaml")
    _c, touch, live, tip = _press_at(ctx, monkeypatch, "flatbot", 0.0, cfg.button_travel_m + 0.0005)
    assert press_push_from_touch(ctx, state(), cfg, touch, live, tip, w) is None
    _c, touch, live, tip = _press_at(ctx, monkeypatch, "flatbot", 0.0, -(PUSH_FK_SLACK_M + 0.002))
    assert press_push_from_touch(ctx, state(), cfg, touch, live, tip, w) is None


def test_a_refused_push_line_falls_back_to_the_planned_push(ctx, monkeypatch):
    from rammp_box_opening.constants import TIP_TO_TOOL_M
    from rammp_box_opening.models.container import load_press_demo
    from rammp_box_opening.tasks import press_demo

    cfg = load_press_demo(PINK)
    chain, touch, _live, tip = _press_at(ctx, monkeypatch, "flatbot", 0.0, 0.0)
    monkeypatch.setattr(chain, "straight_line", lambda *a, **k: (None, "near a singularity"))
    push = press_demo.build_push_legs(ctx, cfg, tip, touch_leg=touch)[0]
    assert push.name == "press:push" and not push.plan_free  # planned (press_push)
    assert push.target[1][2] == pytest.approx(tip[2] - TIP_TO_TOOL_M - cfg.button_travel_m)
    assert push.recoil_via is touch.traj  # the recoil still carries on up the touch
