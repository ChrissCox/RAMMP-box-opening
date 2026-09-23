import math
import time

import pytest
from dataclasses import replace
from pathlib import Path

from conftest import FakeClient, build_demo_legs, predicted_contact

from rammp_box_opening.models.container import ContainerPose, load_press_demo
from rammp_box_opening.primitives.core import tcp_z
from rammp_box_opening.runtime.legs import Kind

CFG = "src/rammp_box_opening/config/containers/oxo_pop.yaml"


def names(legs):
    return [leg.name for leg in legs]


def test_entry_points_registered():
    setup = Path("src/rammp_box_opening/setup.py").read_text()
    for ep in ["press_demo", "home_arm", "preflight", "owl_detector"]:
        assert ep + " = " in setup


def test_home_arm_plans_home_in_the_bench_world(ctx):
    """The isolated recovery home knows no container pose: it plans above
    the unseen-container band, like the mission's own recovery home."""
    from rammp_box_opening.constants import HOME, TRANSIT_SPEED
    from rammp_box_opening.tasks import home_arm

    c = ctx
    c.cpose = None
    legs = home_arm.build_legs(c)
    assert names(legs) == ["home"]
    assert legs[0].world == "bench" and legs[0].speed == TRANSIT_SPEED
    assert legs[0].target == ("joints", list(HOME))


def _demo_cfg():
    from rammp_box_opening.models.container import load_press_demo

    return load_press_demo(CFG)


def test_press_demo_full_composition(ctx):
    """The one-shot composition: staged press, grip, place — and the
    press phase's retreat geometry."""
    import pytest

    from rammp_box_opening.constants import TRANSIT_SPEED
    from rammp_box_opening.models.container import from_container

    c = ctx
    cfg = _demo_cfg()
    legs = build_demo_legs(c, cfg)
    seq = names(legs)
    # press:close is no longer a leg: main() dispatches it at fix commit
    # two-stage press: the touch finds the surface, the push is bounded
    # from the measured contact (offline: the predicted one)
    assert seq[:5] == ["approach:staging", "press:down", "press:push", "retreat", "grip:open"]
    assert seq[5:8] == ["grip:down", "grip:close", "lift"]
    assert seq[8:] == [
        "place:lid:transit",
        "place:lid:down",
        "place:lid:open",
        "retreat",
        "home",
    ]
    button = from_container(c.cpose, c.model.button_offset)
    staging = legs[0]
    assert staging.world.startswith("full") and staging.speed == TRANSIT_SPEED
    assert staging.target[1][2] == pytest.approx(button[2] + cfg.staging_m)
    retreat = legs[3]  # after approach, touch, push
    assert retreat.traj is None  # lazy: planned from live after the push
    assert legs[4].defer_join  # the fingers open on arrival at the hop
    assert retreat.world.startswith("interaction")
    assert retreat.speed == pytest.approx(TRANSIT_SPEED)  # fast up
    # the OPEN-BOX composition re-descends straight away, so the retreat
    # stops at the hop height instead of climbing back to staging — a
    # 253 mm round trip for a 2 mm reposition (speed pass 2026-08-28)
    assert retreat.target[1][2] == pytest.approx(tcp_z(button[2] + cfg.grip_hop_m))
    assert cfg.grip_hop_m < cfg.staging_m


def test_the_press_descends_straight_out_of_its_approach(ctx):
    """The reach and the touch are ONE motion. The descent is planned from
    where the approach ENDS, so the two chain into a single re-timed
    trajectory instead of two goals with a stop between them."""
    import pytest

    from rammp_box_opening.runtime.legs import can_merge

    legs = build_demo_legs(ctx, _demo_cfg())
    approach, down = legs[0], legs[1]
    assert approach.name == "approach:staging" and down.name == "press:down"
    assert list(down.traj.points[0].positions) == pytest.approx(approach.goal_joints)
    assert can_merge(approach, down)


def test_the_search_looks_then_sweeps_each_way(ctx):
    """The look is the arm's own rest pose with the wrist turned down — not a
    bench Cartesian pose, which measured 3.89 s against 1.72 s — and when it
    sees nothing the base pans each way to find the box."""
    import pytest

    from rammp_box_opening.constants import HOME
    from rammp_box_opening.models.container import load_press_demo
    from rammp_box_opening.primitives.look import look_joints
    from rammp_box_opening.tasks import press_demo

    cfg = load_press_demo(CFG)
    steps = press_demo.search_targets(cfg)
    assert [name for name, _q, _speed in steps] == ["look", "sweep:left", "sweep:right"]
    assert steps[0][1] == pytest.approx(look_joints(HOME))
    # the sweep is slower than the look: it exists to see, not to arrive
    assert steps[1][2] < steps[0][2]


def test_a_search_leg_is_joint_space_and_ends_on_a_sighting(ctx):
    """Joint-space so no bench frame is involved, bench world so a transit
    before detection stays above anything that could be standing there, and
    stop_when so the motion ends the moment the camera sees a box."""
    from rammp_box_opening.constants import HOME, TRANSIT_SPEED
    from rammp_box_opening.primitives.look import look_joints
    from rammp_box_opening.tasks import press_demo

    seen = {"fix": None}

    class Watcher:
        def coarse_fix(self):
            return seen["fix"]

    leg = press_demo.build_search_leg(
        ctx, ("look", look_joints(HOME), TRANSIT_SPEED), [0.0] * 7, Watcher()
    )
    assert leg.target[0] == "joints"
    assert leg.world == "bench" and leg.kind is Kind.MOTION
    assert leg.stop_when() is False
    seen["fix"] = ((0.45, 0.0, 0.085), 0.0)
    assert leg.stop_when() is True


FIX = ((0.45, 0.0, 0.085), 0.0)


class _Watcher:
    """Detector double. `sees_when` says when the box is in view — the tests
    express that as a state of the run, e.g. "after the left sweep"."""

    def __init__(self, sees_when=None, coarse=False):
        self.active = False
        self.roi = None
        self.sees_when = sees_when or (lambda: False)
        self.coarse = coarse
        self.last_reject = None  # what the real watcher's frame dump reads

    def coarse_fix(self, now=None):
        return FIX if self.coarse else None

    def last_coarse(self):
        return FIX if self.coarse else None

    def tick_now(self):
        pass  # a double has no frames to process

    def fix(self, now=None):
        return FIX if self.sees_when() else None

    def status(self):
        return "no box"


def _search_env(monkeypatch):
    """press_demo with rclpy stubbed out and the settle beat shortened."""
    from rammp_box_opening.tasks import press_demo

    class Spin:
        @staticmethod
        def spin_once(node, timeout_sec=0.0):
            return None

    monkeypatch.setattr(press_demo, "rclpy", Spin)
    monkeypatch.setattr(press_demo, "SETTLE_S", 0.01)
    return press_demo


def test_the_search_stops_at_the_step_that_sees_the_box(ctx, tmp_path, monkeypatch):
    """A sweep only runs because the look before it saw nothing, and the fix
    the press is aimed with is the precise one, taken at rest."""
    from conftest import runner as make_runner

    press_demo = _search_env(monkeypatch)
    cfg = load_press_demo(CFG)
    r = make_runner(ctx.client, tmp_path)
    # the box is out of the look's view; the left sweep brings it in
    watcher = _Watcher(sees_when=lambda: len(ctx.client.joint_targets) >= 2)
    got, failed = press_demo.search_for_box(None, ctx, cfg, r, watcher, True)
    assert failed is None
    assert got == FIX
    planned = ctx.client.joint_targets
    assert len(planned) == 2  # the right-hand sweep was never needed
    assert planned[1][0] > planned[0][0]  # left first


def test_the_search_skips_a_step_the_arm_already_stands_at(ctx, tmp_path, monkeypatch):
    """Parked at the look pose between runs: no flight, straight to looking."""
    from conftest import runner as make_runner

    from rammp_box_opening.constants import HOME
    from rammp_box_opening.primitives.look import look_joints

    press_demo = _search_env(monkeypatch)
    cfg = load_press_demo(CFG)
    ctx.client.live = look_joints(HOME)
    r = make_runner(ctx.client, tmp_path)
    got, failed = press_demo.search_for_box(
        None, ctx, cfg, r, _Watcher(sees_when=lambda: True), True
    )
    assert got is not None and failed is None
    assert ctx.client.executed == []  # nothing flew


def test_the_search_names_the_step_whose_motion_failed(ctx, tmp_path, monkeypatch):
    from conftest import runner as make_runner

    press_demo = _search_env(monkeypatch)
    cfg = load_press_demo(CFG)
    ctx.client.exec_script = [("failed", {"message": "driver said no", "progress": 0.0})]
    r = make_runner(ctx.client, tmp_path)
    got, failed = press_demo.search_for_box(None, ctx, cfg, r, _Watcher(), True)
    assert got is None and failed == "look"


def test_the_search_that_finds_nothing_reports_no_box(ctx, tmp_path, monkeypatch):
    from conftest import runner as make_runner

    press_demo = _search_env(monkeypatch)
    cfg = load_press_demo(CFG)
    r = make_runner(ctx.client, tmp_path)
    got, failed = press_demo.search_for_box(None, ctx, cfg, r, _Watcher(), True)
    assert got is None and failed is None
    assert len(ctx.client.joint_targets) == 3  # look, then both ways


def test_a_coarse_sighting_that_never_confirms_cannot_eat_the_search(
    ctx, tmp_path, monkeypatch
):
    """A box glimpsed while moving but never confirmed at rest must not
    strand the arm staring at it: every step gets a bounded beat, the whole
    search shares ONE budget, and the directions not yet looked at still
    get looked at."""
    from conftest import runner as make_runner

    press_demo = _search_env(monkeypatch)
    cfg = replace(load_press_demo(CFG), timeout_s=1.0)
    r = make_runner(ctx.client, tmp_path)
    watcher = _Watcher(coarse=True)  # always glimpsed, never confirmed
    t0 = time.monotonic()
    got, failed = press_demo.search_for_box(None, ctx, cfg, r, watcher, True)
    spent = time.monotonic() - t0
    assert got is None and failed is None
    assert len(ctx.client.joint_targets) == 3  # every direction was tried
    assert spent < cfg.timeout_s + 1.0  # and the budget held






def test_press_demo_no_tag_home_uses_the_bench_world(ctx):
    from conftest import FakeStore

    from rammp_box_opening.tasks import press_demo

    c = replace(ctx, worlds=FakeStore())
    home = press_demo.build_home_leg(c, [0.0] * 7)
    assert home.world == "bench"


def test_open_box_grip_and_place_legs(ctx):
    import pytest

    from rammp_box_opening.constants import TRANSIT_SPEED
    from rammp_box_opening.models.container import from_container
    from rammp_box_opening.runtime.legs import VerifyCtx
    from rammp_box_opening.tasks import press_demo

    c = ctx
    cfg = _demo_cfg()
    legs = press_demo.build_grip_legs(c, cfg)
    assert names(legs) == ["grip:down", "grip:close", "lift"]
    down, close, lift = legs
    assert close.gripper_cmd == 0.8
    button = from_container(c.cpose, c.model.button_offset)
    # ABOVE the tag/lid plane (press depth = into the lid — field
    # 2026-08-26), with the bench-measured lateral trim applied
    # the FINGERTIPS land grip_clear_m above the button; the frame the
    # planner is commanded in goes tcp_z higher (constants.TCP_OFFSET_M)
    assert down.target[1][2] == pytest.approx(tcp_z(button[2] + cfg.grip_clear_m))
    assert down.target[1][0] == pytest.approx(button[0] + cfg.grip_offset_xy[0])
    assert down.target[1][1] == pytest.approx(button[1] + cfg.grip_offset_xy[1])
    assert down.guard is not None and down.guard.trip == "obstruction"
    assert close.chain == down.chain + 1  # contact breaks the chain
    # closed-on-air (0.8) fails the band; holding the knob passes
    ok, _ = close.verify(VerifyCtx(outcome="arrived", gripper_pos=0.8))
    assert not ok
    held = (cfg.grip_band[0] + cfg.grip_band[1]) / 2
    ok, _ = close.verify(VerifyCtx(outcome="arrived", gripper_pos=held))
    assert ok
    # bench 2026-08-26: the real knob reads 0.387 — the band must hold it
    ok, _ = close.verify(VerifyCtx(outcome="arrived", gripper_pos=0.387))
    assert ok
    assert lift.speed == pytest.approx(cfg.lift_speed)  # "slowly lift"
    assert lift.target[1][2] == pytest.approx(tcp_z(button[2] + cfg.grip_clear_m) + cfg.lift_m)

    place_legs = press_demo.build_place_legs(c, cfg)
    assert names(place_legs) == [
        "place:lid:transit",
        "place:lid:down",
        "place:lid:open",
        "retreat",
        "home",
    ]
    from rammp_box_opening.primitives.core import CARRY_CLEAR_M

    # carry clears the container even directly overhead (held lid unmodeled)
    carry_floor = c.cpose.xyz[2] + c.model.dims[2] + c.model.lid_dims[2] + CARRY_CLEAR_M
    assert place_legs[0].target[1][2] >= carry_floor - 1e-9
    assert place_legs[1].guard.trip == "setdown"
    assert place_legs[2].gripper_cmd == 0.0
    # a release: dispatched at once, joined before the retreat MOVES
    assert place_legs[2].defer_join and place_legs[2].join_before_motion
    assert c.lid_at is not None  # the placed lid joins later worlds
    assert "lid" in place_legs[4].world  # home plans around the placed lid
    assert place_legs[3].speed == TRANSIT_SPEED
    # both lazy, one chain: planned from live as one group after the touch
    assert place_legs[3].traj is None and place_legs[4].traj is None
    assert place_legs[3].chain == place_legs[4].chain
    # the retreat climbs to the CARRY height, where home is plannable
    assert place_legs[3].target[1][2] == pytest.approx(place_legs[0].target[1][2])


def test_lid_place_clearance_gate_threshold(ctx):
    import pytest

    from rammp_box_opening.tasks import press_demo

    c = ctx
    need = press_demo.lid_place_min_clear(c.model)
    # both footprint half-diagonals + gripper-body room; the field runs on
    # the 2.9-inch box bracketed it: IK_FAIL at 0.073 m separation, clean
    # plan at 0.162 m (0.156 for that box). The shipped 4.1-inch box needs
    # 0.197, and the axis spots at that distance planned 32/32 (2026-09-17)
    assert 0.073 < need < 0.25
    import math

    assert need == pytest.approx(
        (math.hypot(*c.model.dims[:2]) + math.hypot(*c.model.lid_dims[:2])) / 2 + 0.05
    )


def test_the_lid_goes_down_beside_the_box_wherever_the_box_is(ctx):
    """The drop spot is derived from the DETECTED box every run, never
    surveyed: a fixed base-frame point assumes a fixed table, which is the
    thing a wheelchair takes away. The configured lid_place survives only as
    the side to try first, and the table height comes from the camera."""
    from rammp_box_opening.tasks import press_demo

    c = ctx
    m = c.model
    need = press_demo.lid_place_min_clear(m)
    prefer = [0.45, -0.25, -0.027]  # the configured side: -y
    for box_xy in ((0.45, 0.1), (0.396, -0.24), (0.45, -0.25), (0.35, 0.2)):
        box = ContainerPose(xyz=(box_xy[0], box_xy[1], -0.02), yaw=0.0)
        xyz, _side = press_demo.box_relative_lid_drop(m, box, -0.031, prefer)
        assert xyz is not None
        assert math.hypot(xyz[0] - box_xy[0], xyz[1] - box_xy[1]) >= need - 1e-9
        assert press_demo.DROP_X_M[0] <= xyz[0] <= press_demo.DROP_X_M[1]
        assert press_demo.DROP_Y_M[0] <= xyz[1] <= press_demo.DROP_Y_M[1]
        assert xyz[2] == -0.031  # the table the camera measured, not the yaml
    # the preferred side is the one taken when it fits
    middle = ContainerPose(xyz=(0.45, 0.0, -0.02), yaw=0.0)
    xyz, _side = press_demo.box_relative_lid_drop(m, middle, -0.027, prefer)
    assert xyz[1] < 0.0
    # a box well outside the set-down zone leaves nowhere to put the lid:
    # refuse and say so, rather than invent a spot the arm cannot reach
    far_out = ContainerPose(xyz=(0.90, 0.0, -0.02), yaw=0.0)
    xyz, _side = press_demo.box_relative_lid_drop(m, far_out, -0.027, prefer)
    assert xyz is None


def test_the_lid_side_snaps_to_the_base_axis_nearest_the_preferred_one(ctx):
    """Field 2026-09-17: box at (0.5625, -0.0247), lid_place (0.45, -0.25)
    — the exact direction is 27 deg off -y, and the carry pose there had no
    IK in any attitude (the planner's padded container is a base-aligned
    cuboid, so a diagonal spot sits nearer its corner). The -y axis spot
    beside the same box plans."""
    import pytest

    from rammp_box_opening.tasks import press_demo

    m = ctx.model
    need = press_demo.lid_place_min_clear(m)
    box = ContainerPose(xyz=(0.5625, -0.0247, -0.027), yaw=math.radians(10.3))
    xyz, side = press_demo.box_relative_lid_drop(m, box, -0.027, [0.45, -0.25, -0.027])
    assert side == (0.0, -1.0)
    assert xyz[0] == pytest.approx(0.5625) and xyz[1] == pytest.approx(-0.0247 - need)
    # every candidate lies on a base axis through the box, preferred side first
    spots = press_demo.lid_drop_candidates(m, box, -0.027, [0.45, -0.25, -0.027])
    assert [d for _xyz, d in spots][0] == (0.0, -1.0)
    assert all(d in ((0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 0.0)) for _xyz, d in spots)


class _RefusesPlacesAt:
    """A planner that refuses every pose above the given drop spots."""

    def __init__(self, client, refused_xy):
        self._c = client
        self.refused_xy = [tuple(xy) for xy in refused_xy]

    def __getattr__(self, name):
        return getattr(self._c, name)

    def plan_to_pose(self, xyz, quat_xyzw, start_joints, approach_offset_m=0.0):
        for rx, ry in self.refused_xy:
            if abs(xyz[0] - rx) < 1e-3 and abs(xyz[1] - ry) < 1e-3:

                class Bad:
                    success = False
                    message = "MotionGenStatus.IK_FAIL: no collision-free joint solution AT the goal"

                return Bad()
        return self._c.plan_to_pose(xyz, quat_xyzw, start_joints, approach_offset_m)


def test_a_drop_spot_the_planner_refuses_falls_through_to_the_next(ctx):
    import pytest

    from rammp_box_opening.tasks import press_demo

    c = ctx
    need = press_demo.lid_place_min_clear(c.model)
    bx, by = c.cpose.xyz[0], c.cpose.xyz[1]
    first = (bx, by - need)  # the -y spot the config prefers
    c.lid_drop = ContainerPose(xyz=(first[0], first[1], -0.027), yaw=0.0)
    c.client = _RefusesPlacesAt(c.client, [first])
    legs = press_demo.build_place_legs(c, _demo_cfg())
    transit = next(x for x in legs if x.name == "place:lid:transit")
    # the next side in order: +y
    assert transit.target[1][0] == pytest.approx(bx)
    assert transit.target[1][1] == pytest.approx(by + need)
    assert c.lid_drop.xyz[1] == pytest.approx(by + need)
    assert c.lid_at is c.lid_drop  # the placed-lid world follows the switch


def test_no_plannable_drop_spot_is_one_clear_refusal(ctx):
    import pytest

    from rammp_box_opening.tasks import press_demo

    c = ctx
    spots = press_demo.lid_drop_candidates(c.model, c.cpose, -0.027, [0.45, -0.25, -0.027])
    c.lid_drop = ContainerPose(xyz=tuple(spots[0][0]), yaw=0.0)
    c.client = _RefusesPlacesAt(c.client, [xyz[:2] for xyz, _d in spots])
    with pytest.raises(RuntimeError) as e:
        press_demo.build_place_legs(c, _demo_cfg())
    assert "no drop spot beside the box plans" in str(e.value)
    assert c.lid_at is None


def test_place_legs_use_the_resolved_drop_spot(ctx):
    import pytest

    from rammp_box_opening.tasks import press_demo

    c = ctx
    c.lid_drop = ContainerPose(xyz=(0.30, 0.30, -0.027), yaw=0.0)
    legs = press_demo.build_place_legs(c, _demo_cfg())
    transit = legs[0]
    assert transit.target[1][0] == pytest.approx(0.30)
    assert transit.target[1][1] == pytest.approx(0.30)
    assert c.lid_at is c.lid_drop  # the placed-lid world follows the shift


def test_the_lid_comes_off_and_goes_down_in_one_motion(ctx):
    """Lifting the lid, carrying it and setting it down are one continuous
    execution. They used to be three goals with two full stops in the middle,
    which is most of what read as choppy — nothing happens between them that
    the arm needs to stop for."""
    from rammp_box_opening.runtime.legs import merge_groups
    from rammp_box_opening.tasks import press_demo

    legs = press_demo.build_grip_and_place_legs(ctx, _demo_cfg())
    groups = [[x.name for x in g] for g in merge_groups(legs)]
    assert ["lift", "place:lid:transit", "place:lid:down"] in groups
    # ... and the set-down still owns the contact
    setdown = next(x for x in legs if x.name == "place:lid:down")
    assert setdown.guard is not None and setdown.guard.trip == "setdown"
    assert setdown.warp is None  # the group profile does the fast-then-slow


def test_contact_leg_speeds_come_from_config(ctx):
    """grip:down, lift and the set-down each read their own config knob.

    They were three hardcoded 0.15s; the set-down and lift now run at
    0.35 while grip:down stays conservative until the bench ramp."""
    import pytest

    from rammp_box_opening.tasks import press_demo

    c = ctx
    cfg = _demo_cfg()
    grip = press_demo.build_grip_legs(c, cfg)
    down = next(x for x in grip if x.name == "grip:down")
    lift = next(x for x in grip if x.name == "lift")
    # grip:down is time-warped, so its contact scale lives in leg.warp and
    # leg.speed is 1.0 (the profile is already baked into the timing)
    assert down.warp[1] == pytest.approx(cfg.grip_speed)
    assert lift.speed == pytest.approx(cfg.lift_speed)

    c.lid_drop = None
    place = press_demo.build_place_legs(c, cfg)
    setdown = next(x for x in place if x.name == "place:lid:down")
    assert setdown.speed == pytest.approx(cfg.setdown_speed)
    # the set-down keeps its guard: speed rose, the trip=success did not move
    assert setdown.guard is not None and setdown.guard.trip == "setdown"


def test_shipped_config_speeds_are_guard_safe():
    """The shipped values are what actually runs at the bench."""
    import pytest

    from rammp_box_opening.models.container import load_press_demo

    cfg = load_press_demo(CFG)
    assert cfg.lift_speed == pytest.approx(0.35)
    assert cfg.setdown_speed == pytest.approx(0.35)
    # 0.15 -> 0.35 (owner, 2026-09-16): the set-down has run guarded at 0.35
    # since 2026-09-02; the grip descent gets the same
    assert cfg.grip_speed == pytest.approx(0.35)
    assert cfg.detect_period_s == pytest.approx(0.05)
    # every contact/carry speed stays inside the guard-limited band
    for v in (cfg.grip_speed, cfg.setdown_speed, cfg.lift_speed):
        assert 0.0 < v <= 0.5


def test_guarded_descents_are_time_warped_and_rebaseline_the_guard(ctx):
    """grip:down and the set-down run fast through free air and slow into
    contact, and the guard re-baselines where the speed changes."""
    import pytest

    from rammp_box_opening.tasks import press_demo

    c = ctx
    cfg = _demo_cfg()
    grip = press_demo.build_grip_legs(c, cfg)
    down = next(x for x in grip if x.name == "grip:down")
    assert down.warp == (cfg.warp_fast_speed, cfg.grip_speed, cfg.warp_slow_frac)
    # the profile is baked into the timing, so it must NOT be dilated again
    assert down.speed == pytest.approx(1.0)
    # ...and the guard is told where the regime changes
    assert down.guard is not None and down.guard.rebaseline_after is not None
    assert 0.0 < down.guard.rebaseline_after <= 1.0
    assert down.guard.trip == "obstruction"  # semantics unchanged

    # the set-down goes through the same _apply_warp; the fake planner
    # returns a zero-length descent for it, so exercise the hook directly
    # on a trajectory that actually moves
    from rammp_box_opening.runtime.guards import GuardSpec

    moving = next(x for x in grip if x.name == "grip:down")
    probe = replace(moving, warp=None, speed=cfg.setdown_speed)
    probe.guard = GuardSpec(touch_nm=6.0, trip="setdown")
    press_demo._apply_warp(probe, cfg, cfg.setdown_speed)
    assert probe.warp == (cfg.warp_fast_speed, cfg.setdown_speed, cfg.warp_slow_frac)
    assert probe.guard.trip == "setdown" and probe.guard.rebaseline_after is not None


def test_warping_is_off_when_the_config_disables_it(ctx):
    from dataclasses import replace as _replace

    from rammp_box_opening.tasks import press_demo

    c = ctx
    cfg = _replace(_demo_cfg(), warp_fast_speed=0.0)
    down = next(x for x in press_demo.build_grip_legs(c, cfg) if x.name == "grip:down")
    assert down.warp is None
    assert down.speed == cfg.grip_speed  # plain single-scale behaviour
    assert down.guard.rebaseline_after is None














def test_every_descent_carries_a_vertical_final_constraint(ctx):
    """The staged press arched into the button edge exactly like the
    merged press did before it got the grasp-approach constraint (field
    2026-09-01) — and a bowed grip or set-down misses the same way. Every
    contact-bound descent now ends vertical; the constraint rides in the
    target tuple so drift replans preserve it too."""
    import pytest

    c = ctx
    cfg = _demo_cfg()
    legs = build_demo_legs(c, cfg)
    want = {"press:down": 0.06, "grip:down": 0.04, "place:lid:down": 0.05}
    for name, off in want.items():
        leg = next(x for x in legs if x.name == name)
        assert len(leg.target) == 4, name
        assert leg.target[3] == pytest.approx(off), name


def test_post_touch_legs_are_lazy_and_home_needs_no_fallback(ctx):
    """Legs after an expected touch used to be pre-planned and then thrown
    away by the post-touch replan every run; home was even planned twice
    (retreat-end refused, transit-end fallback) and then failed live
    anyway. Now they are LAZY — no plan call at build time — and the
    Runner plans them once, from live, as one group (audit 2026-09-02)."""
    from rammp_box_opening.tasks import press_demo

    class CountingClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.joint_plans = 0

        def plan_to_joints(self, q7, start_joints):
            self.joint_plans += 1
            return super().plan_to_joints(q7, start_joints)

    c = ctx
    c.client = CountingClient()
    legs = press_demo.build_place_legs(c, _demo_cfg())
    assert names(legs)[-2:] == ["retreat", "home"]
    assert c.client.joint_plans == 0  # home is not planned at build time
    assert all(x.traj is None for x in legs[-2:])
    # build-time plans: transit + set-down only
    assert len(c.client.approach_offsets) == 2


def test_place_accounts_for_grip_height_and_gentle_touch(ctx):
    """The fingers hold the knob grip_clear_m above the lid plane, so lid
    contact happens with the TOOL that much above lid-top height — the
    uncompensated target over-travelled by grip_clear_m and crunched the
    lid into the table without tripping the press-strength guard (field
    2026-09-02). The set-down also gets its own gentler threshold."""
    import pytest

    from rammp_box_opening.tasks import press_demo

    c = ctx
    cfg = _demo_cfg()
    legs = press_demo.build_place_legs(c, cfg)
    down = next(x for x in legs if x.name == "place:lid:down")
    from rammp_box_opening.primitives.core import SETDOWN_OVERDRIVE_M
    from rammp_box_opening.tasks.press_demo import load_lid_place

    lid = load_lid_place(CFG)
    want = tcp_z(lid.xyz[2] + c.model.lid_dims[2] + cfg.grip_clear_m) - SETDOWN_OVERDRIVE_M
    assert down.target[1][2] == pytest.approx(want)
    assert down.guard.touch_nm == pytest.approx(cfg.setdown_touch_nm)
    assert cfg.setdown_touch_nm < c.model.touch_nm  # gentler than the press


def test_setdown_verify_rejects_early_trips_and_no_touch(ctx):
    """A trip in the first half of the stroke is a strike, not a set-down
    (the lid was dropped from 110 mm when a fast-segment trip counted as
    touch, field 2026-09-02); arriving without ever feeling the surface
    stays a failure. Only a late trip confirms the set-down."""
    from rammp_box_opening.runtime.legs import VerifyCtx
    from rammp_box_opening.tasks import press_demo

    c = ctx
    cfg = _demo_cfg()
    legs = press_demo.build_place_legs(c, cfg)
    down = next(x for x in legs if x.name == "place:lid:down")
    assert down.guard.arm_after is not None and down.guard.arm_after >= 0.5

    def v(outcome, progress):
        return down.verify(
            VerifyCtx(outcome=outcome, progress=progress, torque_peak=4.2)
        )

    ok, why = v("touch", 0.06)
    assert not ok and "NOT confirmed" in why
    ok, why = v("touch", 0.9)
    assert ok
    ok, why = v("arrived", 1.0)
    assert not ok and "never felt the surface" in why


def test_park_tool_down_rests_at_the_look_pose(ctx):
    """open_box.park_tool_down: the mission ends tool-down instead of at the
    factory HOME, saving the 2.4-2.9 rad wrist flip twice per run. That rest
    is now the LOOK pose — HOME's own wrist turned down — not the surveyed
    joint vector over a fixed bench point it used to be, so parking costs the
    search nothing and survives the arm being moved."""
    import math

    from rammp_box_opening.constants import HOME, REST_TOL_RAD
    from rammp_box_opening.primitives.look import look_joints
    from rammp_box_opening.tasks import press_demo

    c = ctx
    cfg = _demo_cfg()
    assert cfg.park_tool_down is False  # the run ends at HOME (owner 2026-09-04)
    assert press_demo.rest_joints(cfg) == list(HOME)
    on = replace(cfg, park_tool_down=True)
    park = look_joints(HOME)
    assert press_demo.rest_joints(on) == park  # the switch still works
    # parked there, the search's first step is already done
    assert press_demo.search_targets(on)[0][1] == park
    legs = press_demo.build_place_legs(c, on)
    assert legs[-1].name == "home" and legs[-1].target == ("joints", park)
    # "already parked" is judged against the server's own start gate
    assert press_demo.rest_distance(park, park) == 0.0
    nudged = list(park)
    nudged[3] += REST_TOL_RAD * 2
    assert press_demo.rest_distance(nudged, park) > REST_TOL_RAD
    assert press_demo.rest_distance([math.pi] + park[1:], park) > 1.0


def test_home_arm_falls_back_to_the_bare_table_when_the_band_refuses_the_start(ctx):
    """A recovery home usually starts INSIDE the unseen-container band (the
    arm is holding at a press or a grip), which made the guarded bench
    world refuse every post-contact home; it now falls back to the bare
    table with a caution (review 2026-09-02)."""
    from rammp_box_opening.tasks import home_arm

    class RefusesOnce(FakeClient):
        def __init__(self):
            super().__init__()
            self.n = 0

        def plan_to_joints(self, q7, start_joints):
            self.n += 1
            if self.n == 1:
                class Bad:
                    success = False
                    message = "INVALID_START_STATE_WORLD_COLLISION"

                return Bad()
            return super().plan_to_joints(q7, start_joints)

    c = ctx
    c.client = RefusesOnce()
    legs = home_arm.build_legs(c)
    assert legs[0].name == "home" and legs[0].world == "bench_bare"
    kinds = [kw.get("model") is None for kind, kw in c.worlds.pushes if kind == "bench"]
    assert kinds == [False, True]  # guarded band first, then the bare table


def test_press_offset_trims_the_press_target_only(ctx):
    """A base-frame trim for where the closed pads meet the lid relative to
    the tool axis; zero by default, bounded to 2 cm."""
    import pytest

    from rammp_box_opening.models.container import from_container
    from rammp_box_opening.tasks import press_demo

    c = ctx
    cfg = _demo_cfg()
    assert tuple(cfg.press_offset_xy) == (0.0, 0.0)
    button = from_container(c.cpose, c.model.button_offset)
    trimmed = replace(cfg, press_offset_xy=(-0.004, 0.002))
    (press,) = press_demo.build_press_legs(c, trimmed)
    assert press.target[1][0] == pytest.approx(button[0] - 0.004)
    assert press.target[1][1] == pytest.approx(button[1] + 0.002)
    # the grip keeps its own trim
    grip = press_demo.build_grip_legs(c, trimmed)
    assert grip[0].target[1][0] == pytest.approx(button[0] + trimmed.grip_offset_xy[0])




class _Readiness:
    """Client double for the pre-actuation readiness check."""

    def __init__(self, planner=(True,), driver=True):
        self.planner = list(planner)
        self.driver = driver
        self.planner_waits = []

    def planner_reachable(self, timeout_s=5.0):
        self.planner_waits.append(timeout_s)
        return self.planner.pop(0) if len(self.planner) > 1 else self.planner[0]

    def driver_reachable(self, timeout_s=5.0):
        return self.driver


def test_a_run_refuses_before_any_motion_when_the_planner_is_not_up():
    """Field 2026-09-16: the stack had just been restarted, the gripper
    closed, and THEN the run died on `set_world service unavailable` — cuRobo
    takes 17-21 s to load after sheppy starts it, and set_world waits 5 s.
    Readiness is checked before anything is actuated, and says what to do."""
    from rammp_box_opening.tasks import press_demo

    why = press_demo.readiness_refusal(_Readiness(planner=(False,)), execute=True)
    assert why is not None and "planner" in why and "nothing moved" in why.lower()


def test_a_planner_still_loading_is_waited_for_not_failed():
    """A restart race is a pause, not a failure: the first probe is short, and
    a planner that comes up within the load window lets the run proceed."""
    from rammp_box_opening.tasks import press_demo

    client = _Readiness(planner=(False, True))
    assert press_demo.readiness_refusal(client, execute=True) is None
    assert client.planner_waits[0] <= 1.0
    assert client.planner_waits[1] >= 20.0  # longer than cuRobo's measured load


def test_the_driver_is_required_only_when_executing():
    from rammp_box_opening.tasks import press_demo

    down = _Readiness(driver=False)
    why = press_demo.readiness_refusal(down, execute=True)
    assert why is not None and "driver" in why
    assert press_demo.readiness_refusal(_Readiness(driver=False), execute=False) is None


class _SceneFix:
    def __init__(self, xyz, yaw, top):
        from rammp_box_opening.models.container import ContainerPose

        self.pose = ContainerPose(xyz=tuple(xyz), yaw=yaw)
        self.top_xyz = tuple(top)
        self.score = 0.3
        self.n_points = 500


def test_the_scene_residual_is_the_wrist_minus_the_scene_in_mm():
    import pytest

    from rammp_box_opening.tasks import press_demo

    dx, dy, dz, dyaw = press_demo.scene_residual(
        (0.375, -0.106, 0.064), (0.425, -0.114, 0.087), math.radians(72.0), math.radians(70.0)
    )
    assert (round(dx), round(dy), round(dz)) == (50, -8, 23)
    assert dyaw == pytest.approx(-2.0)
    # yaw is mod 90: 88 deg vs 2 deg is a 4 deg disagreement, not 86
    assert press_demo.scene_residual((0, 0, 0), (0, 0, 0), math.radians(88), math.radians(2))[3] == pytest.approx(4.0)


def test_the_preplanned_descent_flies_only_when_the_precise_fix_agrees():
    from rammp_box_opening.models.container import ContainerPose
    from rammp_box_opening.tasks import press_demo

    coarse = ContainerPose(xyz=(0.40, -0.10, -0.027), yaw=0.0)
    legs = ["press:down"]
    assert press_demo.use_preplanned_descent(coarse, ContainerPose(xyz=(0.401, -0.101, -0.027), yaw=0.0), legs)
    # 3.6 mm apart: once "close enough", and the press went to the SCENE
    # camera's spot, that far off the circle's centre. The owner wants the
    # centre, and re-fitting the descent to the aim costs 0.1 s and no plan.
    assert not press_demo.use_preplanned_descent(coarse, ContainerPose(xyz=(0.403, -0.102, -0.027), yaw=0.0), legs)
    assert not press_demo.use_preplanned_descent(coarse, ContainerPose(xyz=(0.41, -0.10, -0.027), yaw=0.0), legs)
    assert not press_demo.use_preplanned_descent(coarse, coarse, None)  # nothing was planned


def test_every_run_records_its_scene_versus_wrist_pair(tmp_path):
    """The pairs are the calibration's refinement data: one per run."""
    import json

    from rammp_box_opening.tasks import press_demo

    f = tmp_path / "residuals.jsonl"
    fix = _SceneFix((0.375, -0.106, -0.027), 1.25, (0.375, -0.106, 0.064))
    press_demo.record_residual(fix, (0.425, -0.114, 0.087), 1.22, path=f)
    press_demo.record_residual(fix, (0.426, -0.113, 0.086), 1.23, path=f)
    rows = [json.loads(l) for l in f.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["scene_top"] == [0.375, -0.106, 0.064] and rows[0]["wrist_top"] == [0.425, -0.114, 0.087]
    assert rows[1]["owl_score"] == 0.3 and rows[1]["n_points"] == 500


def test_the_calibration_report_counts_only_pairs_made_under_this_calibration(tmp_path, capsys):
    """Bench 2026-09-21: "4 pair(s) on file, mean offset [+3, -8, +9] mm" —
    two of the four dated from before the 09-17 correction. What the scene
    camera was actually off by that day was [+0, -12, -6]."""
    import json
    import os

    from rammp_box_opening.tasks import press_demo

    calib = tmp_path / "camera_scene.yaml"
    calib.write_text("xyz: [0, 0, 0]\nquat_xyzw: [0, 0, 0, 1]\n")
    os.utime(calib, (2000.0, 2000.0))
    f = tmp_path / "residuals.jsonl"
    rows = [
        {"t": 1000.0, "scene_top": [0.5738, -0.0294, 0.0651], "wrist_top": [0.5809, -0.033, 0.0919]},
        {"t": 1500.0, "scene_top": [0.5823, -0.0306, 0.0653], "wrist_top": [0.5867, -0.0347, 0.0869]},
        {"t": 3000.0, "scene_top": [0.4353, -0.0314, 0.0907], "wrist_top": [0.4348, -0.0435, 0.0849]},
        {"t": 3500.0, "scene_top": [0.4353, -0.0327, 0.0907], "wrist_top": [0.4357, -0.0451, 0.085]},
    ]
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    press_demo.residual_hint(f, calib)
    said = capsys.readouterr().out
    assert "2 pair(s) on file, mean offset [-0, -12, -6] mm" in said
    assert "2 older pair(s) were made under an earlier calibration" in said


def test_the_scene_approach_falls_back_to_the_search_when_the_wrist_sees_nothing(ctx, tmp_path, monkeypatch):
    """The scene camera is coarse and may be wrong by a few centimetres; the
    wrist must confirm at staging or the old search runs from there."""
    from conftest import runner as make_runner

    press_demo = _search_env(monkeypatch)
    cfg = load_press_demo(CFG)
    r = make_runner(ctx.client, tmp_path)
    fix = _SceneFix((0.45, 0.0, -0.027), 0.0, (0.45, 0.0, 0.085))
    # the staging aim is the button circle on live frames (test_button_aim);
    # here it sees nothing
    monkeypatch.setattr(press_demo, "aim_button_at_staging", lambda *a, **k: (None, "no button"))
    reach = press_demo.approach_from_scene(None, ctx, cfg, r, _Watcher(), True, fix)
    assert reach.got is None and reach.status == "no button"  # nothing confirmed
    assert ctx.client.executed  # ... but the arm did fly to staging first
    # the descent was still planned on the way, from the coarse pose
    assert reach.preplanned is not None and reach.preplanned[0].name == "press:down"


def test_the_scene_approach_returns_the_precise_fix_and_the_preplanned_descent(ctx, tmp_path, monkeypatch):
    from conftest import runner as make_runner

    press_demo = _search_env(monkeypatch)
    cfg = load_press_demo(CFG)
    r = make_runner(ctx.client, tmp_path)
    fix = _SceneFix((0.45, 0.0, -0.027), 0.0, (0.45, 0.0, 0.085))
    monkeypatch.setattr(press_demo, "aim_button_at_staging", lambda *a, **k: (FIX, "3/3 frames"))
    reach = press_demo.approach_from_scene(
        None, ctx, cfg, r, _Watcher(sees_when=lambda: True), True, fix
    )
    assert reach.got == FIX
    assert reach.preplanned is not None and reach.preplanned[0].guard is not None


def test_the_reach_is_cut_only_when_it_must_be():
    """Disagreement cuts it at once; an unconfirmed box cuts it just before
    the descent; a confirmed box lets it fly through the junction."""
    from rammp_box_opening.tasks import press_demo

    stop = press_demo.in_flight_stop
    assert stop(0.10, 0.70, confirmed=False, disagrees=True)
    assert not stop(0.10, 0.70, confirmed=False, disagrees=False)
    assert stop(0.66, 0.70, confirmed=False, disagrees=False)  # about to descend blind
    assert not stop(0.66, 0.70, confirmed=True, disagrees=False)
    assert not stop(0.95, 0.70, confirmed=True, disagrees=False)
    assert not stop(None, 0.70, confirmed=False, disagrees=False)  # no feedback yet


def test_a_confirmed_reach_flies_through_and_presses_without_a_stop(ctx, tmp_path, monkeypatch):
    """With confirm_in_flight, approach and descent are one goal; the wrist
    confirms on the way; the group touches; nothing stopped."""
    from conftest import runner as make_runner

    press_demo = _search_env(monkeypatch)
    cfg = replace(load_press_demo(CFG), confirm_in_flight=True)
    ctx.client.exec_script = [("touch", {"message": "torque guard trip", "progress": 0.9})]
    r = make_runner(ctx.client, tmp_path)
    fix = _SceneFix((0.45, 0.0, -0.027), 0.0, (0.45, 0.0, 0.085))
    watcher = _Watcher(sees_when=lambda: True)
    watcher.to_container_pose = lambda got: fix.pose  # agrees with the scene exactly
    reach = press_demo.approach_from_scene(None, ctx, cfg, r, watcher, True, fix)
    assert reach.in_flight and reach.pressed is not None and reach.got == FIX
    assert len(ctx.client.executed) == 1  # ONE goal for the reach and the touch
    assert reach.press_leg.name == "press:down"


def test_a_passing_glance_may_differ_more_than_an_aim_taken_at_rest(ctx, tmp_path, monkeypatch):
    """Two decisions, two tolerances. At rest at staging the press goes to
    the circle's centre: 2 mm from the scene's spot and the descent is
    re-fitted. In flight nothing can be re-fitted — the descent is already
    flying — so the question is only whether to fly through, and 5 mm still
    does (harness 2026-09-21: tightening the first stopped every in-flight
    reach above the box, its synthetic scene camera being 3.6 mm off)."""
    from conftest import runner as make_runner

    from rammp_box_opening.models.container import ContainerPose

    press_demo = _search_env(monkeypatch)
    cfg = replace(load_press_demo(CFG), confirm_in_flight=True)
    ctx.client.exec_script = [("touch", {"message": "torque guard trip", "progress": 0.9})]
    r = make_runner(ctx.client, tmp_path)
    fix = _SceneFix((0.45, 0.0, -0.027), 0.0, (0.45, 0.0, 0.085))
    seen = ContainerPose(xyz=(0.452, -0.003, -0.027), yaw=0.0)  # 3.6 mm from the scene's
    assert not press_demo.use_preplanned_descent(fix.pose, seen, ["press:down"])  # from rest: re-fit to the aim
    watcher = _Watcher(sees_when=lambda: True)
    watcher.to_container_pose = lambda got: seen
    reach = press_demo.approach_from_scene(None, ctx, cfg, r, watcher, True, fix)
    assert reach.in_flight and reach.pressed is not None  # in flight: through, without a stop
    assert len(ctx.client.executed) == 1


def test_an_unconfirmed_reach_stops_above_the_box_and_looks(ctx, tmp_path, monkeypatch):
    from conftest import runner as make_runner

    press_demo = _search_env(monkeypatch)
    cfg = replace(load_press_demo(CFG), confirm_in_flight=True)
    ctx.client.exec_script = [("stopped", {"message": "stopped part-way", "progress": 0.66})]
    r = make_runner(ctx.client, tmp_path)
    fix = _SceneFix((0.45, 0.0, -0.027), 0.0, (0.45, 0.0, 0.085))
    reach = press_demo.approach_from_scene(None, ctx, cfg, r, _Watcher(), True, fix)
    assert reach.in_flight and reach.pressed is None and reach.got is None
    assert reach.preplanned is None  # the descent must be re-planned from live


def test_a_touch_on_the_raised_knob_means_the_box_is_already_open(ctx):
    """The pads stop ~15-18 mm above the predicted button top when the
    knob is already up; pushing then would seal the lid again (field
    2026-09-17: +17.7 mm, pushed, gripped air). Flush contact is a press."""
    import pytest

    from rammp_box_opening.tasks import press_demo

    c = ctx
    flush = predicted_contact(c)
    verdict = lambda xyz: press_demo.touch_verdict(press_demo.popped_offset_m(xyz, c))  # noqa: E731
    assert press_demo.popped_offset_m(flush, c) == pytest.approx(0.0)
    assert verdict(flush) == "press"
    up = [flush[0], flush[1], flush[2] + 0.0177]
    assert verdict(up) == "popped"
    way_up = [flush[0], flush[1], flush[2] + 0.06]  # something else on the lid
    assert verdict(way_up) == "foreign"


def test_the_touch_height_says_where_the_pads_landed():
    """The verdicts, at the heights the bench has produced. Pads on the
    CLOSED button: -3.7 .. +6.2 mm (a 5-12 mm "rim" band once refused the
    good +6.2 contact — never again). Pads on the knob ALREADY UP: +9.6
    (09-17), +10.8 (09-21 13:35) and +17.7 (09-17). The first two were read
    as "pads on a ring" for four days and pushed: that push is what shuts an
    open box. This knob stands about 10 mm proud with the lid on."""
    from rammp_box_opening.tasks.press_demo import touch_verdict

    for off in (-0.0037, -0.0008, 0.0023, 0.0062, 0.0075):
        assert touch_verdict(off) == "press", off
    for off in (0.0096, 0.0108, 0.0177):
        assert touch_verdict(off) == "popped", off
    assert touch_verdict(0.060) == "foreign"


def test_the_camera_says_the_knob_is_up_at_the_heights_this_knob_reaches():
    """Bench 2026-09-21 14:16, the owner: "it pressed the button and opened
    it but then stopped and went down again and CLOSED the box". The pop
    check had read [11.1, 10.1, 11.1] mm on three fresh frames — the knob
    was up — and called it NOT POPPED, because "popped" began at 12 mm (a
    number from one +17.7 mm touch). Every reading of this knob up is
    9-11 mm (9.2, 10.1, 11.1; +10 against its own lid in an aim frame);
    every reading of it down is within a millimetre of zero (-0.8, 0.2)."""
    from rammp_box_opening.tasks.press_demo import knob_is_up

    for up_mm in (9.2, 10.1, 11.1, 16.0):
        assert knob_is_up(up_mm), up_mm
    for up_mm in (-0.8, 0.2, 1.0, 4.0, None):
        assert not knob_is_up(up_mm), up_mm


def test_a_stop_above_the_box_still_gets_home(ctx, tmp_path, capsys):
    """Bench 2026-09-21 14:16: "home plan refused (... the start state
    collides with the world ...) — arm holds". A recovery home is planned in
    the blind bench world, whose whole placement band is blocked to
    container height — and the arm was at the hop, fingers 50 mm above the
    lid, INSIDE that band: refused by construction, every time a run stops
    over the box. The mission knows where this box is. It rises out of the
    corridor it came down (the re-press's own retreat, in the button's
    interaction world) and goes home from staging height."""
    import re

    import pytest
    from conftest import runner as make_runner

    from rammp_box_opening.tasks import press_demo

    c = ctx
    cfg = _demo_cfg()
    c.press_cfg = cfg

    class Refused:
        success = False
        message = "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION"
        trajectory = None

    c.client.plans = [Refused()]  # the first plan asked for — home from the hop — is refused
    press_demo.try_home(c, make_runner(c.client, tmp_path), True, "STOP: test")
    said = capsys.readouterr().out
    assert "arm holds" not in said and "rising over the box first" in said
    # the rise flows into the home: ONE execution, like the mission's own retreat + home
    assert len(c.client.executed) == 1
    assert re.search(r"retreat:rise .*interaction_button[^\n]*\nhome ", said)
    assert c.client.joint_targets[-1] == pytest.approx(press_demo.rest_joints(cfg))
    assert c.client.live == pytest.approx(press_demo.rest_joints(cfg))  # and it got there
    assert c.lid_at is None


def test_a_stop_with_no_box_located_holds_honestly(ctx, tmp_path, capsys):
    """... and when there is no located box to rise over, a refused home
    still leaves the arm holding with an honest line, as before."""
    from conftest import runner as make_runner

    from rammp_box_opening.tasks import press_demo

    c = ctx
    c.cpose = None

    class Refused:
        success = False
        message = "INVALID_START_STATE_WORLD_COLLISION"
        trajectory = None

    c.client.plans = [Refused()]
    press_demo.try_home(c, make_runner(c.client, tmp_path), True, "STOP: test")
    assert "arm holds" in capsys.readouterr().out and c.client.executed == []


def test_an_open_box_is_not_pressed_shut():
    """What the mission says when the aim finds the button already up — and
    that it says nothing when the button is down or was not measured."""
    from rammp_box_opening.tasks.press_demo import refuse_open_box

    for up_mm in (8.0, 9.0, 10.3):  # the aim frames of 13:35, 14:14:04 and 14:14:42, all pressed shut
        why = refuse_open_box(up_mm)
        assert why is not None and "already" in why and "%.0f mm" % up_mm in why
    for up_mm in (-1.0, 0.0, 1.0, 4.0, None):
        assert refuse_open_box(up_mm) is None


def test_a_popped_knob_is_not_pressed_again(ctx, tmp_path, monkeypatch):
    """... and the run of 14:16 itself: push, read 11.1 mm from the hop —
    that is a pop; nothing presses again."""
    from conftest import runner as make_runner

    from rammp_box_opening.tasks import press_demo

    c = ctx
    r = make_runner(c.client, tmp_path)
    args = type("A", (), {"execute": True, "press_only": False})()
    touch = [type("R", (), {"contact_xyz": predicted_contact(c), "leg_name": "press:down"})()]
    monkeypatch.setattr(press_demo, "knob_height_mm", lambda node, watcher, ctx, **kw: 11.1)
    monkeypatch.setattr(press_demo, "_press_again", lambda *a, **k: (_ for _ in ()).throw(AssertionError("pressed a popped knob again")))
    res, _t = press_demo.run_push(c, _demo_cfg(), r, args, touch, node=object(), watcher=object())
    assert all(x.ok for x in res)


def test_a_contact_a_little_above_the_button_top_still_pushes(ctx, tmp_path):
    """+6.2 mm (14:10, 2026-09-17, pads on the button) must push, not home."""
    import types

    from conftest import runner as make_runner

    from rammp_box_opening.runtime.runner import LegResult
    from rammp_box_opening.tasks import press_demo

    c = ctx
    r = make_runner(c.client, tmp_path)
    args = types.SimpleNamespace(execute=True, press_only=True)
    flush = predicted_contact(c)
    touch = [LegResult("press:down", "touch", True, "surface", contact_xyz=[flush[0], flush[1], flush[2] + 0.0062])]
    res, _t = press_demo.run_push(c, _demo_cfg(), r, args, touch)
    assert [x.leg_name for x in res][0] == "press:push"


def test_the_push_bound_leaves_room_for_the_buttons_travel():
    """The push ends at the button's stop (the force), and 10 mm is the
    room it has to get there — 4 mm from a 3 Nm contact was not enough on
    the 4.1-inch box (owner 2026-09-17)."""
    from rammp_box_opening.models.container import load_press_demo

    cfg = load_press_demo("src/rammp_box_opening/config/containers/oxo_pop.yaml")
    assert cfg.button_travel_m == 0.010
    assert cfg.contact_nm == 3.0 and cfg.touch_nm if hasattr(cfg, "touch_nm") else True


def test_a_press_contact_pushes_and_hands_back_its_touch(ctx, tmp_path):
    import types

    from conftest import runner as make_runner

    from rammp_box_opening.runtime.runner import LegResult
    from rammp_box_opening.tasks import press_demo

    c = ctx
    r = make_runner(c.client, tmp_path)
    args = types.SimpleNamespace(execute=True, press_only=True)
    flush = predicted_contact(c)
    touch = [LegResult("press:down", "touch", True, "surface", contact_xyz=[flush[0], flush[1], flush[2] + 0.0023])]
    res, touch_out = press_demo.run_push(c, _demo_cfg(), r, args, touch)
    assert touch_out is touch
    assert [x.leg_name for x in res][0] == "press:push"


def test_an_already_popped_box_gets_no_push_before_the_grip(ctx):
    from rammp_box_opening.tasks import press_demo

    c = ctx
    cfg = _demo_cfg()
    contact = predicted_contact(c)
    with_push = [x.name for x in press_demo.build_push_legs(c, cfg, contact)]
    no_push = [x.name for x in press_demo.build_push_legs(c, cfg, contact, push=False)]
    assert with_push[0] == "press:push" and "press:push" not in no_push
    assert no_push == with_push[1:]  # the retreat and the open, unchanged


STUCK = [0.031, 0.721, 3.125, -2.281, -0.001, 0.267, 1.568]  # refused as a start, 2026-09-17


def test_home_arm_can_lift_straight_up_before_planning(ctx):
    """The pose the planner refuses as a start (fingers 40 mm above the
    table after a jog) is lifted plan-free, straight up with the attitude
    held, and HOME is planned from where the lift ends."""
    import numpy as np
    import pytest

    from rammp_box_opening.runtime.kinematics import ArmChain
    from rammp_box_opening.tasks import home_arm

    c = ctx
    c.client.live = list(STUCK)
    legs = home_arm.build_legs(c, lift_first_m=0.08)
    assert [x.name for x in legs] == ["lift-free", "home"]
    lift = legs[0]
    assert lift.guard is not None and lift.guard.trip == "obstruction"
    chain = ArmChain()
    R0, t0 = chain.fk(STUCK)
    pts = [p.positions for p in lift.traj.points]
    assert np.allclose(pts[0], STUCK)
    for q in pts:
        R, t = chain.fk(q)
        assert np.hypot(t[0] - t0[0], t[1] - t0[1]) < 1e-4  # straight up
        assert np.allclose(R, R0, atol=2e-3)  # attitude held
    _R, t_end = chain.fk(pts[-1])
    assert t_end[2] - t0[2] == pytest.approx(0.08, abs=1e-4)
    assert legs[1].name == "home"
    # HOME was planned from the lift's END, not from the stuck pose
    assert c.client.joint_starts[-1] == pytest.approx(list(lift.goal_joints))


def test_home_arm_without_a_lift_is_unchanged(ctx):
    from rammp_box_opening.tasks import home_arm

    legs = home_arm.build_legs(ctx)
    assert [x.name for x in legs] == ["home"]


def test_residual_pairs_go_next_to_a_custom_calibration(tmp_path):
    """The e2e harness's synthetic pairs must never land in the real file
    (they did, ~90 of them, 2026-09-17)."""
    from rammp_box_opening.tasks import press_demo

    assert press_demo.residuals_path_for(None) == press_demo.RESIDUALS_FILE
    custom = tmp_path / "camera_scene.yaml"
    assert press_demo.residuals_path_for(str(custom)) == tmp_path / "residuals.jsonl"



def test_the_push_tail_retreats_to_the_hop_with_the_fingers_opening(ctx):
    """After the push: a LAZY retreat to the hop, grip:open riding it — the
    open-box tail re-descends from there at once, and press-only reads the
    knob from there before going home."""
    import pytest

    from rammp_box_opening.models.container import from_container
    from rammp_box_opening.tasks import press_demo

    c = ctx
    cfg = _demo_cfg()
    button = from_container(c.cpose, c.model.button_offset)
    (press,) = press_demo.build_press_legs(c, cfg)
    assert press.guard.trip == "touch" and press.warp is None
    assert press.guard.arm_after >= 0.25  # sits out the launch transient
    tail = press_demo.build_push_legs(c, cfg, predicted_contact(c))
    assert names(tail) == ["press:push", "retreat", "grip:open"]
    push, retreat, open_leg = tail
    assert push.guard.trip == "press"
    assert retreat.traj is None  # lazy: planned from live after the push
    assert open_leg.defer_join and open_leg.send_with_previous_motion and open_leg.gripper_cmd == 0.0
    assert retreat.target[1][2] == pytest.approx(tcp_z(button[2]) + cfg.grip_hop_m)
    # press-only no longer differs here: it too stops at the hop with the
    # fingers opening, so the wrist can confirm the pop (2026-09-23: a
    # press-only run printed PRESSED on a box that had not opened)
    assert push.hold_s == cfg.push_hold_s


def test_a_warped_descent_cannot_trip_before_its_rebaseline_settles(ctx):
    """Bench 2026-09-03: a light threshold tripped at 42 % of a warped
    stroke, in the FAST segment, on the arm's own dynamics. A warped guard
    arms only after the slow-zone rebaseline plus a settle margin — capped,
    so a degenerate warp can never disable a guard outright."""
    import pytest

    from rammp_box_opening.runtime.guards import ARM_AFTER_CAP, WARP_SETTLE_FRAC
    from rammp_box_opening.tasks import press_demo

    cfg = _demo_cfg()
    assert cfg.warp_fast_speed > cfg.grip_speed  # the shipped grip:down IS warped
    down = press_demo.build_grip_legs(ctx, cfg)[0]
    g = down.guard
    assert down.warp is not None and g.rebaseline_after is not None
    assert g.arm_after == pytest.approx(min(g.rebaseline_after + WARP_SETTLE_FRAC, ARM_AFTER_CAP))


def test_a_re_press_rises_over_the_button_whatever_the_lookahead_left_behind(ctx, tmp_path):
    """Bench 2026-09-21: the knob read 9.2 mm, the mission went to press
    once more, and its first motion was REFUSED — "rise ... planned against
    'interaction_place:lid'". The whole grip-and-place tail had been planned
    as a lookahead under the retreat, and it leaves ctx at its LAST leg:
    last_world the lid's set-down world, last_pose a point above the drop
    spot 20 cm away, lid_at the placed lid. The re-press took all three on
    trust. (The refusal was the good outcome: that rise was aimed at the
    drop spot.) It now builds from the button, and nothing else."""
    import pytest
    from conftest import runner as make_runner

    from rammp_box_opening.constants import TRANSIT_SPEED
    from rammp_box_opening.models.container import from_container
    from rammp_box_opening.tasks import press_demo

    c = ctx
    cfg = _demo_cfg()
    button = from_container(c.cpose, c.model.button_offset)
    press_demo.build_grip_and_place_legs(c, cfg)  # the lookahead that ran under the retreat
    assert c.last_world[0].startswith("interaction_place")  # what it leaves behind
    assert c.lid_at is not None and abs(c.last_pose[0][1] - button[1]) > 0.1

    legs = press_demo.build_restage_legs(c, cfg)
    assert names(legs) == ["press:close", "retreat:restage"]
    close, rise = legs
    assert rise.world == "interaction_button" and close.world == "interaction_button"
    assert rise.target[1] == pytest.approx([button[0], button[1], button[2] + cfg.staging_m])
    assert rise.speed == pytest.approx(TRANSIT_SPEED)
    assert c.lid_at is None  # nothing has been placed: no phantom lid in the next worlds
    assert c.last_world[0] == "interaction_button"
    # ... and the Runner's gates let it fly (a rise out of the corridor just
    # descended, like retreat and lift)
    res = make_runner(c.client, tmp_path).run(legs, execute=True)
    assert [r.ok for r in res] == [True, True]


def _hop_frame(ctx, knob_up_m):
    """A synthetic wrist frame from the hop: the camera looking straight
    down from 17.5 cm above the lid, 75 mm off the tool axis (the mount),
    the lid a plane and the button a 45 mm disc standing `knob_up_m` above
    it. Returns (grab, (rot, trans))."""
    import numpy as np
    from types import SimpleNamespace

    from rammp_box_opening.models.container import from_container

    button = np.array(from_container(ctx.cpose, ctx.model.button_offset), float)
    k = np.array([[434.9, 0.0, 424.0], [0.0, 434.3, 237.8], [0.0, 0.0, 1.0]])
    rot = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    trans = button + np.array([0.075, 0.0, 0.175])
    vv, uu = np.mgrid[0:480, 0:848].astype(float)
    depth = np.full((480, 848), 0.175, np.float32)
    # a pixel's ray meets the knob's top plane at range (0.175 - up); it is
    # knob where that point lies within the disc
    zc = 0.175 - knob_up_m
    pts = np.stack([(uu - k[0, 2]) / k[0, 0] * zc, (vv - k[1, 2]) / k[1, 1] * zc, np.full_like(uu, zc)], -1) @ rot.T + trans
    depth[np.hypot(pts[..., 0] - button[0], pts[..., 1] - button[1]) < 0.0225] = zc
    grab = SimpleNamespace(
        color=np.zeros((480, 848, 3), np.uint8), depth=depth, k=k,
        color_stamp=SimpleNamespace(sec=100, nanosec=0),
    )
    return grab, (rot, trans)


def _node_at(now_s):
    """A node whose clock reads `now_s`."""
    from types import SimpleNamespace

    return SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int(now_s * 1e9))))


def test_the_pop_check_judges_only_frames_shot_after_the_arm_came_to_rest(ctx, tmp_path, monkeypatch):
    """Harness 2026-09-21: the check read [-0.0, -0.0, 16.0] mm — two frames
    still in the pipe from BEFORE the push, then the popped knob — and the
    median said "not popped": a needless second press. A frame stamped
    before the check began shows the button as it was, not as it is."""
    from types import SimpleNamespace

    import pytest

    from rammp_box_opening.perception import depth_source
    from rammp_box_opening.tasks import press_demo

    monkeypatch.setattr(press_demo, "CAPTURES_DIR", tmp_path)
    monkeypatch.setattr(press_demo, "_spin_detect", lambda node: None)
    grab, pose = _hop_frame(ctx, 0.0)  # stamped 100 s, the button still down in it
    monkeypatch.setattr(depth_source, "camera_pose_at", lambda g, offset_s=None: pose)
    watcher = SimpleNamespace(grab=grab)
    # the check begins at 200 s: that frame is history, and no verdict is better than its verdict
    assert press_demo.knob_height_mm(_node_at(200.0), watcher, ctx, timeout_s=0.2) is None
    # shot after the check began: it counts
    assert press_demo.knob_height_mm(_node_at(99.0), watcher, ctx, timeout_s=0.2) == pytest.approx(0.0, abs=0.5)


def test_the_pop_check_keeps_the_frame_it_judged(ctx, tmp_path, monkeypatch):
    """Bench 2026-09-21: "NOT POPPED — the knob reads 9.2 mm" after a centred
    push that met the button's stop — the first reading this check ever made
    on the real box, and nothing was kept to say whether the knob was down
    or the reading was low. It now keeps the frame it judged, with the
    window it read, and says what each frame read — like the aim."""
    import numpy as np
    from types import SimpleNamespace

    import pytest

    from rammp_box_opening.perception import depth_source
    from rammp_box_opening.tasks import press_demo

    monkeypatch.setattr(press_demo, "CAPTURES_DIR", tmp_path)
    monkeypatch.setattr(press_demo, "_spin_detect", lambda node: None)
    node = _node_at(50.0)  # the frames below are stamped 100 s: shot after the arm came to rest
    for up_m, tag in ((0.016, "up"), (0.0, "down")):
        grab, pose = _hop_frame(ctx, up_m)
        monkeypatch.setattr(depth_source, "camera_pose_at", lambda g, offset_s=None, pose=pose: pose)
        evidence = {}
        got = press_demo.knob_height_mm(node, SimpleNamespace(grab=grab), ctx, timeout_s=0.2, evidence=evidence)
        assert got == pytest.approx(1000 * up_m, abs=0.5), tag
        assert evidence["frames_mm"] == [pytest.approx(1000 * up_m, abs=0.5)]
        folder = Path(evidence["capture"])
        assert folder.parent == tmp_path and folder.name.startswith("wrist-pop-")
        saved = np.load(folder / "frame_000.npz")
        assert saved["depth"].shape == (480, 848) and np.allclose(saved["trans_cam"], pose[1])
        u, v = evidence["uv"]
        assert 0 <= u < 848 and 0 <= v < 480


def _centring_env(ctx, tmp_path, monkeypatch, tool_xys, aims):
    """press_demo with the tool's position and the wrist's aims scripted:
    tool_xys[i] is where the tool stands before aim i+1; aims[i] is what
    aim i+1 returns (None: the button was not found)."""
    from conftest import runner as make_runner

    from rammp_box_opening.models.container import ContainerPose

    press_demo = _search_env(monkeypatch)
    state = {"n": 0}
    monkeypatch.setattr(press_demo, "tool_xy", lambda ctx: tool_xys[min(state["n"], len(tool_xys) - 1)])

    def aim(node, watcher, yaw, **kw):
        got = aims[state["n"]]
        state["n"] += 1
        return got, "3/3 still frames found the button circle"

    monkeypatch.setattr(press_demo, "aim_button_at_staging", aim)
    watcher = _Watcher()
    watcher.to_container_pose = lambda got: ContainerPose(xyz=(got[0][0], got[0][1], -0.027), yaw=got[1])
    return press_demo, make_runner(ctx.client, tmp_path), watcher, state


def test_the_arm_recentres_over_the_button_before_it_presses(ctx, tmp_path, monkeypatch):
    """Bench 2026-09-23, the owner: the arm started "way too far to the
    right", then "hit the right side of the button and not the center". The
    arm touched within 0.5 mm of where the wrist aimed — but it aimed with
    the button 194 px from the image centre, because staging stood 11 cm
    from the box (a scene camera moved since its calibration). Every press
    that landed dead centre was aimed with the button under the tool. So:
    aim, and if the button is not under the tool, move over it at staging
    height and aim again from there."""
    first, second = ((0.450, 0.000, 0.085), 0.0), ((0.451, 0.001, 0.085), 0.0)
    press_demo, r, watcher, state = _centring_env(
        ctx, tmp_path, monkeypatch, tool_xys=[(0.45, 0.10), (0.451, 0.0015)], aims=[second]
    )
    got, moves, off = press_demo.centre_over_button(None, ctx, _demo_cfg(), r, watcher, True, first)
    assert moves == 1 and state["n"] == 1  # one move over it, one aim from there
    assert got == second  # the press is aimed from over the button
    assert off <= press_demo.RECENTRE_TOL_M
    assert len(ctx.client.executed) == 1  # the one re-staging move, planned (collision-checked)
    assert ctx.cpose.xyz[:2] == pytest.approx(second[0][:2])


def test_a_button_already_under_the_tool_is_pressed_from_where_it_stands(ctx, tmp_path, monkeypatch):
    first = ((0.450, 0.000, 0.085), 0.0)
    press_demo, r, watcher, state = _centring_env(ctx, tmp_path, monkeypatch, tool_xys=[(0.4505, 0.001)], aims=[])
    got, moves, off = press_demo.centre_over_button(None, ctx, _demo_cfg(), r, watcher, True, first)
    assert (got, moves, state["n"]) == (first, 0, 0) and ctx.client.executed == []


def test_recentring_gives_up_after_two_moves_and_presses_from_the_last_aim(ctx, tmp_path, monkeypatch):
    """A tool that never lands over the button (a wrist-mount error it
    cannot see) must not loop: two moves, then the last aim — the best one,
    taken nearest the button — is pressed."""
    a = [((0.450, 0.0, 0.085), 0.0), ((0.452, 0.0, 0.085), 0.0), ((0.454, 0.0, 0.085), 0.0)]
    press_demo, r, watcher, state = _centring_env(
        ctx, tmp_path, monkeypatch, tool_xys=[(0.45, 0.05), (0.45, 0.02), (0.45, 0.02)], aims=a[1:]
    )
    got, moves, off = press_demo.centre_over_button(None, ctx, _demo_cfg(), r, watcher, True, a[0])
    assert moves == press_demo.RECENTRE_MAX == 2
    assert got == a[2] and off > press_demo.RECENTRE_TOL_M


def test_a_lost_button_after_the_move_keeps_the_last_good_aim(ctx, tmp_path, monkeypatch):
    first = ((0.450, 0.0, 0.085), 0.0)
    press_demo, r, watcher, state = _centring_env(
        ctx, tmp_path, monkeypatch, tool_xys=[(0.45, 0.06), (0.45, 0.0)], aims=[None]
    )
    got, moves, off = press_demo.centre_over_button(None, ctx, _demo_cfg(), r, watcher, True, first)
    assert got == first and moves == 1


def test_a_dry_run_says_it_would_recentre_and_moves_nothing(ctx, tmp_path, monkeypatch, capsys):
    first = ((0.450, 0.0, 0.085), 0.0)
    press_demo, r, watcher, state = _centring_env(ctx, tmp_path, monkeypatch, tool_xys=[(0.45, 0.04)], aims=[])
    got, moves, off = press_demo.centre_over_button(None, ctx, _demo_cfg(), r, watcher, False, first)
    assert (got, moves) == (first, 0) and ctx.client.executed == []
    assert "would move 40 mm" in capsys.readouterr().out


def test_a_press_from_staging_without_a_scene_fix_is_planned_fresh(ctx):
    """After the search, or after a re-centring move, nothing pre-planned
    fits: the descent is planned from where the arm stands."""
    from rammp_box_opening.tasks import press_demo

    legs, how = press_demo.descent_from_staging(ctx, _demo_cfg(), None, None)
    assert [leg.name for leg in legs] == ["press:down"] and "planned" in how



def test_press_only_confirms_the_pop_and_presses_again_when_it_did_not(ctx, tmp_path, monkeypatch):
    """Bench 2026-09-23: --press-only printed "PRESSED — met a stop" and the
    box had not opened; press-only never looked. It now reads the knob from
    the hop like the full mission, and presses once more when it is down."""
    from conftest import runner as make_runner

    from rammp_box_opening.tasks import press_demo

    args = type("A", (), {"execute": True, "press_only": True})()
    touch = [type("R", (), {"contact_xyz": predicted_contact(ctx), "leg_name": "press:down"})()]
    again = []
    monkeypatch.setattr(press_demo, "_press_again", lambda *a, **k: again.append(1) or ([], touch))
    for up_mm, pressed_again in ((11.1, False), (0.2, True)):
        again.clear()
        monkeypatch.setattr(press_demo, "knob_height_mm", lambda node, watcher, ctx, up_mm=up_mm, **kw: up_mm)
        press_demo.run_push(ctx, _demo_cfg(), make_runner(ctx.client, tmp_path), args, touch, node=object(), watcher=object())
        assert bool(again) is pressed_again, up_mm


def test_home_from_the_hop_rises_over_the_button_first(ctx):
    """From the hop the fingers stand 5 cm over the lid, inside the band the
    blind home world blocks: the rise out of the corridor comes first, in
    the button's own world, then home."""
    from rammp_box_opening.models.container import from_container
    from rammp_box_opening.tasks import press_demo

    cfg = _demo_cfg()
    legs = press_demo.build_home_from_hop(ctx, cfg)
    assert names(legs) == ["retreat:rise", "home"]
    button = from_container(ctx.cpose, ctx.model.button_offset)
    assert legs[0].world == "interaction_button"
    assert legs[0].target[1] == pytest.approx([button[0], button[1], button[2] + cfg.staging_m])
    assert legs[1].target[1] == pytest.approx(press_demo.rest_joints(cfg))


def test_the_push_holds_on_the_button_before_it_recoils(ctx, tmp_path, monkeypatch):
    """A latch that needs a moment held down was let go the instant the
    push stopped: the recoil fired at once. A push leg with hold_s holds
    still on the button that long first — and only after a good push."""
    from conftest import runner as make_runner

    from rammp_box_opening.runtime import runner as runner_mod

    held = []
    monkeypatch.setattr(runner_mod, "_hold", lambda s: held.append((s, len(ctx.client.executed))))
    cfg = _demo_cfg()
    (press,) = press_demo_mod().build_press_legs(ctx, cfg)
    push, retreat, _open = press_demo_mod().build_push_legs(ctx, cfg, predicted_contact(ctx))
    push.hold_s = 0.3
    ctx.client.exec_script = [("touch", {"message": "stop", "progress": 0.6})]
    make_runner(ctx.client, tmp_path).run([push, retreat], execute=True)
    assert held == [(0.3, 1)]  # after the push flew, before the recoil
    held.clear()
    push.hold_s = 0.0
    ctx.client.exec_script = [("touch", {"message": "stop", "progress": 0.6})]
    make_runner(ctx.client, tmp_path).run([push, retreat], execute=True)
    assert held == []


def press_demo_mod():
    from rammp_box_opening.tasks import press_demo

    return press_demo


def test_the_pink_canister_pushes_firmer_and_holds():
    """The owner, 2026-09-23: "press down harder so it can open the box".
    Its push stopped at the 4.5 Nm backstop after 4.9 mm and the box stayed
    shut. For this box: a 7 Nm backstop (touch_nm 10 - contact 3), 12 mm of
    room, and a 0.3 s hold on the button. The OXOs keep what opened them."""
    from rammp_box_opening.models.container import ContainerModel, load_press_demo

    pink = "src/rammp_box_opening/config/containers/ankou_pink.yaml"
    m, cfg = ContainerModel.load(pink), load_press_demo(pink)
    assert m.touch_nm - cfg.contact_nm == pytest.approx(7.0)
    assert cfg.button_travel_m == pytest.approx(0.012) and cfg.push_hold_s == pytest.approx(0.3)
    oxo = load_press_demo("src/rammp_box_opening/config/containers/oxo_pop.yaml")
    assert oxo.push_hold_s == 0.0 and oxo.button_travel_m == pytest.approx(0.010)



def test_a_coarse_search_fix_is_only_ever_a_place_to_look_from(ctx, tmp_path, monkeypatch):
    """The search may now send the arm to staging on a coarse sighting (a
    top cut by the fingers or the image edge, or one with no button circle
    from high up — both were refused, 2026-09-23, with the box in view).
    But a coarse fix never aims a press: if the close-up aim at staging
    finds no button, the run stops and says so."""
    from conftest import runner as make_runner

    from rammp_box_opening.models.container import ContainerPose

    press_demo = _search_env(monkeypatch)
    monkeypatch.setattr(press_demo, "aim_button_at_staging", lambda *a, **k: (None, "0/20 still frames found the button circle"))
    watcher = _Watcher()
    watcher.to_container_pose = lambda got: ContainerPose(xyz=(got[0][0], got[0][1], -0.027), yaw=got[1])
    r = make_runner(ctx.client, tmp_path)
    with pytest.raises(SystemExit):
        press_demo.stage_over_search_fix(None, ctx, _demo_cfg(), r, watcher, True, FIX, coarse=True)
    # a precise search fix, as before, may still be pressed from
    got, at_staging = press_demo.stage_over_search_fix(None, ctx, _demo_cfg(), r, watcher, True, FIX)
    assert got == FIX and at_staging



def test_a_coarse_sighting_ends_the_search_when_a_close_up_aim_follows(ctx, tmp_path, monkeypatch):
    """Bench 2026-09-23, run 2: the search saw the box in 59 of 94 frames
    and swept on past it anyway, then said NO BOX — its sightings had no
    button circle from half a metre up. With a close-up aim to follow, a
    sighting is all the search is for: it stops at the step that saw it."""
    from conftest import runner as make_runner

    press_demo = _search_env(monkeypatch)
    cfg = load_press_demo(CFG)
    watcher = _Watcher(coarse=True)  # seen, never precisely
    got, failed = press_demo.search_for_box(None, ctx, cfg, make_runner(ctx.client, tmp_path), watcher, True, accept_coarse=True)
    assert (got, failed) == (None, None) and watcher.last_coarse() == FIX
    assert len(ctx.client.joint_targets) == 1  # the look alone: no sweep past the box


def test_without_a_close_up_aim_to_follow_the_search_sweeps_on(ctx, tmp_path, monkeypatch):
    from conftest import runner as make_runner

    press_demo = _search_env(monkeypatch)
    cfg = load_press_demo(CFG)
    watcher = _Watcher(coarse=True)
    got, failed = press_demo.search_for_box(None, ctx, cfg, make_runner(ctx.client, tmp_path), watcher, True)
    assert (got, failed) == (None, None)
    assert len(ctx.client.joint_targets) == 3  # look, left, right — as before
