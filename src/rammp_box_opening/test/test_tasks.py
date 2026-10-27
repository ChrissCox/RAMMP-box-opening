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


def test_home_arm_with_a_known_table_plans_guarded_without_a_caution(ctx, capsys):
    from rammp_box_opening.tasks import home_arm

    legs = home_arm.build_legs(ctx)
    assert legs[0].world == "bench"
    assert "CAUTION" not in capsys.readouterr().out


class _PlanCounter:
    """Wraps a FakeClient's plan_to_joints: counts the plans, and records
    whether anything had been printed before each one."""

    def __init__(self, client, capsys, succeed=True):
        self.n = 0
        self.out = ""
        self.printed_first = []
        inner = client.plan_to_joints

        def plan(q7, start_joints):
            self.n += 1
            self.out += capsys.readouterr().out
            self.printed_first.append(bool(self.out))
            if not succeed:
                class Bad:
                    success = False
                    message = "INVALID_START_STATE_WORLD_COLLISION"

                return Bad()
            return inner(q7, start_joints)

        client.plan_to_joints = plan


def test_home_arm_on_a_tableless_world_cautions_first_and_plans_once(ctx, capsys):
    """Flatbot 2026-10-22: world_flatbot.yaml has no table and home_arm
    measures none, so the 'guarded' bench world was the floor alone, the
    same as the bare one — a recovery home planned blind with no caution,
    and a refused plan re-planned in the identical world."""
    from rammp_box_opening.constants import HOME
    from rammp_box_opening.tasks import home_arm

    c = ctx
    c.worlds.table_top_z = None
    plans = _PlanCounter(c.client, capsys)
    legs = home_arm.build_legs(c)
    assert names(legs) == ["home"] and legs[0].world == "bench_bare"
    assert legs[0].target == ("joints", list(HOME))
    assert [kw.get("model") for kind, kw in c.worlds.pushes if kind == "bench"] == [None]
    assert plans.n == 1 and plans.printed_first == [True]  # the caution came BEFORE the plan
    out = plans.out + capsys.readouterr().out
    assert "CAUTION" in out and "neither the table nor the box" in out and "--lift-first" in out


def test_home_arm_on_a_tableless_world_refused_raises_without_a_second_plan(ctx, capsys):
    import pytest

    from rammp_box_opening.tasks import home_arm

    c = ctx
    c.worlds.table_top_z = None
    plans = _PlanCounter(c.client, capsys, succeed=False)
    with pytest.raises(RuntimeError):
        home_arm.build_legs(c)
    assert plans.n == 1


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


def test_the_search_keeps_the_frame_each_step_stood_at(ctx, tmp_path, monkeypatch):
    """A mission's record (ctx.mission_frames) gets one wrist frame per
    search step, named for the step, in the order flown."""
    from types import SimpleNamespace

    import numpy as np
    from conftest import runner as make_runner

    from rammp_box_opening.detection_set import MissionFrames
    from rammp_box_opening.perception import depth_source

    press_demo = _search_env(monkeypatch)
    monkeypatch.setattr(depth_source, "camera_pose_at", lambda g: (np.eye(3), np.zeros(3)))
    cfg = load_press_demo(CFG)
    ctx.mission_frames = MissionFrames(tmp_path / "missions", "oxo_pop.yaml", 0.05)
    watcher = _Watcher()
    watcher.grab = SimpleNamespace(
        color=np.zeros((4, 6, 3), np.uint8), depth=np.zeros((4, 6), np.float32), k=np.eye(3),
        color_stamp=SimpleNamespace(sec=1, nanosec=2), dist=None,
    )
    got, failed = press_demo.search_for_box(None, ctx, cfg, make_runner(ctx.client, tmp_path), watcher, True)
    assert got is None and failed is None
    assert [n for n, _ in ctx.mission_frames.frames] == ["wrist_look_0", "wrist_sweep:left_0", "wrist_sweep:right_0"]


def test_the_press_point_is_the_aims_only_when_it_is_the_last_close_up_aim():
    from rammp_box_opening.tasks import press_demo

    w = _Watcher()
    assert press_demo.aim_source(w, (0.4, 0.0, 0.15)) == "fix"  # never aimed
    w.last_aim = (0.4, 0.0, 0.15)
    assert press_demo.aim_source(w, [0.4, 0.0, 0.15]) == "aim"
    assert press_demo.aim_source(w, (0.41, 0.0, 0.15)) == "fix"  # a fix from elsewhere


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
    # the base-axis spots through the box first, preferred side first; the
    # diagonals only after every axis spot (2026-10-23)
    spots = press_demo.lid_drop_candidates(m, box, -0.027, [0.45, -0.25, -0.027])
    dirs = [d for _xyz, d in spots]
    assert dirs[0] == (0.0, -1.0)
    axis = [d in ((0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 0.0)) for d in dirs]
    assert axis == sorted(axis, reverse=True)


def test_a_box_at_the_edge_of_the_seen_table_gets_a_diagonal_drop_spot():
    """Flatbot 2026-10-23 16:31: the box at [0.39, -0.20] on the edge of the
    table the scene camera saw, the set-down zone x 0.43-0.63, y -0.63 to
    -0.23: every axis spot fell outside it ("no lid drop spot ... move the
    box"). A diagonal one, LID_DIAG_EXTRA_M farther out, fits."""
    from rammp_box_opening.models.container import ContainerModel, ContainerPose
    from rammp_box_opening.tasks import press_demo

    m = ContainerModel.load("src/rammp_box_opening/config/containers/ankou_pink.yaml")
    box = ContainerPose(xyz=(0.391, -0.196, 0.184), yaw=0.0)
    zone = (0.43, 0.63, -0.63, -0.23)
    (xyz, d), = press_demo.lid_drop_candidates(m, box, 0.184, zone=zone)
    assert d[0] > 0 and d[1] < 0  # toward +x, -y: into the seen table
    assert math.hypot(xyz[0] - 0.391, xyz[1] + 0.196) == pytest.approx(
        press_demo.lid_place_min_clear(m) + press_demo.LID_DIAG_EXTRA_M)
    assert zone[0] <= xyz[0] <= zone[1] and zone[2] <= xyz[1] <= zone[3]


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


def test_the_hand_comes_down_turned_as_the_spot_says_and_falls_through_the_cameras_spots(ctx):
    """A spot the cameras saw room at may need the hand turned about its
    axis (the lid's yaw, clear_lid_drops): the carry and the set-down fly
    that attitude. Refused there, the next spot tried is the next one the
    cameras saw room at, with its own turn, not a spot derived from the box
    alone."""
    import math

    import pytest

    from rammp_box_opening.geometry import yaw_about_world_z
    from rammp_box_opening.tasks import press_demo

    c = ctx
    bx, by = c.cpose.xyz[0], c.cpose.xyz[1]
    turned = ContainerPose(xyz=(bx, by - 0.21, -0.027), yaw=math.pi / 2)
    other = ContainerPose(xyz=(bx - 0.05, by + 0.22, -0.027), yaw=-math.pi / 2)
    c.lid_drops, c.lid_drop = [turned, other], turned
    legs = press_demo.build_place_legs(c, _demo_cfg())
    want = yaw_about_world_z(c.model.press_quat(turned.xyz), math.pi / 2)
    for name in ("place:lid:transit", "place:lid:down"):
        leg = next(x for x in legs if x.name == name)
        assert list(leg.target[2]) == pytest.approx(want)
    c.client = _RefusesPlacesAt(c.client, [turned.xyz[:2]])
    legs = press_demo.build_place_legs(c, _demo_cfg())
    transit = next(x for x in legs if x.name == "place:lid:transit")
    assert transit.target[1][:2] == pytest.approx(other.xyz[:2])
    assert list(transit.target[2]) == pytest.approx(yaw_about_world_z(c.model.press_quat(other.xyz), -math.pi / 2))
    assert c.lid_drop.yaw == pytest.approx(-math.pi / 2)


def test_the_lid_goes_where_the_cameras_saw_room_not_onto_the_microwave(monkeypatch):
    """Flatbot 2026-10-23 17:57 (test/data/table_room_flatbot_20261023_1757
    .npz): with the scene camera's points moved by its error at the box and
    the wrist's frame from staging, the drops are on the open table beyond
    the box, on the table the box stands on — and the 17:57 spot, on the
    microwave's front edge, is none of them."""
    import math

    import numpy as np
    from types import SimpleNamespace

    from rammp_box_opening.models.container import ContainerModel
    from rammp_box_opening.tasks import press_demo

    f = np.load("src/rammp_box_opening/test/data/table_room_flatbot_20261023_1757.npz")
    pink = ContainerModel.load("src/rammp_box_opening/config/containers/ankou_pink.yaml")
    table_z = float(f["table_z"])
    scene_fix = SimpleNamespace(table_points=f["scene_mm"] / 1000.0, table_z_scene=table_z,
                                pose=ContainerPose(xyz=(*f["scene_box"], table_z), yaw=0.0))
    cpose = ContainerPose(xyz=(*f["wrist_box"], table_z), yaw=0.0)
    monkeypatch.setattr(press_demo, "wrist_points", lambda watcher: f["wrist_mm"] / 1000.0)
    tmap = press_demo.room_map(None, scene_fix, cpose, at_staging=True)
    drops, spots, why = press_demo.clear_lid_drops(pink, cpose, tmap, table_z)
    assert drops and why is None
    assert all(d.xyz[2] == table_z for d in drops)
    assert [d.yaw for d in drops] == [s.turn for s in spots]
    assert all(math.hypot(d.xyz[0] - 0.56, d.xyz[1] + 0.37) > 0.15 for d in drops)


def test_what_the_cameras_see_standing_goes_into_the_planners_worlds(monkeypatch, capsys):
    """Owner 2026-10-27: nothing in the environment is hard coded. The map
    the lid's spot comes from becomes the planner's worlds too (adopt_seen):
    the microwave the wrist camera hit at 17:57 is in them, the box and the
    lid's spots are not, and the run's record says how much went in."""
    import numpy as np
    from types import SimpleNamespace

    from rammp_box_opening.models.container import ContainerModel
    from rammp_box_opening.tasks import press_demo

    f = np.load("src/rammp_box_opening/test/data/table_room_flatbot_20261023_1757.npz")
    pink = ContainerModel.load("src/rammp_box_opening/config/containers/ankou_pink.yaml")
    table_z = float(f["table_z"])
    scene_fix = SimpleNamespace(table_points=f["scene_mm"] / 1000.0, table_z_scene=table_z,
                                pose=ContainerPose(xyz=(*f["scene_box"], table_z), yaw=0.0))
    cpose = ContainerPose(xyz=(*f["wrist_box"], table_z), yaw=0.0)
    monkeypatch.setattr(press_demo, "wrist_points", lambda watcher: f["wrist_mm"] / 1000.0)
    tmap = press_demo.room_map(None, scene_fix, cpose, at_staging=True)
    drops, _spots, _why = press_demo.clear_lid_drops(pink, cpose, tmap, table_z)
    notes = []
    ctx = SimpleNamespace(model=pink, client=SimpleNamespace(joints=lambda: [0.0] * 7), worlds=SimpleNamespace(seen=[]))
    runner = SimpleNamespace(note=lambda kind, **kw: notes.append((kind, kw)))
    press_demo.adopt_seen(ctx, runner, tmap, cpose.xyz[:2], table_z, keep_clear=[d.xyz for d in drops],
                          when="scene camera and wrist")
    boxes = ctx.worlds.seen

    def covered(x, y):
        return any(abs(x - b["position"][0]) <= b["dims"][0] / 2 and abs(y - b["position"][1]) <= b["dims"][1] / 2
                   for b in boxes)

    assert boxes and notes == [("seen_obstacles", dict(notes[0][1], when="scene camera and wrist"))]
    assert notes[0][1]["boxes"] == len(boxes)
    assert covered(0.56, -0.37) and not covered(*cpose.xyz[:2])
    assert drops and not any(covered(*d.xyz[:2]) for d in drops)
    assert "the planner keeps the arm clear of what the cameras see standing on the table" in capsys.readouterr().out
    press_demo.adopt_seen(ctx, runner, None, cpose.xyz[:2], table_z)  # no map: the worlds keep what they had
    assert ctx.worlds.seen is boxes and len(notes) == 1


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


def test_after_a_popped_depth_press_the_fingers_open_and_the_lid_comes_off(ctx, tmp_path, capsys):
    """The depth press presses with the pads shut and used to end the run
    there (2026-10-23: "test the full execute stack including pulling the
    lid off"). Now the fingers open above the popped knob, and the tail
    every press ends with flies from where the depth press's rise left the
    arm: down around the knob, close, lift, carry, set down, home."""
    from conftest import runner as make_runner

    from rammp_box_opening.constants import GRIPPER_CMD_CLOSED, GRIPPER_CMD_OPEN
    from rammp_box_opening.tasks import press_demo
    from rammp_box_opening.tasks.depth_press import DepthPress

    sent, held = [], {"pos": 0.0}

    def gripper_cmd(position):
        if position is None:  # a read: what the fingers hold
            return True, held["pos"], False
        sent.append(position)
        held["pos"] = 0.42 if position == GRIPPER_CMD_CLOSED else float(position)  # closed on the knob
        return True, held["pos"], position == GRIPPER_CMD_CLOSED

    ctx.client.gripper_cmd = gripper_cmd
    # grip:down arrives around the knob; the lift-carry-set-down execution
    # ends on the table (the set-down's trip is its success)
    ctx.client.exec_script = [("arrived", {}), ("touch", {"message": "torque guard trip", "progress": 0.95, "torque_peak": 4.1})]
    # ... with the fingertips where the lid meets the table
    from rammp_box_opening.constants import TIP_TO_TOOL_M
    from rammp_box_opening.primitives.core import tcp_z

    lid = ctx.lid_drop or press_demo.load_lid_place(ctx.config_path)
    lands = tcp_z(lid.xyz[2] + ctx.model.lid_dims[2] + press_demo.DEPTH_GRIP_CLEAR_M)
    ctx.client.contact_at = [lid.xyz[0], lid.xyz[1], lands + TIP_TO_TOOL_M]
    result = DepthPress(free=True, contact=True, felt=True, popped=True, touch_xyz=(0.45, 0.0, 0.085))
    code = press_demo.lid_off_after_depth_press(ctx, _demo_cfg(), make_runner(ctx.client, tmp_path), True, result)
    out = capsys.readouterr().out
    assert code == 0 and "LID PULLED" in out and "DONE — box open, lid placed, arm home" in out, out
    assert sent[:2] == [GRIPPER_CMD_OPEN, GRIPPER_CMD_CLOSED]  # open above it first, then the grip
    assert ctx.client.joint_targets[-1] == pytest.approx(press_demo.rest_joints(_demo_cfg()))  # and home


def test_after_a_depth_press_the_lid_is_gripped_low_and_set_down_on_the_measured_table(ctx, tmp_path, monkeypatch):
    """The touch measured where the box stands: the knob is gripped
    DEPTH_GRIP_CLEAR_M above the lid (the Ankou's popped knob stood 4.5-8
    mm), and the lid goes down on the table the box stands on when that is
    lower than the scene camera's (a set-down above the table stops)."""
    from conftest import runner as make_runner

    from rammp_box_opening.models.container import ContainerPose
    from rammp_box_opening.tasks import press_demo
    from rammp_box_opening.tasks.depth_press import DepthPress

    built = []
    monkeypatch.setattr(press_demo, "build_grip_and_place_legs", lambda c, cfg, start_joints=None: built.append(cfg) or [])
    ctx.cpose = ContainerPose(xyz=(0.39, -0.197, 0.182), yaw=0.0)  # the base the pads measured
    ctx.lid_drop = ContainerPose(xyz=(0.552, -0.359, 0.195), yaw=0.0)  # on the scene camera's table
    result = DepthPress(free=True, contact=True, felt=True, popped=True, touch_xyz=(0.39, -0.197, 0.303))
    press_demo.lid_off_after_depth_press(ctx, _demo_cfg(), make_runner(ctx.client, tmp_path), True, result)
    assert built[0].grip_clear_m == pytest.approx(press_demo.DEPTH_GRIP_CLEAR_M)
    assert ctx.lid_drop.xyz == pytest.approx((0.552, -0.359, 0.182))
    # no touch measured: the configured grip and the scene's table, as before
    built.clear()
    ctx.lid_drop = ContainerPose(xyz=(0.552, -0.359, 0.195), yaw=0.0)
    press_demo.lid_off_after_depth_press(ctx, _demo_cfg(), make_runner(ctx.client, tmp_path), True,
                                         DepthPress(free=True, contact=True, felt=True, popped=True))
    assert built[0].grip_clear_m == pytest.approx(_demo_cfg().grip_clear_m) and ctx.lid_drop.xyz[2] == pytest.approx(0.195)


def test_a_grip_refused_after_the_depth_press_goes_home_honestly(ctx, tmp_path, monkeypatch):
    from conftest import runner as make_runner

    from rammp_box_opening.tasks import press_demo
    from rammp_box_opening.tasks.depth_press import DepthPress

    homes = []
    monkeypatch.setattr(press_demo, "build_grip_and_place_legs", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("IK_FAIL")))
    monkeypatch.setattr(press_demo, "try_home", lambda c, runner, execute, why: homes.append(why) or True)
    result = DepthPress(free=True, contact=True, felt=True, popped=True)
    code = press_demo.lid_off_after_depth_press(ctx, _demo_cfg(), make_runner(ctx.client, tmp_path), True, result)
    assert code == 1 and ctx.client.executed == [] and "STOP before the grip" in homes[0]


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


def test_a_set_down_felt_high_above_the_table_struck_something_and_holds_the_lid(ctx):
    """Flatbot 2026-10-23 17:57: the set-down flew merged behind the carry
    and its guard tripped at 70 % of the group — the wrist camera on the
    microwave beside the drop spot, the fingertips 134 mm above where the
    lid meets the table — and that read as "surface felt — set down": the
    fingers opened there. Where the fingertips were at the trip says what
    it was."""
    from rammp_box_opening.constants import TIP_TO_TOOL_M
    from rammp_box_opening.primitives.core import SETDOWN_HIGH_M
    from rammp_box_opening.runtime.legs import VerifyCtx
    from rammp_box_opening.tasks import press_demo

    legs = press_demo.build_place_legs(ctx, _demo_cfg())
    down = next(x for x in legs if x.name == "place:lid:down")
    lands = down.target[1][2] + press_demo.SETDOWN_OVERDRIVE_M  # the tool where the lid meets the table

    def v(tool_z):
        tip = [0.5627, -0.3747, tool_z + TIP_TO_TOOL_M]
        return down.verify(VerifyCtx(outcome="touch", progress=0.70, torque_peak=4.1, contact_xyz=tip))

    ok, why = v(lands + 0.134)
    assert not ok and "134 mm above" in why and "the lid is held" in why
    assert v(lands)[0] and v(lands - 0.004)[0]
    assert v(lands + SETDOWN_HIGH_M - 0.005)[0]  # a table a little higher there, a thicker lid


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

    def __init__(self, planner=(True,), driver=True, tf=True, others=()):
        self.planner = list(planner)
        self.driver = driver
        self.tf = tf
        self.others = list(others)
        self.planner_waits = []
        self._base_frame = "arm_mount"

    def tf_ready(self, target, source, timeout_s=2.0):
        return self.tf

    def other_arm_clients(self):
        return self.others

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


def test_a_run_refuses_before_any_motion_without_the_arms_tf_or_beside_adl():
    """Bench 2026-09-25: box_opening had stopped (no /tf at all) and the run
    flew to staging before it could not place a single wrist frame; and the
    ADL runtime, which moves this arm on its own, was running. Both now stop
    the run before anything moves, and name the one command that fixes both."""
    from rammp_box_opening.tasks import press_demo

    why = press_demo.readiness_refusal(_Readiness(tf=False), execute=True)
    assert why is not None and "TF" in why and "sheppy up box-opening" in why and "Nothing moved" in why
    why = press_demo.readiness_refusal(_Readiness(others=["rammp_adl_runtime"]), execute=True)
    assert why is not None and "rammp_adl_runtime" in why and "sheppy up box-opening" in why
    assert press_demo.readiness_refusal(_Readiness(), execute=True) is None
    assert press_demo.readiness_refusal(_Readiness(tf=False, others=["rammp_adl_runtime"]), execute=False) is None


def test_the_tf_gate_asks_for_the_worlds_wrist_camera_frame():
    """Flatbot 2026-10-22: the gate looked up the D405 default's
    end_effector_link, which flatbot (arm_-prefixed, OAK wrist) never
    publishes, and refused a healthy stack."""
    from rammp_box_opening.tasks import press_demo

    asked = []
    client = _Readiness()
    client.tf_ready = lambda target, source, timeout_s=2.0: asked.append(source) or True
    assert press_demo.readiness_refusal(client, execute=True, camera_cfg_name="camera_oak_wrist.yaml") is None
    assert asked == ["wrist_camera_color_optical_frame"]
    asked.clear()
    press_demo.readiness_refusal(client, execute=True)
    assert asked == ["end_effector_link"]  # the D405 default, unchanged


def test_depth_press_with_confirm_in_flight_is_refused_before_any_motion():
    """With detect.confirm_in_flight the scene approach flies the TF-height
    descent in its own goal; once that touches, --depth-press was skipped
    silently and the TF press ran. The combination stops the run before the
    runtime comes up."""
    import inspect
    from types import SimpleNamespace

    from rammp_box_opening.tasks import press_demo

    why = press_demo.depth_press_refusal(True, SimpleNamespace(confirm_in_flight=True), scene=True, cfg_path="x.yaml")
    assert why is not None and "confirm_in_flight" in why and "Nothing moved" in why and "x.yaml" in why
    src = inspect.getsource(press_demo.main)
    assert src.index("depth_press_refusal(") < src.index("init_runtime(") < src.index("home_first(")


def test_depth_press_without_confirm_in_flight_is_not_refused():
    from types import SimpleNamespace

    from rammp_box_opening.tasks import press_demo

    assert press_demo.depth_press_refusal(True, SimpleNamespace(confirm_in_flight=False), scene=True) is None
    # the shipped containers stay runnable under --depth-press
    for name in ("ankou_pink.yaml", "oxo_pop.yaml", "oxo_pop_small.yaml"):
        cfg = load_press_demo("src/rammp_box_opening/config/containers/" + name)
        assert press_demo.depth_press_refusal(True, cfg, scene=True) is None, name


def test_confirm_in_flight_without_depth_press_is_not_refused():
    from types import SimpleNamespace

    from rammp_box_opening.tasks import press_demo

    assert press_demo.depth_press_refusal(False, SimpleNamespace(confirm_in_flight=True), scene=True) is None


def test_no_scene_depth_press_is_not_refused():
    """--no-scene never flies approach_from_scene: the wrist search stages,
    and the depth press runs from there whatever confirm_in_flight says."""
    from types import SimpleNamespace

    from rammp_box_opening.tasks import press_demo

    assert press_demo.depth_press_refusal(True, SimpleNamespace(confirm_in_flight=True), scene=False) is None


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
    assert "2 pair(s) on file, mean offset [-0, -12] mm" in said  # horizontal: heights are not compared
    assert "2 older pair(s) were made under an earlier calibration" in said


def _pairs_file(tmp_path, offsets_mm, t0=3000.0, spots=((0.325, -0.291), (0.312, -0.158), (0.406, -0.219))):
    """A calibration file (in force from t=2000) and a residuals file with
    one pair per spot: the scene camera's box `offsets_mm` from the wrist's."""
    import json
    import os

    calib = tmp_path / "camera_scene.yaml"
    calib.write_text("xyz: [-0.0767, 0.6836, 0.3883]\nquat_xyzw: [0.026522, 0.043627, -0.28454, 0.957304]\n")
    os.utime(calib, (2000.0, 2000.0))
    f = tmp_path / "residuals.jsonl"
    rows = []
    for i, ((x, y), (dx, dy)) in enumerate(zip(spots, offsets_mm)):
        rows.append({"t": t0 + i, "scene_top": [x + dx / 1000.0, y + dy / 1000.0, 0.062], "wrist_top": [x, y, 0.095]})
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return f, calib


def test_the_scene_camera_is_trusted_on_what_it_last_measured(tmp_path):
    """Bench 2026-09-24: the cabinet-door tag refused the scene camera at
    6 px (a 3 s wrist search) while the camera had not moved — and at 5 px
    it would have flown to a box 12 cm off. What decides now is the last
    pair recorded under the calibration in force."""
    from rammp_box_opening.tasks import press_demo

    f, calib = _pairs_file(tmp_path, [(-126, -28), (-3, 2)])  # 12 cm off, then corrected
    assert press_demo.scene_trust(f, calib) == (True, None)
    f, calib = _pairs_file(tmp_path, [(-3, 2), (-118, -31)])
    trusted, why = press_demo.scene_trust(f, calib)
    assert trusted is False and "122 mm" in why
    f, calib = _pairs_file(tmp_path, [(-118, -31)], t0=1000.0)  # made under an earlier calibration
    assert press_demo.scene_trust(f, calib) == (True, None)
    assert press_demo.scene_trust(tmp_path / "none.jsonl", calib) == (True, None)


def test_the_scene_calibration_corrects_itself_from_pairs_that_agree(tmp_path):
    """Three runs that all found the scene camera ~12 cm off the same way
    (2026-09-24) are a calibration error, not noise: the mission corrects
    it, keeps a backup, and the pairs retire with the calibration they
    measured."""
    import yaml

    from rammp_box_opening.perception.scene_refine import calibrated_at, load_pairs
    from rammp_box_opening.tasks import press_demo

    f, calib = _pairs_file(tmp_path, [(-126, -28), (-119, -30), (-118, -31)])
    before = yaml.safe_load(calib.read_text())
    said = press_demo.auto_refine(f, calib)
    assert said is not None and "REFINED" in said and "3 pairs" in said
    after = yaml.safe_load(calib.read_text())
    assert after["xyz"][0] == pytest.approx(before["xyz"][0] + 0.121, abs=0.002)  # moved by the pairs' mean, in x
    assert after["xyz"][1] == pytest.approx(before["xyz"][1] + 0.0297, abs=0.002)
    assert after["xyz"][2] == pytest.approx(before["xyz"][2])  # heights are the table's, not the pairs'
    assert list(tmp_path.glob("camera_scene.yaml.bak-*"))  # the old one kept
    assert len(load_pairs(f, since=calibrated_at(calib))[0]) == 0  # the pairs retired with it
    # pairs that disagree, or a camera already right, are left alone
    f, calib = _pairs_file(tmp_path, [(-126, -28), (40, 60), (-10, 5)])
    assert press_demo.auto_refine(f, calib) is None
    f, calib = _pairs_file(tmp_path, [(4, -3), (-5, 2), (3, 6)])
    assert press_demo.auto_refine(f, calib) is None
    f, calib = _pairs_file(tmp_path, [(-126, -28), (-119, -30)], spots=((0.325, -0.291), (0.312, -0.158)))
    assert press_demo.auto_refine(f, calib) is None  # two are not enough


def test_a_camera_knocked_since_its_calibration_is_corrected_from_the_pairs_since(tmp_path):
    """Bench 2026-09-25: -2/-6 mm at 11:50, then -39/-53 and -53/-59 at the
    same spot once the camera had moved. Fitted with the old pair, the new
    ones never agree; the last three alone do."""
    from rammp_box_opening.tasks import press_demo

    spot = (0.422, 0.216)
    f, calib = _pairs_file(tmp_path, [(-2, -6), (-39, -53), (-53, -59)], spots=(spot,) * 3)
    assert press_demo.auto_refine(f, calib) is None  # the old pair is still one of the last three
    f, calib = _pairs_file(tmp_path, [(-2, -6), (-39, -53), (-53, -59), (-46, -55)], spots=(spot,) * 4)
    said = press_demo.auto_refine(f, calib)
    assert said is not None and "REFINED" in said


def test_the_scene_approach_says_so_when_the_wrist_sees_nothing_at_staging(ctx, tmp_path, monkeypatch):
    """The scene camera is coarse and may be wrong by a few centimetres; the
    wrist must confirm at staging, or the run stops there — the wrist aims,
    it does not search (owner, 2026-10-22). Since 2026-10-23 it moves over
    a lid it does see (restage_on_lid); with nothing in view it still stops,
    and nothing more moves."""
    from conftest import runner as make_runner

    press_demo = _search_env(monkeypatch)
    cfg = load_press_demo(CFG)
    r = make_runner(ctx.client, tmp_path)
    fix = _SceneFix((0.45, 0.0, -0.027), 0.0, (0.45, 0.0, 0.085))
    # the staging aim is the button circle on live frames (test_button_aim);
    # here it sees nothing
    monkeypatch.setattr(press_demo, "aim_button_at_staging", lambda *a, **k: (None, "no button"))
    reach = press_demo.approach_from_scene(None, ctx, cfg, r, _Watcher(), True, fix)
    assert reach.got is None and reach.status.startswith("no button")  # nothing confirmed
    assert "no lid in the wrist frame" in reach.status  # ... and no lid to move over
    assert not reach.failed and len(ctx.client.executed) == 1  # ... but the arm did fly to staging first
    # the descent was still planned on the way, from the coarse pose
    assert reach.preplanned is not None and reach.preplanned[0].name == "press:down"


def test_the_scene_approach_moves_over_a_lid_the_wrist_sees_and_aims_from_there(ctx, tmp_path, monkeypatch):
    """No button at staging, but the lid in view 7 cm off (flatbot
    2026-10-23): the approach moves over it and the press is aimed from
    there; the descent planned on the way, for the staging it left, is not
    flown."""
    from conftest import runner as make_runner

    press_demo = _search_env(monkeypatch)
    cfg = load_press_demo(CFG)
    r = make_runner(ctx.client, tmp_path)
    fix = _SceneFix((0.45, 0.0, -0.027), 0.0, (0.45, 0.0, 0.085))
    aimed = ((0.452, 0.07, 0.085), 0.0)
    aims = iter([(None, "no button"), (aimed, "3/3 still frames found the button circle")])
    monkeypatch.setattr(press_demo, "aim_button_at_staging", lambda *a, **k: next(aims))
    monkeypatch.setattr(press_demo, "wrist_lid", lambda w: ((0.452, 0.07, 0.085),
                                                            {"frac": 0.3, "cut": True, "table_z": -0.027, "view_xy": (0.52, 0.07)}))
    reach = press_demo.approach_from_scene(None, ctx, cfg, r, _Watcher(), True, fix)
    assert reach.got == aimed and not reach.failed
    assert reach.preplanned is None
    assert len(ctx.client.executed) == 2  # staging, then over the lid


def test_a_box_out_of_straight_down_reach_goes_to_the_reach_out_press_with_nothing_moved(ctx, tmp_path, monkeypatch):
    """Flatbot 2026-10-22: staging above a box 0.89 m out was IK_FAIL (it
    came out of the run as a traceback). The arm's own kinematics finding
    no staging pose, neither at staging_m nor at the clearance floor, hands
    the press to tasks/reach_press — before the planner is asked at all."""
    from conftest import runner as make_runner

    press_demo = _search_env(monkeypatch)
    cfg = load_press_demo(CFG)
    r = make_runner(ctx.client, tmp_path)
    fix = _SceneFix((0.67, -0.58, -0.027), 0.0, (0.67, -0.58, 0.085))

    real = press_demo.build_approach_leg
    why = {"e": "planning failed for approach:staging: MotionGenStatus.IK_FAIL"}
    planned = []

    def refused(ctx, cfg, start_joints=None):
        planned.append(cfg.staging_m)
        raise RuntimeError(why["e"])

    out_of_reach = {"v": True}
    monkeypatch.setattr(press_demo, "build_approach_leg", refused)
    monkeypatch.setattr(press_demo, "staging_out_of_reach", lambda ctx, cfg, live=None: out_of_reach["v"])
    reach = press_demo.approach_from_scene(None, ctx, cfg, r, _Watcher(), True, fix)
    assert reach.reach_out and reach.failed and reach.got is None and "no straight-down staging" in reach.status
    assert reach.cfg is cfg  # the reach-out press gets the config as asked
    assert not planned and not ctx.client.executed  # no doomed plan: the kinematics decided
    # IK_FAIL on a staging the arm CAN reach (a goal in collision): a STOP, not a different press
    out_of_reach["v"] = False
    reach = press_demo.approach_from_scene(None, ctx, cfg, r, _Watcher(), True, fix)
    assert reach.failed and not reach.reach_out and "refused" in reach.status
    # any other refusal of a reachable staging is a STOP too
    why["e"] = "set_world before planning approach:staging failed: set_world timed out"
    reach = press_demo.approach_from_scene(None, ctx, cfg, r, _Watcher(), True, fix)
    assert reach.failed and not reach.reach_out
    assert planned == [cfg.staging_m, cfg.staging_m] and not ctx.client.executed
    # a box straight down DOES reach is pressed the old way
    monkeypatch.setattr(press_demo, "build_approach_leg", real)
    monkeypatch.setattr(press_demo, "aim_button_at_staging", lambda *a, **k: (FIX, "3/3 frames"))
    near = _SceneFix((0.45, 0.0, -0.027), 0.0, (0.45, 0.0, 0.085))
    reach = press_demo.approach_from_scene(None, ctx, cfg, r, _Watcher(), True, near)
    assert not reach.reach_out and reach.got == FIX and reach.cfg is cfg


def _ladder_env(ctx, tmp_path, monkeypatch, ceiling_m, plan_error=None):
    """approach_from_scene with the reach proof a function of the staging
    height alone (in reach at and below `ceiling_m`) and the approach plan
    recorded; `plan_error` makes the planner refuse it."""
    from conftest import runner as make_runner

    press_demo = _search_env(monkeypatch)
    seen = {"proved": [], "planned": [], "proof_live": [], "plan_live": [], "reads": 0}
    real_joints = ctx.client.joints

    def joints():
        seen["reads"] += 1
        return real_joints()

    def proof(ctx, cfg, live=None):
        seen["proved"].append(round(cfg.staging_m, 4))
        seen["proof_live"].append(live)
        return cfg.staging_m > ceiling_m + 1e-9

    real_leg = press_demo.build_approach_leg

    def leg(ctx, cfg, start_joints=None):
        seen["planned"].append(round(cfg.staging_m, 4))
        seen["plan_live"].append(start_joints)
        seen["reads_by_plan"] = seen["reads"]  # the runner reads them again while flying
        if plan_error is not None:
            raise RuntimeError(plan_error)
        return real_leg(ctx, cfg, start_joints=start_joints)

    monkeypatch.setattr(ctx.client, "joints", joints)
    monkeypatch.setattr(press_demo, "staging_out_of_reach", proof)
    monkeypatch.setattr(press_demo, "build_approach_leg", leg)
    monkeypatch.setattr(press_demo, "aim_button_at_staging", lambda *a, **k: (FIX, "3/3 frames"))
    return press_demo, make_runner(ctx.client, tmp_path), seen


def test_a_staging_above_the_reach_ceiling_is_lowered_with_a_step_of_margin(ctx, tmp_path, monkeypatch):
    """Flatbot 2026-10-22: on a high table the 0.12 staging is above the
    straight-down ceiling where 0.10-0.11 reaches. The approach stages at
    the highest step that reaches, one step lower for margin, and the press
    stroke goes by that height."""
    cfg = load_press_demo(CFG)
    assert cfg.staging_m == pytest.approx(0.12)
    press_demo, r, seen = _ladder_env(ctx, tmp_path, monkeypatch, ceiling_m=0.11)
    floor = press_demo.staging_floor(ctx.model)
    assert floor == pytest.approx(0.10)  # oxo_pop: dims.z - button_offset.z + 0.10
    fix = _SceneFix((0.45, 0.0, -0.027), 0.0, (0.45, 0.0, 0.085))
    reach = press_demo.approach_from_scene(None, ctx, cfg, r, _Watcher(), True, fix)
    assert not reach.reach_out and not reach.failed and reach.got == FIX
    assert seen["proved"] == [0.12, 0.10, 0.115, 0.11]  # the floor first, then down from the top
    assert seen["planned"] == [0.105] and reach.cfg.staging_m == pytest.approx(0.105)
    assert min(seen["proved"] + seen["planned"]) >= floor - 1e-9  # never below the clearance floor
    assert cfg.staging_m == pytest.approx(0.12)  # the caller's config untouched
    # the descent pre-planned on the way is the lowered staging's
    down = reach.preplanned[0]
    assert down.contact_path_frac == pytest.approx(0.105 / (0.105 + cfg.travel_m))
    # one read of the joints, the same one to the proof and the plan
    assert seen["reads_by_plan"] == 1 and seen["proof_live"][0] is not None
    assert all(q is seen["proof_live"][0] for q in seen["proof_live"] + seen["plan_live"])


def test_a_box_out_of_reach_even_at_the_floor_costs_two_proofs_and_no_plan(ctx, tmp_path, monkeypatch):
    """Tonight's boxes, 0.59-0.60 m out, reached at neither 0.12 nor the
    0.10 floor: the reach-out press, after two proofs, not the whole ladder."""
    cfg = load_press_demo(CFG)
    press_demo, r, seen = _ladder_env(ctx, tmp_path, monkeypatch, ceiling_m=0.07)
    fix = _SceneFix((0.45, 0.0, -0.027), 0.0, (0.45, 0.0, 0.085))
    reach = press_demo.approach_from_scene(None, ctx, cfg, r, _Watcher(), True, fix)
    assert reach.reach_out and reach.failed and reach.cfg is cfg
    assert seen["proved"] == [0.12, 0.10] and seen["planned"] == []
    assert not ctx.client.executed
    # a staging_m already at the floor is proved once
    seen["proved"].clear()
    reach = press_demo.approach_from_scene(None, ctx, replace(cfg, staging_m=0.10), r, _Watcher(), True, fix)
    assert reach.reach_out and seen["proved"] == [0.10]


def test_a_lowered_staging_the_planner_refuses_goes_to_the_reach_out_press(ctx, tmp_path, monkeypatch):
    """The kinematics reaches a lowered staging, the planner (its padded
    cuboid, its finger spheres) does not: staging_m itself was already out
    of reach, so it is the reach-out press, as it was before the ladder."""
    cfg = load_press_demo(CFG)
    press_demo, r, seen = _ladder_env(
        ctx, tmp_path, monkeypatch, ceiling_m=0.105,
        plan_error="planning failed for approach:staging: MotionGenStatus.IK_FAIL",
    )
    fix = _SceneFix((0.45, 0.0, -0.027), 0.0, (0.45, 0.0, 0.085))
    reach = press_demo.approach_from_scene(None, ctx, cfg, r, _Watcher(), True, fix)
    assert seen["planned"] == [0.10]
    assert reach.reach_out and reach.failed and reach.cfg is cfg and "IK_FAIL" in reach.status
    assert not ctx.client.executed


def test_a_reachable_staging_the_planner_refuses_is_a_stop_proved_once(ctx, tmp_path, monkeypatch):
    cfg = load_press_demo(CFG)
    fix = _SceneFix((0.45, 0.0, -0.027), 0.0, (0.45, 0.0, 0.085))
    for error in ("planning failed for approach:staging: MotionGenStatus.IK_FAIL",
                  "set_world before planning approach:staging failed: set_world timed out"):
        press_demo, r, seen = _ladder_env(ctx, tmp_path, monkeypatch, ceiling_m=0.2, plan_error=error)
        reach = press_demo.approach_from_scene(None, ctx, cfg, r, _Watcher(), True, fix)
        assert reach.failed and not reach.reach_out and "refused" in reach.status
        assert seen["proved"] == [0.12] and seen["planned"] == [0.12]


def test_flatbots_high_table_lowers_the_staging_where_straight_down_reaches(ctx, monkeypatch):
    """Flatbot's URDF, the pink canister on the 2026-10-22 table (0.2933 in
    arm_mount): the 200421 box, 0.59 m out, reaches at no staging down to
    the floor (the reach-out press, as that night); the same table 0.40 m
    out along the same bearing presses straight down from a lower staging."""
    from rammp_box_opening.models.container import ContainerModel
    from rammp_box_opening.runtime import approach
    from rammp_box_opening.runtime.kinematics import ArmChain
    from rammp_box_opening.tasks import press_demo

    pink = "src/rammp_box_opening/config/containers/ankou_pink.yaml"
    chain = ArmChain(path="src/rammp_box_opening/test/data/arm_flatbot_20261021.urdf", root="arm_mount")
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    ctx.model = ContainerModel.load(pink)
    cfg = load_press_demo(pink)
    live = [0.067, -1.897, -3.137, -2.269, -0.211, -0.683, 1.725]  # flatbot's HOME
    x, y, z, yaw = 0.565, -0.1794, 0.2933, math.radians(86.2)
    ctx.cpose = ContainerPose(xyz=(x, y, z), yaw=yaw)
    assert press_demo.staging_for_reach(ctx, cfg, live=live) is None
    k = 0.40 / math.hypot(x, y)
    ctx.cpose = ContainerPose(xyz=(x * k, y * k, z), yaw=yaw)
    lowered = press_demo.staging_for_reach(ctx, cfg, live=live)
    assert lowered is not None and 0.10 - 1e-9 <= lowered.staging_m <= 0.115 + 1e-9


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
    assert press_demo.try_home(c, make_runner(c.client, tmp_path), True, "STOP: test") is True
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
    assert press_demo.try_home(c, make_runner(c.client, tmp_path), True, "STOP: test") is False
    assert "arm holds" in capsys.readouterr().out and c.client.executed == []


def test_a_home_that_was_refused_or_did_not_arrive_says_so(ctx, tmp_path):
    """try_home's answer is whether the arm got home: the depth press exits
    on it (a popped knob with the arm held over the box is not a success)."""
    from conftest import runner as make_runner

    from rammp_box_opening.tasks import press_demo

    class Refused:
        success = False
        message = "INVALID_START_STATE_WORLD_COLLISION"
        trajectory = None

    ctx.press_cfg = _demo_cfg()
    assert press_demo.try_home(ctx, make_runner(ctx.client, tmp_path), True, "STOP: test") is True
    ctx.client.exec_script = [("failed", {"message": "driver aborted the goal", "progress": 0.2})]
    assert press_demo.try_home(ctx, make_runner(ctx.client, tmp_path), True, "STOP: test") is False
    ctx.client.plans = [Refused(), Refused()]  # from here, and the rise over the box
    assert press_demo.try_home(ctx, make_runner(ctx.client, tmp_path), True, "STOP: test") is False


def test_a_stop_with_a_joint_on_its_limit_steps_off_it_and_gets_home(ctx, tmp_path, capsys):
    """Flatbot 2026-10-23: a staging at the reach limit left arm_joint_6
    0.100 rad from its limit. The planner refuses a start inside its margin,
    so the recovery home AND the rise over the box were refused and the arm
    held. It now steps plan-free along the joint line toward the rest pose
    first, as home_arm does, and plans home from where the step ends."""
    import pytest
    from conftest import runner as make_runner

    from rammp_box_opening.constants import PLANNER_LIMIT_MARGIN_RAD
    from rammp_box_opening.primitives.core import CLEAR_MARGIN_FIRST_RAD
    from rammp_box_opening.runtime import approach
    from rammp_box_opening.tasks import press_demo

    c = ctx
    cfg = _demo_cfg()
    c.press_cfg = cfg
    rest = press_demo.rest_joints(cfg)
    lo, hi = approach._chain().limits[5]
    start = list(rest)
    start[5] = lo + 0.0999  # arm_joint_6 a hair inside the planner's margin, as it was
    c.client.live = list(start)

    class InMargin:
        success = False
        message = "START_IN_LIMIT_MARGIN: start: arm_joint_6 is 0.100 rad from its position limit"
        trajectory = None

    def planner(real):  # refuses any start inside its margin, as rammp-curobo does
        def plan(*a, **kw):
            q = kw.get("start_joints", a[-1])
            return InMargin() if min(q[5] - lo, hi - q[5]) < PLANNER_LIMIT_MARGIN_RAD else real(*a, **kw)
        return plan

    c.client.plan_to_joints = planner(c.client.plan_to_joints)
    c.client.plan_to_pose = planner(c.client.plan_to_pose)
    assert press_demo.try_home(c, make_runner(c.client, tmp_path), True, "STOP: test") is True
    said = capsys.readouterr().out
    assert "joint_6 is 0.100 rad from its limit" in said and "a plan-free step" in said
    assert "arm holds" not in said
    # nothing was planned from the refused start: home from the step's end
    end = c.client.joint_starts[0]
    assert min(end[5] - lo, hi - end[5]) == pytest.approx(CLEAR_MARGIN_FIRST_RAD, abs=1e-6)
    # the guarded plan-free step and the planned home fly apart, and it got home
    assert len(c.client.executed) == 2
    assert c.client.live == pytest.approx(rest)


def test_the_mission_record_keeps_the_table_the_scene_camera_measured(ctx, tmp_path):
    """Flatbot (no survey): the record was built before the scene measured
    the table, and every truth.json said table_z NaN."""
    import json
    from types import SimpleNamespace

    import numpy as np

    from rammp_box_opening.detection_set import MissionFrames
    from rammp_box_opening.tasks import press_demo

    worlds = SimpleNamespace(table_top_z=None, table_extent=None)
    watcher = SimpleNamespace(table_z=None)
    ctx.mission_frames = MissionFrames(tmp_path / "set", "ankou_pink.yaml", worlds.table_top_z, now=0)
    fix = SimpleNamespace(table_z_scene=0.2933, table_extent=(0.3, 0.9, -0.5, 0.2))
    press_demo.adopt_scene_table(ctx, worlds, watcher, fix)
    assert worlds.table_top_z == watcher.table_z == 0.2933 and worlds.table_extent == fix.table_extent
    ctx.mission_frames.frames.append(("scene_0", {"x": np.zeros(1)}))
    doc = json.loads((ctx.mission_frames.close(exit_code=1) / "truth.json").read_text())
    assert doc["table_z"] == pytest.approx(0.2933)
    # a surveyed table is the one kept; an unmeasured one changes nothing
    worlds2 = SimpleNamespace(table_top_z=0.05, table_extent=None)
    press_demo.adopt_scene_table(ctx, worlds2, watcher, fix)
    assert worlds2.table_top_z == 0.05
    worlds3 = SimpleNamespace(table_top_z=None, table_extent=None)
    press_demo.adopt_scene_table(ctx, worlds3, watcher, SimpleNamespace(table_z_scene=float("nan"), table_extent=None))
    assert worlds3.table_top_z is None


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
    assert names(legs) == ["retreat:restage", "press:close"]
    rise, close = legs
    assert rise.world == "interaction_button" and close.world == "interaction_button"
    assert rise.target[1] == pytest.approx([button[0], button[1], button[2] + cfg.staging_m])
    assert rise.speed == pytest.approx(TRANSIT_SPEED)
    assert c.lid_at is None  # nothing has been placed: no phantom lid in the next worlds
    assert c.last_world[0] == "interaction_button"
    # ... and the Runner's gates let it fly (a rise out of the corridor just
    # descended, like retreat and lift)
    res = make_runner(c.client, tmp_path).run(legs, execute=True)
    assert [r.ok for r in res] == [True, True]


class _Refused:
    success = False
    message = "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION"
    trajectory = None


def _timeline(client):
    """Instrument a FakeClient: every execution, gripper send and join, in
    order, as (event, detail) — a send inside an execution is tagged with
    the execution it rode."""
    events = []
    execute, send, join = client.execute, client.gripper_send, client.gripper_join

    def ex(traj_, speed, guard=None, while_running=None, stop_when=None):
        events.append(("execute", "guarded" if guard is not None else "free"))
        out = execute(traj_, speed, guard=guard, while_running=while_running, stop_when=stop_when)
        events.append(("arrived", None))
        return out

    client.execute = ex
    client.gripper_send = lambda pos: events.append(("send", pos)) or send(pos)
    client.gripper_join = lambda h: events.append(("join", h[1])) or join(h)
    return events


def test_the_re_press_closes_the_fingers_while_it_rises(ctx, tmp_path):
    """A NOT POPPED retry stood at the hop for the whole blocking close
    before it rose. The close now rides the rise: sent as the rise starts,
    joined when it arrives — before anything guarded can descend."""
    from conftest import runner as make_runner

    from rammp_box_opening.tasks import press_demo

    c = ctx
    rise, close = press_demo.build_restage_legs(c, _demo_cfg())
    assert close.name == "press:close" and close.world == "interaction_button"
    assert close.defer_join and close.send_with_previous_motion and close.verify is None
    events = _timeline(c.client)
    res = make_runner(c.client, tmp_path).run([rise, close], execute=True)
    assert [r.ok for r in res] == [True, True]
    cmd = close.gripper_cmd
    assert events == [("execute", "free"), ("send", cmd), ("arrived", None), ("join", cmd)]


def _press_again_setup(ctx, tmp_path, monkeypatch, build):
    """_press_again with the descent built by `build` and run_push patched
    out: the restage rise is planned to end where the descent starts."""
    from conftest import runner as make_runner
    from conftest import traj

    from rammp_box_opening.tasks import press_demo

    descent = _descent_leg()

    class Rise:
        success, message = True, "ok"
        trajectory = traj(ctx.client.live, list(descent.traj.points[0].positions))

    ctx.client.plans = [Rise()]
    ctx.press_cfg = _demo_cfg()
    pushed = []
    monkeypatch.setattr(press_demo, "build_press_legs", lambda c, cfg, start_joints=None: build(descent, start_joints))
    monkeypatch.setattr(
        press_demo, "run_push", lambda c, cfg, runner, args, touch, touch_leg=None, **k: pushed.append(touch_leg) or ([], touch)
    )
    args = type("A", (), {"execute": True, "press_only": True})()
    return descent, pushed, args, make_runner(ctx.client, tmp_path)


def test_the_re_press_is_planned_under_the_rise_and_flies_its_free_air_fast(ctx, tmp_path, monkeypatch):
    """The second press planned its descent with the arm standing at
    staging, then flew all 75 mm of free air at contact speed (4.2 s on the
    real waypoint descent, against 2.3 s for the first press). It is now
    planned while the rise flies, from the rise's end, and warped like the
    first press — the guarded line and where its guard arms unchanged."""
    from rammp_box_opening.runtime.guards import GROUP_SETTLE_S
    from rammp_box_opening.runtime.stamps import secs
    from rammp_box_opening.tasks import press_demo

    c = ctx
    events = _timeline(c.client)
    built = []

    def build(descent, start_joints):
        built.append((None if start_joints is None else list(start_joints), list(events)))
        return [descent]

    descent, pushed, args, r = _press_again_setup(c, tmp_path, monkeypatch, build)
    c.client.exec_script = [
        ("arrived", {"message": "ok", "progress": 1.0, "torque_peak": 0.0}),  # the rise
        ("touch", {"message": "contact", "progress": 0.97, "torque_peak": 3.2}),  # the press
    ]
    press_demo._press_again(c, _demo_cfg(), r, args, None, None, 0)
    # one build, from the rise's end, while the rise flew (the close riding it)
    assert len(built) == 1
    assert built[0][0] == pytest.approx(list(descent.traj.points[0].positions))
    assert built[0][1][-1] == ("send", press_demo.GRIPPER_CMD_CLOSED)
    # the press handed on is warped as the first press is
    assert pushed == [descent]
    assert descent.warp is not None and descent.speed == 1.0
    idx = descent.waypoint[0]
    start = secs(descent.traj.points[idx].time_from_start)
    total = secs(descent.traj.points[-1].time_from_start)
    assert descent.guard.arm_after * total == pytest.approx(start + GROUP_SETTLE_S, abs=0.01)
    assert c.last_pose == (list(descent.target[1]), list(descent.target[2]))
    # the close is joined before the guarded press descends
    assert [e for e in events if e[0] in ("join", "execute")] == [
        ("execute", "free"), ("join", press_demo.GRIPPER_CMD_CLOSED), ("execute", "guarded")
    ]


def test_a_refused_re_press_goes_home_instead_of_a_traceback(ctx, tmp_path, monkeypatch, capsys):
    """The descent of the second press refused under the rise and again
    from staging: an honest stop and a recovery home, exit 1."""
    from rammp_box_opening.tasks import press_demo

    def build(descent, start_joints):
        raise RuntimeError("planning failed for press:down")

    _d, pushed, args, r = _press_again_setup(ctx, tmp_path, monkeypatch, build)
    homes = []
    monkeypatch.setattr(press_demo, "try_home", lambda c, runner, execute, why: homes.append(why))
    with pytest.raises(SystemExit) as e:
        press_demo._press_again(ctx, _demo_cfg(), r, args, None, None, 0)
    assert e.value.code == 1 and pushed == []
    assert len(homes) == 1 and "STOP before pressing again" in homes[0] and "press:down" in homes[0]


def test_press_only_plans_its_home_from_the_hop_under_the_retreat(ctx, tmp_path, monkeypatch):
    """--press-only planned the rise and home from the hop only after the
    pop check, the arm standing 5 cm over the lid. They are planned while
    the retreat flies, from its end, as the full mission's grip is; home
    then flies with no plan made after the pop check, in one execution."""
    from conftest import runner as make_runner

    from rammp_box_opening.tasks import press_demo

    c = ctx
    cl = c.client
    cfg = _demo_cfg()
    r = make_runner(cl, tmp_path)
    args = type("A", (), {"execute": True, "press_only": True})()
    touch = [type("R", (), {"contact_xyz": predicted_contact(c), "leg_name": "press:down"})()]
    plans = lambda: len(cl.approach_offsets) + len(cl.joint_starts)  # noqa: E731
    at_check = []
    monkeypatch.setattr(press_demo, "knob_height_mm", lambda node, watcher, ctx, **kw: at_check.append(plans()) or 11.0)
    press_demo.run_push(c, cfg, r, args, touch, node=object(), watcher=object())
    legs = r.lookahead_result
    assert names(legs) == ["retreat:rise", "home"]
    assert legs[0].world == "interaction_button" and legs[1].world == "full"
    assert list(legs[0].traj.points[0].positions) == pytest.approx(cl.live)  # from the hop
    n_exec = len(cl.executed)
    assert press_demo.fly_home_from_hop(c, cfg, r, True)
    assert at_check == [plans()]  # nothing planned after the pop check
    assert len(cl.executed) == n_exec + 1  # rise and home: one execution
    assert cl.live == pytest.approx(press_demo.rest_joints(cfg))


def test_press_only_home_refused_from_the_hop_is_an_honest_stop(ctx, tmp_path, capsys):
    """main built the press-only home from the hop outside any guard: a
    refused rise ended the run in a RuntimeError traceback, the arm over
    the open box and no recovery tried. It now goes through try_home and
    reports False (main exits 1); nothing moves on a refused plan."""
    from conftest import runner as make_runner

    from rammp_box_opening.tasks import press_demo

    c = ctx
    cfg = _demo_cfg()
    c.press_cfg = cfg
    r = make_runner(c.client, tmp_path)
    r.lookahead_result = None  # the lookahead failed under the retreat
    # the rise from the hop, try_home's bench-world home, its rise
    c.client.plans = [_Refused(), _Refused(), _Refused()]
    assert not press_demo.fly_home_from_hop(c, cfg, r, True)
    out = capsys.readouterr().out
    assert "home from the hop refused" in out and "returning home" in out and "arm holds" in out
    assert c.client.executed == []


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


def _aim_frame_env(ctx, tmp_path, monkeypatch):
    """press_demo with captures in tmp_path, a hop frame and its pose, a
    sighting on it, and write_frame held until the returned gate is set.
    Returns (press_demo, grab, pose, sighting, gate)."""
    import threading
    from types import SimpleNamespace

    from rammp_box_opening.perception import depth_source
    from rammp_box_opening.tasks import press_demo

    monkeypatch.setattr(press_demo, "CAPTURES_DIR", tmp_path)
    grab, pose = _hop_frame(ctx, 0.0)
    monkeypatch.setattr(depth_source, "camera_pose_at", lambda g, offset_s=None: pose)
    gate, real = threading.Event(), press_demo.write_frame

    def held(folder, arrays, **kw):
        assert gate.wait(5.0)
        return real(folder, arrays, **kw)

    monkeypatch.setattr(press_demo, "write_frame", held)
    sighting = SimpleNamespace(uv=(424.0, 238.0), r_px=40.0, diameter_m=0.045)
    return press_demo, grab, pose, sighting, gate


def test_the_aim_frame_is_written_off_the_critical_path(ctx, tmp_path, monkeypatch):
    """run-20261022-200421: every aim waited ~45 ms for its evidence frame's
    zlib before the next leg could start. The aim returns its folder at
    once; the frame — the one the aim was taken on, not a newer one the
    grabber has since put in its place — lands a moment later."""
    import numpy as np

    press_demo, grab, pose, sighting, gate = _aim_frame_env(ctx, tmp_path, monkeypatch)
    color, depth = grab.color, grab.depth
    folder = press_demo.save_aim_frame(grab, sighting, "aim")
    try:
        assert folder is not None and folder.is_dir() and not (folder / "frame_000.npz").exists()
        # the grabber's next message: new arrays, as d405.py assigns them
        grab.color, grab.depth = np.full_like(color, 255), np.zeros_like(depth)
    finally:
        gate.set()
    press_demo.wait_capture_writes()
    saved = np.load(folder / "frame_000.npz")
    assert np.allclose(saved["trans_cam"], pose[1])
    assert np.array_equal(saved["color"], color) and np.array_equal(saved["depth"], depth)
    assert (folder / "frame_000.jpg").exists()


def test_the_aim_frame_prunes_old_captures_after_its_write(ctx, tmp_path, monkeypatch):
    import numpy as np

    press_demo, grab, pose, sighting, gate = _aim_frame_env(ctx, tmp_path, monkeypatch)
    for i in range(press_demo.AIM_CAPTURES_KEPT + 1):
        (tmp_path / ("wrist-aim-20000101-0000%02d" % i)).mkdir()
    gate.set()
    folder = press_demo.save_aim_frame(grab, sighting, "aim")
    press_demo.wait_capture_writes()
    kept = sorted(tmp_path.glob("wrist-aim-*"))
    assert len(kept) == press_demo.AIM_CAPTURES_KEPT and folder in kept
    assert np.load(folder / "frame_000.npz")["depth"].shape == (480, 848)


def test_the_mission_frames_wait_for_the_aim_frame_they_copy(ctx, tmp_path, monkeypatch):
    """MissionFrames.close copies the aim's folder in as "staging": a run
    that ends straight after the aim must not copy it half written."""
    import threading

    import numpy as np

    from rammp_box_opening.detection_set import MissionFrames

    press_demo, grab, pose, sighting, gate = _aim_frame_env(ctx, tmp_path / "captures", monkeypatch)
    folder = press_demo.save_aim_frame(grab, sighting, "aim")
    frames = MissionFrames(tmp_path / "missions", "ankou_pink.yaml", 0.05)
    frames.located([0.4, 0.0, 0.15], "aim", aim_capture=folder)
    timer = threading.Timer(0.2, gate.set)
    timer.start()
    try:
        press_demo.keep_mission_frames(frames, None)
    finally:
        gate.set()
        timer.cancel()
    staging = frames.dir / "staging"
    assert np.allclose(np.load(staging / "frame_000.npz")["trans_cam"], pose[1])
    assert (staging / "frame_000.jpg").exists()


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


def _restage_env(ctx, tmp_path, monkeypatch, lids, aims):
    """press_demo with the wrist's lid sightings and aims scripted: lids[i]
    is what the i-th wrist_lid returns ((lid xy, where the camera looks), or
    None), aims[i] the i-th aim after a move (None: no button)."""
    from conftest import runner as make_runner

    from rammp_box_opening.models.container import ContainerPose

    press_demo = _search_env(monkeypatch)
    state = {"lid": 0, "aim": 0, "priors": []}

    def lid(watcher):
        got = lids[min(state["lid"], len(lids) - 1)]
        state["lid"] += 1
        if got is None:
            return None, "nothing stands the container's height above the table in view"
        (x, y), view = got
        return (x, y, 0.39), {"frac": 0.2, "cut": True, "table_z": 0.27, "view_xy": view}

    def aim(node, watcher, yaw, **kw):
        state["priors"].append(kw.get("prior_xy"))
        got = aims[min(state["aim"], len(aims) - 1)]
        state["aim"] += 1
        return got, "3/3 still frames found the button circle" if got else "0/9 still frames found the button circle"

    monkeypatch.setattr(press_demo, "wrist_lid", lid)
    monkeypatch.setattr(press_demo, "aim_button_at_staging", aim)
    ctx.cpose = ContainerPose(xyz=(0.395, -0.193, 0.198), yaw=0.0)  # the scene camera's fix, 15:09
    return press_demo, make_runner(ctx.client, tmp_path), _Watcher(), state


def test_no_button_at_staging_moves_the_camera_over_the_lid_it_sees(ctx, tmp_path, monkeypatch, capsys):
    """Flatbot 2026-10-23: at staging the tool stands over the button and
    the wrist camera ~10 cm beside it, so the lid sits at the frame's edge
    and a few centimetres off hide the button. The camera is moved over the
    lid it sees — the lid's place less where the camera looks, both through
    one camera pose, so the wrist mount's error in TF cancels (moving the
    tool over the lid's TF place went 6 cm the wrong way, 15:09) — and the
    button is then looked for on that lid. The world's box stays the scene
    camera's."""
    from rammp_box_opening.runtime.approach import _chain

    aimed = ((0.463, -0.209, 0.32), 0.0)
    lid, view = (0.463, -0.209), (0.554, -0.218)  # the 15:09 frame, through its pose
    press_demo, r, watcher, state = _restage_env(ctx, tmp_path, monkeypatch, lids=[(lid, view)], aims=[aimed])
    tool = _chain().fk(ctx.client.joints())[1]
    got, status, moves, _seen = press_demo.restage_on_lid(None, ctx, _demo_cfg(), r, watcher, True, 0.0,
                                                           "0/39 still frames found the button circle")
    assert got == aimed and moves == 1 and len(ctx.client.executed) == 1
    # the tool moved by the camera's move: -91, +9 mm, at its own height
    assert ctx.last_pose[0] == pytest.approx([tool[0] - 0.091, tool[1] + 0.009, tool[2]], abs=1e-6)
    assert state["priors"] == [lid]  # the button looked for on the lid it saw
    assert ctx.cpose.xyz[:2] == pytest.approx((0.395, -0.193))
    assert "moving the camera over it" in capsys.readouterr().out


def test_the_re_staging_is_bounded(ctx, tmp_path, monkeypatch):
    """No lid in view, a lid more than RESTAGE_MAX_M from where the camera
    looks, or one where it looks already (the button is hidden by something
    else): nothing moves. A button still not found after RESTAGE_TRIES
    moves stops."""
    for lids, word in (([None], "no lid in the wrist frame"),
                       ([((0.40, -0.20), (0.55, -0.20))], "more than a re-staging"),
                       ([((0.40, -0.20), (0.405, -0.20))], "moving would not show the button")):
        ctx.client.executed.clear()
        press_demo, r, watcher, state = _restage_env(ctx, tmp_path, monkeypatch, lids=lids, aims=[None])
        got, status, moves, _seen = press_demo.restage_on_lid(None, ctx, _demo_cfg(), r, watcher, True, 0.0, "why")
        assert got is None and moves == 0 and word in status and ctx.client.executed == []
    ctx.client.executed.clear()
    press_demo, r, watcher, state = _restage_env(
        ctx, tmp_path, monkeypatch, lids=[((0.46, -0.21), (0.55, -0.22)), ((0.46, -0.21), (0.48, -0.21))], aims=[None, None])
    got, status, moves, _seen = press_demo.restage_on_lid(None, ctx, _demo_cfg(), r, watcher, True, 0.0, "why")
    assert got is None and moves == press_demo.RESTAGE_TRIES == 2 and "after 2 moves" in status
    assert len(ctx.client.executed) == 2


def test_an_aim_far_from_under_the_tool_is_not_centred_on(ctx, tmp_path, monkeypatch):
    """Flatbot 2026-10-23 15:09: an aim 313 mm from under the tool (a circle
    on a cabinet) and the centring flew there. Farther than
    RECENTRE_MAX_MOVE_M, nothing moves to it and the run stops."""
    homes = []
    first = ((0.673, 0.005, 0.32), 0.0)
    press_demo, r, watcher, state = _centring_env(ctx, tmp_path, monkeypatch, tool_xys=[(0.395, -0.193)], aims=[])
    monkeypatch.setattr(press_demo, "try_home", lambda c, runner, execute, why: homes.append(why))
    with pytest.raises(SystemExit):
        press_demo.centre_over_button(None, ctx, _demo_cfg(), r, watcher, True, first)
    assert ctx.client.executed == [] and state["n"] == 0
    assert "more than a re-centre" in homes[0]


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
        ctx, tmp_path, monkeypatch, tool_xys=[(0.45, 0.05), (0.45, 0.04), (0.45, 0.04)], aims=a[1:]
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


class _IkFail:
    success = False
    message = "MotionGenStatus.IK_FAIL"
    trajectory = None


def test_a_refused_recentre_plan_presses_from_the_last_aim(ctx, tmp_path, monkeypatch, capsys):
    """Near the reach edge the move over the button can be IK_FAIL: a
    refused plan sends nothing, so the arm still stands at staging and the
    last aim stands — it used to escape main() as a RuntimeError traceback,
    the arm left hanging over the lid."""
    first = ((0.450, 0.0, 0.085), 0.0)
    press_demo, r, watcher, state = _centring_env(ctx, tmp_path, monkeypatch, tool_xys=[(0.45, 0.06)], aims=[])
    ctx.client.plans = [_IkFail()]
    got, moves, off = press_demo.centre_over_button(None, ctx, _demo_cfg(), r, watcher, True, first)
    assert (got, moves, state["n"]) == (first, 0, 0) and ctx.client.executed == []
    out = capsys.readouterr().out
    assert "move over the button was refused" in out and "IK_FAIL" in out and "pressing from the last aim" in out


def test_a_recentre_the_runner_refused_presses_from_the_last_aim(ctx, tmp_path, monkeypatch):
    """The runner refusing the move (outcome 'refused': nothing sent) is the
    same: the arm is still at staging, the last aim stands."""
    first = ((0.450, 0.0, 0.085), 0.0)
    press_demo, r, watcher, state = _centring_env(ctx, tmp_path, monkeypatch, tool_xys=[(0.45, 0.06)], aims=[])
    monkeypatch.setattr(r, "_refusal", lambda leg: "wrist efforts stale")
    got, moves, off = press_demo.centre_over_button(None, ctx, _demo_cfg(), r, watcher, True, first)
    assert (got, moves, state["n"]) == (first, 0, 0) and ctx.client.executed == []


def test_a_recentre_that_failed_in_flight_does_not_go_on_to_press(ctx, tmp_path, monkeypatch, capsys):
    """A move that went out and failed partway (an aborted goal, a following
    error) leaves the arm wherever it stopped: the run ends there with the
    arm holding, as for every other failed motion — no press after a motion
    fault, and no automatic recovery move either."""
    first = ((0.450, 0.0, 0.085), 0.0)
    press_demo, r, watcher, state = _centring_env(ctx, tmp_path, monkeypatch, tool_xys=[(0.45, 0.06)], aims=[first])
    ctx.client.exec_script = [("failed", {"message": "goal aborted: following error", "progress": 0.4})]
    with pytest.raises(SystemExit) as e:
        press_demo.centre_over_button(None, ctx, _demo_cfg(), r, watcher, True, first)
    assert e.value.code == 1
    assert len(ctx.client.executed) == 1 and state["n"] == 0  # the failed move alone: no aim, no home
    out = capsys.readouterr().out
    assert "the move over the button failed" in out and "returning home" not in out


def test_the_re_aim_over_the_button_is_asked_whether_the_box_is_open(ctx, tmp_path, monkeypatch, capsys):
    """The aim from straight over the button is the best look at the knob:
    an open box it shows (10 mm up) stops the run and goes home, like the
    first aim's. A flush reading (2 mm) carries on."""
    from rammp_box_opening.models.container import ContainerPose

    first, second = ((0.450, 0.0, 0.085), 0.0), ((0.451, 0.001, 0.085), 0.0)
    for up_mm, stops in ((10.0, True), (2.0, False)):
        press_demo, r, watcher, state = _centring_env(
            ctx, tmp_path, monkeypatch, tool_xys=[(0.45, 0.06), (0.451, 0.0015)], aims=[second]
        )
        asked = []

        def aim(node, watcher, yaw, out=None, up_mm=up_mm, asked=asked, state=state, **kw):
            state["n"] += 1  # the tool stands over the button from here
            asked.append(out is not None)
            if out is not None:
                out["button_up_mm"] = up_mm
            return second, "3/3 still frames found the button circle"

        monkeypatch.setattr(press_demo, "aim_button_at_staging", aim)
        ctx.cpose = ContainerPose(xyz=(0.45, 0.0, -0.07), yaw=0.0)
        ctx.client.executed.clear()
        if stops:
            with pytest.raises(SystemExit) as e:
                press_demo.centre_over_button(None, ctx, _demo_cfg(), r, watcher, True, first)
            assert e.value.code == 1
            out = capsys.readouterr().out
            assert "already OPEN" in out and "returning home" in out
            assert len(ctx.client.executed) == 2  # the move over it, then home
        else:
            got, moves, _off = press_demo.centre_over_button(None, ctx, _demo_cfg(), r, watcher, True, first)
            assert (got, moves) == (second, 1)
        assert asked == [True], up_mm


def test_a_refused_staging_over_the_search_fix_goes_home(ctx, tmp_path, monkeypatch, capsys):
    from conftest import runner as make_runner

    from rammp_box_opening.models.container import ContainerPose

    press_demo = _search_env(monkeypatch)
    monkeypatch.setattr(press_demo, "aim_button_at_staging", lambda *a, **k: (FIX, "3/3"))
    watcher = _Watcher()
    watcher.to_container_pose = lambda got: ContainerPose(xyz=(got[0][0], got[0][1], -0.027), yaw=got[1])
    ctx.client.plans = [_IkFail()]  # approach:staging
    with pytest.raises(SystemExit) as e:
        press_demo.stage_over_search_fix(None, ctx, _demo_cfg(), make_runner(ctx.client, tmp_path), watcher, True, FIX)
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "over the search's fix was refused" in out and "returning home" in out


def test_a_refused_second_press_plan_goes_home_not_a_traceback(ctx, tmp_path, monkeypatch, capsys):
    """NOT POPPED: the arm rises back to staging and the second press is
    planned there. A refusal of that plan stops with a STOP line and the
    recovery home, not a traceback over the box."""
    from conftest import runner as make_runner

    from rammp_box_opening.tasks import press_demo

    args = type("A", (), {"execute": True, "press_only": True})()
    touch = [type("R", (), {"contact_xyz": predicted_contact(ctx), "leg_name": "press:down"})()]
    monkeypatch.setattr(press_demo, "knob_height_mm", lambda node, watcher, ctx, **kw: 0.2)

    def refused(ctx, cfg, **kw):
        raise RuntimeError("planning failed for press:down: MotionGenStatus.IK_FAIL")

    monkeypatch.setattr(press_demo, "build_press_legs", refused)
    with pytest.raises(SystemExit) as e:
        press_demo.run_push(ctx, _demo_cfg(), make_runner(ctx.client, tmp_path), args, touch, node=object(), watcher=object())
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "NOT POPPED" in out and "STOP before pressing again" in out and "returning home" in out


def test_before_the_depth_press_staging_does_not_recentre_through_the_wrist_mount(ctx, tmp_path, monkeypatch):
    """--depth-press centres the fingers itself from straight down, by the
    wrist camera's own frame (tasks/depth_press CENTRE); the staging aim's
    recentre goes through the wrist mount's TF, wrong on flatbot (83-85 mm
    off the scene camera's box, 2026-10-22). Nothing moves for it — but the
    box is still put where the aim saw it, for every world from here."""
    first = ((0.450, 0.000, 0.085), 0.0)
    press_demo, r, watcher, state = _centring_env(ctx, tmp_path, monkeypatch, tool_xys=[(0.45, 0.08)], aims=[])
    got, moves, off = press_demo.centre_for_press(None, ctx, _demo_cfg(), r, watcher, True, first, depth_press=True)
    assert (got, moves, state["n"]) == (first, 0, 0) and ctx.client.executed == []
    assert ctx.cpose == watcher.to_container_pose(first) and off == pytest.approx(0.08)


def test_before_the_depth_press_an_aim_beyond_its_reach_gets_one_planned_move(ctx, tmp_path, monkeypatch):
    from rammp_box_opening.tasks.depth_press import CENTRE_MAX_M

    a = [((0.450, 0.0, 0.085), 0.0), ((0.452, 0.0, 0.085), 0.0), ((0.454, 0.0, 0.085), 0.0)]
    press_demo, r, watcher, state = _centring_env(
        # beyond the depth press's own reach, inside a re-centre (RECENTRE_MAX_MOVE_M)
        ctx, tmp_path, monkeypatch, tool_xys=[(0.45, CENTRE_MAX_M + 0.03), (0.45, 0.04), (0.45, 0.04)], aims=a[1:]
    )
    got, moves, off = press_demo.centre_for_press(None, ctx, _demo_cfg(), r, watcher, True, a[0], depth_press=True)
    assert moves == 1 and state["n"] == 1 and got == a[1] and len(ctx.client.executed) == 1
    assert ctx.cpose == watcher.to_container_pose(a[1])


def test_without_the_depth_press_staging_recentres_as_before(ctx, tmp_path, monkeypatch):
    first, second = ((0.450, 0.000, 0.085), 0.0), ((0.451, 0.001, 0.085), 0.0)
    press_demo, r, watcher, state = _centring_env(
        ctx, tmp_path, monkeypatch, tool_xys=[(0.45, 0.08), (0.451, 0.0015)], aims=[second]
    )
    got, moves, off = press_demo.centre_for_press(None, ctx, _demo_cfg(), r, watcher, True, first)
    assert (got, moves) == (second, 1) and len(ctx.client.executed) == 1


def test_the_depth_press_plans_no_descent_on_the_way_to_staging(ctx, tmp_path, monkeypatch):
    """It flies its own touch: the approach's lookahead was a cuRobo plan
    (and GPU) for a descent never flown."""
    from conftest import runner as make_runner

    from rammp_box_opening.models.container import ContainerPose

    press_demo = _search_env(monkeypatch)
    cfg = load_press_demo(CFG)
    monkeypatch.setattr(press_demo, "aim_button_at_staging", lambda *a, **k: (FIX, "3/3 frames"))
    fix = _SceneFix((0.45, 0.0, -0.027), 0.0, (0.45, 0.0, 0.085))
    r = make_runner(ctx.client, tmp_path)
    assert press_demo.approach_from_scene(None, ctx, cfg, r, _Watcher(), True, fix).preplanned is not None
    reach = press_demo.approach_from_scene(None, ctx, cfg, r, _Watcher(), True, fix, lookahead=False)
    assert reach.got == FIX and reach.preplanned is None
    watcher = _Watcher()
    watcher.to_container_pose = lambda got: ContainerPose(xyz=(got[0][0], got[0][1], -0.027), yaw=got[1])
    monkeypatch.setattr(press_demo, "tool_xy", lambda ctx: (FIX[0][0], FIX[0][1] + 0.12))  # staged beside it
    _got, at_staging, preplanned, _pf = press_demo.stage_over_search_fix(None, ctx, _demo_cfg(), r, watcher, True, FIX,
                                                                          depth_press=True)
    assert at_staging and preplanned is None


def _descent_leg():
    """A press descent as the planner client builds it: a planned segment
    from staging to the waypoint 60 mm above the target, at rest there,
    then the arm's own straight line (runtime/approach), guarded."""
    from rammp_box_opening.models.container import attitude_quat
    from rammp_box_opening.runtime.approach import plan_with_vertical_approach
    from rammp_box_opening.runtime.guards import GuardSpec
    from rammp_box_opening.runtime.kinematics import ArmChain
    from rammp_box_opening.runtime.legs import Kind, Leg
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    chain = ArmChain()
    q_wp = [0.1208, 0.5375, 3.3522, -1.8057, -0.1478, -0.8136, 0.1407]  # a live planner waypoint
    goal = [0.4264, -0.1151, 0.0815]
    quat = attitude_quat([180.0, 0.0, 0.0], math.atan2(goal[1], goal[0]))
    _R, t_wp = chain.fk(q_wp)
    up, _why = chain.straight_line(q_wp, [t_wp[0], t_wp[1], t_wp[2] + 0.075], quat)
    first = JointTrajectory()
    first.joint_names = ["joint_%d" % i for i in range(1, 8)]
    for i, q in enumerate(list(reversed(up))):  # staging down to the waypoint, 1 s, at rest at both ends
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q]
        pt.velocities = [0.0] * 7
        t = i / (len(up) - 1)
        pt.time_from_start.sec, pt.time_from_start.nanosec = int(t), int(round((t - int(t)) * 1e9))
        first.points.append(pt)

    class Plan:
        success, message, trajectory, planning_time = True, "ok", first, 0.2

    plan = plan_with_vertical_approach(lambda xyz, q, start: Plan(), goal, quat, list(up[-1]), 0.06, chain_model=chain)
    return Leg(name="press:down", kind=Kind.MOTION, traj=plan.trajectory, speed=0.35,
               guard=GuardSpec(touch_nm=3.0, trip="touch", arm_after=0.25), world="interaction_button", chain=0,
               target=("pose", goal, list(quat), 0.06), goal_joints=list(plan.trajectory.points[-1].positions),
               waypoint=plan.waypoint, guard_from=plan.waypoint[0])


def test_the_press_descent_flies_its_free_air_fast_and_its_guard_arms_where_it_did():
    """Bench 2026-09-24: 2.6 s from staging to the touch, all of it at
    contact speed — though the guard of a descent flown on its own cannot
    trip above its straight line (it takes its baseline and arms where the
    line begins, plus a settle). Now the free air above the line flies at
    warp_fast_speed; the line, and where the guard arms on it, are as
    before."""
    from rammp_box_opening.runtime.guards import GROUP_SETTLE_S
    from rammp_box_opening.runtime.runner import _restore_execution_profile
    from rammp_box_opening.runtime.stamps import secs
    from rammp_box_opening.tasks import press_demo

    cfg = _demo_cfg()
    leg = _descent_leg()
    idx0, waypoint0 = leg.waypoint[0], leg.waypoint
    unwarped = leg.traj
    # as the runner flew it unwarped: dilated by the contact speed, armed a
    # settle into the line
    total0 = secs(unwarped.points[-1].time_from_start) / leg.speed
    line0 = total0 - secs(unwarped.points[idx0].time_from_start) / leg.speed
    assert press_demo.fly_free_air_fast(leg, cfg)
    idx = leg.waypoint[0]
    total1 = secs(leg.traj.points[-1].time_from_start)  # baked in: flown at the 1.0 sentinel
    start1 = secs(leg.traj.points[idx].time_from_start)
    assert leg.speed == 1.0 and leg.warp[:2] == (cfg.warp_fast_speed, 0.35)
    assert total1 < total0 - 0.5  # the free air is faster ...
    assert total1 - start1 == pytest.approx(line0, rel=0.02)  # ... the line is not
    assert list(leg.traj.points[idx].positions) == pytest.approx(list(waypoint0[1]))  # the line starts where it did
    assert max(abs(v) for v in leg.traj.points[idx].velocities) < 1e-9  # ... from rest
    # the guard takes its baseline and arms a settle into the line, never above it
    assert leg.guard.rebaseline_after == leg.guard.arm_after
    assert leg.guard.arm_after * total1 == pytest.approx(start1 + GROUP_SETTLE_S, abs=0.01)
    # positions untouched: only samples of the planned, checked path are flown
    planned = {tuple(round(v, 9) for v in p.positions) for p in unwarped.points}
    assert all(tuple(round(v, 9) for v in p.positions) in planned for p in leg.traj.points)
    # a replan re-warps it the same way (runner._restore_execution_profile)
    leg.traj, leg.waypoint = unwarped, waypoint0
    _restore_execution_profile(leg)
    start_r = secs(leg.traj.points[idx0].time_from_start)
    assert leg.guard.arm_after * secs(leg.traj.points[-1].time_from_start) == pytest.approx(start_r + GROUP_SETTLE_S, abs=0.01)
    # warp_fast_speed 0 turns it off, as for grip:down
    off = _descent_leg()
    assert not press_demo.fly_free_air_fast(off, replace(cfg, warp_fast_speed=0.0)) and off.speed == 0.35


def test_the_free_air_is_only_ever_re_timed_to_be_faster():
    """The planner's part of a descent is re-timed on the human profile
    only when that is quicker; a brisk plan is left as it was."""
    from rammp_box_opening.runtime.approach import human_timed_above_waypoint
    from rammp_box_opening.runtime.stamps import secs, set_stamp

    leg = _descent_leg()
    idx = leg.waypoint[0]
    t_line = [secs(p.time_from_start) for p in leg.traj.points[idx:]]
    assert human_timed_above_waypoint(leg)
    idx2 = leg.waypoint[0]
    assert [secs(p.time_from_start) - secs(leg.traj.points[idx2].time_from_start) for p in leg.traj.points[idx2:]] == (
        pytest.approx([t - t_line[0] for t in t_line], abs=1e-6))  # the line itself untouched
    brisk = _descent_leg()
    for i, p in enumerate(brisk.traj.points[: brisk.waypoint[0] + 1]):
        set_stamp(p.time_from_start, 1e-3 * i)  # already faster than any profile
    before = [secs(p.time_from_start) for p in brisk.traj.points]
    assert not human_timed_above_waypoint(brisk)
    assert [secs(p.time_from_start) for p in brisk.traj.points] == before


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
    got, at_staging, preplanned, planned_for = press_demo.stage_over_search_fix(None, ctx, _demo_cfg(), r, watcher, True, FIX)
    assert got == FIX and at_staging
    # the descent was planned while the arm flew to staging, for the pose it flew to
    assert [leg.name for leg in preplanned] == ["press:down"]
    assert planned_for.xyz[:2] == pytest.approx(FIX[0][:2])



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


def test_the_run_starts_by_going_home_unless_already_there(ctx, tmp_path, capsys):
    """Owner, 2026-09-25: the first thing a run does is send the arm to its
    exact rest pose, if it is not already there — one joint goal to HOME,
    planned in the bench world like home_arm's."""
    from conftest import runner as make_runner

    from rammp_box_opening.constants import HOME
    from rammp_box_opening.tasks import press_demo

    ctx.client.live = list(HOME)
    ctx.client.live[2] = HOME[2] - 2 * math.pi + 0.004  # home as the driver reports it (-3.1412): already there
    assert press_demo.home_first(ctx, make_runner(ctx.client, tmp_path), True, list(HOME))
    assert ctx.client.executed == [] and ctx.client.joint_targets == []

    ctx.client.live = [v + 0.3 for v in HOME]  # anywhere else
    assert press_demo.home_first(ctx, make_runner(ctx.client, tmp_path), True, list(HOME))
    assert len(ctx.client.executed) == 1
    assert ctx.client.joint_targets[-1] == pytest.approx(list(HOME), abs=1e-9)
    assert "HOME FIRST" in capsys.readouterr().out


def test_a_home_the_bench_world_refuses_moves_nothing_and_names_the_recovery(ctx, tmp_path, capsys):
    """An arm down at the table is home_arm's case (bare-world and
    lift-first fallbacks, the operator's eye on the bench): the run does not
    start, and says so."""
    from conftest import runner as make_runner

    from rammp_box_opening.constants import HOME
    from rammp_box_opening.tasks import press_demo

    class Refused:
        success = False
        message = "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION"
        trajectory = None

    ctx.client.live = [v + 0.3 for v in HOME]
    ctx.client.plans = [Refused()]
    assert not press_demo.home_first(ctx, make_runner(ctx.client, tmp_path), True, list(HOME))
    assert ctx.client.executed == []
    assert "home_arm --execute" in capsys.readouterr().out


def test_detect_only_reports_the_scene_cameras_fixes_with_nothing_moving(monkeypatch, capsys):
    """--detect-only is the scene camera's answer, repeated: the wrist does
    not find the box (owner, 2026-10-22), and nothing is flown to ask."""
    from types import SimpleNamespace

    from rammp_box_opening.tasks import press_demo

    clock = [0.0]
    monkeypatch.setattr(press_demo.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(press_demo.time, "sleep", lambda s: None)

    class Locator:
        last_why = None
        calls = 0

        def locate(self, timeout_s):
            clock[0] += 2.0  # a locate takes OWL ticks
            self.calls += 1
            if self.calls == 2:
                self.last_why = "the scene OWL saw no box in 5.0 s"
                return None
            top = (0.42, 0.22 + 0.001 * self.calls, 0.21)
            return SimpleNamespace(pose=SimpleNamespace(xyz=(0.42, 0.22, 0.09), yaw=0.1), top_xyz=top,
                                   n_points=120, score=0.44, table_z_scene=0.089, table_extent=(0.3, 0.9, -0.2, 0.6))

    assert press_demo.scene_detect_report(Locator(), duration_s=15.0) == 7
    out = capsys.readouterr().out
    assert "nothing moves" in out and "no fix: the scene OWL saw no box" in out
    assert "7 fix(es), spread [0, 7, 0] mm" in out

    class Blind:
        last_why = "scene camera streams missing: depth"

        def locate(self, timeout_s):
            clock[0] += 5.0
            return None

    assert press_demo.scene_detect_report(Blind(), duration_s=15.0) == 0
    out = capsys.readouterr().out
    assert out.count("no fix: scene camera streams missing") == 1  # said once, not every try
    assert "NO BOX — scene camera streams missing: depth" in out


def test_the_lid_is_set_down_on_the_table_the_camera_measured():
    """Flatbot: no surveyed bench, so no fixed set-down zone either — the
    lid goes beside the box on the table measured this run, whole."""
    from types import SimpleNamespace

    from rammp_box_opening.models.container import ContainerModel
    from rammp_box_opening.tasks import press_demo

    m = ContainerModel.load(CFG)
    measured = SimpleNamespace(table_extent=(0.45, 1.10, -0.90, -0.10))
    x0, x1, y0, y1 = press_demo.drop_zone(m, measured)
    inset = 0.5 * max(m.lid_dims[:2]) + press_demo.DROP_EDGE_M
    assert (x0, x1, y0, y1) == pytest.approx((0.45 + inset, 1.10 - inset, -0.90 + inset, -0.10 - inset))
    box = ContainerPose(xyz=(0.70, -0.55, 0.30), yaw=0.0)
    spots = press_demo.lid_drop_candidates(m, box, 0.30, zone=(x0, x1, y0, y1))
    assert spots and all(x0 <= s[0] <= x1 and y0 <= s[1] <= y1 for s, _d in spots)
    assert [d for _s, d in spots] != [d for _s, d in press_demo.lid_drop_candidates(m, box, 0.30)]  # not the bench's zone
    assert press_demo.drop_zone(m, SimpleNamespace(table_extent=None)) == (*press_demo.DROP_X_M, *press_demo.DROP_Y_M)
    # too few table points for an extent: the ring square assumed around the
    # pink canister still leaves the lid somewhere to go
    import numpy as np

    from rammp_box_opening.perception.scene_source import seen_table_extent

    pink = ContainerModel.load("src/rammp_box_opening/config/containers/ankou_pink.yaml")
    ring, assumed = seen_table_extent(np.zeros((0, 3)), 0.30, box.xyz[:2])
    assert assumed and press_demo.lid_drop_candidates(pink, box, 0.30, zone=press_demo.drop_zone(pink, SimpleNamespace(table_extent=ring)))


def test_the_staging_aim_builds_its_aimer_the_straight_down_way_when_no_reach_out_priors_are_given(monkeypatch):
    """Review 2026-10-22: aim_button_at_staging always passed lid_z/prior_xy,
    and a ButtonAimer without them raised TypeError at every straight-down
    aim — with the arm already at staging. Unset, they are not passed."""
    from types import SimpleNamespace

    from rammp_box_opening.perception import button_aim
    from rammp_box_opening.tasks import press_demo

    made = []

    class Aimer:
        last_why = None

        def __init__(self, grab, model, table_z, **kw):
            made.append(kw)

        def status(self):
            return "still frames"

    monkeypatch.setattr(button_aim, "ButtonAimer", Aimer)
    w = SimpleNamespace(grab=SimpleNamespace(color=None, depth=None, k=None, color_stamp=None), model=None, table_z=0.1)
    got, status = press_demo.aim_button_at_staging(None, w, 0.0, timeout_s=0)
    assert got is None and "still frames" in status and made == [{}]
    press_demo.aim_button_at_staging(None, w, 0.0, timeout_s=0, lid_z=0.2, prior_xy=(0.8, 0.0))
    assert made[-1] == {"lid_z": 0.2, "prior_xy": (0.8, 0.0)}


def test_staging_reach_is_judged_by_the_arms_own_kinematics(ctx, monkeypatch):
    from rammp_box_opening.runtime import approach
    from rammp_box_opening.tasks import press_demo

    cfg = load_press_demo(CFG)
    asked = []

    class Chain:
        def __init__(self, walk=None, seeded=None):
            self.walk, self.seeded = walk, seeded

        def nearest_solution(self, q, xyz, quat, seeds=()):
            asked.append(list(xyz))
            return self.walk

        def solve_pose(self, seed, xyz, R):
            return self.seeded

        def equivalents(self, q_start, q):
            return [q]

    monkeypatch.setattr(approach, "_chain", lambda: Chain())
    assert press_demo.staging_out_of_reach(ctx, cfg)
    button = press_demo.from_container(ctx.cpose, ctx.model.button_offset)
    assert asked[-1] == pytest.approx([button[0], button[1], button[2] + cfg.staging_m])  # build_approach_leg's staging
    monkeypatch.setattr(approach, "_chain", lambda: Chain(walk=[0.0] * 7))
    assert not press_demo.staging_out_of_reach(ctx, cfg)
    # the walk misses it, a seed on another branch finds it: not out of reach
    monkeypatch.setattr(approach, "_chain", lambda: Chain(seeded=[0.0] * 7))
    assert not press_demo.staging_out_of_reach(ctx, cfg)

    def broken():
        raise KeyError("arm_mount")  # the bench URDF under flatbot's root

    monkeypatch.setattr(approach, "_chain", broken)
    assert not press_demo.staging_out_of_reach(ctx, cfg)  # a kinematics problem proves nothing


def test_flatbots_own_kinematics_separate_a_box_straight_down_reaches_from_one_it_cannot(ctx, monkeypatch):
    """On flatbot's URDF (arm_mount, 30 deg mount), with the pink canister's
    straight-down attitude: a box 0.45 m out on a low table solves; the
    2026-10-22 box 0.89 m out does not, from any seed."""
    from rammp_box_opening.models.container import ContainerModel
    from rammp_box_opening.runtime import approach
    from rammp_box_opening.runtime.kinematics import ArmChain
    from rammp_box_opening.tasks import press_demo

    pink = "src/rammp_box_opening/config/containers/ankou_pink.yaml"
    chain = ArmChain(path="src/rammp_box_opening/test/data/arm_flatbot_20261021.urdf", root="arm_mount")
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    ctx.model = ContainerModel.load(pink)
    cfg = load_press_demo(pink)
    ctx.client.live = [0.067, -1.897, -3.137, -2.269, -0.211, -0.683, 1.725]  # flatbot's HOME
    oz = ctx.model.button_offset[2]
    ctx.cpose = ContainerPose(xyz=(0.45, 0.0, 0.25 - oz), yaw=0.0)
    assert not press_demo.staging_out_of_reach(ctx, cfg)
    ctx.cpose = ContainerPose(xyz=(0.675, -0.578, 0.40 - oz), yaw=0.0)
    assert press_demo.staging_out_of_reach(ctx, cfg)


def test_an_in_flight_touch_the_wrist_never_confirmed_is_a_stop(ctx, tmp_path, monkeypatch):
    from conftest import runner as make_runner

    press_demo = _search_env(monkeypatch)
    cfg = replace(load_press_demo(CFG), confirm_in_flight=True)
    ctx.client.exec_script = [("touch", {"message": "torque guard trip", "progress": 0.9})]
    r = make_runner(ctx.client, tmp_path)
    fix = _SceneFix((0.45, 0.0, -0.027), 0.0, (0.45, 0.0, 0.085))
    reach = press_demo.approach_from_scene(None, ctx, cfg, r, _Watcher(), True, fix)
    assert reach.failed and reach.pressed is not None and reach.got is None and not reach.reach_out
    assert "never confirmed" in reach.status
