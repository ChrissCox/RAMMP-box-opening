"""The reach-out press (tasks/reach_press): flatbot's box 0.80-0.89 m out on
a table 0.30 m above the arm's mount, where no straight-down pose reaches."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from rammp_box_opening.constants import TCP_OFFSET_M, TIP_TO_TOOL_M
from rammp_box_opening.models.container import ContainerModel, load_press_demo
from rammp_box_opening.perception.d405 import quat_to_mat
from rammp_box_opening.tasks import reach_press

PINK = "src/rammp_box_opening/config/containers/ankou_pink.yaml"
BOX = (0.675, -0.578)  # where the scene camera put it, 2026-10-22


@pytest.fixture(scope="module")
def model():
    return ContainerModel.load(PINK)


def _axes(quat):
    R = np.asarray(quat_to_mat(*quat), float)
    return R[:, 0], R[:, 1], R[:, 2]


def test_the_reach_attitude_leans_back_with_the_camera_above_the_hand(model):
    u = np.array([BOX[0], BOX[1], 0.0]) / math.hypot(*BOX)
    for tilt in reach_press.REACH_TILTS_DEG:
        x, y, z = _axes(reach_press.reach_quat(model, BOX, tilt))
        assert math.degrees(math.acos(float(np.clip(-z[2], -1, 1)))) == pytest.approx(tilt, abs=1e-6)  # off straight down by tilt
        assert z @ u > 0  # the fingertips lead away from the arm: the hand leans back toward it
        assert y[2] > 0 and abs(y @ np.cross([0, 0, 1], u)) < 1e-9  # the wrist camera's side is up, in the bearing's plane
        assert abs(x[2]) < 1e-9 and abs(x @ u) < 1e-9  # the fingers side by side across the line to the box
    assert reach_press.REACH_TILTS_DEG[-1] <= 45  # past that the knuckles, then the palm, meet the flush lid


def test_the_fingertip_lands_where_it_is_aimed_and_is_read_back_from_the_tips(model):
    for tilt in (0, 30, 45):
        q = reach_press.reach_quat(model, BOX, tilt)
        R = np.asarray(quat_to_mat(*q), float)
        aim = np.array([0.66, -0.57, 0.40])
        tool = np.asarray(reach_press.tool_for_contact(aim, q))
        assert tool + R @ np.asarray(reach_press.CONTACT_TOOL) == pytest.approx(aim, abs=1e-12)
        tips = tool + R @ np.asarray(reach_press.TIP_MID_TOOL)  # what client.contact_xyz reads
        assert reach_press.contact_from_tips(tips, q) == pytest.approx(list(aim), abs=1e-12)
    # straight down, the same numbers the straight-down press uses
    q0 = reach_press.reach_quat(model, BOX, 0)
    tool = np.asarray(reach_press.tool_for_contact((0.5, -0.4, 0.4), q0))
    assert tool[2] - 0.4 == pytest.approx(TCP_OFFSET_M, abs=1e-9)
    tips = tool + np.asarray(quat_to_mat(*q0), float) @ np.asarray(reach_press.TIP_MID_TOOL)
    assert tips[2] - tool[2] == pytest.approx(TIP_TO_TOOL_M, abs=1e-9)


STRAIGHT_DOWN = np.diag([1.0, -1.0, -1.0])  # tool_frame's z along the world's -z


class _Chain:
    """Straight lines that always fly, through two waypoints; no joint
    limits (ClearMarginFree finds none near). fk stands still, the hand at
    R: np.eye(3) unless a test says otherwise — a lean of 180 deg, whose
    sine is ~0, so a test that means straight down sets STRAIGHT_DOWN."""

    limits = [(None, None)] * 7
    R = np.eye(3)

    def __init__(self, refuse=()):
        self.refuse = refuse
        self.lines = []

    def straight_line(self, q0, xyz, quat):
        self.lines.append((list(xyz), list(quat)))
        for what in self.refuse:
            if what(xyz, quat):
                return None, "IK did not converge"
        return [list(q0), [v + 0.01 for v in q0]], None

    def walk(self, q0, xyz, quat, step_m=None, step_rad=None, max_step_rad=None):
        return None, "no turning walk modelled"  # the depth press's slide in, then its turn

    def fk(self, q):
        return np.array(self.R, float), np.array([0.5, -0.4, 0.45])


def _tilt_of(quat):
    _x, _y, z = _axes(quat)
    return round(math.degrees(math.acos(float(np.clip(-z[2], -1, 1)))))


def test_the_reach_plan_takes_the_smallest_tilt_that_plans(ctx, monkeypatch):
    from rammp_box_opening.runtime import approach

    cfg = load_press_demo(PINK)
    ctx.model = ContainerModel.load(PINK)
    chain = _Chain()
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    asked = []

    def look_plans_from_40(ctx_, state, name, target, world, speed):
        asked.append(_tilt_of(target[2]))
        if _tilt_of(target[2]) < 40:
            raise RuntimeError("planning failed for reach:look: MotionGenStatus.IK_FAIL")
        return SimpleNamespace(goal_joints=[0.0] * 7), state

    monkeypatch.setattr(reach_press, "_plan_motion", look_plans_from_40)
    button = (0.66, -0.57, 0.40)
    plan, why = reach_press.plan_reach(ctx, cfg, button, [0.0] * 7)
    assert plan is not None and plan.tilt_deg == 40 and asked == [30, 35, 40] and why is None
    d = math.hypot(button[0], button[1])
    back = np.subtract(plan.start_xyz[:2], plan.look_xyz[:2])
    assert np.linalg.norm(back) == pytest.approx(reach_press.REACH_PULLBACK_M)  # pulled back toward the arm ...
    assert back @ (np.asarray(button[:2]) / d) == pytest.approx(reach_press.REACH_PULLBACK_M)
    # ... and up from the stroke's start: the wrist camera ~0.30 m off the lid, not ~0.245
    assert plan.look_xyz[2] == pytest.approx(plan.start_xyz[2] + reach_press.REACH_LOOK_RISE_M)
    # the start and the bottom are straight above/below each other: the stroke is vertical
    assert plan.start_xyz[:2] == pytest.approx(plan.bottom_xyz[:2], abs=1e-12)
    assert plan.start_xyz[2] - plan.bottom_xyz[2] == pytest.approx(reach_press.REACH_START_M + cfg.travel_m)
    # nothing plans: every tilt is named in the refusal
    monkeypatch.setattr(reach_press, "_plan_motion", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x: IK_FAIL")))
    plan, why = reach_press.plan_reach(ctx, cfg, button, [0.0] * 7)
    assert plan is None and all("%d deg" % t in why for t in reach_press.REACH_TILTS_DEG)


def test_a_tilt_whose_stroke_the_arm_cannot_fly_straight_is_passed_over(ctx, monkeypatch):
    from rammp_box_opening.runtime import approach

    cfg = load_press_demo(PINK)
    ctx.model = ContainerModel.load(PINK)
    chain = _Chain(refuse=[lambda xyz, quat: _tilt_of(quat) == 30 and xyz[2] < 0.41])  # the 30 deg stroke down
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    monkeypatch.setattr(reach_press, "_plan_motion", lambda c, s, n, t, w, sp: (SimpleNamespace(goal_joints=[0.0] * 7), s))
    plan, why = reach_press.plan_reach(ctx, cfg, (0.66, -0.57, 0.40), [0.0] * 7)
    assert plan is not None and plan.tilt_deg == 35


DOWN_QUAT = [1.0, 0.0, 0.0, 0.0]  # STRAIGHT_DOWN as a quaternion (x, y, z, w)


def _way_down(asked, solves=lambda tilt, tip: False):
    """reach_press.way_down for the fake chain: each ask kept as (tilt, the
    xy under the slid-in fingertip — the button's —, top_z); `solves(tilt,
    fingertip)` says where the hand turns down and strokes: the turn in
    place where the slide ends, then the press stroke from its end, each a
    step on from the last as _Chain's lines are, the spring 440 N/m. A
    refusal reads as 16:23's did."""
    from rammp_box_opening.tasks import depth_press as dp

    def way_down(ctx_, cfg_, q_look, slide, start, quat, top_z):
        tip = np.asarray(start, float) + np.asarray(quat_to_mat(*quat), float) @ np.asarray(reach_press.CONTACT_TOOL)
        asked.append((_tilt_of(quat), [float(tip[0]), float(tip[1])], float(top_z)))
        if solves(_tilt_of(quat), tip):
            turn = [[float(v) for v in slide[-1]], [float(v) + 0.01 for v in slide[-1]]]
            down = dp.WayDown(turn, list(DOWN_QUAT), 0.0, False)
            t0 = np.array([0.5, -0.4, 0.45])  # the fake fk's
            stroke = dp.StrokePlan([("depth:press", [turn[-1], [v + 0.01 for v in turn[-1]]], dp.TOUCH_SPEED)], 440.0,
                                   0.03 - cfg_.button_travel_m, float(top_z) - 0.03, t0,
                                   t0 + STRAIGHT_DOWN @ np.asarray(dp.DOWN_CONTACT_TOOL), turn[-1], list(DOWN_QUAT))
            return down, stroke, None
        return None, None, ("walk in: 0 deg short: a joint jump of 0.19 rad 44% of the way — near a singularity; "
                            "turn after the slide: 0 deg short: IK did not converge 73% of the way")

    return way_down


def test_a_compliant_depth_plan_passes_over_a_tilt_whose_way_down_refuses(ctx, monkeypatch):
    """The compliant depth press never flies the tilted stroke: from the
    look pose the hand turns down and strokes straight down (way_down).
    plan_reach(depth=True) asks that instead, at the scene button's top,
    passes over a tilt where it does not solve, and keeps that tilt's look
    leg and slide (`looks`). Without depth, the tilted stroke as before."""
    from rammp_box_opening.runtime import approach

    cfg = load_press_demo(PINK)
    ctx.model = ContainerModel.load(PINK)
    chain = _Chain(refuse=[lambda xyz, quat: xyz[2] < 0.41])  # every tilt's stroke down (the fake fk stands at 0.45)
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    monkeypatch.setattr(reach_press, "_plan_motion",
                        lambda c, s, n, t, w, sp: (SimpleNamespace(goal_joints=[0.0] * 7), s))
    asked = []
    monkeypatch.setattr(reach_press, "way_down", _way_down(asked, solves=lambda tilt, tip: tilt > 30))
    looks = []
    plan, why = reach_press.plan_reach(ctx, cfg, BUTTON, [0.0] * 7, depth=True, looks=looks)
    assert plan is not None and why is None and plan.tilt_deg == 35 and plan.stroke_pts is None
    assert plan.down.walked is False and plan.stroke.end_z == pytest.approx(BUTTON[2] - 0.03)
    assert [(t, z) for t, _xy, z in asked] == [(30, BUTTON[2]), (35, BUTTON[2])]
    assert all(xy == pytest.approx(list(BUTTON[:2])) for _t, xy, _z in asked)  # the slide ends over the button
    assert not any(xyz[2] < 0.41 for xyz, _q in chain.lines)  # no tilted stroke asked for
    (look,) = looks
    assert look.tilt_deg == 30 and look.down is None and look.look_leg is not None and look.slide_pts
    plan, why = reach_press.plan_reach(ctx, cfg, BUTTON, [0.0] * 7)
    assert plan is None and all("%d deg: the stroke (" % t in why for t in reach_press.REACH_TILTS_DEG)


def test_where_no_tilt_turns_down_and_strokes_the_depth_plan_names_every_tilt_and_keeps_their_looks(ctx, monkeypatch):
    """A tilt whose look pose does not plan has no look to keep."""
    from rammp_box_opening.runtime import approach

    cfg = load_press_demo(PINK)
    ctx.model = ContainerModel.load(PINK)
    monkeypatch.setattr(approach, "_chain", lambda: _Chain())

    def look_plans_from_35(ctx_, state, name, target, world, speed):
        if _tilt_of(target[2]) < 35:
            raise RuntimeError("planning failed for reach:look: MotionGenStatus.IK_FAIL")
        return SimpleNamespace(goal_joints=[0.0] * 7), state

    monkeypatch.setattr(reach_press, "_plan_motion", look_plans_from_35)
    monkeypatch.setattr(reach_press, "way_down", _way_down([]))
    looks = []
    plan, why = reach_press.plan_reach(ctx, cfg, BUTTON, [0.0] * 7, depth=True, looks=looks)
    assert plan is None and "30 deg: the look pose (MotionGenStatus.IK_FAIL)" in why
    for tilt in (35, 40, 45):
        assert ("%d deg: the way down (walk in: 0 deg short: a joint jump of 0.19 rad 44%% of the way — near a "
                "singularity; turn after the slide: 0 deg short: IK did not converge 73%% of the way)" % tilt) in why
    assert [lk.tilt_deg for lk in looks] == [35, 40, 45]
    assert all(lk.down is None and lk.stroke is None and lk.look_leg is not None and lk.slide_pts for lk in looks)


class _Gravity:
    """No weight anywhere; a pull at the fingertip reads like flatbot's."""

    K = [0.48, 0.0, 0.2, 0.0]

    def wrist_holding(self, q):
        return [0.0] * 4

    def tip_load(self, q, point_tool):
        return list(self.K)


def test_the_reach_touch_takes_every_trip_as_a_surface_and_leaves_the_height_to_the_verdict(ctx, monkeypatch):
    from rammp_box_opening.runtime import approach

    cfg = load_press_demo(PINK)
    monkeypatch.setattr(approach, "_chain", lambda: _Chain())
    monkeypatch.setattr(approach, "_gravity", lambda: _Gravity())
    q0 = [0.0, 0.3, 0.0, 1.9, 0.0, 0.9, 0.0]
    pts = [[a + 0.002 * k for a in q0] for k in range(30)]
    quat = reach_press.reach_quat(ContainerModel.load(PINK), BOX, 30)
    ref = ([0.0] * 4, q0)
    leg = reach_press.touch_leg(ctx, cfg, pts, quat, ("interaction", "w.yaml"), ref)
    assert leg.guard.trip == "touch" and leg.guard.touch_nm == cfg.contact_nm
    assert leg.guard.reference == ref and leg.guard.gravity is not None and leg.plan_free
    assert leg.guard.arm_after is None
    v = lambda outcome, progress: SimpleNamespace(outcome=outcome, progress=progress, torque_peak=3.2)  # noqa: E731
    ok, _why = leg.verify(v("touch", 0.7))
    assert ok
    ok, _why = leg.verify(v("touch", 0.05))  # high above the button: touch_verdict calls it foreign
    assert ok
    ok, why = leg.verify(v("arrived", 1.0))
    assert not ok and "no surface" in why
    # 40 mm above the aimed top is not this button: the run backs out instead of pressing
    from rammp_box_opening.tasks.press_demo import touch_verdict

    assert touch_verdict(0.040) == "foreign" and touch_verdict(0.012) == "popped" and touch_verdict(0.001) == "press"


def test_the_way_out_is_judged_on_what_gravity_does_not_explain(monkeypatch):
    from rammp_box_opening.runtime import approach

    monkeypatch.setattr(approach, "_gravity", lambda: _Gravity())
    ref = ([0.0] * 4, [0.0] * 7)
    g = reach_press.drag_guard(ref, _Gravity.K)
    assert g.trip == "obstruction" and g.touch_nm <= 1.5 and g.gravity is not None and g.reference == ref
    assert g.hang == (_Gravity.K, reach_press.HANG_N) and reach_press.HANG_N < 0.3 * 9.81  # under a 0.3 kg canister
    assert g.arm_after is None  # armed from the first sample: a catch is felt from the first millimetre
    u = reach_press.unload_guard(ContainerModel.load(PINK), ref, _Gravity.K)
    # it starts ON the button: only the push's backstop bounds the residual, the pull check does the rest
    assert u.touch_nm == ContainerModel.load(PINK).touch_nm and u.hang[1] == reach_press.HANG_N


def test_a_pull_at_the_fingertip_is_read_off_the_residual():
    k = _Gravity.K
    grav = lambda q: [2.0 * q[0], 0.0, 0.0, 0.0]  # noqa: E731
    ref = ([1.0, 0.0, 0.0, 0.0], [0.0] * 7)
    now = ([1.0 + 2.0 * 0.3 + 3.0 * k[0], 0.0, 3.0 * k[2], 0.0], [0.3] + [0.0] * 6)
    r = reach_press.residual(grav, ref, now)
    assert r == pytest.approx([3.0 * k[0], 0.0, 3.0 * k[2], 0.0])  # gravity's 0.6 Nm taken out
    assert reach_press.down_force(k, r) == pytest.approx(3.0)
    assert reach_press.down_force(k, [-v for v in r]) == pytest.approx(-3.0)  # a surface pushing up


def test_the_gravity_model_is_checked_against_the_arm_before_any_guarded_line():
    grav = lambda q: [3.0 * q[0], -1.0 * q[0], 0.0, 0.0]  # noqa: E731
    a = ([0.0, 0.0, 0.0, 0.0], [0.0] * 7)
    good = ([0.9 + 0.2, -0.3, 0.1, 0.0], [0.3] + [0.0] * 6)
    assert reach_press.gravity_mismatch(grav, a, good) is None
    flipped = ([-0.9, 0.3, 0.0, 0.0], [0.3] + [0.0] * 6)  # the driver's sign the other way round
    why = reach_press.gravity_mismatch(grav, a, flipped)
    assert why is not None and "joint_4" in why
    payload = ([0.9 + 1.5, -0.3, 0.0, 0.0], [0.3] + [0.0] * 6)  # something in the hand
    assert reach_press.gravity_mismatch(grav, a, payload) is not None


def _drift_of(monkeypatch, f, slide, stroke):
    calls = []

    def g(q):
        calls.append(tuple(q))
        return [f(q), 0.0, 0.0, 0.0]

    monkeypatch.setattr(reach_press, "_wrist_gravity", lambda: g)
    return reach_press._predicted_drift(SimpleNamespace(slide_pts=slide, stroke_pts=stroke)), calls


def _line(n, tag=0.0):
    return [[i / max(n - 1, 1), tag] + [0.0] * 5 for i in range(n)]


def test_the_predicted_drift_line_samples_the_gravity_model_and_never_misses_the_last_point(monkeypatch):
    # a display line computed with the arm at rest before the look leg: not
    # every point (232 gravity evaluations, ~0.2 s), the start once per line
    slide, stroke = _line(77), _line(39, tag=1.0)
    (s, t), calls = _drift_of(monkeypatch, lambda q: 3.0 * q[0], slide, stroke)
    every = reach_press.DRIFT_SAMPLE_EVERY
    for pts, tag in ((slide, 0.0), (stroke, 1.0)):
        mine = [q for q in calls if q[1] == tag]
        assert len(mine) <= 2 + math.ceil(len(pts) / every) < len(pts)
        assert mine.count(tuple(pts[0])) == 1 and len(set(mine)) == len(mine)  # the start once, nothing twice
        assert tuple(pts[-1]) in mine
    # rising along the path: the peak is at the last point, 77 and 39 not multiples of the stride
    assert s == pytest.approx(3.0) and t == pytest.approx(3.0)


def test_the_predicted_drift_on_a_smooth_path_is_the_per_point_peak(monkeypatch):
    f = lambda q: 2.0 * math.sin(math.pi * q[0])  # noqa: E731 - peaks mid-path
    slide, stroke = _line(77), _line(39)
    (s, t), _calls = _drift_of(monkeypatch, f, slide, stroke)
    full = [max(abs(f(q) - f(pts[0])) for q in pts) for pts in (slide, stroke)]
    assert s == pytest.approx(full[0], rel=0.02) and t == pytest.approx(full[1], rel=0.02)


def test_the_predicted_drift_of_a_one_point_or_empty_line_is_zero(monkeypatch):
    (s, t), _calls = _drift_of(monkeypatch, lambda q: 5.0 + q[0], _line(1), [])
    assert s == 0.0 and t == 0.0


def test_the_box_is_put_where_its_aimed_button_is(model):
    from rammp_box_opening.models.container import from_container

    for offset in ((0.0, 0.0, 0.121), (0.02, -0.01, 0.1)):
        for yaw in (0.0, 0.7, -2.0):
            pose = reach_press.cpose_for_button((0.66, -0.57, 0.40), yaw, offset)
            assert from_container(pose, offset) == pytest.approx([0.66, -0.57, 0.40], abs=1e-12)
            assert pose.yaw == yaw


def test_only_the_measured_attitude_is_pressed_reaching_out(model):
    assert reach_press.unsupported(model) is None
    other = SimpleNamespace(press_attitude_rpy_deg=(180.0, 0.0, 0.0))
    assert "measured for" in reach_press.unsupported(other)


def test_a_press_that_moved_the_box_is_measured_by_two_aims():
    assert reach_press.judge_box_moved((0.66, -0.57), (0.66, -0.57)) == 0.0
    assert reach_press.judge_box_moved((0.66, -0.57), (0.663, -0.574)) == pytest.approx(0.005)
    assert reach_press.judge_box_moved((0.66, -0.57, 0.4), (0.67, -0.57, 0.1)) == pytest.approx(0.01)  # heights ignored


def test_the_push_never_goes_deeper_than_the_gripper_clears_the_lid():
    cfg = load_press_demo(PINK)
    for tilt in reach_press.REACH_TILTS_DEG:
        depth = reach_press.reach_push_m(cfg, tilt)
        assert depth <= cfg.button_travel_m
        assert depth <= reach_press.REACH_LID_CLEAR_M[tilt] - reach_press.PUSH_MARGIN_M + 1e-12
        assert depth >= 0.004  # still past the button's own ~4 mm travel
    assert reach_press.reach_push_m(cfg, 45) < reach_press.reach_push_m(cfg, 30)


# ---- the whole reach-out press, the arm faked -------------------------------

BUTTON = (0.66, -0.57, 0.40)
POPPED = (BUTTON[0], BUTTON[1], BUTTON[2] + 0.012)  # the aim's press point on a popped knob: its top
Q = [0.0] * 7
# the rest readings, in the order the flow takes them; distinct so each
# guard's reference can be traced to the reading it must be
AT_REST = ([0.0, 0.0, 0.0, 0.0], Q)
AT_LOOK = ([0.01, 0.0, 0.0, 0.0], Q)
AT_LOOK2 = ([0.02, 0.0, 0.0, 0.0], Q)
AT_START = ([0.03, 0.0, 0.0, 0.0], Q)
FREE = [AT_REST, AT_LOOK, AT_LOOK2, AT_START, AT_START, AT_START]  # ... then 15 mm up and at the stroke's start


class _Rec:
    """What happened, in order: every leg the runner was handed, every aim."""

    def __init__(self):
        self.events = []
        self.legs = {}  # name -> the last leg of that name
        self.owl = []  # the scene OWL's ("on"/"off", events before it)
        self.reads = 0  # at-rest reads (client.settled_efforts)
        self.runs = []  # per runner.run: (its legs' names, the gains each execution it took was sent)


def _flow(ctx, monkeypatch, tmp_path, after=(POPPED, 12.0), first=(BUTTON, 0.4), scene_after="same", settled=None,
          touch=None, contact=BUTTON, home_refusals=0, run_args=None, chain=None, home_turns=0):
    """Run run_reach_press with the arm, the cameras and the planner faked
    (`run_args`: press_demo's args, a guarded TF press by default; `chain`,
    a _Chain). Every plan starts at the joints it is handed (PlanState), as
    the planner's do; each home plan's joint_3 is written `home_turns`
    whole turns off them, as a plan handed planner_branch(start) can be.
    Returns (exit code, the record, the chain, home plan's cpose and pad —
    and its start, the record's length and the live joints when it was
    planned)."""
    from conftest import leg as make_leg

    from rammp_box_opening.models.container import ContainerPose
    from rammp_box_opening.runtime import approach
    from rammp_box_opening.runtime.runner import Runner

    rec = _Rec()
    cfg = load_press_demo(PINK)
    ctx.model = ContainerModel.load(PINK)
    ctx.press_cfg = cfg
    chain = chain or _Chain()
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    monkeypatch.setattr(approach, "_gravity", lambda: _Gravity())
    monkeypatch.setattr(reach_press, "oblique_aim_available", lambda: True)
    home = {}

    refusals = {"home": home_refusals}

    def plan(ctx_, state, name, target, world, speed):
        if name == "home" and refusals["home"] > 0:
            refusals["home"] -= 1
            raise RuntimeError("planning failed for home: MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION: the start")
        start = list(state.joints)
        end = [0.0] * 7 if name == "home" else [0.05] * 7
        if name == "home":
            home.update(cpose=ctx_.cpose, pad=ctx_.contact_pad, start=list(start), at=len(rec.events),
                        live=list(ctx_.client.joints()))
            start[2] += 2.0 * math.pi * home_turns
            end = list(end)
            end[2] += 2.0 * math.pi * home_turns
        else:
            home.setdefault("look_cposes", []).append(ctx_.cpose)
        lg = make_leg(name, start=start, end=end, speed=speed, world=world[0])
        lg.target = target
        return lg, state

    monkeypatch.setattr(reach_press, "_plan_motion", plan)

    class R(Runner):
        def run(self, legs, execute, lookahead=None):
            for lg in legs:
                rec.events.append(lg.name)
                rec.legs[lg.name] = lg
            sent = len(getattr(self.client, "gains_sent", []))
            out = super().run(legs, execute, lookahead)
            rec.runs.append(([lg.name for lg in legs], getattr(self.client, "gains_sent", [])[sent:]))
            return out

    runner = R(ctx.client, log_dir=tmp_path)
    aims = [first, after]

    def aim(node, watcher, yaw, lid_z, prior_xy, runner_, tag):
        rec.events.append("aim:" + tag)
        got = aims.pop(0)
        return (None, "no circle", None) if got is None else ((got[0], 0.0), "3/3 frames", got[1])

    monkeypatch.setattr(reach_press, "_aim", aim)
    seq = list(FREE if settled is None else settled)
    def settled(*a, **k):
        rec.reads += 1
        return seq.pop(0) if seq else AT_START

    ctx.client.settled_efforts = settled
    # the touch trips with the fingertip exactly on `contact`
    quat = reach_press.reach_quat(ctx.model, BUTTON[:2], reach_press.REACH_TILTS_DEG[0])
    R_ = np.asarray(quat_to_mat(*quat), float)
    tips = np.asarray(contact) - R_ @ (np.asarray(reach_press.CONTACT_TOOL) - np.asarray(reach_press.TIP_MID_TOOL))
    ctx.client.contact_at = [float(v) for v in tips]
    ctx.client.exec_script = [
        ("arrived", {"message": "ok", "progress": 1.0}),  # look
        ("arrived", {"message": "ok", "progress": 1.0}),  # look, for the aimed button
        ("arrived", {"message": "ok", "progress": 1.0}),  # slide in
        touch or ("touch", {"message": "torque guard trip", "progress": 0.8, "torque_peak": 3.1}),
    ]
    fix = SimpleNamespace(pose=ContainerPose(xyz=(BUTTON[0], BUTTON[1], BUTTON[2] - 0.121), yaw=0.0), top_xyz=BUTTON)
    moved_pose = ContainerPose(xyz=(BUTTON[0] + 0.05, BUTTON[1], BUTTON[2] - 0.121), yaw=0.0)
    seen = {"same": fix, "none": None, "moved": SimpleNamespace(pose=moved_pose, top_xyz=(BUTTON[0] + 0.05, BUTTON[1], BUTTON[2]))}
    # the scene OWL's enables/disables, each with how many events came before it
    owl = SimpleNamespace(enable=lambda: rec.owl.append(("on", len(rec.events))),
                          disable=lambda: rec.owl.append(("off", len(rec.events))))
    locator = SimpleNamespace(owl=owl, locate=lambda s: (rec.events.append("scene"), seen[scene_after])[1])
    args = SimpleNamespace(execute=True, press_only=True)
    code = reach_press.run_reach_press(None, ctx, cfg, runner, SimpleNamespace(), run_args or args, fix, locator)
    return code, rec, chain, home


def test_the_reach_out_press_flies_in_presses_unloads_judges_and_only_then_goes_home(ctx, monkeypatch, tmp_path):
    code, rec, chain, home = _flow(ctx, monkeypatch, tmp_path)
    assert code == 0
    assert rec.events == [
        "reach:look", "aim:reach-aim", "reach:re-aim", "reach:slide-in", "press:down", "press:push",
        "reach:unload", "reach:rise", "reach:slide-out", "aim:reach-after", "scene", "home",
    ]
    assert home["pad"] > 0.0  # contact happened: home pads the box for a shove
    # each guard is judged against the free-air reading it must be: never a loaded first sample
    L = rec.legs
    assert L["reach:re-aim"].guard.reference == AT_LOOK and L["reach:re-aim"].guard.touch_nm == reach_press.DRAG_NM
    assert L["reach:slide-in"].guard.reference == AT_LOOK2 and L["reach:slide-in"].guard.hang is None
    assert L["press:down"].guard.reference == AT_START
    for name in ("reach:unload", "reach:rise", "reach:slide-out"):
        g = L[name].guard
        assert g.reference == AT_START, name  # the same pose, read before the touch
        assert g.gravity is not None and g.hang == (_Gravity.K, reach_press.HANG_N), name
    assert L["reach:unload"].guard.touch_nm == ctx.model.touch_nm  # starts on the button: the backstop only
    assert L["reach:slide-out"].guard.touch_nm == reach_press.DRAG_NM
    # the push's backstop bounds the TOTAL off the free-air reading at the
    # stroke's start, not what it adds to the touch's loaded first sample
    gp = L["press:push"].guard
    assert gp.trip == "press" and gp.touch_nm == ctx.model.touch_nm
    assert gp.gravity is not None and gp.reference == AT_START
    assert L["press:push"].recoil_via is L["press:down"].traj  # the recoil carries on up the touch
    # the push flown is the lid-clear depth, not button_travel_m (the fake fk stands at z 0.45)
    cfg = load_press_demo(PINK)
    depth = reach_press.reach_push_m(cfg, reach_press.REACH_TILTS_DEG[0])
    assert depth < cfg.button_travel_m
    assert any(xyz[2] == pytest.approx(0.45 - depth) for xyz, _q in chain.lines)
    assert not any(xyz[2] == pytest.approx(0.45 - cfg.button_travel_m) for xyz, _q in chain.lines)
    # the first stop on the way up is UNLOAD_CHECK_M above where the arm stood
    assert any(xyz[2] == pytest.approx(0.45 + reach_press.UNLOAD_CHECK_M) for xyz, _q in chain.lines)
    # every world from the aim on holds the box where the aim put it
    assert _reach_world_button(ctx) == pytest.approx(list(BUTTON))
    # toward the box, or loaded, at CONTACT_SPEED; up and away after an at-rest check, faster
    from rammp_box_opening.constants import CONTACT_SPEED, FREE_LINE_SPEED

    assert {n: L[n].speed for n in L if n.startswith(("reach:", "press:")) and n != "reach:look"} == {
        "reach:re-aim": CONTACT_SPEED, "reach:slide-in": CONTACT_SPEED, "press:down": cfg.press_speed,
        "press:push": cfg.press_speed, "reach:unload": CONTACT_SPEED, "reach:rise": FREE_LINE_SPEED,
        "reach:slide-out": FREE_LINE_SPEED,
    }
    assert CONTACT_SPEED < FREE_LINE_SPEED < cfg.press_speed
    # the slide and the stroke reach_from_here solved are flown as solved: the arm stood at their starts
    quat = reach_press.reach_quat(ctx.model, BUTTON[:2], reach_press.REACH_TILTS_DEG[0])
    start = reach_press.tool_for_contact((BUTTON[0], BUTTON[1], BUTTON[2] + reach_press.REACH_START_M), quat)
    bottom = reach_press.tool_for_contact((BUTTON[0], BUTTON[1], BUTTON[2] - cfg.travel_m), quat)
    # plan_reach's and reach_from_here's (and the rise's, back up to the start), none from the live joints
    assert sum(x == pytest.approx(start) for x, _q in chain.lines) == 3
    assert sum(x == pytest.approx(bottom) for x, _q in chain.lines) == 2


def test_a_solved_line_is_flown_as_solved_only_from_where_it_starts(ctx, monkeypatch):
    from rammp_box_opening.runtime import approach

    chain = _Chain()
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    pts = [[0.1] * 7, [0.2] * 7]
    ctx.client.live = [0.1 + 0.0009] + [0.1] * 6
    assert reach_press._line_from(ctx, pts, [0.9, 0.0, 0.3], [0.0, 0.0, 0.0, 1.0]) == (pts, None) and chain.lines == []
    ctx.client.live = [0.1 + 0.002] + [0.1] * 6  # drifted: solved again from where the arm is
    got, why = reach_press._line_from(ctx, pts, [0.9, 0.0, 0.3], [0.0, 0.0, 0.0, 1.0])
    assert got[0] == ctx.client.live and chain.lines == [([0.9, 0.0, 0.3], [0.0, 0.0, 0.0, 1.0])]


def test_the_scene_owl_is_enabled_before_the_post_press_wrist_aim(ctx, monkeypatch, tmp_path):
    # its first bbox (~1.2 s) comes while the wrist re-aims, not after it
    code, rec, _c, _h = _flow(ctx, monkeypatch, tmp_path)
    assert code == 0
    assert rec.events == [
        "reach:look", "aim:reach-aim", "reach:re-aim", "reach:slide-in", "press:down", "press:push",
        "reach:unload", "reach:rise", "reach:slide-out", "aim:reach-after", "scene", "home",
    ]
    ons = [i for word, i in rec.owl if word == "on"]
    assert ons and min(ons) <= rec.events.index("aim:reach-after")


def test_an_aim_that_raises_after_the_early_enable_disables_the_scene_owl(ctx, monkeypatch):
    # the enable is latched: a Ctrl-C in the re-aim must not leave the scene OWL inferring
    ctx.model = ContainerModel.load(PINK)
    calls = []
    owl = SimpleNamespace(enable=lambda: calls.append("on"), disable=lambda: calls.append("off"))
    locator = SimpleNamespace(owl=owl, locate=lambda s: calls.append("scene"))

    def aim(node, watcher, yaw, lid_z, prior_xy, runner_, tag):
        assert tag == "reach-after"
        raise KeyboardInterrupt

    monkeypatch.setattr(reach_press, "_aim", aim)
    fix = SimpleNamespace(top_xyz=BUTTON)
    with pytest.raises(KeyboardInterrupt):
        reach_press._after_check(None, ctx, SimpleNamespace(), SimpleNamespace(), fix, locator, BUTTON, None, 0.0)
    assert calls == ["on", "off"]


def _reach_world_button(ctx):
    from rammp_box_opening.models.container import from_container

    kw = [kw for kind, kw in ctx.worlds.pushes if kind == "interaction" and kw.get("tag") == "reach"][-1]
    return from_container(kw["cpose"], ctx.model.button_offset)


def test_the_world_follows_the_aimed_button_not_the_scene_cameras(ctx, monkeypatch, tmp_path):
    aimed = (BUTTON[0] + 0.03, BUTTON[1], BUTTON[2] - 0.02)
    code, rec, _c, home = _flow(ctx, monkeypatch, tmp_path, first=(aimed, 0.4), after=((aimed[0], aimed[1], aimed[2] + 0.012), 12.0),
                                contact=aimed)
    assert _reach_world_button(ctx) == pytest.approx(list(aimed))
    assert code == 0


def test_when_no_straight_line_reaches_the_aimed_look_pose_the_planner_plans_it(ctx, monkeypatch, tmp_path):
    monkeypatch.setattr(reach_press, "reach_from_here", lambda *a, **k: (None, None, "IK did not converge"))
    code, rec, _c, home = _flow(ctx, monkeypatch, tmp_path)
    assert code == 0 and rec.events[:4] == ["reach:look", "aim:reach-aim", "reach:look", "reach:slide-in"]
    from rammp_box_opening.models.container import from_container

    assert from_container(home["look_cposes"][-1], ctx.model.button_offset) == pytest.approx(list(BUTTON))


def test_a_home_refused_from_inside_the_boxs_padded_volume_backs_off_toward_the_arm_first(ctx, monkeypatch, tmp_path,
                                                                                          capsys):
    """2026-10-22: the scene camera's box 85 mm off; the aim moved the box
    under the arm and every plan from the look pose was INVALID_START. The
    back-off says what the planner refused: INVALID_START_STATE_JOINT_LIMITS
    takes it too, and that start is not inside the box's padded volume."""
    code, rec, chain, _h = _flow(ctx, monkeypatch, tmp_path, home_refusals=1,
                                 settled=[AT_REST, ([3.0, 0.0, 0.0, 0.0], Q)])  # a gravity mismatch: home from the look pose
    assert code == 1 and rec.events == ["reach:look", "reach:back-off", "home"]
    said = capsys.readouterr().out
    assert ("the planner refused home from here (MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION: the start) — "
            "backing 15 cm straight off toward the arm first") in said
    assert "inside the box's padded volume" not in said
    back = rec.legs["reach:back-off"]
    assert back.plan_free and back.guard.trip == "obstruction" and back.guard.gravity is not None
    from rammp_box_opening.constants import FREE_LINE_SPEED

    assert back.speed == FREE_LINE_SPEED and back.guard.touch_nm == reach_press.DRAG_NM and back.guard.hang is None
    # straight back toward the arm, level: from the fake fk's (0.5, -0.4, 0.45), against the box's bearing
    xyz, _q = chain.lines[-1]
    b = np.asarray(ctx.cpose.xyz[:2]) / np.hypot(*ctx.cpose.xyz[:2])
    assert np.subtract([0.5, -0.4], xyz[:2]) == pytest.approx(reach_press.REACH_PULLBACK_M * b) and xyz[2] == pytest.approx(0.45)


def test_a_home_refused_for_a_joint_on_its_limit_steps_off_it_first(ctx, tmp_path, capsys):
    """Flatbot 2026-10-27 14:33: the depth press ended over the box with
    arm_joint_6 0.043 rad from its limit, home was refused
    START_IN_LIMIT_MARGIN, and the arm held until home_arm stepped it off.
    The reach-out's own home steps off first, as home_arm and try_home do:
    plan-free along the joint line toward the rest pose, the shortest step
    home plans from, and home planned from where it ends."""
    from conftest import runner as make_runner

    from rammp_box_opening.constants import PLANNER_LIMIT_MARGIN_RAD
    from rammp_box_opening.primitives.core import CLEAR_MARGIN_FIRST_RAD
    from rammp_box_opening.runtime import approach
    from rammp_box_opening.tasks import press_demo

    cfg = load_press_demo(PINK)
    ctx.press_cfg = cfg
    rest = press_demo.rest_joints(cfg)
    lo, hi = approach._chain().limits[5]
    start = list(rest)
    start[5] = hi - 0.043  # arm_joint_6 where 14:33 left it
    ctx.client.live = list(start)

    class InMargin:
        success = False
        message = "START_IN_LIMIT_MARGIN: start: arm_joint_6 is 0.043 rad from its position limit"
        trajectory = None

    real = ctx.client.plan_to_joints

    def plan(q7, start_joints):  # refuses any start inside its margin, as rammp-curobo does
        if min(start_joints[5] - lo, hi - start_joints[5]) < PLANNER_LIMIT_MARGIN_RAD:
            return InMargin()
        return real(q7, start_joints)

    ctx.client.plan_to_joints = plan
    assert reach_press._go_home(ctx, cfg, make_runner(ctx.client, tmp_path), "STOP: test") is True
    said = capsys.readouterr().out
    assert "joint_6 is 0.043 rad from its limit" in said and "a plan-free step" in said and "holds" not in said
    # nothing planned from the refused start: home from the step's end, just clear of the margin
    end = ctx.client.joint_starts[0]
    assert hi - end[5] == pytest.approx(CLEAR_MARGIN_FIRST_RAD, abs=1e-6)
    assert len(ctx.client.executed) == 2 and ctx.client.live == pytest.approx(rest)


def test_an_open_box_is_not_reached_for(ctx, monkeypatch, tmp_path):
    code, rec, _c, _h = _flow(ctx, monkeypatch, tmp_path, first=(POPPED, 14.0))
    assert code == 1 and "reach:slide-in" not in rec.events and rec.events[-1] == "home"


def test_a_press_that_does_not_pop_the_knob_is_a_failure(ctx, monkeypatch, tmp_path):
    code, rec, _c, _h = _flow(ctx, monkeypatch, tmp_path, after=(BUTTON, 0.5))
    assert code == 1 and rec.events[-1] == "home"


def test_when_neither_camera_finds_the_box_after_the_press_the_arm_holds(ctx, monkeypatch, tmp_path):
    code, rec, _c, _h = _flow(ctx, monkeypatch, tmp_path, after=None, scene_after="none")
    assert code == 1 and "home" not in rec.events and rec.events[-1] == "scene"


def test_a_moved_box_stops_the_run_and_home_is_planned_around_where_it_now_is(ctx, monkeypatch, tmp_path):
    moved = (POPPED[0] + 0.012, POPPED[1], POPPED[2])
    code, rec, _c, home = _flow(ctx, monkeypatch, tmp_path, after=(moved, 12.0))
    assert code == 1 and rec.events.index("aim:reach-after") < rec.events.index("home")
    from rammp_box_opening.models.container import from_container

    # the popped knob's top is not the box's height: it is placed by its xy, at the button's height
    assert from_container(home["cpose"], ctx.model.button_offset) == pytest.approx([moved[0], moved[1], BUTTON[2]])


def test_a_box_the_scene_camera_saw_move_stops_the_run_even_when_the_wrist_sees_it_in_place(ctx, monkeypatch, tmp_path):
    code, rec, _c, _h = _flow(ctx, monkeypatch, tmp_path, scene_after="moved")
    assert code == 1 and rec.events[-1] == "home"


def test_home_is_planned_around_the_scene_cameras_box_when_the_wrist_cannot_find_it(ctx, monkeypatch, tmp_path):
    code, rec, _c, home = _flow(ctx, monkeypatch, tmp_path, after=None, scene_after="moved")
    assert code == 1 and home["cpose"].xyz[0] == pytest.approx(BUTTON[0] + 0.05)


def test_a_lid_that_came_up_or_went_down_is_a_moved_box(ctx, monkeypatch, tmp_path):
    tipped = (POPPED[0], POPPED[1], POPPED[2] + 0.015)  # the knob reads 12 mm up, its lid 15 mm higher
    code, _rec, _c, _h = _flow(ctx, monkeypatch, tmp_path, after=(tipped, 12.0))
    assert code == 1


def test_a_touch_that_finds_no_surface_backs_out_before_going_home(ctx, monkeypatch, tmp_path):
    code, rec, _c, _h = _flow(ctx, monkeypatch, tmp_path, touch=("arrived", {"message": "ok", "progress": 1.0}))
    assert code == 1 and "press:push" not in rec.events
    assert rec.events[-4:] == ["reach:unload", "reach:rise", "reach:slide-out", "home"]  # nothing touched: no after-check


def test_a_foreign_surface_is_not_pressed_and_the_box_is_judged_before_home(ctx, monkeypatch, tmp_path):
    high = (BUTTON[0], BUTTON[1], BUTTON[2] + 0.040)
    code, rec, _c, _h = _flow(ctx, monkeypatch, tmp_path, contact=high, after=(BUTTON, 0.4))
    assert code == 1 and "press:push" not in rec.events
    assert rec.events[-6:] == ["reach:unload", "reach:rise", "reach:slide-out", "aim:reach-after", "scene", "home"]


def test_a_box_latched_on_the_finger_is_lifted_no_more_than_the_first_stop(ctx, monkeypatch, tmp_path):
    k = _Gravity.K
    hanging = ([0.03 + 3.0 * k[0], 0.0, 3.0 * k[2], 0.0], Q)  # 3 N pulling down at the fingertip
    code, rec, _c, _h = _flow(ctx, monkeypatch, tmp_path, settled=FREE[:4] + [hanging])
    assert code == 1 and rec.events[-1] == "reach:unload"  # stopped 15 mm up: no rise, no slide out, no home


def test_anything_left_on_the_hand_at_the_strokes_start_holds_the_arm(ctx, monkeypatch, tmp_path):
    # no pull at all, but joint_6 is 1 Nm off its free-air reading
    off = ([0.03, 0.0, 0.0, 1.0], Q)
    code, rec, _c, _h = _flow(ctx, monkeypatch, tmp_path, settled=FREE[:5] + [off])
    assert code == 1 and rec.events[-1] == "reach:rise"


def test_a_gravity_model_that_does_not_fit_the_arm_stops_before_any_guarded_line(ctx, monkeypatch, tmp_path):
    code, rec, _c, _h = _flow(ctx, monkeypatch, tmp_path, settled=[AT_REST, ([3.0, 0.0, 0.0, 0.0], Q)])
    assert code == 1 and rec.events == ["reach:look", "home"]


@pytest.mark.parametrize("soft", [False, True])
def test_without_the_oblique_aim_nothing_moves(ctx, monkeypatch, tmp_path, soft):
    """Nor, in a compliant run, is the joint impedance primed: this refusal
    comes first, and its "nothing moved" stays true."""
    from rammp_box_opening.models.container import ContainerPose
    from rammp_box_opening.perception import button_aim
    from rammp_box_opening.runtime.runner import Runner

    class OldAimer:  # the straight-down aim's constructor
        def __init__(self, grab, model, table_z):
            pass

    monkeypatch.setattr(button_aim, "ButtonAimer", OldAimer)
    assert not reach_press.oblique_aim_available()
    cfg = load_press_demo(PINK)
    ctx.model = ContainerModel.load(PINK)
    fix = SimpleNamespace(pose=ContainerPose(xyz=(BUTTON[0], BUTTON[1], BUTTON[2] - 0.121), yaw=0.0), top_xyz=BUTTON)
    args = SimpleNamespace(execute=True, press_only=True, compliant=soft, depth_press=soft)
    code = reach_press.run_reach_press(None, ctx, cfg, Runner(ctx.client, log_dir=tmp_path), None, args, fix, None)
    assert code == 1 and not ctx.client.executed and ctx.client.primes == []


@pytest.mark.parametrize("soft", [False, True])
def test_the_reach_out_preview_moves_nothing(ctx, monkeypatch, tmp_path, capsys, soft):
    """Guarded, or compliant with its press proven: the preview flies
    nothing, and primes no joint impedance either."""
    from rammp_box_opening.models.container import ContainerPose
    from rammp_box_opening.runtime import approach
    from rammp_box_opening.runtime.runner import Runner

    cfg = load_press_demo(PINK)
    ctx.model = ContainerModel.load(PINK)
    monkeypatch.setattr(approach, "_chain", lambda: _Chain())
    monkeypatch.setattr(approach, "_gravity", lambda: _Gravity())
    monkeypatch.setattr(reach_press, "oblique_aim_available", lambda: True)
    monkeypatch.setattr(reach_press, "_plan_motion", lambda c, s, n, t, w, sp: (SimpleNamespace(
        goal_joints=[0.0] * 7, name=n, traj=None, speed=sp, world="full", kind=None, guard=None, warp=None,
        plan_s=0.0, plan_server_s=None, waypoint=None, target=t, verify=None), s))
    monkeypatch.setattr(Runner, "preview", lambda self, legs: "preview")
    if soft:
        monkeypatch.setattr(reach_press, "way_down", _way_down([], solves=lambda tilt, tip: True))
    fix = SimpleNamespace(pose=ContainerPose(xyz=(BUTTON[0], BUTTON[1], BUTTON[2] - 0.121), yaw=0.0), top_xyz=BUTTON)
    args = SimpleNamespace(execute=False, press_only=True, compliant=soft, depth_press=soft)
    code = reach_press.run_reach_press(None, ctx, cfg, Runner(ctx.client, log_dir=tmp_path), None, args, fix, None)
    assert code == 0 and not ctx.client.executed and ctx.client.primes == []
    out = capsys.readouterr().out
    assert "gravity alone moves joint_4 0.00 Nm along the slide" in out
    assert ("COMPLIANT: every move" in out) is soft and "enters its joint impedance" not in out


def test_a_compliant_run_says_which_gains_every_move_flies(ctx, monkeypatch, capsys):
    """Owner 2026-10-27: "impedance mode needs to be firmer". The banner
    names the tuning in force — the client's compliant_gains — and the
    client is back in position once the run is over; a guarded run has no
    banner."""
    from rammp_box_opening.constants import GAINS_MEDIUM

    during = []
    monkeypatch.setattr(reach_press, "_reach_press", lambda *a: during.append(ctx.client.compliant) or 0)
    args = SimpleNamespace(compliant=True, depth_press=True, execute=True)
    assert reach_press.run_reach_press(None, ctx, None, None, None, args, None, None) == 0
    out = capsys.readouterr().out
    assert ("COMPLIANT: every move in the arm's joint impedance, STIFF (Kq 160/60 N·m/rad on joints 1-4/5-7), "
            "and nothing the run reads stops it") in out
    ctx.client.compliant_gains = GAINS_MEDIUM
    reach_press.run_reach_press(None, ctx, None, None, None, args, None, None)
    assert "impedance, MEDIUM (Kq 80/30 N·m/rad on joints 1-4/5-7), and" in capsys.readouterr().out
    assert during == [True, True] and not ctx.client.compliant
    reach_press.run_reach_press(None, ctx, None, None, None, SimpleNamespace(compliant=False, depth_press=True),
                                None, None)
    assert "COMPLIANT" not in capsys.readouterr().out and during[-1] is False


def test_a_stop_says_what_stopped_the_leg_not_always_something_met():
    """15:34 2026-10-27: the re-aim's guard lost its stream and the STOP
    line said the move met something; nothing had been touched."""
    from rammp_box_opening.runtime.runner import LegResult

    lost = LegResult("reach:re-aim", "failed", False, "effort stream lost during guarded leg — cancelled")
    assert reach_press._stopped_by([lost]) == "stopped (effort stream lost during guarded leg — cancelled)"
    assert reach_press._stopped_by([LegResult("reach:re-aim", "touch", False, "torque guard trip")]) == "met something"


# ---- the compliant depth press, proven before the arm moves toward the box ------------------

def _depth_press_flow(ctx, monkeypatch, tmp_path, way_down, compliant=True, **kw):
    """_flow with --depth-press — compliant, as press_demo runs by default
    since 2026-10-27, unless `compliant` is False (--guarded) — the way down
    `way_down`'s, the press itself faked: the guarded press, and a compliant
    one's hold at the end of its one-execution way in, which judges it at
    the aimed button the box was put at (ctx.cpose). Returns _flow's, and
    what the press was handed ({} when it was not called)."""
    from rammp_box_opening.models.container import from_container
    from rammp_box_opening.tasks import depth_press as dp

    got = {}

    def press(node, ctx_, cfg, runner, watcher, contact_tool, here=None, turned=None, knob_up_mm=None, aimed=None,
              compliant=False, top_z=None):
        got.update(aimed=aimed, top_z=top_z, compliant=compliant)
        return dp.DepthPress(free=True, contact=True, felt=True, judged=False, press_n=14.0)

    def hold(ctx_, cfg_, runner_, stroke, top_z, seen):
        got.update(aimed=from_container(ctx_.cpose, ctx_.model.button_offset), top_z=top_z, compliant=True)
        return dp.DepthPress(free=True, contact=True, felt=True, judged=False, press_n=14.0)

    monkeypatch.setattr(dp, "press_down_from_here", press)
    monkeypatch.setattr(dp, "compliant_hold", hold)
    monkeypatch.setattr(reach_press, "way_down", way_down)
    kw.setdefault("touch", ("arrived", {"message": "ok", "progress": 1.0}))  # no TF stroke: the 4th motion is the clear
    args = SimpleNamespace(execute=True, press_only=True, depth_press=True, compliant=compliant)
    return (*_flow(ctx, monkeypatch, tmp_path, run_args=args, **kw), got)


def test_a_scene_button_no_tilt_presses_even_30_mm_off_stops_before_anything_moves(ctx, monkeypatch, tmp_path, capsys):
    """16:23 2026-10-27 flew the look, the re-aim and a 7.4 s slide in
    before it asked for the turn down, which no IK reaches there. Now the
    proof refuses at the rest pose: every tilt is named, then the button
    PROOF_SHIFT_M nearer the arm and to either side, at the smallest
    tilt's look, and nothing is sent to the arm."""
    asked = []
    code, rec, _chain, _h, got = _depth_press_flow(ctx, monkeypatch, tmp_path, _way_down(asked))
    assert code == 1 and rec.events == [] and ctx.client.executed == [] and got == {}
    # the joint impedance was primed while the proof was planned: a hold
    # where the arm stands, which the run leaves it in
    assert ctx.client.primes == [[0.0] * 7]
    out = capsys.readouterr().out
    assert ("STOP: no reach-out press for the scene camera's box — no tilt turns the hand down and strokes there, "
            "nor with the button 30 mm nearer the arm or to either side (30 deg: the way down (walk in: ") in out
    assert out.rstrip().endswith("; nothing moved")
    for tilt in reach_press.REACH_TILTS_DEG:
        assert "%d deg: the way down (walk in: " % tilt in out
    for name in reach_press.SHIFTED:
        assert "30 mm %s: the way down (walk in: " % name in out
    assert [t for t, _xy, _z in asked] == list(reach_press.REACH_TILTS_DEG) + [30] * 3
    u = np.asarray(BUTTON[:2]) / np.hypot(*BUTTON[:2])
    across = np.array([-u[1], u[0]])  # to its left, seen from the arm
    for (_t, xy, z), want in zip(asked[4:], (BUTTON[:2] - 0.03 * u, BUTTON[:2] + 0.03 * across,
                                             BUTTON[:2] - 0.03 * across)):
        assert xy == pytest.approx(list(want), abs=1e-9) and z == BUTTON[2]


def test_a_scene_button_the_press_reaches_30_mm_off_flies_the_look_and_the_aimed_button_decides(
        ctx, monkeypatch, tmp_path, capsys):
    """As on flatbot's URDF for a scene button at r 0.766 m, az -15 deg
    (test_on_flatbot_a_scene_button_out_of_reach_by_a_little_still_flies_the_look):
    no tilt presses it, 30 mm nearer the arm the press reaches — and the
    aim, 9 mm nearer and 21 mm aside, too. The look flies, and the aimed
    button's own way down, solved after the aim, takes the press in — the
    re-aim, the slide in, the turn and the stroke as one execution."""
    import json

    from rammp_box_opening.tasks import depth_press as dp

    u = np.asarray(BUTTON[:2]) / np.hypot(*BUTTON[:2])
    aimed = (BUTTON[0] - 0.02 * u[0], BUTTON[1] - 0.02 * u[1], BUTTON[2])
    off_scene = lambda tilt, tip: np.hypot(tip[0] - BUTTON[0], tip[1] - BUTTON[1]) > 0.015  # noqa: E731
    asked = []
    code, rec, _chain, _h, got = _depth_press_flow(ctx, monkeypatch, tmp_path, _way_down(asked, off_scene),
                                                   first=(aimed, 0.4))
    assert code == 0
    way_in = ["reach:re-aim", "reach:slide-in", "depth:turn", "depth:press"]
    assert rec.events == ["reach:look", "aim:reach-aim", *way_in, "depth:clear", "home"]
    assert (way_in, [dp.COMPLIANT_GAINS]) in rec.runs
    assert got["aimed"] == pytest.approx(aimed) and got["top_z"] == pytest.approx(BUTTON[2]) and got["compliant"]
    out = capsys.readouterr().out
    assert ("the press does not reach the scene camera's button itself (30 deg: the way down (walk in: ") in out
    assert "but at 30 deg it does with the button 30 mm nearer the arm — inside what the aim corrects" in out
    # the scene button at each tilt, the first shifted one, then the aimed button after the aim
    assert [t for t, _xy, _z in asked] == list(reach_press.REACH_TILTS_DEG) + [30, 30]
    assert asked[-1][1] == pytest.approx(list(aimed[:2]), abs=1e-9)
    rows = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl")))]
    plan, = [r for r in rows if r["leg"] == "reach_plan"]
    assert plan["proof"] == "shifted" and plan["down"] is None and plan["tilt_deg"] == 30
    way, = [r for r in rows if r["leg"] == "reach_way_in"]
    assert (way["tilt_deg"], way["walked"], way["short_deg"], way["k"]) == (30, False, 0.0, 440.0)
    assert way["end_z"] == pytest.approx(BUTTON[2] - 0.03)


def test_where_no_straight_line_reaches_the_aimed_look_pose_the_planned_one_brings_its_own_proof(
        ctx, monkeypatch, tmp_path):
    """The planner plans the move to the aimed button's look pose where no
    straight line flies there, as before; plan_reach proves the way down
    from that leg's end, which is where the arm then stands — no second
    solve, no re-aim after it: from there the slide in, the turn and the
    stroke fly as one execution."""
    import json

    from rammp_box_opening.tasks import depth_press as dp

    asked = []
    chain = _Chain(refuse=[lambda xyz, quat: xyz[2] > 0.52])  # the move to the look pose (0.54), not the slide (0.47)
    code, rec, _c, _h, got = _depth_press_flow(ctx, monkeypatch, tmp_path, _way_down(asked, lambda tilt, tip: True),
                                               chain=chain)
    assert code == 0 and got["aimed"] == pytest.approx(BUTTON)
    way_in = ["reach:slide-in", "depth:turn", "depth:press"]
    assert rec.events == ["reach:look", "aim:reach-aim", "reach:look", *way_in, "depth:clear", "home"]
    assert (way_in, [dp.COMPLIANT_GAINS]) in rec.runs
    assert [t for t, _xy, _z in asked] == [30, 30]  # the scene button's, then the planned look's for the aimed one
    rows = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl")))]
    way, = [r for r in rows if r["leg"] == "reach_way_in"]
    assert way["tilt_deg"] == 30 and way["walked"] is False


def test_a_planned_look_the_arm_stands_off_flies_its_way_in_leg_by_leg(ctx, monkeypatch, tmp_path, capsys):
    """After the planner's look leg the way in starts at that leg's planned
    end. An arrival a few mrad off it (impedance) is not jumped from: the
    way in flies leg by leg, each line solved again from where the arm
    stands, as before the one execution — the slide in from the live
    joints, then the press, which turns and strokes from there."""
    chain = _Chain(refuse=[lambda xyz, quat: xyz[2] > 0.52])  # the move to the look pose (0.54), not the slide (0.47)
    real = ctx.client.execute

    def execute(traj, speed, guard=None, while_running=None, stop_when=None, gains=None):
        out = real(traj, speed, guard=guard, while_running=while_running, stop_when=stop_when, gains=gains)
        if len(ctx.client.executed) == 2:  # the planned look for the aimed button: 5 mrad short on joint_2
            ctx.client.live[1] -= 0.005
        return out

    monkeypatch.setattr(ctx.client, "execute", execute)
    code, rec, _c, _h, got = _depth_press_flow(ctx, monkeypatch, tmp_path, _way_down([], lambda tilt, tip: True),
                                               chain=chain)
    assert code == 0 and got["compliant"] and got["aimed"] == pytest.approx(BUTTON)  # the press, handed the aim
    assert rec.events == ["reach:look", "aim:reach-aim", "reach:look", "reach:slide-in", "depth:clear", "home"]
    assert "the way in does not fly as one execution (the arm stands 0.005 rad from where reach:slide-in starts)" in (
        capsys.readouterr().out)
    slide = rec.legs["reach:slide-in"]
    assert list(slide.traj.points[0].positions) == pytest.approx([0.05, 0.045] + [0.05] * 5)  # from the live joints
    assert (["reach:slide-in"], [None]) in rec.runs and slide.gains is None  # a goal of its own, as before


def test_where_neither_a_line_nor_a_planned_look_brings_the_aimed_button_a_way_down_the_arm_goes_home_from_the_look(
        ctx, monkeypatch, tmp_path, capsys):
    """No straight line reaches the aimed button's look pose, and at no
    tilt the planner offers does the hand turn down and stroke there: the
    arm goes home from the look pose, in free air. No planned look leg for
    the aimed button flies, no slide in, nothing pressed; the console says
    why at every tilt."""
    import json

    asked = []
    scene_only = lambda tilt, tip: len(asked) == 1  # noqa: E731 - plan_reach's, for the scene camera's button
    chain = _Chain(refuse=[lambda xyz, quat: xyz[2] > 0.52])  # the move to the look pose (0.54), not the slide (0.47)
    code, rec, _c, _h, got = _depth_press_flow(ctx, monkeypatch, tmp_path, _way_down(asked, scene_only), chain=chain)
    assert code == 1 and got == {}
    assert rec.events == ["reach:look", "aim:reach-aim", "home"]
    assert [t for t, _xy, _z in asked] == [30] + list(reach_press.REACH_TILTS_DEG)  # the scene's, then each planned
    out = capsys.readouterr().out
    assert ("STOP: no press reaches the aimed button (as straight lines: the move to its look pose (IK did not "
            "converge); planned: 30 deg: the way down (walk in: ") in out
    assert all("%d deg: the way down (walk in: " % t in out for t in reach_press.REACH_TILTS_DEG)
    assert "— nothing was pressed — returning home" in out
    rows = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl")))]
    assert not [r for r in rows if r["leg"] in ("reach_way_in", "home_refused")]


@pytest.mark.parametrize("depth", [False, True])
def test_a_guarded_reach_out_asks_for_no_way_down(ctx, monkeypatch, tmp_path, depth):
    """--guarded, with or without the depth press, is planned as before:
    no way down is asked for, and the plan's note says it was not proven."""
    import json

    asked = []
    if depth:
        code, rec, _chain, _h, got = _depth_press_flow(ctx, monkeypatch, tmp_path, _way_down(asked), compliant=False)
        assert got["aimed"] == pytest.approx(BUTTON) and not got["compliant"]
    else:
        monkeypatch.setattr(reach_press, "way_down", _way_down(asked))
        code, rec, _chain, _h = _flow(ctx, monkeypatch, tmp_path)
        assert "press:down" in rec.events
    assert code == 0 and asked == [] and rec.events[:4] == ["reach:look", "aim:reach-aim", "reach:re-aim",
                                                            "reach:slide-in"]
    rows = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl")))]
    plan, = [r for r in rows if r["leg"] == "reach_plan"]
    assert plan["proof"] is None and plan["down"] is None
    assert not [r for r in rows if r["leg"] == "reach_way_in"]


# Flatbot's URDF, the scene camera's buttons of 2026-10-27 (run records' scene_fix top_xyz)
SCENE_1623 = (0.7248, -0.2860, 0.3125)
SCENE_1519 = (0.6754, -0.1812, 0.3111)


def _on_flatbot(ctx, monkeypatch):
    """Flatbot's own kinematics and the pink canister, each look leg
    'planned' to the joints client._plan_pose_near hands the planner. Pure
    kinematics: no planner, no arm."""
    from rammp_box_opening.constants import HOME
    from rammp_box_opening.primitives.look import look_joints
    from rammp_box_opening.runtime import approach
    from rammp_box_opening.runtime.branches import planner_branch
    from rammp_box_opening.runtime.kinematics import ArmChain

    chain = ArmChain(path="src/rammp_box_opening/test/data/arm_flatbot_20261021.urdf", root="arm_mount")
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    ctx.model = ContainerModel.load(PINK)

    def plan(ctx_, state, name, target, world, speed):
        _kind, look, quat = target
        q = chain.nearest_solution(planner_branch(HOME), look, quat, seeds=(look_joints(HOME),))
        if q is None:
            raise RuntimeError("planning failed for %s: MotionGenStatus.IK_FAIL" % name)
        return SimpleNamespace(goal_joints=[float(v) for v in q]), state

    monkeypatch.setattr(reach_press, "_plan_motion", plan)
    return chain, load_press_demo(PINK), list(HOME)


def test_on_flatbot_1623s_scene_button_is_refused_before_the_look(ctx, monkeypatch):
    """16:23 2026-10-27, replayed with the leans its turn tried then
    (TURN_SHORT_DEG, the guarded press's): the plan without the proof takes
    the tilted stroke at 30 deg, as the run did — 20.2 s from the look to
    home with nothing pressed, no straight-down hand at the slide's end. At
    every tilt the look pose and the slide solve, and neither the walk in
    nor the turn after the slide turns the hand down: refused, and 30 mm
    nearer the arm or to either side as well."""
    from rammp_box_opening.tasks import depth_press as dp

    monkeypatch.setattr(dp, "COMPLIANT_TURN_SHORT_DEG", dp.TURN_SHORT_DEG)
    _chain, cfg, home = _on_flatbot(ctx, monkeypatch)
    plan, _why = reach_press.plan_reach(ctx, cfg, SCENE_1623, home)
    assert plan is not None and plan.tilt_deg == 30  # what 16:23 flew
    looks = []
    plan, why = reach_press.plan_reach(ctx, cfg, SCENE_1623, home, depth=True, looks=looks)
    assert plan is None
    assert why.count("the way down (walk in: ") == len(reach_press.REACH_TILTS_DEG), why
    assert all("%d deg: the way down (walk in: " % t in why for t in reach_press.REACH_TILTS_DEG)
    assert [lk.tilt_deg for lk in looks] == list(reach_press.REACH_TILTS_DEG)
    near, why = reach_press.shifted_way_down(ctx, cfg, looks[0], SCENE_1623)
    assert near is None and all("30 mm %s: the way down (walk in: " % n in why for n in reach_press.SHIFTED), why


# 15:19's aimed button (the touch met it at 0.312; the clear's replay, test_depth_press)
AIMED_1519 = (0.6617, -0.1972, 0.3111)


def _at(r_az, z=0.3125):
    """A button `r_az` (r m, az deg) from the arm, the lid at 16:23's height."""
    r, az = r_az
    return (r * math.cos(math.radians(az)), r * math.sin(math.radians(az)), z)


def test_on_flatbot_1623s_scene_button_is_proven_turning_short_after_the_slide(ctx, monkeypatch):
    """The compliant press's turn after the slide may stop up to 15 deg
    short of straight down (depth_press.COMPLIANT_TURN_SHORT_DEG): 16:23's
    scene button is proven at the first tilt, the slide in and a turn 10
    deg short, 1.77 rad of joint arc — no walk in reaches it — then the
    stroke from there. 15:19's aimed button, which the guarded leans
    already reached, is as it was: one walk in, exactly down."""
    from rammp_box_opening.tasks import depth_press as dp

    chain, cfg, home = _on_flatbot(ctx, monkeypatch)
    plan, why = reach_press.plan_reach(ctx, cfg, SCENE_1623, home, depth=True)
    assert why is None and plan.tilt_deg == 30 and plan.stroke_pts is None
    assert not plan.down.walked and max(dp.TURN_SHORT_DEG) < plan.down.short_deg <= 15.0
    assert plan.down.short_deg == 10.0  # as modelled before this was flown
    assert plan.down.pts[0] == pytest.approx(plan.slide_pts[-1])  # in place, where the slide ends
    arc = dp.joint_arc(plan.down.pts)
    assert arc <= dp.TURN_MAX_ARC_RAD and arc == pytest.approx(1.77, abs=0.02)
    R, _t = chain.fk(plan.down.pts[-1])
    assert math.degrees(reach_press.hand_lean(np.asarray(R, float))) == pytest.approx(plan.down.short_deg, abs=0.05)
    assert plan.stroke.q0 == pytest.approx([float(v) for v in plan.down.pts[-1]])
    assert [name for name, _pts, _speed in plan.stroke.legs] == ["depth:approach", "depth:press"]
    assert 400.0 <= plan.stroke.k <= 600.0 and plan.stroke.end_z < SCENE_1623[2] - cfg.button_travel_m
    plan, why = reach_press.plan_reach(ctx, cfg, AIMED_1519, home, depth=True)
    assert why is None and plan.tilt_deg == 30 and plan.down.walked and plan.down.short_deg == 0.0
    # a box the guarded leans reached only 30 mm off (r 0.766 m, az -15 deg) is reached as it stands
    plan, why = reach_press.plan_reach(ctx, cfg, _at((0.766, -15.0)), home, depth=True)
    assert why is None and plan.tilt_deg == 30 and not plan.down.walked and plan.down.short_deg == 8.0


def test_on_flatbot_1519s_scene_button_walks_in_exactly_down_at_30_deg(ctx, monkeypatch):
    """15:19 2026-10-27, the button pressed with ~13.5 N: proven at the
    first tilt, one walk in from the look pose to the hand exactly down,
    the stroke from its end past the button's bottom."""
    _chain, cfg, home = _on_flatbot(ctx, monkeypatch)
    plan, why = reach_press.plan_reach(ctx, cfg, SCENE_1519, home, depth=True)
    assert why is None and plan.tilt_deg == 30 and plan.stroke_pts is None
    assert plan.down.walked and plan.down.short_deg == 0.0
    assert plan.stroke.q0 == pytest.approx([float(v) for v in plan.down.pts[-1]])
    assert [name for name, _pts, _speed in plan.stroke.legs] == ["depth:approach", "depth:press"]
    assert plan.stroke.end_z < SCENE_1519[2] - cfg.button_travel_m


@pytest.mark.parametrize("scene, aimed, near, leans", [
    ((0.700, -22.0), (0.692, -23.6), "to its left", "guarded"),
    ((0.766, -15.0), (0.757, -16.6), "nearer the arm", "guarded"),
    ((0.700, -22.0), (0.692, -23.6), "nearer the arm", "compliant"),
])
def test_on_flatbot_a_scene_button_out_of_reach_by_a_little_still_flies_the_look(ctx, monkeypatch, scene, aimed, near,
                                                                                 leans):
    """Two boxes found offline (r m, az deg; the lid at 16:23's 0.3125)
    with the turn after the slide leaning no further than the guarded
    press's TURN_SHORT_DEG: no tilt presses the scene camera's button, but
    one PROOF_SHIFT_M off it the press reaches — and the aim's, 12-23 mm
    off as on 10-27, it does. With the compliant press's own leans the
    first is still such a box, reached 30 mm nearer the arm; the second is
    reached as it stands (test_on_flatbot_1623s_scene_button_is_proven_turning_short_after_the_slide)."""
    from rammp_box_opening.tasks import depth_press as dp

    if leans == "guarded":
        monkeypatch.setattr(dp, "COMPLIANT_TURN_SHORT_DEG", dp.TURN_SHORT_DEG)
    _chain, cfg, home = _on_flatbot(ctx, monkeypatch)
    looks = []
    plan, _why = reach_press.plan_reach(ctx, cfg, _at(scene), home, depth=True, looks=looks)
    assert plan is None and looks
    assert reach_press.shifted_way_down(ctx, cfg, looks[0], _at(scene)) == (near, None)
    way, why, kind = reach_press.solve_way_in(ctx, cfg, _at(aimed), looks[0].tilt_deg, looks[0].look_leg.goal_joints)
    assert way is not None, (why, kind)
    assert way.move and way.down.walked and way.slide is None and way.stroke.legs
