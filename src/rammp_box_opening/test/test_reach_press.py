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


class _Chain:
    """Straight lines that always fly, through two waypoints."""

    def __init__(self, refuse=()):
        self.refuse = refuse
        self.lines = []

    def straight_line(self, q0, xyz, quat):
        self.lines.append((list(xyz), list(quat)))
        for what in self.refuse:
            if what(xyz, quat):
                return None, "IK did not converge"
        return [list(q0), [v + 0.01 for v in q0]], None

    def fk(self, q):
        return np.eye(3), np.array([0.5, -0.4, 0.45])


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
    assert plan.look_xyz[2] == pytest.approx(plan.start_xyz[2])  # ... at the stroke's start height
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


def _flow(ctx, monkeypatch, tmp_path, after=(POPPED, 12.0), first=(BUTTON, 0.4), scene_after="same", settled=None,
          touch=None, contact=BUTTON, home_refusals=0):
    """Run run_reach_press with the arm, the cameras and the planner faked.
    Returns (exit code, the record, the chain, home plan's cpose and pad)."""
    from conftest import leg as make_leg

    from rammp_box_opening.models.container import ContainerPose
    from rammp_box_opening.runtime import approach
    from rammp_box_opening.runtime.runner import Runner

    rec = _Rec()
    cfg = load_press_demo(PINK)
    ctx.model = ContainerModel.load(PINK)
    ctx.press_cfg = cfg
    chain = _Chain()
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    monkeypatch.setattr(approach, "_gravity", lambda: _Gravity())
    monkeypatch.setattr(reach_press, "oblique_aim_available", lambda: True)
    home = {}

    refusals = {"home": home_refusals}

    def plan(ctx_, state, name, target, world, speed):
        if name == "home" and refusals["home"] > 0:
            refusals["home"] -= 1
            raise RuntimeError("planning failed for home: MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION: the start")
        if name == "home":
            home["cpose"], home["pad"] = ctx_.cpose, ctx_.contact_pad
        else:
            home.setdefault("look_cposes", []).append(ctx_.cpose)
        end = [0.0] * 7 if name == "home" else [0.05] * 7
        lg = make_leg(name, start=list(ctx_.client.joints()), end=end, speed=speed, world=world[0])
        lg.target = target
        return lg, state

    monkeypatch.setattr(reach_press, "_plan_motion", plan)

    class R(Runner):
        def run(self, legs, execute, lookahead=None):
            for lg in legs:
                rec.events.append(lg.name)
                rec.legs[lg.name] = lg
            return super().run(legs, execute, lookahead)

    runner = R(ctx.client, log_dir=tmp_path)
    aims = [first, after]

    def aim(node, watcher, yaw, lid_z, prior_xy, runner_, tag):
        rec.events.append("aim:" + tag)
        got = aims.pop(0)
        return (None, "no circle", None) if got is None else ((got[0], 0.0), "3/3 frames", got[1])

    monkeypatch.setattr(reach_press, "_aim", aim)
    seq = list(FREE if settled is None else settled)
    ctx.client.settled_efforts = lambda *a, **k: (seq.pop(0) if seq else AT_START)
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
    locator = SimpleNamespace(locate=lambda s: (rec.events.append("scene"), seen[scene_after])[1])
    args = SimpleNamespace(execute=True, press_only=True)
    code = reach_press.run_reach_press(None, ctx, cfg, runner, SimpleNamespace(), args, fix, locator)
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
    assert L["press:push"].guard.gravity is None  # the push's backstop, as straight down
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
    monkeypatch.setattr(reach_press, "reach_from_here", lambda *a: (None, None, "IK did not converge"))
    code, rec, _c, home = _flow(ctx, monkeypatch, tmp_path)
    assert code == 0 and rec.events[:4] == ["reach:look", "aim:reach-aim", "reach:look", "reach:slide-in"]
    from rammp_box_opening.models.container import from_container

    assert from_container(home["look_cposes"][-1], ctx.model.button_offset) == pytest.approx(list(BUTTON))


def test_a_home_refused_from_inside_the_boxs_padded_volume_backs_off_toward_the_arm_first(ctx, monkeypatch, tmp_path):
    """2026-10-22: the scene camera's box 85 mm off; the aim moved the box
    under the arm and every plan from the look pose was INVALID_START."""
    code, rec, chain, _h = _flow(ctx, monkeypatch, tmp_path, home_refusals=1,
                                 settled=[AT_REST, ([3.0, 0.0, 0.0, 0.0], Q)])  # a gravity mismatch: home from the look pose
    assert code == 1 and rec.events == ["reach:look", "reach:back-off", "home"]
    back = rec.legs["reach:back-off"]
    assert back.plan_free and back.guard.trip == "obstruction" and back.guard.gravity is not None
    # straight back toward the arm, level: from the fake fk's (0.5, -0.4, 0.45), against the box's bearing
    xyz, _q = chain.lines[-1]
    b = np.asarray(ctx.cpose.xyz[:2]) / np.hypot(*ctx.cpose.xyz[:2])
    assert np.subtract([0.5, -0.4], xyz[:2]) == pytest.approx(reach_press.REACH_PULLBACK_M * b) and xyz[2] == pytest.approx(0.45)


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


def test_without_the_oblique_aim_nothing_moves(ctx, monkeypatch, tmp_path):
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
    code = reach_press.run_reach_press(None, ctx, cfg, Runner(ctx.client, log_dir=tmp_path), None,
                                       SimpleNamespace(execute=True, press_only=True), fix, None)
    assert code == 1 and not ctx.client.executed


def test_the_reach_out_preview_moves_nothing(ctx, monkeypatch, tmp_path, capsys):
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
    fix = SimpleNamespace(pose=ContainerPose(xyz=(BUTTON[0], BUTTON[1], BUTTON[2] - 0.121), yaw=0.0), top_xyz=BUTTON)
    code = reach_press.run_reach_press(None, ctx, cfg, Runner(ctx.client, log_dir=tmp_path), None,
                                       SimpleNamespace(execute=False, press_only=True), fix, None)
    assert code == 0 and not ctx.client.executed
    assert "gravity alone moves joint_4 0.00 Nm along the slide" in capsys.readouterr().out
